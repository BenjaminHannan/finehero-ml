# engineer.py - Cleans the merged NYC parking dataset (nc67-uf89 + pvqr FY22-FY25)
# and engineers pre-hearing features:
#   - Temporal: hour, day_of_week, month, is_holiday
#   - Financial: fine_amount
#   - Weather: precipitation, visibility, wind_speed, is_bad_weather (Open-Meteo)
#       (weather_code is fetched only to derive is_bad_weather, not emitted as a feature)
#   - Text: keyword flags from violation_description
#   - Categorical (raw strings, for CatBoost native handling):
#       violation_code, precinct, county, issuing_agency, license_type, state,
#       issuer_code, issuer_command, issuer_squad, street_name,
#       vehicle_make, vehicle_body_type,
#       violation_legal_code, law_section, sub_division,
#       violation_in_front_of_or_opposite, days_parking_in_effect,
#       viol_x_precinct, viol_x_license, hour_x_dow   (cross features)
#   - Plate history (leave-one-out chronological):
#       plate_prior_ticket_count, plate_prior_win_rate
#   - pvqr numerics: feet_from_curb, vehicle_year, from_hour, to_hour, within_posted_hours
#
# CatBoost handles categoricals natively via ordered target statistics, so we
# do NOT ordinal-encode anymore. We save the list of cat_features for train.py.

import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

VIOLATIONS_PATH    = os.path.join(DATA_DIR, "violations_raw.csv")
FEATURES_PATH      = os.path.join(DATA_DIR, "features.csv")
CAT_FEATURES_PATH  = os.path.join(MODELS_DIR, "cat_features.joblib")
PLATE_HISTORY_PATH = os.path.join(MODELS_DIR, "plate_history_map.joblib")
WEATHER_CACHE_PATH = os.path.join(DATA_DIR, "weather_cache.csv")

# Plate leave-one-out smoothing
PLATE_SMOOTH_K = 10

WIN_PATTERN = re.compile(r"DISMISS|NOT GUILTY|NOT LIABLE", re.IGNORECASE)

FIXED_HOLIDAYS = {(1, 1), (7, 4), (11, 11), (12, 24), (12, 25), (12, 31)}

COUNTY_COORDS = {
    "MANHATTAN": (40.7831, -73.9712), "BRONX": (40.8448, -73.8648),
    "BROOKLYN":  (40.6782, -73.9442), "QUEENS": (40.7282, -73.7949),
    "STATEN":    (40.5795, -74.1502),
}
DEFAULT_COORDS = (40.7128, -74.0060)

COUNTY_NORM = {
    "NY": "MANHATTAN", "MN": "MANHATTAN", "NEW YORK": "MANHATTAN",
    "BX": "BRONX", "BRONX": "BRONX",
    "BK": "BROOKLYN", "K": "BROOKLYN", "KINGS": "BROOKLYN", "BROOKLYN": "BROOKLYN",
    "QN": "QUEENS", "Q": "QUEENS", "QNS": "QUEENS", "QUEEN": "QUEENS", "QUEENS": "QUEENS",
    "ST": "STATEN", "R": "STATEN", "RICH": "STATEN", "RICHMOND": "STATEN",
}

# Case-INSENSITIVE keyword patterns (bug fix: text was uppercased before matching)
KEYWORD_GROUPS = {
    "kw_meter":    r"(?i)meter|muni\s*meter",
    "kw_hydrant":  r"(?i)hydrant|fire\s*plug",
    "kw_bus_stop": r"(?i)bus\s*stop|bus\s*lane",
    "kw_sign":     r"(?i)\bsign\b|no\s*standing|no\s*parking|no\s*stopping",
    "kw_blocking": r"(?i)blocking|crosswalk|driveway|sidewalk",
    "kw_expired":  r"(?i)expir|registr",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VTIME_RE = re.compile(r"(\d{1,2})[:\.]?(\d{0,2})\s*([AP])?", re.IGNORECASE)

def _parse_vtime(s) -> float:
    """Parse NYC violation_time strings. Handles '0156A', '01:56P', '1:56PM', etc."""
    if pd.isnull(s):
        return np.nan
    s = str(s).strip().upper().replace(" ", "")
    if not s:
        return np.nan
    m = _VTIME_RE.match(s)
    if not m:
        return np.nan
    h_str, _mm, suffix = m.group(1), m.group(2), m.group(3)
    # If no colon/dot and 3-4 digits, first 1-2 are hour (e.g. "0156" -> 01)
    if not _mm and len(h_str) >= 3:
        h = int(h_str[:-2])
    else:
        h = int(h_str)
    if suffix == "P" and h != 12:
        h += 12
    elif suffix == "A" and h == 12:
        h = 0
    return float(h % 24) if 0 <= h <= 23 else np.nan


def _parse_pvqr_hour(s) -> float:
    """pvqr from_hours_in_effect / to_hours_in_effect are like '0700' or '1900'."""
    if pd.isnull(s):
        return np.nan
    s = str(s).strip()
    if not s or s.upper() == "ALL":
        return np.nan
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return np.nan
    s = s.zfill(4)
    try:
        h = int(s[:2])
        return float(h) if 0 <= h <= 23 else np.nan
    except ValueError:
        return np.nan


def _is_holiday(dt: pd.Series) -> pd.Series:
    def _check(d):
        if pd.isnull(d):
            return 0.0
        if (d.month, d.day) in FIXED_HOLIDAYS:
            return 1.0
        m, wd = d.month, d.weekday()
        wom = (d.day - 1) // 7 + 1
        if (m == 1 and wd == 0 and wom == 3) or \
           (m == 2 and wd == 0 and wom == 3) or \
           (m == 5 and wd == 0 and d.day >= 25) or \
           (m == 9 and wd == 0 and wom == 1) or \
           (m == 11 and wd == 3 and wom == 4):
            return 1.0
        return 0.0
    return dt.apply(_check)


def _fetch_weather_county(county, lat, lon, start, end) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "precipitation,visibility,wind_speed_10m,weather_code",
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "weather_dt":    pd.to_datetime(h["time"]),
        "precipitation": h["precipitation"],
        "visibility":    h["visibility"],
        "wind_speed":    h["wind_speed_10m"],
        "weather_code":  h["weather_code"],
    })
    df["county"] = county
    df["weather_date"] = df["weather_dt"].dt.date.astype(str)
    df["weather_hour"] = df["weather_dt"].dt.hour
    return df[["county", "weather_date", "weather_hour",
               "precipitation", "visibility", "wind_speed", "weather_code"]]


def _build_weather_cache(df, date_col, county_col) -> pd.DataFrame:
    parsed_dt = pd.to_datetime(df[date_col], errors="coerce")
    df["_date_str"] = parsed_dt.dt.strftime("%Y-%m-%d")
    df["_canon"] = df[county_col].str.strip().str.upper().map(COUNTY_NORM).fillna("MANHATTAN")
    today = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for canon in COUNTY_COORDS:
        lat, lon = COUNTY_COORDS[canon]
        mask = df["_canon"] == canon
        dates = df.loc[mask, "_date_str"].dropna()
        if dates.empty:
            continue
        start = min(dates.min(), today)
        end = min(dates.max(), today)
        if start > end:
            continue
        try:
            print(f"    Fetching weather: {canon} ({start} -> {end})")
            frames.append(_fetch_weather_county(canon, lat, lon, start, end))
            time.sleep(2)
        except Exception as exc:
            print(f"    [WARN] Weather fetch failed for {canon}: {exc}")
    df.drop(columns=["_date_str", "_canon"], inplace=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_weather(df, date_col, hour_col, county_col) -> pd.DataFrame:
    if os.path.exists(WEATHER_CACHE_PATH):
        print("  Loading weather from cache...")
        weather = pd.read_csv(WEATHER_CACHE_PATH)
    else:
        print("  Fetching weather from Open-Meteo...")
        weather = _build_weather_cache(df.copy(), date_col, county_col)
        if not weather.empty:
            weather.to_csv(WEATHER_CACHE_PATH, index=False)

    if weather.empty:
        for col in ("precipitation", "visibility", "wind_speed", "weather_code", "is_bad_weather"):
            df[col] = 0.0
        return df

    parsed_dt = pd.to_datetime(df[date_col], errors="coerce")
    df["_weather_date"] = parsed_dt.dt.strftime("%Y-%m-%d")
    df["_weather_hour"] = df[hour_col].fillna(12).astype(int).clip(0, 23)
    df["_canon"] = df[county_col].str.strip().str.upper().map(COUNTY_NORM).fillna("MANHATTAN")
    weather = weather.rename(columns={"county": "_w_county",
                                       "weather_date": "_w_date",
                                       "weather_hour": "_w_hour"})
    weather["_w_hour"] = weather["_w_hour"].astype(int)
    merged = df.merge(
        weather,
        left_on=["_canon", "_weather_date", "_weather_hour"],
        right_on=["_w_county", "_w_date", "_w_hour"],
        how="left",
    )
    merged.drop(columns=["_weather_date", "_weather_hour", "_canon",
                          "_w_county", "_w_date", "_w_hour"],
                inplace=True, errors="ignore")
    merged["is_bad_weather"] = (
        (merged["precipitation"].fillna(0) > 2) |
        (merged["weather_code"].fillna(0) >= 71) |
        (merged["weather_code"].fillna(0).isin([45, 48]))
    ).astype(float)
    return merged


def _keyword_features(text_series) -> pd.DataFrame:
    """Extract binary keyword flags. Patterns are case-insensitive (bug fix)."""
    text = text_series.fillna("").astype(str)
    out = {feat: text.str.contains(pat, regex=True).astype(float)
           for feat, pat in KEYWORD_GROUPS.items()}
    return pd.DataFrame(out)


def _compute_plate_history(df: pd.DataFrame, plate_col: str, date_col: str,
                           global_mean: float) -> tuple[pd.DataFrame, dict]:
    """Chronological plate history features with STRICT `<` on issue_date.

    Per FineHero AUC playbook §7 item 2: two tickets on the same day for the
    same plate must not see each other in the prior stats. We enforce this by
    ranking on `_dt` with `method='min'` (all ties share the smallest rank)
    for `prior_count`, and by aggregating wins per (plate, date) pair then
    cumsum+shift for `prior_wins`.

    Returns (features_df, history_map) where history_map[plate] = {wins, count}
    so predict.py can apply the full (non-LOO) value at inference.
    """
    print("  Computing plate history features (strict-< on date)...")
    work = df[[plate_col, date_col, "won"]].copy()
    work["_dt"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_orig"] = np.arange(len(work))

    # prior_count: count of this plate's tickets with STRICTLY earlier date.
    # rank(method='min') assigns all ties the same (smallest) rank.
    work["plate_prior_ticket_count"] = (
        work.groupby(plate_col)["_dt"].rank(method="min", ascending=True) - 1.0
    ).fillna(0.0).astype(float)

    # prior_wins: sum of wins on strictly-earlier dates for this plate.
    # Aggregate wins per (plate, date), cumsum within plate sorted by date,
    # then shift(1) so each row sees only wins from earlier dates.
    daily = (
        work.groupby([plate_col, "_dt"], dropna=False)["won"]
        .sum()
        .reset_index()
        .sort_values([plate_col, "_dt"], kind="mergesort")
    )
    daily["_cumwins"] = daily.groupby(plate_col)["won"].cumsum()
    daily["_prior_wins"] = daily.groupby(plate_col)["_cumwins"].shift(1).fillna(0.0)

    work = work.merge(
        daily[[plate_col, "_dt", "_prior_wins"]]
            .rename(columns={"_prior_wins": "plate_prior_wins"}),
        on=[plate_col, "_dt"],
        how="left",
    )

    work["plate_prior_win_rate"] = (
        (work["plate_prior_wins"].fillna(0.0) + PLATE_SMOOTH_K * global_mean) /
        (work["plate_prior_ticket_count"] + PLATE_SMOOTH_K)
    )

    # Reorder back to original row indexing
    work = work.sort_values("_orig", kind="mergesort")
    feats = work[["plate_prior_ticket_count", "plate_prior_win_rate"]].reset_index(drop=True)

    # Build history map for inference: full (not-LOO) totals per plate
    totals = df.groupby(plate_col)["won"].agg(["sum", "count"])
    history_map = {
        "per_plate": totals.rename(columns={"sum": "wins", "count": "count"}).to_dict(orient="index"),
        "global_mean": global_mean,
        "smooth_k": PLATE_SMOOTH_K,
    }
    return feats, history_map


def _compute_rolling_group_history(df: pd.DataFrame, group_col: str, date_col: str,
                                    global_mean: float, windows=("30D", "90D"),
                                    prefix: str = None, k: int = 5) -> pd.DataFrame:
    """Time-aware rolling win/count per group with closed='left' (playbook §3.6).

    For each row, computes the group's sum and count of wins in the window
    ENDING STRICTLY BEFORE the current row's date. `closed='left'` is mandatory
    to avoid leakage. NaN dates are excluded (those rows get 0/0).

    Returns a DataFrame with columns:
        {prefix}_prior_wins_30D, {prefix}_prior_count_30D, {prefix}_prior_win_rate_30D,
        {prefix}_prior_wins_90D, {prefix}_prior_count_90D, {prefix}_prior_win_rate_90D
    indexed like the input df.
    """
    prefix = prefix or group_col
    n = len(df)
    print(f"  Rolling history for '{group_col}' ({windows})...")

    work = pd.DataFrame({
        "_grp":  df[group_col].values,
        "_won":  df["won"].values,
        "_dt":   pd.to_datetime(df[date_col], errors="coerce"),
        "_orig": np.arange(n),
    })
    valid = work["_dt"].notna() & work["_grp"].notna()
    work_v = work.loc[valid].sort_values(["_grp", "_dt"], kind="mergesort")
    work_v_indexed = work_v.set_index("_dt")
    orig_vals = work_v["_orig"].values

    out = {}
    for window in windows:
        grp = work_v_indexed.groupby("_grp")["_won"]
        roll_sum = np.nan_to_num(
            grp.rolling(window, closed="left").sum().values, nan=0.0
        )
        roll_cnt = np.nan_to_num(
            grp.rolling(window, closed="left").count().values, nan=0.0
        )
        wins_full = np.zeros(n)
        cnt_full  = np.zeros(n)
        wins_full[orig_vals] = roll_sum
        cnt_full[orig_vals]  = roll_cnt
        out[f"{prefix}_prior_wins_{window}"]     = wins_full
        out[f"{prefix}_prior_count_{window}"]    = cnt_full
        out[f"{prefix}_prior_win_rate_{window}"] = (
            wins_full + k * global_mean
        ) / (cnt_full + k)

    return pd.DataFrame(out, index=df.index)


def _pick_col(df, *candidates):
    for cand in candidates:
        for c in df.columns:
            if c.lower() == cand.lower():
                return c
    return None


def _find_col(df, substr):
    for c in df.columns:
        if substr.lower() in c.lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def engineer_features() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("  Loading violations data...")
    df = pd.read_csv(VIOLATIONS_PATH, low_memory=False)
    print(f"  Rows loaded: {len(df):,}  |  Columns: {len(df.columns)}")

    # --- Outcome ---
    outcome_col = _pick_col(df, "violation_status", "status")
    if outcome_col is None:
        raise ValueError(f"No outcome column. Columns: {list(df.columns)}")

    df = df.dropna(subset=[outcome_col])
    unresolved = df[outcome_col].astype(str).str.upper().str.strip().isin(
        ["OUTSTANDING", "IN PROCESS", ""]
    )
    df = df[~unresolved].copy().reset_index(drop=True)
    print(f"  After dropping unresolved: {len(df):,} rows")

    df["won"] = df[outcome_col].apply(lambda v: 1 if WIN_PATTERN.search(str(v)) else 0)

    # --- Column discovery ---
    vcode_col      = _pick_col(df, "violation", "violation_code")
    precinct_col   = _pick_col(df, "precinct") or _find_col(df, "precinct")
    county_col     = _find_col(df, "county")
    agency_col     = _pick_col(df, "issuing_agency") or _find_col(df, "agency")
    license_col    = _pick_col(df, "license_type", "plate_type")
    state_col      = _pick_col(df, "state", "registration_state")
    plate_col      = _pick_col(df, "plate", "plate_id")
    vtime_col      = _find_col(df, "violation_time")
    date_col       = _find_col(df, "issue_date")
    fine_col       = _pick_col(df, "fine_amount")

    # pvqr fields
    issuer_col         = _pick_col(df, "issuer_code")
    issuer_cmd_col     = _pick_col(df, "issuer_command")
    issuer_sqd_col     = _pick_col(df, "issuer_squad")
    street_col         = _pick_col(df, "street_name")
    vmake_col          = _pick_col(df, "vehicle_make")
    vbody_col          = _pick_col(df, "vehicle_body_type")
    vyear_col          = _pick_col(df, "vehicle_year")
    vdesc_col          = _pick_col(df, "violation_description")
    vlegal_col         = _pick_col(df, "violation_legal_code")
    lawsec_col         = _pick_col(df, "law_section")
    subdiv_col         = _pick_col(df, "sub_division")
    fromhrs_col        = _pick_col(df, "from_hours_in_effect")
    tohrs_col          = _pick_col(df, "to_hours_in_effect")
    feet_col           = _pick_col(df, "feet_from_curb")
    front_col          = _pick_col(df, "violation_in_front_of_or_opposite")
    dayseff_col        = _pick_col(df, "days_parking_in_effect")

    def _str(col, default="UNKNOWN"):
        return df[col].astype(str).str.strip() if col else pd.Series([default] * len(df))

    # --- Core categoricals (raw strings for CatBoost) ---
    df["violation_code"]  = _str(vcode_col)
    df["precinct"]        = _str(precinct_col)
    df["county"]          = _str(county_col).str.upper().map(COUNTY_NORM).fillna(_str(county_col))
    df["issuing_agency"]  = _str(agency_col)
    df["license_type"]    = _str(license_col)
    df["state"]           = _str(state_col)

    # --- pvqr categoricals ---
    df["issuer_code"]                      = _str(issuer_col)
    df["issuer_command"]                   = _str(issuer_cmd_col)
    df["issuer_squad"]                     = _str(issuer_sqd_col)
    df["street_name"]                      = _str(street_col).str.upper()
    df["vehicle_make"]                     = _str(vmake_col).str.upper()
    df["vehicle_body_type"]                = _str(vbody_col).str.upper()
    df["violation_legal_code"]             = _str(vlegal_col)
    df["law_section"]                      = _str(lawsec_col)
    df["sub_division"]                     = _str(subdiv_col)
    df["violation_in_front_of_or_opposite"] = _str(front_col).str.upper()
    df["days_parking_in_effect"]           = _str(dayseff_col).str.upper()

    # --- Temporal ---
    df["hour_of_offense"] = df[vtime_col].apply(_parse_vtime) if vtime_col else np.nan
    if date_col:
        parsed_dt = pd.to_datetime(df[date_col], errors="coerce")
        df["day_of_week"] = parsed_dt.dt.dayofweek.astype(float)
        df["month"]       = parsed_dt.dt.month.astype(float)
        df["is_holiday"]  = _is_holiday(parsed_dt)
    else:
        for c in ("day_of_week", "month", "is_holiday"):
            df[c] = np.nan

    # Cyclical encodings (FineHero AUC playbook §3.6). Trees don't strictly
    # need these, but they give the model a smooth notion of adjacency
    # (hour 23 ≈ hour 0) that plain integers miss.
    two_pi = 2.0 * np.pi
    hour = df["hour_of_offense"].fillna(-1).astype(float)
    dow  = df["day_of_week"].fillna(-1).astype(float)
    mon  = df["month"].fillna(-1).astype(float)
    df["hour_sin"]  = np.where(hour >= 0, np.sin(two_pi * hour / 24.0), 0.0)
    df["hour_cos"]  = np.where(hour >= 0, np.cos(two_pi * hour / 24.0), 0.0)
    df["dow_sin"]   = np.where(dow  >= 0, np.sin(two_pi * dow  / 7.0),  0.0)
    df["dow_cos"]   = np.where(dow  >= 0, np.cos(two_pi * dow  / 7.0),  0.0)
    df["month_sin"] = np.where(mon  >= 0, np.sin(two_pi * mon  / 12.0), 0.0)
    df["month_cos"] = np.where(mon  >= 0, np.cos(two_pi * mon  / 12.0), 0.0)

    # --- Financial ---
    df["fine_amount"] = pd.to_numeric(df[fine_col], errors="coerce") if fine_col else np.nan

    # --- Keyword features (from description if available, else violation_code) ---
    print("  Extracting keyword features...")
    kw_source = df[vdesc_col] if vdesc_col else df["violation_code"]
    kw_df = _keyword_features(kw_source)
    for col in kw_df.columns:
        df[col] = kw_df[col].values
    print(f"  Keyword features: {list(kw_df.columns)} "
          f"(positive rates: {kw_df.mean().round(3).to_dict()})")

    # --- pvqr numerics ---
    df["feet_from_curb"] = pd.to_numeric(df[feet_col], errors="coerce") if feet_col else np.nan
    df["vehicle_year"]   = pd.to_numeric(df[vyear_col], errors="coerce") if vyear_col else np.nan
    df["from_hour"]      = df[fromhrs_col].apply(_parse_pvqr_hour) if fromhrs_col else np.nan
    df["to_hour"]        = df[tohrs_col].apply(_parse_pvqr_hour) if tohrs_col else np.nan
    # Was the ticket within posted hours-in-effect? (big dismissal signal)
    df["within_posted_hours"] = np.where(
        df["hour_of_offense"].notna() & df["from_hour"].notna() & df["to_hour"].notna(),
        (((df["from_hour"] <= df["to_hour"]) &
          (df["hour_of_offense"] >= df["from_hour"]) &
          (df["hour_of_offense"] <= df["to_hour"])) |
         ((df["from_hour"] > df["to_hour"]) &   # overnight range
          ((df["hour_of_offense"] >= df["from_hour"]) |
           (df["hour_of_offense"] <= df["to_hour"])))).astype(float),
        np.nan,
    )

    # --- Weather ---
    if date_col and county_col:
        print("  Attaching weather data...")
        df = _attach_weather(df, date_col, "hour_of_offense", county_col)
    else:
        for c in ("precipitation", "visibility", "wind_speed", "weather_code", "is_bad_weather"):
            df[c] = 0.0

    # --- Cross-features (CatBoost will target-encode these via ordered TS) ---
    print("  Building cross-features...")
    df["viol_x_precinct"] = df["violation_code"] + "_" + df["precinct"]
    df["viol_x_license"]  = df["violation_code"] + "_" + df["license_type"]
    hour_bin = df["hour_of_offense"].fillna(-1).astype(int).astype(str)
    dow_str  = df["day_of_week"].fillna(-1).astype(int).astype(str)
    df["hour_x_dow"]      = hour_bin + "_" + dow_str

    # --- Plate history (leave-one-out, chronological) ---
    rolling_feature_cols: list[str] = []
    if plate_col and date_col:
        global_mean = float(df["won"].mean())
        plate_feats, history_map = _compute_plate_history(df, plate_col, date_col, global_mean)
        df["plate_prior_ticket_count"] = plate_feats["plate_prior_ticket_count"].values
        df["plate_prior_win_rate"]     = plate_feats["plate_prior_win_rate"].values
        joblib.dump(history_map, PLATE_HISTORY_PATH)
        print(f"  Plate history map saved -> {PLATE_HISTORY_PATH}  "
              f"({len(history_map['per_plate']):,} plates)")

        # --- Rolling-window features (playbook §3.6 — #2 AUC move) ---
        # Time-aware rolling aggregates with closed='left'. For plate, precinct
        # and issuer_code — the three group keys with the highest signal on
        # dispute-style problems.
        for group_col, prefix in [
            (plate_col,    "plate"),
            (precinct_col, "precinct"),
            (issuer_col,   "issuer"),
        ]:
            if group_col is None or group_col not in df.columns:
                print(f"  [WARN] skipping rolling for missing column: {prefix}")
                continue
            rf = _compute_rolling_group_history(
                df, group_col=group_col, date_col=date_col,
                global_mean=global_mean, windows=("30D", "90D"), prefix=prefix,
            )
            for c in rf.columns:
                df[c] = rf[c].values
                rolling_feature_cols.append(c)
        print(f"  Rolling features added: {len(rolling_feature_cols)}")
    else:
        df["plate_prior_ticket_count"] = 0.0
        df["plate_prior_win_rate"]     = float(df["won"].mean())
        print("  [WARN] No plate column — skipped plate history")

    # --- Final feature list ---
    cat_features = [
        "violation_code", "precinct", "county", "issuing_agency",
        "license_type", "state",
        "issuer_code", "issuer_command", "issuer_squad",
        "street_name",
        "vehicle_make", "vehicle_body_type",
        "violation_legal_code", "law_section", "sub_division",
        "violation_in_front_of_or_opposite", "days_parking_in_effect",
        "viol_x_precinct", "viol_x_license", "hour_x_dow",
    ]
    numeric_features = [
        "hour_of_offense", "day_of_week", "month", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "fine_amount",
        "feet_from_curb", "vehicle_year",
        "from_hour", "to_hour", "within_posted_hours",
        "precipitation", "visibility", "wind_speed", "is_bad_weather",
        "plate_prior_ticket_count", "plate_prior_win_rate",
    ] + list(kw_df.columns) + rolling_feature_cols

    # Normalize missing-value markers in categoricals so CatBoost sees one bucket
    for c in cat_features:
        df[c] = df[c].replace({"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN",
                               np.nan: "UNKNOWN"}).astype(str)

    # Save the cat_features list for train.py and predict.py
    joblib.dump(cat_features, CAT_FEATURES_PATH)
    print(f"  Cat features list saved -> {CAT_FEATURES_PATH}  "
          f"({len(cat_features)} cat, {len(numeric_features)} num)")

    # Preserve issue_date as a META column (not a feature). train.py and
    # predict.py must drop it before fitting. The audit script uses it for
    # time-aware splits and leakage probes.
    meta_cols = []
    if date_col:
        df["issue_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        meta_cols.append("issue_date")

    feature_cols = meta_cols + cat_features + numeric_features + ["won"]
    features = df[feature_cols].copy()

    # Numeric NaN handling — CatBoost tolerates NaN natively, but fill sentinels
    # for consistency with predict.py
    for c in numeric_features:
        if features[c].isna().all():
            features[c] = 0.0

    features.to_csv(FEATURES_PATH, index=False)
    print(f"  Saved {len(features):,} rows x {features.shape[1]} cols -> {FEATURES_PATH}")

    total = len(features)
    won = int(features["won"].sum())
    print(f"\n  Class balance: won={won:,} ({100*won/total:.1f}%) "
          f"| lost={total-won:,} ({100*(total-won)/total:.1f}%)")


if __name__ == "__main__":
    engineer_features()
