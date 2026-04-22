# predict.py - Scores a random unseen ticket through the CatBoost model,
# applying the same feature engineering as training (including weather,
# keyword flags, pvqr fields, cross-features, and plate history lookup).

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

CATBOOST_PATH      = os.path.join(MODELS_DIR, "catboost_model.cbm")
METADATA_PATH      = os.path.join(MODELS_DIR, "metadata.json")
CAT_FEATURES_PATH  = os.path.join(MODELS_DIR, "cat_features.joblib")
PLATE_HISTORY_PATH = os.path.join(MODELS_DIR, "plate_history_map.joblib")
VIOLATIONS_PATH    = os.path.join(DATA_DIR, "violations_raw.csv")

DISPUTE_THRESHOLD = 0.40
PLATE_SMOOTH_K = 10

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

KEYWORD_GROUPS = {
    "kw_meter":    r"(?i)meter|muni\s*meter",
    "kw_hydrant":  r"(?i)hydrant|fire\s*plug",
    "kw_bus_stop": r"(?i)bus\s*stop|bus\s*lane",
    "kw_sign":     r"(?i)\bsign\b|no\s*standing|no\s*parking|no\s*stopping",
    "kw_blocking": r"(?i)blocking|crosswalk|driveway|sidewalk",
    "kw_expired":  r"(?i)expir|registr",
}

WIN_STR   = r"DISMISS|NOT GUILTY|NOT LIABLE"
HEARD_STR = r"HEARING HELD|DISMISS|NOT GUILTY|NOT LIABLE|APPEAL"

_VTIME_RE = re.compile(r"(\d{1,2})[:\.]?(\d{0,2})\s*([AP])?", re.IGNORECASE)


def _parse_vtime(s):
    if pd.isnull(s):
        return np.nan
    s = str(s).strip().upper().replace(" ", "")
    m = _VTIME_RE.match(s)
    if not m:
        return np.nan
    h_str, _mm, suf = m.group(1), m.group(2), m.group(3)
    h = int(h_str[:-2]) if (not _mm and len(h_str) >= 3) else int(h_str)
    if suf == "P" and h != 12:
        h += 12
    elif suf == "A" and h == 12:
        h = 0
    return float(h % 24) if 0 <= h <= 23 else np.nan


def _parse_pvqr_hour(s):
    if pd.isnull(s):
        return np.nan
    s = re.sub(r"[^\d]", "", str(s).strip())
    if not s:
        return np.nan
    s = s.zfill(4)
    try:
        h = int(s[:2])
        return float(h) if 0 <= h <= 23 else np.nan
    except ValueError:
        return np.nan


def _is_holiday(dt):
    if pd.isnull(dt):
        return 0.0
    if (dt.month, dt.day) in FIXED_HOLIDAYS:
        return 1.0
    m, wd, wom = dt.month, dt.weekday(), (dt.day - 1) // 7 + 1
    if (m == 1 and wd == 0 and wom == 3) or (m == 2 and wd == 0 and wom == 3) or \
       (m == 5 and wd == 0 and dt.day >= 25) or (m == 9 and wd == 0 and wom == 1) or \
       (m == 11 and wd == 3 and wom == 4):
        return 1.0
    return 0.0


def _fetch_weather_point(lat, lon, date_str, hour):
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": date_str, "end_date": date_str,
                "hourly": "precipitation,visibility,wind_speed_10m,weather_code",
                "timezone": "America/New_York",
            }, timeout=10,
        )
        resp.raise_for_status()
        h = resp.json()["hourly"]
        i = min(hour, len(h["precipitation"]) - 1)
        precip = h["precipitation"][i] or 0.0
        vis    = h["visibility"][i] or 10000.0
        wind   = h["wind_speed_10m"][i] or 0.0
        code   = h["weather_code"][i] or 0
        return {"precipitation": precip, "visibility": vis, "wind_speed": wind,
                "weather_code": code,
                "is_bad_weather": float(precip > 2 or code >= 71 or code in (45, 48))}
    except Exception as exc:
        print(f"  [WARN] Weather fetch failed: {exc}")
        return {"precipitation": 0.0, "visibility": 10000.0, "wind_speed": 0.0,
                "weather_code": 0, "is_bad_weather": 0.0}


def _pick_disputed_row(raw):
    status_col = next((c for c in raw.columns if "violation_status" in c.lower()), None)
    if status_col is None:
        return raw.sample(1).iloc[0], None
    heard = raw[raw[status_col].astype(str).str.contains(HEARD_STR, na=False, case=False)].copy()
    if heard.empty:
        heard = raw.copy()
    heard["_won"] = heard[status_col].astype(str).str.contains(WIN_STR, na=False, case=False)
    won_pool, lost_pool = heard[heard["_won"]], heard[~heard["_won"]]
    pick_won = np.random.random() < 0.5
    if pick_won and not won_pool.empty:
        return won_pool.sample(1).iloc[0], True
    if not lost_pool.empty:
        return lost_pool.sample(1).iloc[0], False
    row = heard.sample(1).iloc[0]
    return row, bool(row["_won"])


def _get(row, col, default=""):
    return str(row[col]).strip() if col and col in row.index and pd.notna(row[col]) else default


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


def predict_ticket() -> float:
    if not os.path.exists(CATBOOST_PATH):
        raise FileNotFoundError(f"Model not found at {CATBOOST_PATH}. Run pipeline.py first.")

    print("  Loading model + supporting data...")
    model = CatBoostClassifier()
    model.load_model(CATBOOST_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    plate_history = joblib.load(PLATE_HISTORY_PATH) if os.path.exists(PLATE_HISTORY_PATH) else None
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    print(f"  Model: {meta['model_type']}  AUC {meta['auc_score']:.4f}  {meta['row_count']:,} rows trained")

    print("  Picking a random disputed ticket (50/50 won vs lost)...")
    raw = pd.read_csv(VIOLATIONS_PATH, low_memory=False)
    row, actually_won = _pick_disputed_row(raw)

    # --- Column discovery ---
    vcode_col    = _pick_col(raw, "violation", "violation_code")
    precinct_col = _pick_col(raw, "precinct") or _find_col(raw, "precinct")
    county_col   = _find_col(raw, "county")
    agency_col   = _pick_col(raw, "issuing_agency") or _find_col(raw, "agency")
    license_col  = _pick_col(raw, "license_type", "plate_type")
    state_col    = _pick_col(raw, "state", "registration_state")
    plate_col    = _pick_col(raw, "plate", "plate_id")
    vtime_col    = _find_col(raw, "violation_time")
    date_col     = _find_col(raw, "issue_date")
    fine_col     = _pick_col(raw, "fine_amount")
    vdesc_col    = _pick_col(raw, "violation_description")

    # --- Core categoricals ---
    vcode = _get(row, vcode_col, "UNKNOWN")
    precinct = _get(row, precinct_col, "UNKNOWN")
    county_raw = _get(row, county_col, "UNKNOWN").upper()
    county = COUNTY_NORM.get(county_raw, county_raw)
    agency = _get(row, agency_col, "UNKNOWN")
    license_type = _get(row, license_col, "UNKNOWN")
    state = _get(row, state_col, "UNKNOWN")
    plate = _get(row, plate_col, "UNKNOWN")

    # pvqr categoricals
    issuer_code                      = _get(row, _pick_col(raw, "issuer_code"), "UNKNOWN")
    issuer_command                   = _get(row, _pick_col(raw, "issuer_command"), "UNKNOWN")
    issuer_squad                     = _get(row, _pick_col(raw, "issuer_squad"), "UNKNOWN")
    street_name                      = _get(row, _pick_col(raw, "street_name"), "UNKNOWN").upper()
    vehicle_make                     = _get(row, _pick_col(raw, "vehicle_make"), "UNKNOWN").upper()
    vehicle_body_type                = _get(row, _pick_col(raw, "vehicle_body_type"), "UNKNOWN").upper()
    violation_legal_code             = _get(row, _pick_col(raw, "violation_legal_code"), "UNKNOWN")
    law_section                      = _get(row, _pick_col(raw, "law_section"), "UNKNOWN")
    sub_division                     = _get(row, _pick_col(raw, "sub_division"), "UNKNOWN")
    violation_in_front_of_or_opposite = _get(row, _pick_col(raw, "violation_in_front_of_or_opposite"), "UNKNOWN").upper()
    days_parking_in_effect           = _get(row, _pick_col(raw, "days_parking_in_effect"), "UNKNOWN").upper()

    # --- Temporal ---
    hour_val = _parse_vtime(row[vtime_col]) if vtime_col else np.nan
    parsed_dt = pd.to_datetime(row[date_col], errors="coerce") if date_col else None
    if parsed_dt is not None and not pd.isnull(parsed_dt):
        day_of_week = float(parsed_dt.dayofweek)
        month = float(parsed_dt.month)
        is_holiday = _is_holiday(parsed_dt)
    else:
        day_of_week = month = is_holiday = np.nan

    # --- pvqr numerics ---
    feet_from_curb = pd.to_numeric(row.get(_pick_col(raw, "feet_from_curb")), errors="coerce")
    vehicle_year   = pd.to_numeric(row.get(_pick_col(raw, "vehicle_year")), errors="coerce")
    from_hour = _parse_pvqr_hour(row.get(_pick_col(raw, "from_hours_in_effect")))
    to_hour   = _parse_pvqr_hour(row.get(_pick_col(raw, "to_hours_in_effect")))
    within_posted_hours = np.nan
    if not np.isnan(hour_val) and not np.isnan(from_hour) and not np.isnan(to_hour):
        if from_hour <= to_hour:
            within_posted_hours = float(from_hour <= hour_val <= to_hour)
        else:
            within_posted_hours = float(hour_val >= from_hour or hour_val <= to_hour)

    # --- Financial ---
    fine_amount = pd.to_numeric(row.get("fine_amount"), errors="coerce") if fine_col else np.nan

    # --- Keyword flags ---
    kw_source = _get(row, vdesc_col, vcode)
    kw_values = {feat: float(bool(re.search(pat, kw_source)))
                 for feat, pat in KEYWORD_GROUPS.items()}

    # --- Weather ---
    weather = {"precipitation": 0.0, "visibility": 10000.0, "wind_speed": 0.0,
               "weather_code": 0, "is_bad_weather": 0.0}
    if parsed_dt is not None and not pd.isnull(parsed_dt):
        today = datetime.now().strftime("%Y-%m-%d")
        ds = parsed_dt.strftime("%Y-%m-%d")
        if ds < today:
            lat, lon = COUNTY_COORDS.get(county, DEFAULT_COORDS)
            h_int = int(hour_val) if not np.isnan(hour_val) else 12
            print(f"  Fetching weather {ds} h={h_int} {county}...")
            weather = _fetch_weather_point(lat, lon, ds, h_int)

    # --- Cross-features ---
    viol_x_precinct = f"{vcode}_{precinct}"
    viol_x_license  = f"{vcode}_{license_type}"
    hour_bin = int(hour_val) if not np.isnan(hour_val) else -1
    dow_bin  = int(day_of_week) if not np.isnan(day_of_week) else -1
    hour_x_dow = f"{hour_bin}_{dow_bin}"

    # --- Plate history lookup ---
    if plate_history and plate in plate_history["per_plate"]:
        rec = plate_history["per_plate"][plate]
        wins, cnt = rec["wins"], rec["count"]
        gmean, k = plate_history["global_mean"], plate_history["smooth_k"]
        plate_prior_ticket_count = float(cnt)
        plate_prior_win_rate = (wins + k * gmean) / (cnt + k)
    elif plate_history:
        plate_prior_ticket_count = 0.0
        plate_prior_win_rate = plate_history["global_mean"]
    else:
        plate_prior_ticket_count = 0.0
        plate_prior_win_rate = 0.3  # fallback

    # --- Assemble model row ---
    values = {
        "violation_code": vcode, "precinct": precinct, "county": county,
        "issuing_agency": agency, "license_type": license_type, "state": state,
        "issuer_code": issuer_code, "issuer_command": issuer_command,
        "issuer_squad": issuer_squad,
        "street_name": street_name,
        "vehicle_make": vehicle_make, "vehicle_body_type": vehicle_body_type,
        "violation_legal_code": violation_legal_code, "law_section": law_section,
        "sub_division": sub_division,
        "violation_in_front_of_or_opposite": violation_in_front_of_or_opposite,
        "days_parking_in_effect": days_parking_in_effect,
        "viol_x_precinct": viol_x_precinct, "viol_x_license": viol_x_license,
        "hour_x_dow": hour_x_dow,
        "hour_of_offense": hour_val, "day_of_week": day_of_week, "month": month,
        "is_holiday": is_holiday, "fine_amount": fine_amount,
        "feet_from_curb": feet_from_curb, "vehicle_year": vehicle_year,
        "from_hour": from_hour, "to_hour": to_hour,
        "within_posted_hours": within_posted_hours,
        "plate_prior_ticket_count": plate_prior_ticket_count,
        "plate_prior_win_rate": plate_prior_win_rate,
        **kw_values, **weather,
    }

    # Cyclical time encodings (match src/engineer.py)
    two_pi = 2.0 * np.pi
    hf = float(hour_val) if not np.isnan(hour_val) else -1.0
    df_ = float(day_of_week) if not np.isnan(day_of_week) else -1.0
    mf = float(month) if not np.isnan(month) else -1.0
    values.update({
        "hour_sin":  np.sin(two_pi * hf / 24.0) if hf >= 0 else 0.0,
        "hour_cos":  np.cos(two_pi * hf / 24.0) if hf >= 0 else 0.0,
        "dow_sin":   np.sin(two_pi * df_ / 7.0) if df_ >= 0 else 0.0,
        "dow_cos":   np.cos(two_pi * df_ / 7.0) if df_ >= 0 else 0.0,
        "month_sin": np.sin(two_pi * mf / 12.0) if mf >= 0 else 0.0,
        "month_cos": np.cos(two_pi * mf / 12.0) if mf >= 0 else 0.0,
    })

    model_row = pd.DataFrame([{f: values.get(f, np.nan) for f in feature_names}])
    # CatBoost wants strings for cat columns
    for c in cat_features:
        if c in model_row.columns:
            model_row[c] = model_row[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    prob = float(model.predict_proba(model_row)[0, 1])
    pct = prob * 100

    # --- Display ---
    print(f"\n{'-'*56}")
    print("  Ticket details:")
    display = {
        "violation":          vcode,
        "precinct":           precinct,
        "county":             county,
        "agency":             agency,
        "license_type":       license_type,
        "state":              state,
        "plate":              plate,
        "street":             street_name,
        "side":               violation_in_front_of_or_opposite,
        "vehicle":            f"{vehicle_year if not pd.isna(vehicle_year) else '?'} "
                              f"{vehicle_make} ({vehicle_body_type})",
        "fine_amount":        fine_amount,
        "issue_date":         row.get("issue_date", "?"),
        "violation_time":     row.get("violation_time", "?"),
        "hours_in_effect":    f"{from_hour}\u2013{to_hour}  within={within_posted_hours}",
        "days_in_effect":     days_parking_in_effect,
        "feet_from_curb":     feet_from_curb,
        "plate_history":      f"{int(plate_prior_ticket_count)} prior tickets, "
                              f"win rate {plate_prior_win_rate*100:.1f}%",
        "weather":            f"precip={weather['precipitation']}mm  vis={weather['visibility']}m",
        "keywords":           ", ".join(k for k, v in kw_values.items() if v) or "(none)",
        "status":             row.get("violation_status", "?"),
    }
    for k, v in display.items():
        print(f"    {k:<22} {v}")
    print(f"{'-'*56}")
    print(f"\n  Win probability:  {pct:.1f}%")

    if prob >= DISPUTE_THRESHOLD:
        verdict = "Worth disputing"
        advice = "Gather evidence (photos, receipts, signage) and submit a dispute."
    else:
        verdict = "Pay this one"
        advice = "The odds are against you. Paying avoids late fees and penalties."
    print(f"  Verdict:          {verdict}")
    print(f"\n  {advice}")

    # --- Reveal actual outcome ---
    print(f"\n{'-'*56}")
    if actually_won is None:
        print("  Actual outcome:   Unknown")
    elif actually_won:
        print("  Actual outcome:   WON")
        print(f"  Model was:        {'CORRECT' if prob >= DISPUTE_THRESHOLD else 'WRONG'}")
    else:
        print("  Actual outcome:   LOST")
        print(f"  Model was:        {'CORRECT' if prob < DISPUTE_THRESHOLD else 'WRONG'}")
    print(f"{'-'*56}\n")
    return prob


if __name__ == "__main__":
    predict_ticket()
