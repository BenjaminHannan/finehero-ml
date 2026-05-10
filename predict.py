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

CATBOOST_PATH         = os.path.join(MODELS_DIR, "catboost_model.cbm")
METADATA_PATH         = os.path.join(MODELS_DIR, "metadata.json")
CAT_FEATURES_PATH     = os.path.join(MODELS_DIR, "cat_features.joblib")
PLATE_HISTORY_PATH    = os.path.join(MODELS_DIR, "plate_history_map.joblib")
CALIBRATOR_PATH       = os.path.join(MODELS_DIR, "isotonic_calibrator.joblib")
THRESHOLD_PATH        = os.path.join(MODELS_DIR, "dispute_threshold.json")
ROLLING_MEANS_PATH    = os.path.join(MODELS_DIR, "rolling_prior_means.joblib")
FEATURE_AUDIT_PATH    = os.path.join(MODELS_DIR, "feature_audit.joblib")
INFERENCE_LOG_PATH    = os.path.join(DATA_DIR, "inference_log.jsonl")
VIOLATIONS_PATH       = os.path.join(DATA_DIR, "violations_raw.csv")

# Fallback if dispute_threshold.json is missing. The tuned default lives in
# that file under policies.f1.threshold (~0.32 on calibrated probabilities).
DISPUTE_THRESHOLD = 0.40
PLATE_SMOOTH_K = 10


def _load_calibrator():
    """Return the fitted IsotonicRegression, or None if unavailable.

    The calibrator was fit on the chronological last 10% of training (n=80k).
    Applying it to raw CatBoost predict_proba outputs takes ECE from 0.152 to
    0.023 on the held-out test set; AUC is preserved (isotonic is monotone).
    See LIMITATIONS.md §"Probability threshold".
    """
    if not os.path.exists(CALIBRATOR_PATH):
        return None
    pkg = joblib.load(CALIBRATOR_PATH)
    return pkg["calibrator"] if isinstance(pkg, dict) else pkg


def _load_threshold():
    """Return (threshold, policy_name). Falls back to (DISPUTE_THRESHOLD, 'fallback')."""
    if not os.path.exists(THRESHOLD_PATH):
        return DISPUTE_THRESHOLD, "fallback"
    try:
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        return float(cfg["default_threshold"]), str(cfg.get("default_policy", "default"))
    except (KeyError, ValueError, json.JSONDecodeError):
        return DISPUTE_THRESHOLD, "fallback"


def _load_rolling_prior_means():
    """Return the rolling-prior fallback artifact, or None if unavailable.

    The 27 rolling-prior features (`plate_prior_*_30D/90D/365D`,
    `precinct_prior_*_*`, `issuer_prior_*_*`) are time-dependent windowed
    aggregates that `engineer.py` builds via running tallies during training.
    `predict.py` has no inference-time equivalent, so absent this fallback
    they would all be NaN — and the model has learned that NaN-rolling-priors
    correlates strongly with losses, collapsing predictions toward zero.

    Build the artifact with `python -m src.build_rolling_prior_means`.
    See LIMITATIONS.md §"Live-data drift" for the full diagnosis.
    """
    if not os.path.exists(ROLLING_MEANS_PATH):
        return None
    return joblib.load(ROLLING_MEANS_PATH)


def _apply_rolling_prior_fallback(df, fallback):
    """Fill NaN rolling-prior, days_since, and issuer_bayes columns in `df`.

    Operates in-place on the passed DataFrame; returns the count of cells
    filled so callers can report it. Pass `fallback=None` to no-op.
    """
    if fallback is None:
        return 0
    filled = 0
    for col, mean in fallback.get("rolling_prior_means", {}).items():
        if col in df.columns:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = mean
                filled += int(mask.sum())
    for col, default in fallback.get("days_since_defaults", {}).items():
        if col in df.columns:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = default
                filled += int(mask.sum())
    if "issuer_bayes_rate" in df.columns:
        mask = df["issuer_bayes_rate"].isna()
        if mask.any():
            df.loc[mask, "issuer_bayes_rate"] = fallback.get("issuer_bayes_default", 0.18)
            filled += int(mask.sum())
    return filled


# ---------------------------------------------------------------------------
# Mirrors of engineer.py constructors that test_live_nyc_tickets.py and
# predict_ticket() were missing. The audit added in build_feature_audit.py
# surfaced 19 features (8 cross/format + 10 is_missing_* + missing_fields_count)
# that engineer.py builds during training but the inference path never set,
# leaving them NaN at scoring time. Same training-serving skew class as the
# rolling priors. These helpers must stay byte-identical to engineer.py's
# logic — when engineer.py changes, mirror the change here.

# Mirrors engineer._SUMMONS_FORMAT_MAP. Leading digit of the summons number
# identifies how the ticket was issued. Whitepaper #1 feature (3.18× lift
# between handwritten and electronic).
_SUMMONS_FORMAT_MAP = {
    "1": "handwritten", "4": "camera_speed", "5": "camera_redlight",
    "7": "muni_meter",  "8": "electronic",   "9": "electronic",
}

# Mirrors engineer._MISS_SENTINELS — tokens that mean "categorical effectively missing"
_MISS_SENTINELS = {"", "NAN", "NONE", "UNKNOWN", "0", "00000000", "0000"}

# Mirrors engineer._DT_OBS_SENTINELS for first_observed_filled
_DT_OBS_SENTINELS = {"", "0", "00000000", "0000", "0000-00-00",
                     "00:00", "00:00:00", "NAN", "NONE"}

# Fields that get is_missing_<name> flags (from engineer.py:1015-1023)
_MISSING_FIELD_SPECS = [
    "vehicle_make", "vehicle_body_type", "street_name",
    "issuer_command", "issuer_squad", "violation_legal_code",
    "days_parking_in_effect",
]


def _summons_format(summons) -> str:
    """Map a summons number to its issuance-channel format. Mirror of
    engineer._summons_format (whitepaper Tier 1.1)."""
    if summons is None or pd.isnull(summons):
        return "UNKNOWN"
    txt = str(summons).strip()
    if not txt or txt.upper() in ("NAN", "NONE"):
        return "UNKNOWN"
    head = txt.lstrip()[:1]
    if not head.isdigit():
        return "UNKNOWN"
    return _SUMMONS_FORMAT_MAP.get(head, f"prefix_{head}")


def _is_missing_value(v) -> bool:
    """Single-value mirror of engineer._is_missing_str. Returns True when a
    stringified categorical is effectively missing (NaN, blank, or sentinel)."""
    if v is None:
        return True
    try:
        if pd.isnull(v):
            return True
    except (TypeError, ValueError):
        pass
    s = str(v).strip().upper()
    return (not s) or (s in _MISS_SENTINELS)


def _parse_hours_kind(s) -> str:
    """Mirror of engineer._parse_hours_token's kind output. Returns
    'all', 'numeric', or 'missing'. Used for hours_kind_from/to and
    hours_in_effect_all (whitepaper Tier 1.2: 3.2× lift on dismissal)."""
    if s is None or pd.isnull(s):
        return "missing"
    txt = str(s).strip()
    if not txt:
        return "missing"
    if txt.upper() == "ALL":
        return "all"
    digits = re.sub(r"[^\d]", "", txt)
    if not digits:
        return "missing"
    return "numeric"


def _first_observed_filled(date_val, time_val) -> float:
    """Mirror of engineer._is_first_observed_filled — 1.0 iff both date and
    time are non-sentinel. Whitepaper Tier 1.5: +1.33× dismissal lift when
    the officer recorded a separate observation timestamp."""
    def _filled(v) -> bool:
        if v is None:
            return False
        try:
            if pd.isnull(v):
                return False
        except Exception:
            pass
        return str(v).strip().upper() not in _DT_OBS_SENTINELS
    return 1.0 if (_filled(date_val) and _filled(time_val)) else 0.0


def _build_engineered_extras(row, vcode, issuing_agency, issuer_command,
                             from_hours_token, to_hours_token,
                             vehicle_year, plate_id):
    """Construct the 19 features that engineer.py builds but the original
    inference helper did not. Returns a dict suitable for merging into the
    `values` dict before DataFrame construction.

    Args:
        row: the joined source Series (used for sentinel-checking categorical
            fields by name and for the date/time observation fields).
        vcode: the violation code string (already extracted upstream).
        issuing_agency, issuer_command: the agency/command strings.
        from_hours_token, to_hours_token: the raw `from_hours_in_effect` /
            `to_hours_in_effect` values from pvqr (NOT the parsed numeric
            from_hour/to_hour). Engineer.py needs the original tokens to
            distinguish "ALL" from missing.
        vehicle_year: numeric vehicle year (already coerced).
        plate_id: the canonicalized plate string.
    """
    summons = row.get("summons_number") if hasattr(row, "get") else None
    sfmt = _summons_format(summons)

    # hours kinds + hours_in_effect_all (whitepaper Tier 1.2)
    hk_from = _parse_hours_kind(from_hours_token)
    hk_to   = _parse_hours_kind(to_hours_token)
    hours_all = float((hk_from == "all") or (hk_to == "all"))

    # first_observed_filled (whitepaper Tier 1.5)
    date_obs = row.get("date_first_observed") if hasattr(row, "get") else None
    time_obs = row.get("time_first_observed") if hasattr(row, "get") else None
    first_obs = _first_observed_filled(date_obs, time_obs)

    # Per-field missing flags (whitepaper Tier 1.3)
    miss_flags = {}
    miss_total = 0.0
    for field in _MISSING_FIELD_SPECS:
        col_val = row.get(field) if hasattr(row, "get") else None
        flag = float(_is_missing_value(col_val))
        miss_flags[f"is_missing_{field}"] = flag
        miss_total += flag

    # vehicle_year: missing if NaN or 0 (engineer.py:1033)
    try:
        vy = float(vehicle_year) if vehicle_year is not None else float("nan")
    except (TypeError, ValueError):
        vy = float("nan")
    flag_vy = float(np.isnan(vy) or vy == 0)
    miss_flags["is_missing_vehicle_year"] = flag_vy
    miss_total += flag_vy

    # plate: missing iff canonicalized plate is missing/empty
    flag_plate = float(_is_missing_value(plate_id) or plate_id == "UNKNOWN")
    miss_flags["is_missing_plate"] = flag_plate
    miss_total += flag_plate

    # Cross features
    fmt_x_viol = f"{sfmt}_{vcode}"
    agency_x_cmd = f"{issuing_agency}__{issuer_command}"
    # format_x_missing_plate (engineer.py:1055)
    fmt_x_miss_plate = f"{sfmt}_pm{int(flag_plate)}"

    return {
        "summons_format":         sfmt,
        "format_x_viol":          fmt_x_viol,
        "hours_kind_from":        hk_from,
        "hours_kind_to":          hk_to,
        "hours_in_effect_all":    hours_all,
        "first_observed_filled":  first_obs,
        "agency_x_command":       agency_x_cmd,
        "format_x_missing_plate": fmt_x_miss_plate,
        "missing_fields_count":   miss_total,
        **miss_flags,
    }


def _load_feature_audit():
    """Return the feature audit artifact, or None if unavailable.

    Built by `python -m src.build_feature_audit`. Holds per-feature null rates
    and numeric ranges from training data. Used to detect training-serving
    skew at inference — the most common form is a feature that was rarely
    null in training going to 100% null at inference (the rolling-prior bug
    documented in PROJECT.md §11).
    """
    if not os.path.exists(FEATURE_AUDIT_PATH):
        return None
    return joblib.load(FEATURE_AUDIT_PATH)


def _check_feature_drift(df, audit, *, sample_label="row"):
    """Compare a live feature DataFrame against training stats; return a list
    of warnings as strings. Empty list = clean.

    Two checks per feature:
      1. Null-rate drift. If a feature was rarely null in training (≤5%) but
         is null on this row, flag it. This is the rolling-prior bug signature.
      2. Numeric range. If a numeric feature's value falls outside the
         training [p01, p99] expanded by `range_padding`, flag it.

    Categorical features only get the null check; OOV categoricals are
    handled by CatBoost's encoder and aren't a skew signal.
    """
    if audit is None:
        return ["[skew] feature_audit.joblib missing — drift checks disabled"]
    feature_audit = audit.get("feature_audit", {})
    null_threshold = audit.get("null_rate_drift_threshold", 0.5)
    range_padding  = audit.get("range_padding", 0.10)

    warnings = []
    for col in df.columns:
        info = feature_audit.get(col)
        if info is None or not info.get("present_in_training"):
            continue

        live_null_rate = float(df[col].isna().mean())
        train_null_rate = float(info.get("null_rate", 0.0))
        # Only alert when the live feature is *more* null than training expected
        # by a wide margin. (Less-null at inference is fine.)
        if live_null_rate - train_null_rate > null_threshold:
            warnings.append(
                f"[skew] {col}: live null-rate {live_null_rate*100:.1f}% vs "
                f"training {train_null_rate*100:.1f}%"
            )

        if info.get("is_numeric") and "p01" in info and "p99" in info:
            # Coerce to numeric — live row may have object dtype if it carries
            # mixed types from upstream feature building.
            non_null = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()
            if len(non_null) == 0:
                continue
            p01, p99 = info["p01"], info["p99"]
            # Skip degenerate distributions (binaries, constants, near-degenerate).
            # If 99% of training values are within a 1e-6 span, range checks
            # produce mostly noise, especially for 0/1 indicator features.
            span_raw = p99 - p01
            if span_raw < 1e-6:
                continue
            span = max(span_raw, 1e-9)
            lo = p01 - range_padding * span
            hi = p99 + range_padding * span
            out_of_range = ((non_null < lo) | (non_null > hi)).any()
            if out_of_range:
                warnings.append(
                    f"[range] {col}: live value(s) outside training "
                    f"[{p01:.3g}, {p99:.3g}] expanded by ±{range_padding*100:.0f}%"
                )
    return warnings


def _log_inference(payload, path=None):
    """Append a single-line JSON record to the inference log.

    No-op (with a printed warning the first time) if the log directory
    doesn't exist or isn't writable. We never want logging to break a
    prediction. Each record is the served feature vector + raw and
    calibrated probabilities + threshold + verdict + timestamp.

    Per Google Rule #29: log served features so future training runs can
    use them, eliminating training-serving skew by construction.
    """
    if path is None:
        path = INFERENCE_LOG_PATH
    try:
        if not os.path.exists(os.path.dirname(path)):
            return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        return True
    except (OSError, TypeError) as exc:
        # Don't crash predictions over logging failures
        if not getattr(_log_inference, "_warned", False):
            print(f"  [WARN] inference logging disabled: {exc}")
            _log_inference._warned = True
        return False


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
    # Keep raw tokens for hours_kind_*: _parse_pvqr_hour collapses "ALL" and missing
    # to the same NaN, but engineer.py distinguishes them in hours_kind_from/to.
    _from_hours_raw = row.get(_pick_col(raw, "from_hours_in_effect"))
    _to_hours_raw   = row.get(_pick_col(raw, "to_hours_in_effect"))
    from_hour = _parse_pvqr_hour(_from_hours_raw)
    to_hour   = _parse_pvqr_hour(_to_hours_raw)
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

    # 19 engineered features that engineer.py builds during training but the
    # original predict.py never set: summons_format, format_x_viol,
    # hours_kind_from/to, hours_in_effect_all, first_observed_filled,
    # agency_x_command, format_x_missing_plate, plus 10 is_missing_* flags
    # and missing_fields_count. Surfaced by the feature audit in May 2026.
    values.update(_build_engineered_extras(
        row=row,
        vcode=vcode,
        issuing_agency=agency,
        issuer_command=issuer_command,
        from_hours_token=_from_hours_raw,
        to_hours_token=_to_hours_raw,
        vehicle_year=vehicle_year,
        plate_id=plate,
    ))

    model_row = pd.DataFrame([{f: values.get(f, np.nan) for f in feature_names}])

    # Apply the rolling-prior fallback BEFORE categorical normalization. Without
    # this, the 27 rolling-prior features are NaN at inference and the model
    # collapses every live ticket to ~1% calibrated probability (it learned
    # NaN-rolling-priors == loss during training). See LIMITATIONS.md
    # §"Live-data drift" — this is a stand-in fix; the long-term fix is
    # MaskTab-style training (random NaN masking on these columns) so the
    # model handles missing rolling priors gracefully.
    rolling_fallback = _load_rolling_prior_means()
    cells_filled = _apply_rolling_prior_fallback(model_row, rolling_fallback)
    if rolling_fallback is None:
        print("  [WARN] models/rolling_prior_means.joblib not found — rolling "
              "priors stay NaN, predictions will collapse toward zero. Run "
              "`python -m src.build_rolling_prior_means`.")
    else:
        print(f"  Rolling-prior fallback filled {cells_filled} NaN cells.")

    # Drift checks: compare the post-fallback feature row against training-time
    # null rates and numeric ranges. Will catch any future "feature flipped to
    # 100% NaN" bug at the moment it ships, instead of waiting for a calibration
    # collapse to be discovered manually.
    audit = _load_feature_audit()
    drift_warnings = _check_feature_drift(model_row, audit)
    for w in drift_warnings:
        print(f"  {w}")

    # CatBoost wants strings for cat columns
    for c in cat_features:
        if c in model_row.columns:
            model_row[c] = model_row[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    prob_raw = float(model.predict_proba(model_row)[0, 1])

    # Apply isotonic calibration. Raw CatBoost probabilities over-predict
    # win rate by ~10–35 points in the [0.40, 0.85) range (ECE 0.152 raw vs
    # 0.023 calibrated). Skip silently if the calibrator file is missing.
    calibrator = _load_calibrator()
    if calibrator is not None:
        prob = float(calibrator.predict(np.asarray([prob_raw]))[0])
    else:
        prob = prob_raw
    pct = prob * 100

    threshold, policy = _load_threshold()

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
    cal_label = "calibrated" if calibrator is not None else "raw (no calibrator found)"
    print(f"\n  Win probability:  {pct:.1f}%  ({cal_label}; raw was {prob_raw*100:.1f}%)")
    print(f"  Threshold:        {threshold*100:.1f}%  (policy: {policy})")

    if prob >= threshold:
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
        print(f"  Model was:        {'CORRECT' if prob >= threshold else 'WRONG'}")
    else:
        print("  Actual outcome:   LOST")
        print(f"  Model was:        {'CORRECT' if prob < threshold else 'WRONG'}")
    print(f"{'-'*56}\n")

    # Inference logging — Google Rule #29. Records the served feature vector
    # plus prediction so future training runs can use what was actually scored
    # and any future training-serving skew can be diagnosed from logs.
    _log_inference({
        "ts":            datetime.utcnow().isoformat() + "Z",
        "summons":       row.get("summons_number") if hasattr(row, "get") else None,
        "plate":         plate,
        "violation":     vcode,
        "precinct":      precinct,
        "county":        county,
        "prob_raw":      float(prob_raw),
        "prob_calibrated": float(prob),
        "threshold":     float(threshold),
        "policy":        policy,
        "verdict":       "DISPUTE" if prob >= threshold else "PAY",
        "actually_won":  None if actually_won is None else bool(actually_won),
        "drift_warnings": drift_warnings,
        "cells_filled":  int(cells_filled),
        # Full served feature row, for future skew diagnosis. Keep last to
        # make truncated logs still useful.
        "features":      {k: (None if pd.isna(v) else v)
                          for k, v in model_row.iloc[0].to_dict().items()},
    })

    return prob


if __name__ == "__main__":
    predict_ticket()
