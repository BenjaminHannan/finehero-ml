# predict_ensemble.py — Score a random ticket with the full CatBoost + LightGBM
# + XGBoost rank-blended ensemble. Requires all three models to have been
# trained (src/train.py, src/train_lgb.py, src/train_xgb.py) and their test
# predictions saved (used as reference distributions for rank-percentile).
#
# For a SINGLE ticket, pure rank-averaging isn't meaningful (there's nothing to
# rank against), so we compute each model's percentile *within its saved
# test-set distribution* and average those percentiles. That is the rank-blend
# equivalent for single-point inference.
#
# Shows: three raw probabilities, simple probability average, rank-percentile
# blend, and a final verdict.

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier

import predict  # reuse helpers from the main predict.py

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

CATBOOST_PATH      = os.path.join(MODELS_DIR, "catboost_model.cbm")
LGB_MODEL_PATH     = os.path.join(MODELS_DIR, "lgb_model.txt")
XGB_MODEL_PATH     = os.path.join(MODELS_DIR, "xgb_model.json")
LGB_ENCODER_PATH   = os.path.join(MODELS_DIR, "lgb_ord_encoder.joblib")
XGB_ENCODER_PATH   = os.path.join(MODELS_DIR, "xgb_ord_encoder.joblib")
LGB_PREDS_PATH     = os.path.join(MODELS_DIR, "lgb_test_preds.joblib")
XGB_PREDS_PATH     = os.path.join(MODELS_DIR, "xgb_test_preds.joblib")
METADATA_PATH      = os.path.join(MODELS_DIR, "metadata.json")
CAT_FEATURES_PATH  = os.path.join(MODELS_DIR, "cat_features.joblib")
PLATE_HISTORY_PATH = os.path.join(MODELS_DIR, "plate_history_map.joblib")
VIOLATIONS_PATH    = os.path.join(DATA_DIR, "violations_raw.csv")

DISPUTE_THRESHOLD = predict.DISPUTE_THRESHOLD


def _build_values_for_row(row, raw, plate_history):
    """Replicate predict.py's row → values dict (minus the CatBoost scoring).

    Returns (values, display, plate, actually_won_hint).
    """
    vcode_col    = predict._pick_col(raw, "violation", "violation_code")
    precinct_col = predict._pick_col(raw, "precinct") or predict._find_col(raw, "precinct")
    county_col   = predict._find_col(raw, "county")
    agency_col   = predict._pick_col(raw, "issuing_agency") or predict._find_col(raw, "agency")
    license_col  = predict._pick_col(raw, "license_type", "plate_type")
    state_col    = predict._pick_col(raw, "state", "registration_state")
    plate_col    = predict._pick_col(raw, "plate", "plate_id")
    vtime_col    = predict._find_col(raw, "violation_time")
    date_col     = predict._find_col(raw, "issue_date")
    fine_col     = predict._pick_col(raw, "fine_amount")
    vdesc_col    = predict._pick_col(raw, "violation_description")

    vcode        = predict._get(row, vcode_col, "UNKNOWN")
    precinct     = predict._get(row, precinct_col, "UNKNOWN")
    county_raw   = predict._get(row, county_col, "UNKNOWN").upper()
    county       = predict.COUNTY_NORM.get(county_raw, county_raw)
    agency       = predict._get(row, agency_col, "UNKNOWN")
    license_type = predict._get(row, license_col, "UNKNOWN")
    state        = predict._get(row, state_col, "UNKNOWN")
    plate        = predict._get(row, plate_col, "UNKNOWN")

    issuer_code                       = predict._get(row, predict._pick_col(raw, "issuer_code"), "UNKNOWN")
    issuer_command                    = predict._get(row, predict._pick_col(raw, "issuer_command"), "UNKNOWN")
    issuer_squad                      = predict._get(row, predict._pick_col(raw, "issuer_squad"), "UNKNOWN")
    street_name                       = predict._get(row, predict._pick_col(raw, "street_name"), "UNKNOWN").upper()
    vehicle_make                      = predict._get(row, predict._pick_col(raw, "vehicle_make"), "UNKNOWN").upper()
    vehicle_body_type                 = predict._get(row, predict._pick_col(raw, "vehicle_body_type"), "UNKNOWN").upper()
    violation_legal_code              = predict._get(row, predict._pick_col(raw, "violation_legal_code"), "UNKNOWN")
    law_section                       = predict._get(row, predict._pick_col(raw, "law_section"), "UNKNOWN")
    sub_division                      = predict._get(row, predict._pick_col(raw, "sub_division"), "UNKNOWN")
    violation_in_front_of_or_opposite = predict._get(row, predict._pick_col(raw, "violation_in_front_of_or_opposite"), "UNKNOWN").upper()
    days_parking_in_effect            = predict._get(row, predict._pick_col(raw, "days_parking_in_effect"), "UNKNOWN").upper()

    hour_val = predict._parse_vtime(row[vtime_col]) if vtime_col else np.nan
    parsed_dt = pd.to_datetime(row[date_col], errors="coerce") if date_col else None
    if parsed_dt is not None and not pd.isnull(parsed_dt):
        day_of_week = float(parsed_dt.dayofweek)
        month       = float(parsed_dt.month)
        is_holiday  = predict._is_holiday(parsed_dt)
    else:
        day_of_week = month = is_holiday = np.nan

    feet_from_curb = pd.to_numeric(row.get(predict._pick_col(raw, "feet_from_curb")), errors="coerce")
    vehicle_year   = pd.to_numeric(row.get(predict._pick_col(raw, "vehicle_year")),   errors="coerce")
    from_hour      = predict._parse_pvqr_hour(row.get(predict._pick_col(raw, "from_hours_in_effect")))
    to_hour        = predict._parse_pvqr_hour(row.get(predict._pick_col(raw, "to_hours_in_effect")))
    within_posted_hours = np.nan
    if not np.isnan(hour_val) and not np.isnan(from_hour) and not np.isnan(to_hour):
        if from_hour <= to_hour:
            within_posted_hours = float(from_hour <= hour_val <= to_hour)
        else:
            within_posted_hours = float(hour_val >= from_hour or hour_val <= to_hour)

    fine_amount = pd.to_numeric(row.get("fine_amount"), errors="coerce") if fine_col else np.nan

    kw_source = predict._get(row, vdesc_col, vcode)
    kw_values = {feat: float(bool(re.search(pat, kw_source)))
                 for feat, pat in predict.KEYWORD_GROUPS.items()}

    weather = {"precipitation": 0.0, "visibility": 10000.0, "wind_speed": 0.0,
               "weather_code": 0, "is_bad_weather": 0.0}
    if parsed_dt is not None and not pd.isnull(parsed_dt):
        today = datetime.now().strftime("%Y-%m-%d")
        ds = parsed_dt.strftime("%Y-%m-%d")
        if ds < today:
            lat, lon = predict.COUNTY_COORDS.get(county, predict.DEFAULT_COORDS)
            h_int = int(hour_val) if not np.isnan(hour_val) else 12
            print(f"  Fetching weather {ds} h={h_int} {county}...")
            weather = predict._fetch_weather_point(lat, lon, ds, h_int)

    viol_x_precinct = f"{vcode}_{precinct}"
    viol_x_license  = f"{vcode}_{license_type}"
    hour_bin = int(hour_val) if not np.isnan(hour_val) else -1
    dow_bin  = int(day_of_week) if not np.isnan(day_of_week) else -1
    hour_x_dow = f"{hour_bin}_{dow_bin}"

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
        plate_prior_win_rate = 0.3

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

    # Cyclical time encodings
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

    display = {
        "violation":       vcode,
        "precinct":        precinct,
        "county":          county,
        "agency":          agency,
        "license_type":    license_type,
        "state":           state,
        "plate":           plate,
        "street":          street_name,
        "side":            violation_in_front_of_or_opposite,
        "vehicle":         f"{vehicle_year if not pd.isna(vehicle_year) else '?'} "
                           f"{vehicle_make} ({vehicle_body_type})",
        "fine_amount":     fine_amount,
        "issue_date":      row.get("issue_date", "?"),
        "violation_time":  row.get("violation_time", "?"),
        "hours_in_effect": f"{from_hour}\u2013{to_hour}  within={within_posted_hours}",
        "days_in_effect":  days_parking_in_effect,
        "feet_from_curb":  feet_from_curb,
        "plate_history":   f"{int(plate_prior_ticket_count)} prior tickets, "
                           f"win rate {plate_prior_win_rate*100:.1f}%",
        "weather":         f"precip={weather['precipitation']}mm  vis={weather['visibility']}m",
        "keywords":        ", ".join(k for k, v in kw_values.items() if v) or "(none)",
        "status":          row.get("violation_status", "?"),
    }
    return values, display, plate


def _percentile_in(ref_probs, new_prob):
    """Fraction of reference predictions BELOW new_prob. In [0, 1]."""
    return float(np.mean(ref_probs < new_prob))


def predict_ticket_ensemble():
    for p in (CATBOOST_PATH, LGB_MODEL_PATH, XGB_MODEL_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run src/train.py, src/train_lgb.py and src/train_xgb.py first."
            )

    print("  Loading CatBoost + LightGBM + XGBoost...")
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)

    import lightgbm as lgb
    import xgboost as xgb
    lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    lgb_enc = joblib.load(LGB_ENCODER_PATH)
    xgb_enc = joblib.load(XGB_ENCODER_PATH)
    lgb_ref = joblib.load(LGB_PREDS_PATH)["probs"]
    xgb_ref = joblib.load(XGB_PREDS_PATH)["probs"]

    cat_features = joblib.load(CAT_FEATURES_PATH)
    plate_history = joblib.load(PLATE_HISTORY_PATH) if os.path.exists(PLATE_HISTORY_PATH) else None
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    print(f"  CatBoost time-aware AUC: {meta['auc_score']:.4f}")

    # Reference CatBoost distribution: score the saved test set
    test_path = os.path.join(DATA_DIR, "test_set.joblib")
    if os.path.exists(test_path):
        X_test, y_test, _ = joblib.load(test_path)
        X_ref = X_test.copy()
        for c in cat_features:
            if c in X_ref.columns:
                X_ref[c] = X_ref[c].fillna("UNKNOWN").astype(str)
        cb_ref = cb.predict_proba(X_ref)[:, 1]
    else:
        cb_ref = None
        print("  [WARN] data/test_set.joblib missing — rank percentile disabled for CatBoost.")

    print("  Picking a random disputed ticket (50/50 won vs lost)...")
    raw = pd.read_csv(VIOLATIONS_PATH, low_memory=False)
    row, actually_won = predict._pick_disputed_row(raw)
    values, display, plate = _build_values_for_row(row, raw, plate_history)

    # --- CatBoost row (strings for cats) ---
    cb_row = pd.DataFrame([{f: values.get(f, np.nan) for f in feature_names}])
    for c in cat_features:
        if c in cb_row.columns:
            cb_row[c] = cb_row[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    prob_cb = float(cb.predict_proba(cb_row)[0, 1])

    # --- LGB row (ordinal-encoded cats) ---
    # LGB's feature order is the same as CatBoost's feature_names (both trained
    # on the same features.csv with meta cols dropped).
    lgb_row = pd.DataFrame([{f: values.get(f, np.nan) for f in feature_names}])
    for c in cat_features:
        if c in lgb_row.columns:
            lgb_row[c] = lgb_row[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    lgb_cat_present = [c for c in cat_features if c in lgb_row.columns]
    lgb_row[lgb_cat_present] = lgb_enc.transform(lgb_row[lgb_cat_present])
    prob_lgb = float(lgb_model.predict(lgb_row)[0])

    # --- XGB row (ordinal-encoded + category dtype) ---
    xgb_row = pd.DataFrame([{f: values.get(f, np.nan) for f in feature_names}])
    for c in cat_features:
        if c in xgb_row.columns:
            xgb_row[c] = xgb_row[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    xgb_cat_present = [c for c in cat_features if c in xgb_row.columns]
    xgb_row[xgb_cat_present] = xgb_enc.transform(xgb_row[xgb_cat_present])
    for c in xgb_cat_present:
        xgb_row[c] = xgb_row[c].astype("category")
    dmat = xgb.DMatrix(xgb_row, enable_categorical=True)
    prob_xgb = float(xgb_model.predict(dmat)[0])

    # --- Rank-percentile blend ---
    pct_cb  = _percentile_in(cb_ref, prob_cb) if cb_ref is not None else None
    pct_lgb = _percentile_in(lgb_ref, prob_lgb)
    pct_xgb = _percentile_in(xgb_ref, prob_xgb)

    avail = [p for p in (pct_cb, pct_lgb, pct_xgb) if p is not None]
    rank_blend = float(np.mean(avail)) if avail else None
    prob_avg   = float(np.mean([prob_cb, prob_lgb, prob_xgb]))

    # --- Display ---
    print(f"\n{'-'*56}")
    print("  Ticket details:")
    for k, v in display.items():
        print(f"    {k:<22} {v}")
    print(f"{'-'*56}")
    print("\n  Per-model win probability:")
    print(f"    CatBoost        {prob_cb*100:5.1f}%"
          f"   (test-set percentile: {pct_cb*100:5.1f}%)" if pct_cb is not None else
          f"    CatBoost        {prob_cb*100:5.1f}%")
    print(f"    LightGBM        {prob_lgb*100:5.1f}%   (test-set percentile: {pct_lgb*100:5.1f}%)")
    print(f"    XGBoost         {prob_xgb*100:5.1f}%   (test-set percentile: {pct_xgb*100:5.1f}%)")
    print(f"\n  Blends:")
    print(f"    Probability avg  {prob_avg*100:5.1f}%")
    if rank_blend is not None:
        print(f"    Rank-percentile  {rank_blend*100:5.1f}%  <- this is the AUC-optimal blend")

    # Verdict off the rank-percentile blend — AUC-optimal (§5.3).
    # XGBoost is systematically overconfident due to scale_pos_weight, so
    # prob_avg is pulled high on almost every ticket. The rank-blend is
    # calibrated against each model's actual test-set distribution, making
    # it a true percentile signal. Threshold 0.50 = "above median disputed
    # ticket" which maps cleanly to "worth disputing".
    verdict_prob = rank_blend if rank_blend is not None else prob_avg
    RANK_THRESHOLD = 0.50
    if verdict_prob >= RANK_THRESHOLD:
        verdict = "Worth disputing"
        advice = "Gather evidence (photos, receipts, signage) and submit a dispute."
    else:
        verdict = "Pay this one"
        advice = "The odds are against you. Paying avoids late fees and penalties."
    print(f"\n  Verdict:          {verdict}")
    print(f"  {advice}")

    print(f"\n{'-'*56}")
    if actually_won is None:
        print("  Actual outcome:   Unknown")
    elif actually_won:
        print("  Actual outcome:   WON")
        print(f"  Ensemble was:     "
              f"{'CORRECT' if verdict_prob >= RANK_THRESHOLD else 'WRONG'}")
    else:
        print("  Actual outcome:   LOST")
        print(f"  Ensemble was:     "
              f"{'CORRECT' if verdict_prob < RANK_THRESHOLD else 'WRONG'}")
    print(f"{'-'*56}\n")

    return {"catboost": prob_cb, "lightgbm": prob_lgb, "xgboost": prob_xgb,
            "prob_avg": prob_avg, "rank_blend": rank_blend,
            "actually_won": actually_won}


if __name__ == "__main__":
    predict_ticket_ensemble()
