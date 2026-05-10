# test_live_nyc_tickets.py
#
# Fetches fresh tickets from NYC Open Data (the same Socrata APIs the model
# was trained on), dedupes against the training set's summons_numbers, attempts
# pvqr enrichment per fiscal year, runs them through the same feature pipeline
# as predict.py, and scores them with the calibrated CatBoost model.
#
# This is the most honest possible OOS test: real tickets the model has never
# seen, with ground-truth hearing outcomes.

import os
import re
import io
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict  # reuse helpers

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

# NYC fiscal year endpoints (FY = Jul 1 – Jun 30)
PVQR_FISCAL_ENDPOINTS = {
    "FY2022": ("7mxj-7a6y", "2021-07-01", "2022-06-30"),
    "FY2023": ("869v-vr48", "2022-07-01", "2023-06-30"),
    "FY2024": ("8zf9-spf8", "2023-07-01", "2024-06-30"),
    "FY2025": ("m5vz-tzqv", "2024-07-01", "2025-06-30"),
    "FY2026": ("pvqr-7yc4", "2025-07-01", "2026-06-30"),
}
OPEN_PARKING_URL = "https://data.cityofnewyork.us/resource/nc67-uf89.csv"


def fetch_outcomes(target_rows: int = 500) -> pd.DataFrame:
    """Pull tickets with clean WIN/LOSS hearing outcomes, newest first."""
    print(f" Fetching outcomes (newest first, target {target_rows} rows)...")
    where = "violation_status IN ('HEARING HELD-NOT GUILTY','HEARING HELD-GUILTY')"
    r = requests.get(
        OPEN_PARKING_URL,
        params={"$limit": target_rows, "$where": where, "$order": "issue_date DESC"},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    print(f"   got {len(df)} rows; status mix: {df['violation_status'].value_counts().to_dict()}")
    return df


def fiscal_year(dt) -> str:
    if pd.isnull(dt):
        return None
    fy_year = dt.year if dt.month < 7 else dt.year + 1
    return f"FY{fy_year}"


def fetch_pvqr_for_summons(summons_numbers, fy: str) -> pd.DataFrame:
    """Fetch pvqr rows for a list of summons_numbers from one FY endpoint."""
    if fy not in PVQR_FISCAL_ENDPOINTS or not summons_numbers:
        return pd.DataFrame()
    endpoint = PVQR_FISCAL_ENDPOINTS[fy][0]
    url = f"https://data.cityofnewyork.us/resource/{endpoint}.csv"
    # Socrata accepts IN-list filters; cap at ~100 per query to keep URL sane
    out = []
    for i in range(0, len(summons_numbers), 80):
        batch = summons_numbers[i:i + 80]
        in_list = ",".join(str(s) for s in batch)
        try:
            r = requests.get(url, params={
                "$limit": 200,
                "$where": f"summons_number IN ({in_list})"
            }, timeout=60)
            if r.status_code == 200:
                chunk = pd.read_csv(io.StringIO(r.text), low_memory=False)
                if not chunk.empty:
                    out.append(chunk)
            time.sleep(0.3)
        except Exception as exc:
            print(f"   [WARN] pvqr fetch failed for {fy}: {exc}")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def dedupe_against_training(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose summons_number appears in violations_raw.csv."""
    train_path = os.path.join(DATA_DIR, "violations_raw.csv")
    train_summons = set(
        pd.read_csv(train_path, usecols=["summons_number"], low_memory=False)
        ["summons_number"].astype(str)
    )
    before = len(df)
    df = df[~df["summons_number"].astype(str).isin(train_summons)].copy()
    print(f" Deduped against training: {before} -> {len(df)} fresh tickets")
    return df


def build_feature_row(row, plate_history, feature_names):
    """Build a single-row feature dict using predict.py's helpers."""
    # Pretend `row` is the joined row; pass it to the same column-discovery logic.
    # We construct a one-row DataFrame to satisfy `_pick_col` calls.
    df_one = pd.DataFrame([row])
    raw_cols = df_one  # alias for readability

    vcode_col    = predict._pick_col(raw_cols, "violation", "violation_code")
    precinct_col = predict._pick_col(raw_cols, "precinct") or predict._find_col(raw_cols, "precinct")
    county_col   = predict._find_col(raw_cols, "county")
    agency_col   = predict._pick_col(raw_cols, "issuing_agency") or predict._find_col(raw_cols, "agency")
    license_col  = predict._pick_col(raw_cols, "license_type", "plate_type")
    state_col    = predict._pick_col(raw_cols, "state", "registration_state")
    plate_col    = predict._pick_col(raw_cols, "plate", "plate_id")
    vtime_col    = predict._find_col(raw_cols, "violation_time")
    date_col     = predict._find_col(raw_cols, "issue_date")
    fine_col     = predict._pick_col(raw_cols, "fine_amount")
    vdesc_col    = predict._pick_col(raw_cols, "violation_description")

    vcode = predict._get(row, vcode_col, "UNKNOWN")
    precinct = predict._get(row, precinct_col, "UNKNOWN")
    county_raw = predict._get(row, county_col, "UNKNOWN").upper()
    county = predict.COUNTY_NORM.get(county_raw, county_raw)
    agency = predict._get(row, agency_col, "UNKNOWN")
    license_type = predict._get(row, license_col, "UNKNOWN")
    state = predict._get(row, state_col, "UNKNOWN")
    # Canonicalize plate the same way engineer.py does so plate_history_map hits.
    plate = predict.canonicalize_plate(predict._get(row, plate_col, "UNKNOWN"))

    issuer_code         = predict._get(row, predict._pick_col(raw_cols, "issuer_code"), "UNKNOWN")
    issuer_command      = predict._get(row, predict._pick_col(raw_cols, "issuer_command"), "UNKNOWN")
    issuer_squad        = predict._get(row, predict._pick_col(raw_cols, "issuer_squad"), "UNKNOWN")
    street_name         = predict._get(row, predict._pick_col(raw_cols, "street_name"), "UNKNOWN").upper()
    vehicle_make        = predict._get(row, predict._pick_col(raw_cols, "vehicle_make"), "UNKNOWN").upper()
    vehicle_body_type   = predict._get(row, predict._pick_col(raw_cols, "vehicle_body_type"), "UNKNOWN").upper()
    violation_legal_code = predict._get(row, predict._pick_col(raw_cols, "violation_legal_code"), "UNKNOWN")
    law_section         = predict._get(row, predict._pick_col(raw_cols, "law_section"), "UNKNOWN")
    sub_division        = predict._get(row, predict._pick_col(raw_cols, "sub_division"), "UNKNOWN")
    side                = predict._get(row, predict._pick_col(raw_cols, "violation_in_front_of_or_opposite"), "UNKNOWN").upper()
    days_in_effect      = predict._get(row, predict._pick_col(raw_cols, "days_parking_in_effect"), "UNKNOWN").upper()

    hour_val = predict._parse_vtime(row[vtime_col]) if vtime_col else np.nan
    parsed_dt = pd.to_datetime(row[date_col], errors="coerce") if date_col else None
    if parsed_dt is not None and not pd.isnull(parsed_dt):
        day_of_week = float(parsed_dt.dayofweek)
        month = float(parsed_dt.month)
        is_holiday = predict._is_holiday(parsed_dt)
    else:
        day_of_week = month = is_holiday = np.nan

    feet_from_curb = pd.to_numeric(row.get(predict._pick_col(raw_cols, "feet_from_curb")), errors="coerce")
    vehicle_year   = pd.to_numeric(row.get(predict._pick_col(raw_cols, "vehicle_year")),   errors="coerce")
    # Keep raw tokens for hours_kind_*: _parse_pvqr_hour collapses "ALL" and missing
    # to the same NaN, but engineer.py distinguishes them in hours_kind_from/to.
    _from_hours_raw = row.get(predict._pick_col(raw_cols, "from_hours_in_effect"))
    _to_hours_raw   = row.get(predict._pick_col(raw_cols, "to_hours_in_effect"))
    from_hour = predict._parse_pvqr_hour(_from_hours_raw)
    to_hour   = predict._parse_pvqr_hour(_to_hours_raw)
    within = np.nan
    if not np.isnan(hour_val) and not np.isnan(from_hour) and not np.isnan(to_hour):
        if from_hour <= to_hour:
            within = float(from_hour <= hour_val <= to_hour)
        else:
            within = float(hour_val >= from_hour or hour_val <= to_hour)

    fine_amount = pd.to_numeric(row.get("fine_amount"), errors="coerce") if fine_col else np.nan

    kw_source = predict._get(row, vdesc_col, vcode)
    kw_values = {feat: float(bool(re.search(pat, kw_source)))
                 for feat, pat in predict.KEYWORD_GROUPS.items()}

    # Skip live weather fetch for speed — pass zeros (slightly disadvantages
    # bad-weather tickets, but most won't hit that signal anyway).
    weather = {"precipitation": 0.0, "visibility": 10000.0, "wind_speed": 0.0,
               "weather_code": 0, "is_bad_weather": 0.0}

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
        "issuer_squad": issuer_squad, "street_name": street_name,
        "vehicle_make": vehicle_make, "vehicle_body_type": vehicle_body_type,
        "violation_legal_code": violation_legal_code, "law_section": law_section,
        "sub_division": sub_division,
        "violation_in_front_of_or_opposite": side,
        "days_parking_in_effect": days_in_effect,
        "viol_x_precinct": viol_x_precinct, "viol_x_license": viol_x_license,
        "hour_x_dow": hour_x_dow,
        "hour_of_offense": hour_val, "day_of_week": day_of_week, "month": month,
        "is_holiday": is_holiday, "fine_amount": fine_amount,
        "feet_from_curb": feet_from_curb, "vehicle_year": vehicle_year,
        "from_hour": from_hour, "to_hour": to_hour,
        "within_posted_hours": within,
        "plate_prior_ticket_count": plate_prior_ticket_count,
        "plate_prior_win_rate": plate_prior_win_rate,
        **kw_values, **weather,
    }
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

    # Mirror engineer.py's 19 additional features (summons_format, format_x_viol,
    # hours_kind_from/to, hours_in_effect_all, first_observed_filled,
    # agency_x_command, format_x_missing_plate, 10 is_missing_* flags,
    # missing_fields_count). Surfaced by the feature audit in May 2026.
    values.update(predict._build_engineered_extras(
        row=row,
        vcode=vcode,
        issuing_agency=agency,
        issuer_command=issuer_command,
        from_hours_token=_from_hours_raw,
        to_hours_token=_to_hours_raw,
        vehicle_year=vehicle_year,
        plate_id=plate,
    ))

    return {f: values.get(f, np.nan) for f in feature_names}, {
        "summons_number": row.get("summons_number"),
        "violation": vcode, "precinct": precinct, "county": county,
        "issue_date": row.get("issue_date"), "fine_amount": fine_amount,
        "within_hours": within, "kw": ",".join(k for k, v in kw_values.items() if v) or "(none)",
    }


def main():
    print(" Loading model + calibrator + supporting maps...")
    model = CatBoostClassifier()
    model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
    cat_features = joblib.load(os.path.join(MODELS_DIR, "cat_features.joblib"))
    plate_history = joblib.load(os.path.join(MODELS_DIR, "plate_history_map.joblib"))
    cal_pkg = joblib.load(os.path.join(MODELS_DIR, "isotonic_calibrator.joblib"))
    calibrator = cal_pkg["calibrator"]
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    with open(os.path.join(MODELS_DIR, "dispute_threshold.json")) as f:
        thr_cfg = json.load(f)
    threshold = thr_cfg["default_threshold"]
    print(f" Model AUC {meta['auc_score']:.4f}, threshold {threshold:.3f} (policy: {thr_cfg['default_policy']})\n")

    # 1. Fetch outcomes
    outcomes = fetch_outcomes(target_rows=500)
    outcomes["won"] = (outcomes["violation_status"] == "HEARING HELD-NOT GUILTY").astype(int)

    # 2. Dedupe vs training
    fresh = dedupe_against_training(outcomes)
    if len(fresh) == 0:
        print(" No fresh tickets after dedupe — bailing.")
        return

    # 3. Group by FY and fetch pvqr enrichment
    fresh["fy"] = fresh["issue_date"].apply(fiscal_year)
    print(f" Fiscal-year split: {fresh['fy'].value_counts().to_dict()}")
    pvqr_frames = []
    for fy, group in fresh.groupby("fy"):
        if fy not in PVQR_FISCAL_ENDPOINTS:
            print(f"   no pvqr endpoint for {fy}, skipping enrichment ({len(group)} rows)")
            continue
        ss = group["summons_number"].astype(int).tolist()
        print(f"   fetching pvqr for {fy} ({len(ss)} summons)...")
        pv = fetch_pvqr_for_summons(ss, fy)
        if not pv.empty:
            pvqr_frames.append(pv)
            print(f"     got {len(pv)} pvqr rows")
    pvqr = pd.concat(pvqr_frames, ignore_index=True) if pvqr_frames else pd.DataFrame()

    # 4. Join
    if not pvqr.empty:
        fresh["summons_str"] = fresh["summons_number"].astype(str)
        pvqr["summons_str"] = pvqr["summons_number"].astype(str)
        pvqr_to_join = pvqr.drop(columns=[c for c in pvqr.columns
                                          if c in fresh.columns and c != "summons_str"])
        joined = fresh.merge(pvqr_to_join, on="summons_str", how="left")
    else:
        joined = fresh.copy()
    n_pvqr = joined["from_hours_in_effect"].notna().sum() if "from_hours_in_effect" in joined.columns else 0
    print(f" pvqr enrichment hit rate: {n_pvqr}/{len(joined)} ({n_pvqr/len(joined)*100:.1f}%)\n")

    # 5. Build feature rows
    rows, displays = [], []
    for _, row in joined.iterrows():
        feat, disp = build_feature_row(row, plate_history, feature_names)
        rows.append(feat)
        displays.append(disp)
    df_feat = pd.DataFrame(rows)
    for c in cat_features:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    # 6. Score
    raw_probs = model.predict_proba(df_feat)[:, 1]
    cal_probs = calibrator.predict(raw_probs)
    y = joined["won"].values

    # 7. Per-ticket display (cap at 25 for readability)
    print(" Per-ticket scores (showing first 25):")
    print(f" {'#':<3} {'Actual':<5} {'Raw':>6} {'Cal':>6} {'Verdict':<8} {'OK':<3} "
          f"{'viol':<7} {'prec':<5} {'within':<7} kw")
    print(" " + "-" * 90)
    correct = 0
    for i, (rp, cp, actual, disp) in enumerate(
        list(zip(raw_probs, cal_probs, y, displays))[:25], 1
    ):
        verdict = "DISPUTE" if cp >= threshold else "PAY"
        ok = (verdict == "DISPUTE") == (actual == 1)
        actual_str = "WON" if actual == 1 else "LOST"
        print(f" {i:<3} {actual_str:<5} {rp*100:>5.1f}% {cp*100:>5.1f}% {verdict:<8} "
              f"{('OK' if ok else 'MISS'):<3} {str(disp['violation'])[:7]:<7} "
              f"{str(disp['precinct'])[:5]:<5} "
              f"{('?' if pd.isna(disp['within_hours']) else ('Y' if disp['within_hours']==1 else 'N')):<7} "
              f"{disp['kw']}")
    print(" " + "-" * 90)

    # 8. Aggregate metrics
    pred = (cal_probs >= threshold).astype(int)
    correct = (pred == y).sum()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = correct / len(y)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    try:
        auc = roc_auc_score(y, cal_probs)
    except ValueError:
        auc = float("nan")

    print(f"\n Aggregate on {len(y)} fresh tickets:")
    print(f"   Base rate (true win rate): {y.mean()*100:.1f}%")
    print(f"   AUC:        {auc:.4f}  (model card claims 0.8694 on internal holdout)")
    print(f"   Accuracy:   {acc*100:.1f}%")
    print(f"   Precision:  {prec*100:.1f}%  (P(won | DISPUTE))")
    print(f"   Recall:     {rec*100:.1f}%  (P(DISPUTE | won))")
    print(f"   Confusion:  TP={tp}  FN={fn}  FP={fp}  TN={tn}")

    # Calibration check on this fresh sample
    bins = [0, 0.10, 0.30, 0.50, 0.85, 1.01]
    print(f"\n Calibration on fresh sample:")
    print(f"   {'Cal bin':<14} {'count':>6} {'mean cal':>10} {'empirical':>10}")
    df_c = pd.DataFrame({"cal": cal_probs, "y": y})
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df_c["cal"] >= lo) & (df_c["cal"] < hi)
        if mask.sum() == 0:
            continue
        print(f"   [{lo:.2f},{hi:.2f})    {int(mask.sum()):>6} "
              f"{df_c.loc[mask,'cal'].mean()*100:>9.1f}% "
              f"{df_c.loc[mask,'y'].mean()*100:>9.1f}%")


if __name__ == "__main__":
    main()
