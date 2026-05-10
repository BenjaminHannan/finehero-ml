# test_calibrated_comparison.py
#
# Re-runs both the synthetic-ticket scoring and the held-out anchor with
# the fitted isotonic calibrator applied. The calibrator was fit on the
# chronological last 10% of training data (n=80,000) and reports:
#   - ECE: 0.152 raw -> 0.023 calibrated
#   - Brier: 0.137 raw -> 0.101 calibrated
# AUC is preserved (isotonic is monotone).
#
# After applying the calibrator, the synthetic-ticket numbers should
# represent literal probabilities, not just ranking scores.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Reuse the synthetic ticket builder
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_hypothetical_tickets import make_scenarios

MODELS_DIR = r"C:\Users\benja\Downloads\finehero-ml\models"
DATA_DIR   = r"C:\Users\benja\Downloads\finehero-ml\data"
CATBOOST_PATH      = os.path.join(MODELS_DIR, "catboost_model.cbm")
METADATA_PATH      = os.path.join(MODELS_DIR, "metadata.json")
CAT_FEATURES_PATH  = os.path.join(MODELS_DIR, "cat_features.joblib")
CALIBRATOR_PATH    = os.path.join(MODELS_DIR, "isotonic_calibrator.joblib")
TEST_SET_PATH      = os.path.join(DATA_DIR, "test_set.joblib")

DISPUTE_THRESHOLD = 0.40


def main():
    print(" Loading model + calibrator...")
    model = CatBoostClassifier()
    model.load_model(CATBOOST_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    cal_pkg = joblib.load(CALIBRATOR_PATH)
    calibrator = cal_pkg["calibrator"]
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    print(f" Calibrator fitted on: {cal_pkg['fitted_on']}, n={cal_pkg['n_calibration']:,}")
    print(f" ECE: {cal_pkg['ece_test_raw']:.3f} raw -> {cal_pkg['ece_test_cal']:.3f} calibrated")
    print(f" Brier: {cal_pkg['brier_test_raw']:.3f} raw -> {cal_pkg['brier_test_cal']:.3f} calibrated\n")

    # ---------------- Synthetic tickets ----------------
    print(" === SYNTHETIC TICKETS (raw vs calibrated) ===")
    scenarios = make_scenarios()
    rows = [{f: row.get(f, np.nan) for f in feature_names}
            for _, _, _, row in scenarios]
    df_syn = pd.DataFrame(rows)
    for c in cat_features:
        if c in df_syn.columns:
            df_syn[c] = df_syn[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    raw_probs = model.predict_proba(df_syn)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    print(f" {'#':<3} {'Scenario':<32} {'Expected':<10} {'Raw':>7} {'Calibrated':>11} {'Verdict'}")
    print(" " + "-" * 80)
    for i, ((label, expected, _, _), rp, cp) in enumerate(zip(scenarios, raw_probs, cal_probs), 1):
        verdict = "DISPUTE" if cp >= DISPUTE_THRESHOLD else "PAY"
        print(f" {i:<3} {label:<32} {expected:<10} {rp*100:>6.1f}% {cp*100:>10.1f}%   {verdict}")
    print()

    # ---------------- Real held-out anchor ----------------
    print(" === HELD-OUT TEST SET (200,000 real tickets) ===")
    X_test, y_test, _ = joblib.load(TEST_SET_PATH)
    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(raw_test)

    auc_raw = roc_auc_score(y_test, raw_test)
    auc_cal = roc_auc_score(y_test, cal_test)
    print(f" AUC raw: {auc_raw:.4f}, calibrated: {auc_cal:.4f}  (should match — isotonic is monotone)")
    print(f" Base rate: {y_test.mean()*100:.1f}% wins\n")

    print(" Calibration table (raw vs calibrated):")
    print(f" {'Bin':<14} {'Count':>8} {'Raw':>10} {'Cal':>10} {'Empirical':>10}")
    print(" " + "-" * 56)
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.01]
    df_t = pd.DataFrame({"raw": raw_test, "cal": cal_test, "y": y_test.values})
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df_t["raw"] >= lo) & (df_t["raw"] < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_raw = df_t.loc[mask, "raw"].mean()
        mean_cal = df_t.loc[mask, "cal"].mean()
        emp = df_t.loc[mask, "y"].mean()
        bin_label = f"[{lo:.2f},{hi:.2f})"
        print(f" {bin_label:<14} {n:>8,} {mean_raw*100:>9.1f}% {mean_cal*100:>9.1f}% {emp*100:>9.1f}%")
    print()

    n_dispute_raw = int((raw_test >= DISPUTE_THRESHOLD).sum())
    n_dispute_cal = int((cal_test >= DISPUTE_THRESHOLD).sum())
    print(f" Tickets crossing 0.4 threshold:")
    print(f"   Raw:        {n_dispute_raw:>7,} ({n_dispute_raw/len(raw_test)*100:.1f}%) — win rate among them: "
          f"{y_test.values[raw_test >= DISPUTE_THRESHOLD].mean()*100:.1f}%")
    print(f"   Calibrated: {n_dispute_cal:>7,} ({n_dispute_cal/len(raw_test)*100:.1f}%) — win rate among them: "
          f"{y_test.values[cal_test >= DISPUTE_THRESHOLD].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
