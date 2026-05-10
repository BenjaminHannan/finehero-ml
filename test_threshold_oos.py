# test_threshold_oos.py
#
# Out-of-sample threshold tuning. The existing `dispute_threshold.json` was
# tuned on the entire 200k held-out test set, which is the chronological last
# 20% of the data (per src/train.py:140). Numbers reported in that file are
# from-fold — the same data was used to pick the threshold and to report its
# performance.
#
# This script splits the chronological test set in half:
#   - Tune slice  = older 100k (X_test rows [0, 100000))
#   - Report slice = newer 100k (X_test rows [100000, 200000))
#
# We tune each policy on the tune slice, then evaluate on the report slice.
# The report numbers are honest out-of-sample.
#
# We also report calibration ECE on the report slice — this tells us whether
# the existing isotonic calibrator (fit on the eval slice that's
# chronologically adjacent to the start of the test set) is still well-
# calibrated on the most recent slice, or whether temporal drift has
# accumulated.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)

MODELS_DIR = r"C:\Users\benja\Downloads\finehero-ml\models"
DATA_DIR   = r"C:\Users\benja\Downloads\finehero-ml\data"

CATBOOST_PATH   = os.path.join(MODELS_DIR, "catboost_model.cbm")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "isotonic_calibrator.joblib")
TEST_SET_PATH   = os.path.join(DATA_DIR, "test_set.joblib")
EXISTING_THR_PATH = os.path.join(MODELS_DIR, "dispute_threshold.json")


def expected_calibration_error(probs, y, n_bins=15):
    """ECE with equal-frequency bins."""
    order = np.argsort(probs)
    p_sorted = probs[order]
    y_sorted = y[order]
    bin_edges = np.linspace(0, len(p_sorted), n_bins + 1, dtype=int)
    n = len(p_sorted)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        mean_p = p_sorted[lo:hi].mean()
        emp = y_sorted[lo:hi].mean()
        weight = (hi - lo) / n
        ece += weight * abs(mean_p - emp)
    return ece


def best_threshold_by(probs, y, metric_fn):
    """Find threshold in [0.01, 0.99] that maximizes metric_fn(y, pred)."""
    candidates = np.linspace(0.01, 0.99, 99)
    best_thr, best_val = candidates[0], -np.inf
    for thr in candidates:
        pred = (probs >= thr).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        v = metric_fn(y, pred)
        if v > best_val:
            best_val, best_thr = v, thr
    return float(best_thr), float(best_val)


def youden_j(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return tpr - fpr


def match_baserate_threshold(probs, y):
    """Threshold s.t. predicted-positive rate equals empirical positive rate."""
    target = y.mean()
    sorted_probs = np.sort(probs)[::-1]
    k = int(round(target * len(probs)))
    if k <= 0 or k >= len(probs):
        return 0.5, np.nan
    thr = (sorted_probs[k - 1] + sorted_probs[k]) / 2
    pred = (probs >= thr).astype(int)
    return float(thr), float(f1_score(y, pred))


def report_metrics(probs, y, thr, label):
    pred = (probs >= thr).astype(int)
    base = y.mean()
    if pred.sum() == 0:
        prec, rec, f1, lift = float("nan"), 0.0, 0.0, float("nan")
    else:
        prec = precision_score(y, pred)
        rec  = recall_score(y, pred)
        f1   = f1_score(y, pred)
        lift = prec / base
    pct = pred.mean() * 100
    return {"label": label, "thr": thr, "pct_dispute": pct,
            "precision": prec, "recall": rec, "f1": f1, "lift": lift}


def main():
    print(" Loading model + calibrator + chronological test set...")
    m = CatBoostClassifier(); m.load_model(CATBOOST_PATH)
    cal = joblib.load(CALIBRATOR_PATH)["calibrator"]
    X, y, _ = joblib.load(TEST_SET_PATH)
    y = y.values
    n = len(X)
    print(f" Test set: {n:,} rows (chronological tail, in date order)")

    print(" Scoring full test set...")
    raw_full = m.predict_proba(X)[:, 1]
    cal_full = cal.predict(raw_full)

    cut = n // 2
    print(f" Split: tune = rows [0, {cut:,})  |  report = rows [{cut:,}, {n:,})\n")

    raw_tune,  raw_rep  = raw_full[:cut],  raw_full[cut:]
    cal_tune,  cal_rep  = cal_full[:cut],  cal_full[cut:]
    y_tune,    y_rep    = y[:cut],         y[cut:]

    # AUC on each half — sanity check + drift signal
    auc_tune = roc_auc_score(y_tune, cal_tune)
    auc_rep  = roc_auc_score(y_rep,  cal_rep)
    print(f" AUC tune slice:   {auc_tune:.4f}  (base rate {y_tune.mean()*100:.2f}%)")
    print(f" AUC report slice: {auc_rep:.4f}  (base rate {y_rep.mean()*100:.2f}%)")
    if abs(auc_tune - auc_rep) > 0.01:
        print(f"   note: AUC gap of {abs(auc_tune-auc_rep):.4f} between halves "
              f"suggests temporal drift")
    print()

    # Calibration check on report slice
    ece_rep_raw = expected_calibration_error(raw_rep, y_rep)
    ece_rep_cal = expected_calibration_error(cal_rep, y_rep)
    print(f" ECE on report slice — raw: {ece_rep_raw:.4f}, calibrated: {ece_rep_cal:.4f}")
    if ece_rep_cal > 0.05:
        print(f"   note: calibrated ECE > 0.05 suggests refit may be warranted")
    print()

    # ----- Tune each policy on the tune slice -----
    f1_thr,     f1_val_tune     = best_threshold_by(cal_tune, y_tune, f1_score)
    youden_thr, youden_val_tune = best_threshold_by(cal_tune, y_tune, youden_j)
    acc_thr,    acc_val_tune    = best_threshold_by(cal_tune, y_tune, accuracy_score)
    base_thr,   base_val_tune   = match_baserate_threshold(cal_tune, y_tune)

    # Existing thresholds (from full-test tuning) for comparison
    with open(EXISTING_THR_PATH) as f:
        existing = json.load(f)
    existing_f1     = existing["policies"]["f1"]["threshold"]
    existing_youden = existing["policies"]["youden"]["threshold"]
    existing_acc    = existing["policies"]["accuracy"]["threshold"]
    existing_base   = existing["policies"]["match_baserate"]["threshold"]

    print(" Threshold comparison (existing was tuned on the full 200k):")
    print(f"   {'policy':<18} {'existing':>10} {'tune-half':>10} {'shift':>8}")
    print(" " + "-" * 50)
    for label, ex, new in [
        ("f1",             existing_f1,     f1_thr),
        ("youden",         existing_youden, youden_thr),
        ("accuracy",       existing_acc,    acc_thr),
        ("match_baserate", existing_base,   base_thr),
    ]:
        print(f"   {label:<18} {ex:>10.4f} {new:>10.4f} {new-ex:>+8.4f}")
    print()

    # ----- Evaluate everything on the report slice -----
    print(" Out-of-sample evaluation (report slice, never seen by tuning):")
    print(f" {'policy':<28} {'thr':>7} {'%disp':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'lift':>6}")
    print(" " + "-" * 75)
    rows = [
        report_metrics(cal_rep, y_rep, f1_thr,      "f1 (oos-tuned)"),
        report_metrics(cal_rep, y_rep, youden_thr,  "youden (oos-tuned)"),
        report_metrics(cal_rep, y_rep, acc_thr,     "accuracy (oos-tuned)"),
        report_metrics(cal_rep, y_rep, base_thr,    "match_baserate (oos-tuned)"),
        # Compare against the existing (full-fold) thresholds on the same report slice
        report_metrics(cal_rep, y_rep, existing_f1, "f1 (existing full-fold)"),
        # Plus the legacy raw 0.40 (apply to RAW probs, not calibrated)
        report_metrics(raw_rep, y_rep, 0.40,        "legacy raw 0.40"),
    ]
    for r in rows:
        print(f" {r['label']:<28} {r['thr']:>7.4f} {r['pct_dispute']:>6.1f}% "
              f"{r['precision']*100:>6.1f}% {r['recall']*100:>6.1f}% "
              f"{r['f1']:>7.3f} {r['lift']:>5.2f}x")

    # Compare existing-policy F1 vs OOS-tuned F1 on report slice — measures
    # how much overfitting the full-fold tuning had.
    print()
    print(" Tune-half optimum vs same-policy on report slice (overfit gap):")
    for label, val_tune, thr_oos in [
        ("f1",       f1_val_tune,     f1_thr),
        ("accuracy", acc_val_tune,    acc_thr),
    ]:
        pred = (cal_rep >= thr_oos).astype(int)
        if label == "f1":
            val_rep = f1_score(y_rep, pred)
        else:
            val_rep = accuracy_score(y_rep, pred)
        gap = val_tune - val_rep
        print(f"   {label:<10} tune={val_tune:.4f}  report={val_rep:.4f}  gap={gap:+.4f}")


if __name__ == "__main__":
    main()
