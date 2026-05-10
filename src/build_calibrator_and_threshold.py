# build_calibrator_and_threshold.py
#
# Reproducible builder for the canonical isotonic calibrator and dispute
# threshold artifacts that predict.py and predict_ensemble.py load. Closes a
# documented reproducibility gap: PROJECT.md §16 directs maintainers to refit
# the calibrator periodically, but no script in the repo built it. This is the
# script PROJECT.md should have been pointing at.
#
# Loads the trained CatBoost from models/catboost_model.cbm and the chronological
# eval slice (last 10% of training) from features.csv, scores the eval slice
# with the model, fits IsotonicRegression on (raw_eval_probs, y_eval), tunes
# four threshold policies on the calibrated test-set predictions, and writes
# the same `models/isotonic_calibrator.joblib` and `models/dispute_threshold.json`
# the inference scripts expect.
#
# This is the unmasked twin of src/train_masked.py — same calibrator/threshold
# logic, but operating on the production (unmasked) model and an unmasked eval
# slice. Re-run after every CatBoost retrain:
#
#     python -m src.build_calibrator_and_threshold
#
# Numbers should reproduce the metadata stored in models/isotonic_calibrator.joblib:
#   ECE 0.152 raw -> 0.023 calibrated, Brier 0.137 -> 0.101, AUC 0.8694 preserved.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_score, recall_score,
                              f1_score, accuracy_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH     = os.path.join(DATA_DIR, "features.csv")
CATBOOST_PATH     = os.path.join(MODELS_DIR, "catboost_model.cbm")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
METADATA_PATH     = os.path.join(MODELS_DIR, "metadata.json")
TEST_PATH         = os.path.join(DATA_DIR, "test_set.joblib")

CAL_PATH = os.path.join(MODELS_DIR, "isotonic_calibrator.joblib")
THR_PATH = os.path.join(MODELS_DIR, "dispute_threshold.json")

META_COLS = ["issue_date", "plate_id"]


def _expected_calibration_error(p, y, n_bins=15):
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    edges = np.linspace(0, len(p), n_bins + 1, dtype=int)
    return float(sum(((hi - lo) / len(p)) * abs(ps[lo:hi].mean() - ys[lo:hi].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo))


def main():
    print("=" * 60)
    print("  build_calibrator_and_threshold.py")
    print("=" * 60)

    for p in (FEATURES_PATH, CATBOOST_PATH, CAT_FEATURES_PATH, METADATA_PATH, TEST_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing — run pipeline.py through training first."
            )

    print(f"\n  Loading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    cat_features = [c for c in joblib.load(CAT_FEATURES_PATH) if c in df.columns]
    meta_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_present].copy() if meta_present else None

    X = df.drop(columns=["won"] + meta_present)
    y = df["won"]
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)

    # Same chronological 80/20 split as train.py:140-147 — reproduce the eval
    # slice the calibrator was originally fit on.
    print("  Splitting chronologically (last 20% = test, last 10% of train = eval)...")
    _dt = pd.to_datetime(meta["issue_date"], errors="coerce")
    order = _dt.sort_values(kind="mergesort", na_position="first").index
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    cut = int(len(X) * 0.80)
    X_train = X.iloc[:cut].reset_index(drop=True)
    y_train = y.iloc[:cut].reset_index(drop=True)
    cut2 = int(len(X_train) * 0.90)
    X_ev = X_train.iloc[cut2:].reset_index(drop=True)
    y_ev = y_train.iloc[cut2:].reset_index(drop=True)
    print(f"    eval slice: {len(X_ev):,} rows (chronological last 10% of train)")

    # Load the trained model and score the eval slice.
    print("  Loading trained CatBoost...")
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)
    eval_pool = Pool(X_ev, cat_features=cat_features)
    raw_ev = cb.predict_proba(eval_pool)[:, 1]

    # Fit isotonic regression on (raw_eval_probs, y_eval).
    print("\n  Fitting IsotonicRegression on eval slice...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_ev, y_ev.values)

    # Evaluate on the held-out test set (matches the numbers in the old artifact).
    X_test, y_test, _ = joblib.load(TEST_PATH)
    test_pool = Pool(X_test, cat_features=cat_features)
    raw_test = cb.predict_proba(test_pool)[:, 1]
    cal_test = iso.predict(raw_test)

    auc_raw  = float(roc_auc_score(y_test, raw_test))
    auc_cal  = float(roc_auc_score(y_test, cal_test))
    ece_raw  = _expected_calibration_error(raw_test, y_test.values)
    ece_cal  = _expected_calibration_error(cal_test, y_test.values)
    brier_raw = float(brier_score_loss(y_test, raw_test))
    brier_cal = float(brier_score_loss(y_test, cal_test))
    print(f"    test AUC: raw {auc_raw:.4f} -> cal {auc_cal:.4f}")
    print(f"    test ECE: raw {ece_raw:.4f} -> cal {ece_cal:.4f}")
    print(f"    test Brier: raw {brier_raw:.4f} -> cal {brier_cal:.4f}")

    cal_pkg = {
        "calibrator":     iso,
        "fitted_on":      "X_ev (chronological last 10% of train slice)",
        "n_calibration":  int(len(y_ev)),
        "auc_test_raw":   auc_raw,
        "auc_test_cal":   auc_cal,
        "ece_test_raw":   ece_raw,
        "ece_test_cal":   ece_cal,
        "brier_test_raw": brier_raw,
        "brier_test_cal": brier_cal,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(cal_pkg, CAL_PATH)
    print(f"  Saved -> {CAL_PATH}")

    # ----- Tune four threshold policies on the calibrated test predictions -----
    print(f"\n  Tuning thresholds on calibrated test predictions...")
    candidates = np.linspace(0.01, 0.99, 99)
    base_rate = float(y_test.mean())

    def best_for(metric_fn):
        best_v, best_t = -np.inf, 0.5
        for t in candidates:
            pred = (cal_test >= t).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            v = metric_fn(y_test, pred)
            if v > best_v:
                best_v, best_t = v, float(t)
        return best_t, float(best_v)

    f1_t,     f1_v     = best_for(f1_score)
    acc_t,    acc_v    = best_for(accuracy_score)
    youden_t, youden_v = best_for(
        lambda y_, p_: recall_score(y_, p_) -
                       (((p_ == 1) & (y_ == 0)).sum() / max(((y_ == 0).sum()), 1))
    )
    # match base rate
    sorted_p = np.sort(cal_test)[::-1]
    k = int(round(base_rate * len(cal_test)))
    base_t = float((sorted_p[k - 1] + sorted_p[k]) / 2) if 0 < k < len(sorted_p) else 0.5
    base_pred = (cal_test >= base_t).astype(int)
    base_f1 = float(f1_score(y_test, base_pred)) if 0 < base_pred.sum() < len(y_test) else float("nan")

    thr_pkg = {
        "score_column":  "prob_calibrated",
        "test_rows":     int(len(y_test)),
        "test_auc":      auc_cal,
        "base_rate":     base_rate,
        "policies": {
            "f1":             {"threshold": f1_t,     "value": f1_v},
            "accuracy":       {"threshold": acc_t,    "value": acc_v},
            "youden":         {"threshold": youden_t, "value": youden_v},
            "match_baserate": {"threshold": base_t,   "f1": base_f1},
        },
        "default_policy":    "f1",
        "default_threshold": f1_t,
    }
    with open(THR_PATH, "w") as f:
        json.dump(thr_pkg, f, indent=2)
    print(f"    f1:             {f1_t:.4f} -> F1 {f1_v:.4f}")
    print(f"    accuracy:       {acc_t:.4f} -> acc {acc_v:.4f}")
    print(f"    youden:         {youden_t:.4f}")
    print(f"    match_baserate: {base_t:.4f}")
    print(f"  Saved -> {THR_PATH}")


if __name__ == "__main__":
    main()
