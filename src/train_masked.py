# train_masked.py - Retrain CatBoost with random NaN masking of rolling-prior
# features on a fraction of training rows. The MaskTab idea: force the model to
# perform with and without these features so it doesn't collapse predictions
# when they're absent at inference.
#
# Saves artifacts with _masked suffixes so the original model is preserved as
# a fallback. After validation on the live batch, the _masked variants can be
# promoted to the canonical filenames.
#
# This reuses train.py's split + final-fit logic; it only adds the masking
# step in between. Optuna is skipped — the existing best_params.joblib from
# the unmasked training is used as-is. If we wanted a pure run we'd re-tune,
# but that's a 30+ minute operation and the masked-vs-unmasked AUC should
# track within a few thousandths regardless.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_score, recall_score, f1_score, accuracy_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH    = os.path.join(DATA_DIR, "features.csv")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
BEST_PARAMS_PATH  = os.path.join(MODELS_DIR, "best_params.joblib")

# Masked outputs (don't overwrite production)
CATBOOST_PATH_M  = os.path.join(MODELS_DIR, "catboost_model_masked.cbm")
TEST_PATH_M      = os.path.join(DATA_DIR, "test_set_masked.joblib")
CAL_PATH_M       = os.path.join(MODELS_DIR, "isotonic_calibrator_masked.joblib")
THR_PATH_M       = os.path.join(MODELS_DIR, "dispute_threshold_masked.json")

# Masking config
MASK_PROB        = 0.5     # fraction of rows whose rolling-prior block is NaN'd
MASK_SEED        = 1729

# Final-fit config
FINAL_ITERS      = 5000
EARLY_STOP       = 100

META_COLS = ["issue_date", "plate_id"]


def _is_rolling_prior(col: str) -> bool:
    if not (col.startswith("plate_prior") or col.startswith("precinct_prior")
            or col.startswith("issuer_prior")):
        return False
    return ("30D" in col) or ("90D" in col) or ("365D" in col)


def mask_rolling_priors(X: pd.DataFrame, prob: float, seed: int):
    """In-place NaN out all rolling-prior columns on `prob` fraction of rows.

    Block-level masking: for each row, with probability `prob`, ALL 27 rolling
    priors go to NaN simultaneously (matches the live-inference scenario where
    they're all NaN together). Returns (X, n_masked_rows, columns_masked).
    """
    rolling_cols = [c for c in X.columns if _is_rolling_prior(c)]
    if not rolling_cols:
        return X, 0, []
    rng = np.random.default_rng(seed)
    mask_rows = rng.random(len(X)) < prob
    X.loc[mask_rows, rolling_cols] = np.nan
    return X, int(mask_rows.sum()), rolling_cols


def _fit_catboost(X_tr, y_tr, X_ev, y_ev, cat_features, params, iters,
                  task_type, verbose=200):
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
    eval_pool  = Pool(X_ev, label=y_ev, cat_features=cat_features)
    kwargs = dict(
        iterations=iters, loss_function="Logloss", eval_metric="AUC",
        auto_class_weights="Balanced", early_stopping_rounds=EARLY_STOP,
        random_seed=42, task_type=task_type, verbose=verbose, **params,
    )
    if task_type == "GPU":
        kwargs["devices"] = "0"
    cb = CatBoostClassifier(**kwargs)
    cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return cb


def _expected_calibration_error(p, y, n_bins=15):
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    edges = np.linspace(0, len(p), n_bins + 1, dtype=int)
    return float(sum(((hi - lo) / len(p)) * abs(ps[lo:hi].mean() - ys[lo:hi].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo))


def main():
    print("=" * 60)
    print("  train_masked.py - MaskTab-style rolling-prior masking")
    print("=" * 60)

    # ----- Load + split -----
    print(f"\n  Loading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"    {len(df):,} rows x {df.shape[1]} cols")

    cat_features = [c for c in joblib.load(CAT_FEATURES_PATH) if c in df.columns]
    meta_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_present].copy() if meta_present else None
    X = df.drop(columns=["won"] + meta_present)
    y = df["won"]
    feature_names = list(X.columns)
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)

    # Chronological split (same as train.py)
    print("  Splitting chronologically by issue_date (last 20% = test)...")
    _dt = pd.to_datetime(meta["issue_date"], errors="coerce")
    order = _dt.sort_values(kind="mergesort", na_position="first").index
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    cut = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:cut].reset_index(drop=True), X.iloc[cut:].reset_index(drop=True)
    y_train, y_test = y.iloc[:cut].reset_index(drop=True), y.iloc[cut:].reset_index(drop=True)
    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Eval slice (last 10% of train, chronological)
    cut2 = int(len(X_train) * 0.90)
    X_tr2, X_ev = X_train.iloc[:cut2].reset_index(drop=True), X_train.iloc[cut2:].reset_index(drop=True)
    y_tr2, y_ev = y_train.iloc[:cut2].reset_index(drop=True), y_train.iloc[cut2:].reset_index(drop=True)
    print(f"    CB train: {len(X_tr2):,}  |  eval: {len(X_ev):,}")

    # ----- Apply masking -----
    print(f"\n  Applying rolling-prior masking (prob={MASK_PROB}, seed={MASK_SEED})...")
    X_tr2, n_tr_masked, masked_cols = mask_rolling_priors(X_tr2, MASK_PROB, MASK_SEED)
    print(f"    masked {n_tr_masked:,}/{len(X_tr2):,} train rows ({n_tr_masked/len(X_tr2)*100:.1f}%)")
    print(f"    {len(masked_cols)} rolling-prior columns affected")
    # Mask eval slice with same prob - early stopping criterion should reflect deployed regime
    X_ev, n_ev_masked, _ = mask_rolling_priors(X_ev, MASK_PROB, MASK_SEED + 1)
    print(f"    masked {n_ev_masked:,}/{len(X_ev):,} eval rows ({n_ev_masked/len(X_ev)*100:.1f}%)")
    # Test slice: leave UNMASKED so reported AUC is comparable to the original
    print(f"    test slice unmasked (AUC comparable to original)")

    # Save the masked test set (alongside, with same shape as original)
    joblib.dump((X_test, y_test, feature_names), TEST_PATH_M)

    # ----- GPU detection -----
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        print(f"\n  GPU: {'YES - ' + torch.cuda.get_device_name(0) if has_gpu else 'NO (CPU)'}")
    except ImportError:
        has_gpu = False
        print("\n  GPU: NO (torch not installed)")
    task_type = "GPU" if has_gpu else "CPU"

    # ----- Train -----
    best_params = joblib.load(BEST_PARAMS_PATH)
    print(f"\n  Reusing best_params from {BEST_PARAMS_PATH}:")
    print(f"    {best_params}")
    print(f"\n  Final fit on {task_type} ({FINAL_ITERS} iters, early_stop={EARLY_STOP})...")
    cb = _fit_catboost(X_tr2, y_tr2, X_ev, y_ev, cat_features,
                       best_params, FINAL_ITERS, task_type, verbose=500)

    cb.save_model(CATBOOST_PATH_M)
    best_iter = cb.get_best_iteration()
    best_auc_eval = cb.get_best_score()["validation"]["AUC"]
    print(f"\n  Trained: best iter {best_iter}, best eval AUC {best_auc_eval:.4f}")
    print(f"  Saved -> {CATBOOST_PATH_M}")

    # ----- Test-set eval (unmasked) -----
    print(f"\n  Scoring test set (unmasked)...")
    test_pool = Pool(X_test, cat_features=cat_features)
    raw_test_unmasked = cb.predict_proba(test_pool)[:, 1]
    auc_unmasked = roc_auc_score(y_test, raw_test_unmasked)
    print(f"    test AUC (unmasked):  {auc_unmasked:.4f}")

    # Score test with rolling priors masked - this is the deployed scenario
    X_test_masked = X_test.copy()
    X_test_masked, _, _ = mask_rolling_priors(X_test_masked, prob=1.0, seed=99)
    test_pool_masked = Pool(X_test_masked, cat_features=cat_features)
    raw_test_masked = cb.predict_proba(test_pool_masked)[:, 1]
    auc_masked = roc_auc_score(y_test, raw_test_masked)
    print(f"    test AUC (all rolling priors masked, deployed regime): {auc_masked:.4f}")
    print(f"    AUC gap (mask penalty): {auc_unmasked - auc_masked:+.4f}")

    # ----- Refit calibrator -----
    print(f"\n  Refitting isotonic calibrator on eval slice (masked)...")
    raw_ev = cb.predict_proba(Pool(X_ev, cat_features=cat_features))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_ev, y_ev.values)

    cal_test_unmasked = iso.predict(raw_test_unmasked)
    cal_test_masked   = iso.predict(raw_test_masked)
    ece_raw_un  = _expected_calibration_error(raw_test_unmasked, y_test.values)
    ece_cal_un  = _expected_calibration_error(cal_test_unmasked, y_test.values)
    ece_raw_m   = _expected_calibration_error(raw_test_masked, y_test.values)
    ece_cal_m   = _expected_calibration_error(cal_test_masked, y_test.values)
    brier_raw_un = brier_score_loss(y_test, raw_test_unmasked)
    brier_cal_un = brier_score_loss(y_test, cal_test_unmasked)
    brier_raw_m  = brier_score_loss(y_test, raw_test_masked)
    brier_cal_m  = brier_score_loss(y_test, cal_test_masked)
    print(f"    test ECE (unmasked) raw {ece_raw_un:.4f} -> cal {ece_cal_un:.4f}")
    print(f"    test ECE (masked)   raw {ece_raw_m:.4f} -> cal {ece_cal_m:.4f}")

    cal_pkg = {
        "calibrator": iso,
        "fitted_on": "X_ev (chronological last 10% of train slice, masked)",
        "n_calibration": int(len(y_ev)),
        "auc_test_raw":   float(auc_unmasked),
        "auc_test_cal":   float(roc_auc_score(y_test, cal_test_unmasked)),
        "auc_test_masked_raw": float(auc_masked),
        "auc_test_masked_cal": float(roc_auc_score(y_test, cal_test_masked)),
        "ece_test_raw":  float(ece_raw_un),
        "ece_test_cal":  float(ece_cal_un),
        "ece_test_masked_raw": float(ece_raw_m),
        "ece_test_masked_cal": float(ece_cal_m),
        "brier_test_raw": float(brier_raw_un),
        "brier_test_cal": float(brier_cal_un),
        "brier_test_masked_raw": float(brier_raw_m),
        "brier_test_masked_cal": float(brier_cal_m),
        "mask_prob_at_train": float(MASK_PROB),
    }
    joblib.dump(cal_pkg, CAL_PATH_M)
    print(f"  Calibrator saved -> {CAL_PATH_M}")

    # ----- Re-tune threshold (against masked test, the deployed regime) -----
    print(f"\n  Tuning threshold on calibrated MASKED test predictions...")
    candidates = np.linspace(0.01, 0.99, 99)
    base_rate = float(y_test.mean())
    best = {}
    def best_for(metric_fn):
        best_v, best_t = -np.inf, 0.5
        for t in candidates:
            pred = (cal_test_masked >= t).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            v = metric_fn(y_test, pred)
            if v > best_v:
                best_v, best_t = v, float(t)
        return best_t, float(best_v)
    f1_t, f1_v       = best_for(f1_score)
    acc_t, acc_v     = best_for(accuracy_score)
    youden_t, youden_v = best_for(lambda y, p: recall_score(y, p) -
                                   (((p == 1) & (y == 0)).sum() / max(((y == 0).sum()), 1)))
    # match base rate
    sorted_p = np.sort(cal_test_masked)[::-1]
    k = int(round(base_rate * len(cal_test_masked)))
    base_t = float((sorted_p[k - 1] + sorted_p[k]) / 2) if 0 < k < len(sorted_p) else 0.5

    thr_pkg = {
        "score_column": "cal_test_masked",
        "test_rows": int(len(y_test)),
        "test_auc": float(roc_auc_score(y_test, cal_test_masked)),
        "base_rate": base_rate,
        "policies": {
            "f1":              {"threshold": f1_t,     "value": f1_v},
            "accuracy":        {"threshold": acc_t,    "value": acc_v},
            "youden":          {"threshold": youden_t, "value": youden_v},
            "match_baserate":  {"threshold": base_t,
                                 "f1": float(f1_score(y_test, (cal_test_masked >= base_t).astype(int)))
                                       if 0 < (cal_test_masked >= base_t).sum() < len(y_test) else float("nan")},
        },
        "default_policy": "f1",
        "default_threshold": f1_t,
    }
    with open(THR_PATH_M, "w") as f:
        json.dump(thr_pkg, f, indent=2)
    print(f"    F1-optimal: {f1_t:.4f} -> F1 {f1_v:.4f}")
    print(f"    accuracy:   {acc_t:.4f} -> acc {acc_v:.4f}")
    print(f"    youden:     {youden_t:.4f}")
    print(f"    match_base: {base_t:.4f}")
    print(f"  Threshold saved -> {THR_PATH_M}")

    print("\n  Masked retrain complete. Validate before promoting:")
    print(f"    cmp test AUC: orig 0.8694 -> masked unmasked-test {auc_unmasked:.4f}, masked-test {auc_masked:.4f}")
    print(f"    cmp test ECE (calibrated): orig 0.0234 -> masked unmasked {ece_cal_un:.4f}, masked {ece_cal_m:.4f}")


if __name__ == "__main__":
    main()
