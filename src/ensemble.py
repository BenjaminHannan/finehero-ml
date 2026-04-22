# ensemble.py — Rank-averaged blend of CatBoost + LightGBM + XGBoost
# (FineHero AUC Playbook §4.2, §5.3). AUC = Mann-Whitney U statistic, so blend
# in RANK space, not probability space. Otherwise the model with the widest
# score range dominates.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

CATBOOST_PATH     = os.path.join(MODELS_DIR, "catboost_model.cbm")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
TEST_PATH         = os.path.join(DATA_DIR, "test_set.joblib")
LGB_PREDS_PATH    = os.path.join(MODELS_DIR, "lgb_test_preds.joblib")
XGB_PREDS_PATH    = os.path.join(MODELS_DIR, "xgb_test_preds.joblib")
ENSEMBLE_PATH     = os.path.join(MODELS_DIR, "ensemble_test_preds.joblib")


def _catboost_test_probs():
    """Re-score CatBoost on the saved time-aware test set."""
    X_test, y_test, feat_names = joblib.load(TEST_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    cat_features = [c for c in cat_features if c in X_test.columns]
    X = X_test.copy()
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)
    probs = cb.predict_proba(X)[:, 1]
    return probs, y_test.values if hasattr(y_test, "values") else np.asarray(y_test)


def rank_mean(arrays):
    ranks = np.stack([rankdata(a) / len(a) for a in arrays], axis=0)
    return ranks.mean(axis=0)


def main():
    print("  Loading CatBoost, LGB, XGB test predictions...")
    cb_probs, y_cb = _catboost_test_probs()
    lgb_pack = joblib.load(LGB_PREDS_PATH)
    xgb_pack = joblib.load(XGB_PREDS_PATH)

    # Sanity: all three should be over the same test rows in the same order
    # (all use the time-aware chronological last-20% split).
    if not (len(cb_probs) == len(lgb_pack["probs"]) == len(xgb_pack["probs"])):
        raise ValueError(
            f"Test-set length mismatch: CB={len(cb_probs)}, "
            f"LGB={len(lgb_pack['probs'])}, XGB={len(xgb_pack['probs'])}. "
            "Re-run train/train_lgb/train_xgb against the same features.csv."
        )
    if not np.array_equal(y_cb, lgb_pack["y_test"]):
        print("  [WARN] y_test arrays differ between CatBoost and LGB — double-check splits")
    y_test = y_cb

    # Individual AUCs
    aucs = {
        "CatBoost": roc_auc_score(y_test, cb_probs),
        "LightGBM": roc_auc_score(y_test, lgb_pack["probs"]),
        "XGBoost":  roc_auc_score(y_test, xgb_pack["probs"]),
    }
    print("\n  Individual time-aware test AUCs:")
    for name, auc in aucs.items():
        print(f"    {name:10s} {auc:.4f}")

    # Rank-averaged blends
    rb_all = rank_mean([cb_probs, lgb_pack["probs"], xgb_pack["probs"]])
    rb_cb_lgb = rank_mean([cb_probs, lgb_pack["probs"]])
    rb_cb_xgb = rank_mean([cb_probs, xgb_pack["probs"]])

    auc_all    = roc_auc_score(y_test, rb_all)
    auc_cb_lgb = roc_auc_score(y_test, rb_cb_lgb)
    auc_cb_xgb = roc_auc_score(y_test, rb_cb_xgb)

    print("\n  Rank-averaged blends:")
    print(f"    CatBoost+LGB+XGB   {auc_all:.4f}")
    print(f"    CatBoost+LGB       {auc_cb_lgb:.4f}")
    print(f"    CatBoost+XGB       {auc_cb_xgb:.4f}")

    best_auc = max(auc_all, auc_cb_lgb, auc_cb_xgb)
    best_single = max(aucs.values())
    print(f"\n  Best blend AUC: {best_auc:.4f}")
    print(f"  Best single:    {best_single:.4f}")
    print(f"  Lift:           {best_auc - best_single:+.4f}")

    joblib.dump({
        "probs_cb":        cb_probs,
        "probs_lgb":       lgb_pack["probs"],
        "probs_xgb":       xgb_pack["probs"],
        "rank_blend_all":  rb_all,
        "y_test":          y_test,
        "aucs":            aucs,
        "blended_auc_all": auc_all,
    }, ENSEMBLE_PATH)
    print(f"\n  Ensemble saved -> {ENSEMBLE_PATH}")


if __name__ == "__main__":
    main()
