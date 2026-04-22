# cleanlab_scan.py — Confident-learning label-noise scan on FineHero
# (FineHero AUC Playbook §2.5, Northcutt, Jiang, Chuang 2022 arXiv:1911.00068).
# Runs a time-respecting 4-fold expanding-window CV with a FAST CatBoost, feeds
# OOF predicted probabilities into cleanlab.filter.find_label_issues, and
# writes the flagged rows to data/cleanlab_flagged.csv for manual review.

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH     = os.path.join(DATA_DIR, "features.csv")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
FLAGGED_PATH      = os.path.join(DATA_DIR, "cleanlab_flagged.csv")
OOF_PATH          = os.path.join(MODELS_DIR, "cleanlab_oof.joblib")

# Fast CatBoost config for OOF generation (not headline quality).
CB_ITERS       = 600
CB_EARLY_STOP  = 50
CB_DEPTH       = 6
CB_LR          = 0.08
N_FOLDS        = 4
SUBSAMPLE_ROWS = None  # None = full dataset; integer = cap for speed


def _expanding_window_folds(n, n_folds):
    """Yield (train_idx, val_idx) pairs for an expanding-window time split.

    Rows are assumed already sorted chronologically. For n_folds=4 we use
    the last 4 quarters as validation folds; train is the prefix up to each.
    """
    fold_size = n // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds else n
        yield np.arange(0, train_end), np.arange(train_end, val_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE_ROWS,
                        help="Cap dataset size for speed (rows, chronologically truncated).")
    parser.add_argument("--top", type=int, default=2000,
                        help="Keep the top-N flagged rows in the output CSV.")
    args = parser.parse_args()

    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print("[ERROR] cleanlab not installed. Run: pip install cleanlab")
        raise SystemExit(1)

    print("  Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Dataset: {len(df):,} rows x {df.shape[1]} columns")

    if "issue_date" not in df.columns:
        raise RuntimeError("features.csv lacks issue_date — re-run src/engineer.py first")

    df = df.iloc[pd.to_datetime(df["issue_date"], errors="coerce").argsort()].reset_index(drop=True)

    if args.subsample and args.subsample < len(df):
        df = df.iloc[:args.subsample].reset_index(drop=True)
        print(f"  Subsampled to first {len(df):,} rows chronologically")

    cat_features = joblib.load(CAT_FEATURES_PATH)
    cat_features = [c for c in cat_features if c in df.columns]

    y = df["won"].values.astype(int)
    drop_cols = [c for c in ("won", "issue_date") if c in df.columns]
    X = df.drop(columns=drop_cols)
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)

    # --- GPU detection ---
    try:
        import torch
        task_type = "GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        task_type = "CPU"
    print(f"  CatBoost task_type: {task_type}")

    n = len(df)
    oof_probs = np.full(n, np.nan, dtype=float)

    print(f"\n  Running {N_FOLDS}-fold expanding-window CV...")
    for fold_i, (tr_idx, va_idx) in enumerate(_expanding_window_folds(n, N_FOLDS), 1):
        print(f"\n  Fold {fold_i}/{N_FOLDS}: train={len(tr_idx):,}  val={len(va_idx):,}")
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        val_pool   = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_features)
        cb = CatBoostClassifier(
            iterations=CB_ITERS, depth=CB_DEPTH, learning_rate=CB_LR,
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", early_stopping_rounds=CB_EARLY_STOP,
            task_type=task_type, verbose=0,
        )
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        fold_probs = cb.predict_proba(X.iloc[va_idx])[:, 1]
        oof_probs[va_idx] = fold_probs
        fold_auc = roc_auc_score(y[va_idx], fold_probs)
        print(f"    fold AUC = {fold_auc:.4f}")

    mask = ~np.isnan(oof_probs)
    print(f"\n  OOF coverage: {mask.sum():,} / {n:,} rows")
    overall_auc = roc_auc_score(y[mask], oof_probs[mask])
    print(f"  Overall OOF AUC: {overall_auc:.4f}")

    joblib.dump({"oof_probs": oof_probs, "y": y, "mask": mask}, OOF_PATH)

    # --- cleanlab ---
    pred_probs = np.column_stack([1.0 - oof_probs[mask], oof_probs[mask]])
    print("\n  Running cleanlab.filter.find_label_issues...")
    issue_indices = find_label_issues(
        labels=y[mask],
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    pct = 100.0 * len(issue_indices) / mask.sum()
    print(f"  Flagged {len(issue_indices):,} potential label issues "
          f"({pct:.2f}% of OOF rows)")

    keep = min(args.top, len(issue_indices))
    if keep == 0:
        print("  [WARN] No flags — cleanlab found no label issues.")
        return
    issue_indices = issue_indices[:keep]

    flagged_global = np.where(mask)[0][issue_indices]
    out = df.iloc[flagged_global].copy()
    out["oof_prob"]     = oof_probs[flagged_global]
    out["label"]        = y[flagged_global]
    out["model_disagreement"] = np.abs(out["oof_prob"] - out["label"])
    out = out.sort_values("model_disagreement", ascending=False)
    out.to_csv(FLAGGED_PATH, index=False)
    print(f"\n  Top {keep:,} flagged rows saved -> {FLAGGED_PATH}")
    print("  Inspect manually for systematic mislabeling patterns "
          "(e.g., 'DEFAULTED' coded inconsistently, dismissal-on-procedural "
          "vs merits).")


if __name__ == "__main__":
    main()
