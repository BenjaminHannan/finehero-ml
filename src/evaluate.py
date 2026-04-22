# evaluate.py - Loads the CatBoost model + held-out test set, prints accuracy,
# ROC-AUC, classification report, confusion matrix, and top features.
# Writes models/metadata.json consumed by predict.py.

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

CATBOOST_PATH = os.path.join(MODELS_DIR, "catboost_model.cbm")
LR_PATH       = os.path.join(MODELS_DIR, "lr_model.joblib")
TEST_PATH     = os.path.join(DATA_DIR, "test_set.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")


def _print_banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_confusion(cm: np.ndarray, label: str) -> None:
    print(f"\n  Confusion Matrix - {label}:")
    print(f"               Predicted Lost  Predicted Won")
    print(f"  Actual Lost       {cm[0,0]:>6}         {cm[0,1]:>6}")
    print(f"  Actual Won        {cm[1,0]:>6}         {cm[1,1]:>6}")


def evaluate() -> dict:
    print("  Loading test set and models...")
    X_test, y_test, feature_names = joblib.load(TEST_PATH)
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)
    lr_bundle = joblib.load(LR_PATH)
    lr            = lr_bundle["pipeline"]
    lr_encoder    = lr_bundle["encoder"]
    lr_cat_feats  = lr_bundle["cat_features"]

    # Build LR-compatible X_test (ordinal-encoded cats)
    X_test_lr = X_test.copy()
    X_test_lr[lr_cat_feats] = lr_encoder.transform(X_test_lr[lr_cat_feats].astype(str))

    total_rows = len(pd.read_csv(FEATURES_PATH))
    results = {}

    # --- CatBoost ---
    _print_banner("CatBoost (GPU)")
    y_prob_cb = cb.predict_proba(X_test)[:, 1]
    y_pred_cb = (y_prob_cb >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_cb)
    auc = roc_auc_score(y_test, y_prob_cb)
    cm  = confusion_matrix(y_test, y_pred_cb)

    print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Classification Report:\n")
    report = classification_report(y_test, y_pred_cb, target_names=["Lost", "Won"])
    print("\n".join("    " + line for line in report.splitlines()))
    _print_confusion(cm, "CatBoost")
    results["CatBoost"] = {"accuracy": acc, "auc": auc}

    # --- Feature importance ---
    try:
        print(f"\n  Top 15 Feature Importances:")
        importances = cb.get_feature_importance()
        fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15]
        for rank, (feat, imp) in enumerate(fi, 1):
            print(f"    {rank:>2}. {feat:<32} {imp:.4f}")
    except Exception as exc:
        print(f"  [WARN] Feature importance failed: {exc}")

    # --- Baseline Logistic Regression ---
    _print_banner("LogisticRegression (baseline)")
    y_prob_lr = lr.predict_proba(X_test_lr)[:, 1]
    y_pred_lr = lr.predict(X_test_lr)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    cm_lr  = confusion_matrix(y_test, y_pred_lr)

    print(f"\n  Accuracy : {acc_lr:.4f}  ({acc_lr*100:.2f}%)")
    print(f"  ROC-AUC  : {auc_lr:.4f}")
    print(f"\n  Classification Report:\n")
    report_lr = classification_report(y_test, y_pred_lr, target_names=["Lost", "Won"])
    print("\n".join("    " + line for line in report_lr.splitlines()))
    _print_confusion(cm_lr, "LogisticRegression")
    results["LogisticRegression"] = {"accuracy": acc_lr, "auc": auc_lr}

    # --- Summary ---
    _print_banner("Summary")
    print(f"  CatBoost AUC:       {auc:.4f}")
    print(f"  CatBoost Accuracy:  {acc*100:.2f}%")
    print(f"  LR AUC:             {auc_lr:.4f}  (baseline)")
    print(f"  AUC gain over baseline: {auc - auc_lr:+.4f}")

    # --- Write metadata.json ---
    metadata = {
        "model_type": "CatBoost_GPU",
        "auc_score": round(auc, 6),
        "accuracy": round(acc, 6),
        "feature_names": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "row_count": total_rows,
        "baseline_lr_auc": round(auc_lr, 6),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved -> {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    evaluate()
