# build_feature_audit.py
#
# One-shot builder for `models/feature_audit.joblib`. Captures the training-time
# null rate and basic numeric range for every model feature, so predict.py can
# detect training-serving skew at inference (a feature that was rarely null in
# training but is 100% null at inference is the textbook signature of the bug
# this whole codebase exists to manage — see PROJECT.md §11).
#
# Run after every training-data refresh:
#     python -m src.build_feature_audit

import os
import joblib
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
OUT_PATH      = os.path.join(MODELS_DIR, "feature_audit.joblib")


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} missing — run pipeline.py first.")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"{METADATA_PATH} missing — train the model first.")

    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    print(f"  Auditing {len(feature_names)} model features against {FEATURES_PATH}...")

    df = pd.read_csv(FEATURES_PATH, usecols=lambda c: c in feature_names, low_memory=False)
    print(f"    {len(df):,} training rows scanned")

    audit = {}
    for col in feature_names:
        if col not in df.columns:
            audit[col] = {"present_in_training": False, "null_rate": float("nan")}
            continue
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        entry = {
            "present_in_training": True,
            "null_rate": float(s.isna().mean()),
            "is_numeric": is_numeric,
        }
        if is_numeric:
            sn = s.dropna()
            if len(sn):
                entry.update({
                    "mean":  float(sn.mean()),
                    "std":   float(sn.std()),
                    "min":   float(sn.min()),
                    "max":   float(sn.max()),
                    "p01":   float(sn.quantile(0.01)),
                    "p99":   float(sn.quantile(0.99)),
                })
        else:
            # Categorical: top-K values + "other" rate
            counts = s.dropna().value_counts(normalize=True)
            top = counts.head(20).to_dict()
            entry.update({
                "top_values": {str(k): float(v) for k, v in top.items()},
                "n_unique":   int(s.nunique(dropna=True)),
            })
        audit[col] = entry

    artifact = {
        "feature_audit": audit,
        "n_training_rows": int(len(df)),
        "feature_names": feature_names,
        # Default drift threshold for null rate — features whose live null rate
        # differs from training by more than this trigger a warning.
        "null_rate_drift_threshold": 0.5,
        # If a feature's live value falls outside [training p01 - delta, training p99 + delta]
        # for delta = drift_range_padding * (p99 - p01), warn.
        "range_padding":             0.10,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(artifact, OUT_PATH)

    # Diagnostic summary
    nulls = [(c, audit[c]["null_rate"]) for c in feature_names
             if audit[c].get("present_in_training") and not pd.isna(audit[c]["null_rate"])]
    nulls.sort(key=lambda t: -t[1])
    print(f"  Top-10 features by training-time null rate:")
    for c, r in nulls[:10]:
        print(f"    {c:<40} {r*100:>5.1f}%")
    high_null = sum(1 for _, r in nulls if r > 0.5)
    print(f"  {high_null} features have null rate > 50% in training")
    print(f"  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
