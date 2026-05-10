# build_rolling_prior_means.py
#
# One-shot script that scans features.csv and saves the population means
# (and a few defensible defaults) for the inference-time rolling-prior
# fallback used by predict.py and predict_ensemble.py.
#
# Run this once after every training data refresh. The artifact is consumed
# by predict._load_rolling_prior_means(), which fills NaN rolling-prior
# columns at inference. See LIMITATIONS.md §"Live-data drift" for the
# diagnosis this is closing.
#
# Usage:
#   python -m src.build_rolling_prior_means
#
# Output:
#   models/rolling_prior_means.joblib

import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
OUT_PATH      = os.path.join(MODELS_DIR, "rolling_prior_means.joblib")

# The 27 rolling-prior columns produced by engineer.py for each window.
ROLLING_PRIOR_PREFIXES = ("plate_prior", "precinct_prior", "issuer_prior")
ROLLING_WINDOWS = ("30D", "90D", "365D")
ROLLING_STATS = ("wins", "count", "win_rate")

# These are not rolling priors but they're computed during training the same
# way (running tally) and have no inference-time equivalent. We supply
# sensible defaults so the model isn't routed down a "no history at all"
# branch.
DAYS_SINCE_DEFAULTS = {
    "days_since_plate_last_ticket":  365.0,  # ~1 year
    "days_since_plate_last_win":     365.0,
    "days_since_issuer_last_ticket":  30.0,  # issuers are active, so shorter default
}


def is_rolling_prior(col: str) -> bool:
    """True if column matches the rolling-prior naming pattern."""
    if not any(col.startswith(p) for p in ROLLING_PRIOR_PREFIXES):
        return False
    if not any(w in col for w in ROLLING_WINDOWS):
        return False
    return True


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} missing — run pipeline.py through the engineering "
            "step before building rolling-prior means."
        )

    print(f"  Reading {FEATURES_PATH}...")
    header = pd.read_csv(FEATURES_PATH, nrows=0).columns.tolist()
    rolling_cols = [c for c in header if is_rolling_prior(c)]
    days_since_cols = [c for c in header if c.startswith("days_since_")]
    issuer_bayes_cols = [c for c in header if c == "issuer_bayes_rate"]
    needed = rolling_cols + days_since_cols + issuer_bayes_cols
    if not rolling_cols:
        raise RuntimeError(
            "No rolling-prior columns found in features.csv. Either the schema "
            "changed or feature engineering didn't produce them."
        )
    print(f"    {len(rolling_cols)} rolling-prior cols + {len(days_since_cols)} "
          f"days_since cols + {len(issuer_bayes_cols)} issuer_bayes")

    # We need `won` to compute the base rate, which is the principled fallback
    # for `issuer_bayes_rate` on unseen issuers. engineer.py defines
    # issuer_bayes_rate as (wins + alpha*global_mean) / (count + alpha), which
    # collapses to the global mean for count=0 — i.e. the base rate, NOT the
    # ticket-weighted training mean (which is biased by high-volume issuers).
    df = pd.read_csv(FEATURES_PATH, usecols=needed + (["won"] if "won" not in needed else []),
                     low_memory=False)
    print(f"    {len(df):,} training rows scanned")

    means = {c: float(df[c].mean()) for c in rolling_cols}
    medians = {c: float(df[c].median()) for c in rolling_cols}
    days_since_means = {c: float(df[c].mean()) for c in days_since_cols}
    issuer_bayes = {c: float(df[c].mean()) for c in issuer_bayes_cols}
    base_rate = float(df["won"].mean()) if "won" in df.columns else 0.18

    artifact = {
        "rolling_prior_means":   means,
        "rolling_prior_medians": medians,
        "days_since_means":      days_since_means,
        "days_since_defaults":   {k: DAYS_SINCE_DEFAULTS.get(k, v)
                                  for k, v in days_since_means.items()},
        "issuer_bayes_means":    issuer_bayes,
        # Base rate, not training-mean of issuer_bayes_rate. engineer.py would
        # assign the base rate to a count=0 (unseen) issuer.
        "issuer_bayes_default":  base_rate,
        "base_rate":             base_rate,
        "n_training_rows":       int(len(df)),
        "rolling_prior_columns": rolling_cols,
        "days_since_columns":    days_since_cols,
    }

    print(f"    sample: plate_prior_win_rate_30D mean = "
          f"{means.get('plate_prior_win_rate_30D', float('nan')):.4f}")
    print(f"            precinct_prior_win_rate_365D mean = "
          f"{means.get('precinct_prior_win_rate_365D', float('nan')):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(artifact, OUT_PATH)
    print(f"  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
