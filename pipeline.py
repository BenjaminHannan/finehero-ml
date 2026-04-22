# pipeline.py - Orchestrates the full FineHero ML pipeline in order:
#   1. Fetch raw data from NYC Open Data (or generate synthetic fallback)
#   2. Engineer features and build the training dataset
#   3. Train GradientBoosting + LogisticRegression models
#   4. Evaluate both models and save metadata

import sys
import time

DEFAULT_ROWS = 500_000


def _ask_row_count() -> int:
    print(f"\n  How many rows to fetch? (default {DEFAULT_ROWS:,}, max ~1,000,000)")
    val = input("  Rows: ").strip().replace(",", "")
    if not val:
        return DEFAULT_ROWS
    try:
        n = int(val)
        if n < 1000:
            print(f"  Too low — using minimum of 1,000")
            return 1_000
        return n
    except ValueError:
        print(f"  Invalid input — using default {DEFAULT_ROWS:,}")
        return DEFAULT_ROWS


def step(n: int, title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'-'*60}")


def main() -> None:
    print("=" * 60)
    print("  FineHero ML Pipeline - Parking Ticket Win Predictor")
    print("=" * 60)

    row_count = _ask_row_count()
    print(f"  Fetching {row_count:,} rows.\n")

    total_start = time.time()

    # --- Step 1: Fetch Data ---
    step(1, "Fetching raw data from NYC Open Data")
    t = time.time()
    try:
        from src.fetch_data import fetch_all
        fetch_all(row_count)
    except Exception as exc:
        print(f"  [ERROR] fetch_data failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Done in {time.time()-t:.1f}s")

    # --- Step 2: Feature Engineering ---
    step(2, "Engineering features")
    t = time.time()
    try:
        from src.engineer import engineer_features
        engineer_features()
    except Exception as exc:
        print(f"  [ERROR] engineer failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Done in {time.time()-t:.1f}s")

    # --- Step 3: Train Models ---
    step(3, "Training models")
    t = time.time()
    try:
        from src.train import train_models
        train_models()
    except Exception as exc:
        print(f"  [ERROR] train failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Done in {time.time()-t:.1f}s")

    # --- Step 4: Evaluate ---
    step(4, "Evaluating models")
    t = time.time()
    try:
        from src.evaluate import evaluate
        evaluate()
    except Exception as exc:
        print(f"  [ERROR] evaluate failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Done in {time.time()-t:.1f}s")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Run  python predict.py  to score a ticket.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
