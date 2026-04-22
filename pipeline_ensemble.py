# pipeline_ensemble.py — Full FineHero ensemble pipeline, end to end.
#
#   python pipeline_ensemble.py                 # everything except cleanlab
#   python pipeline_ensemble.py --cleanlab      # + label-noise scan
#   python pipeline_ensemble.py --fetch         # also refetch raw NYC data
#   python pipeline_ensemble.py --skip-engineer # use existing features.csv
#
# Order:
#   1. (optional) fetch raw data
#   2. engineer features (with rolling-window features, playbook §3.6)
#   3. train CatBoost (+ LR baseline)
#   4. evaluate → writes models/metadata.json (so predict.py works)
#   5. train LightGBM
#   6. train XGBoost
#   7. ensemble → rank-averaged blend, reports individual + blended AUCs
#   8. (optional) cleanlab label-noise scan

import argparse
import sys
import time


def step(n, title):
    print(f"\n{'-'*60}\n  STEP {n}: {title}\n{'-'*60}")


def run(label, fn):
    t = time.time()
    try:
        fn()
    except Exception as exc:
        print(f"  [ERROR] {label} failed: {exc}", file=sys.stderr)
        raise
    dur = time.time() - t
    mins = dur / 60.0
    print(f"  {label} done in {dur:.1f}s ({mins:.1f} min)")
    return dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Re-fetch raw NYC Open Data (slow, default: skip)")
    parser.add_argument("--rows", type=int, default=1_000_000,
                        help="Rows to fetch if --fetch is set")
    parser.add_argument("--skip-engineer", action="store_true",
                        help="Use existing features.csv (skip src.engineer)")
    parser.add_argument("--skip-catboost", action="store_true",
                        help="Skip CatBoost training (use existing model)")
    parser.add_argument("--cleanlab", action="store_true",
                        help="Run cleanlab label-noise scan after ensemble")
    parser.add_argument("--cleanlab-subsample", type=int, default=None,
                        help="Subsample size for cleanlab (default: full)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FineHero ML - Ensemble Pipeline")
    print("=" * 60)
    t0 = time.time()

    n = 0
    if args.fetch:
        n += 1
        step(n, "Fetching raw data")
        from src.fetch_data import fetch_all
        run("fetch", lambda: fetch_all(args.rows))

    if not args.skip_engineer:
        n += 1
        step(n, "Engineering features (incl. rolling-window)")
        from src.engineer import engineer_features
        run("engineer", engineer_features)

    if not args.skip_catboost:
        n += 1
        step(n, "Training CatBoost (+ LR baseline)")
        from src.train import train_models
        run("train", train_models)

    n += 1
    step(n, "Evaluating CatBoost → metadata.json")
    from src.evaluate import evaluate
    run("evaluate", evaluate)

    n += 1
    step(n, "Training LightGBM")
    import src.train_lgb as train_lgb
    run("train_lgb", train_lgb.main)

    n += 1
    step(n, "Training XGBoost")
    import src.train_xgb as train_xgb
    run("train_xgb", train_xgb.main)

    n += 1
    step(n, "Ensemble: rank-averaged blend")
    import src.ensemble as ensemble
    run("ensemble", ensemble.main)

    if args.cleanlab:
        n += 1
        step(n, "Cleanlab label-noise scan")
        import src.cleanlab_scan as cleanlab_scan
        saved_subsample = cleanlab_scan.SUBSAMPLE_ROWS
        if args.cleanlab_subsample:
            cleanlab_scan.SUBSAMPLE_ROWS = args.cleanlab_subsample
        try:
            run("cleanlab", cleanlab_scan.main)
        finally:
            cleanlab_scan.SUBSAMPLE_ROWS = saved_subsample

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Score a ticket:         python predict_ensemble.py")
    print(f"  Single-model (CatBoost): python predict.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
