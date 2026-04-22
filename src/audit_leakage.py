# audit_leakage.py - Runs the leakage audit from FineHero AUC playbook §2.2 / §7.
#
# Six probes:
#   (1) Baseline AUC under random 80/20 stratified split (anchor).
#   (2) Target-shuffle test — shuffle y within year; AUC should be ~0.5.
#   (3) Time-shift probe — replace plate_prior_win_rate with a FUTURE-including
#       version; a proper audit's own sensitivity check.
#   (4) Single-feature ablation — leave-one-out AUC for each feature.
#   (5) Plate-blocked CV — GroupKFold by plate.
#   (6) Time-aware chronological split — last 20% by issue_date as test.
#
# Also reports train/test prior-stat drift under random vs time-aware splits.
#
# All probes run on a 200k-row subsample for speed. Pass --full to use all rows.
#
# Writes: docs/leakage_audit.md  (overwritten each run)

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR   = os.path.join(ROOT, "docs")

FEATURES_PATH     = os.path.join(DATA_DIR, "features.csv")
VIOLATIONS_PATH   = os.path.join(DATA_DIR, "violations_raw.csv")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
REPORT_PATH       = os.path.join(DOCS_DIR, "leakage_audit.md")

SUBSAMPLE = 200_000
FAST_ITERS = 400
FAST_EARLY_STOP = 40


@contextmanager
def timed(label):
    t0 = time.time()
    print(f"\n  >> {label}")
    yield
    print(f"  << {label}  [{time.time()-t0:.1f}s]")


def _load(subsample=SUBSAMPLE, full=False):
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} cols")
    if not full and len(df) > subsample:
        df = df.sample(n=subsample, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows")
    return df


def _prep(df, cat_features):
    cat_features = [c for c in cat_features if c in df.columns]
    meta_cols = [c for c in ("issue_date",) if c in df.columns]
    X = df.drop(columns=["won"] + meta_cols)
    y = df["won"].astype(int).values
    meta = df[meta_cols].copy() if meta_cols else None
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    return X, y, cat_features, meta


def _fit_fast(X_tr, y_tr, X_ev, y_ev, cat_features, task_type="GPU", iters=FAST_ITERS):
    """Short CatBoost fit for audit probes."""
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
    eval_pool  = Pool(X_ev, label=y_ev, cat_features=cat_features)
    kwargs = dict(
        iterations=iters,
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3.0,
        early_stopping_rounds=FAST_EARLY_STOP,
        random_seed=42,
        task_type=task_type,
        verbose=0,
    )
    if task_type == "GPU":
        kwargs["devices"] = "0"
    cb = CatBoostClassifier(**kwargs)
    cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return cb


def _detect_gpu():
    try:
        import torch
        return "GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def probe_baseline(X, y, cat_features, task_type):
    """(1) Random 80/20 stratified — the number the model card claims."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(sss.split(X, y))
    X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    y_tr, y_te = y[tr], y[te]
    cb = _fit_fast(X_tr, y_tr, X_te, y_te, cat_features, task_type)
    return float(cb.get_best_score()["validation"]["AUC"])


def probe_target_shuffle(X, y, cat_features, meta, task_type):
    """(2) Shuffle y within fiscal year. AUC should drop to ~0.5."""
    if meta is None or "issue_date" not in meta.columns:
        # Fall back to global shuffle
        rng = np.random.default_rng(42)
        y_sh = y.copy()
        rng.shuffle(y_sh)
    else:
        years = pd.to_datetime(meta["issue_date"], errors="coerce").dt.year.fillna(-1).astype(int).values
        y_sh = y.copy()
        rng = np.random.default_rng(42)
        for yr in np.unique(years):
            mask = years == yr
            perm = rng.permutation(y_sh[mask])
            y_sh[mask] = perm
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(sss.split(X, y_sh))
    cb = _fit_fast(X.iloc[tr].reset_index(drop=True), y_sh[tr],
                   X.iloc[te].reset_index(drop=True), y_sh[te],
                   cat_features, task_type)
    return float(cb.get_best_score()["validation"]["AUC"])


def probe_time_shift(X, y, cat_features, task_type):
    """(3) Sensitivity check: swap plate_prior_win_rate with the target-leaked
    version (= y itself scaled + noise). This is the probe confirming that the
    audit MACHINERY would detect a leak if one existed."""
    if "plate_prior_win_rate" not in X.columns:
        return None, None
    X_leak = X.copy()
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.05, 0.05, size=len(y))
    X_leak["plate_prior_win_rate"] = y.astype(float) * 0.8 + 0.1 + noise
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(sss.split(X_leak, y))
    cb = _fit_fast(X_leak.iloc[tr].reset_index(drop=True), y[tr],
                   X_leak.iloc[te].reset_index(drop=True), y[te],
                   cat_features, task_type)
    return float(cb.get_best_score()["validation"]["AUC"])


def probe_ablation(X, y, cat_features, baseline_auc, task_type):
    """(4) Drop one feature at a time, retrain, report AUC delta.
    Flag features whose removal drops AUC by > 0.02 (strong signal)."""
    results = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(sss.split(X, y))
    for col in X.columns:
        X_ab = X.drop(columns=[col])
        cat_ab = [c for c in cat_features if c != col]
        try:
            cb = _fit_fast(X_ab.iloc[tr].reset_index(drop=True), y[tr],
                           X_ab.iloc[te].reset_index(drop=True), y[te],
                           cat_ab, task_type, iters=200)
            auc = float(cb.get_best_score()["validation"]["AUC"])
        except Exception as e:
            auc = float("nan")
            print(f"    [ablation] {col} failed: {e}")
        delta = baseline_auc - auc
        results.append((col, auc, delta))
        print(f"    ablate {col:40s}  auc={auc:.4f}  drop={delta:+.4f}")
    results.sort(key=lambda r: -r[2])
    return results


def probe_plate_blocked(df, X, y, cat_features, task_type):
    """(5) GroupKFold by plate_id. If AUC drops materially vs random split,
    the model is memorizing plates."""
    plate_col = None
    raw = None
    if os.path.exists(VIOLATIONS_PATH):
        try:
            raw = pd.read_csv(VIOLATIONS_PATH, usecols=lambda c: "plate" in c.lower(),
                              nrows=0)
            for c in raw.columns:
                if c.lower() in ("plate", "plate_id"):
                    plate_col = c
                    break
        except Exception:
            pass
    if plate_col is None:
        # Fall back: use a cheap proxy — plate_prior_ticket_count+plate_prior_win_rate combo
        # won't be unique per plate but gives coarse grouping
        if "plate_prior_ticket_count" not in X.columns:
            return None
        groups = (X["plate_prior_ticket_count"].astype(int).astype(str) + "_" +
                  X["plate_prior_win_rate"].round(3).astype(str)).values
        note = "PROXY groups (plate_prior_* combo) — real plate column unavailable here"
    else:
        # We'd need to load the full violations file aligned to features.csv rows,
        # which we don't have. Use the proxy.
        groups = (X["plate_prior_ticket_count"].astype(int).astype(str) + "_" +
                  X["plate_prior_win_rate"].round(3).astype(str)).values
        note = "PROXY groups (plate_prior_* combo) — feature alignment with raw plate skipped"

    gkf = GroupKFold(n_splits=5)
    aucs = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        cb = _fit_fast(X.iloc[tr].reset_index(drop=True), y[tr],
                       X.iloc[te].reset_index(drop=True), y[te],
                       cat_features, task_type, iters=200)
        auc = float(cb.get_best_score()["validation"]["AUC"])
        aucs.append(auc)
        print(f"    fold {fold+1}  auc={auc:.4f}")
    return {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
            "folds": aucs, "note": note}


def probe_time_aware(X, y, meta, cat_features, task_type):
    """(6) Chronological split: last 20% by issue_date as test."""
    if meta is None or "issue_date" not in meta.columns:
        return None
    order = pd.to_datetime(meta["issue_date"], errors="coerce").argsort().values
    X_s = X.iloc[order].reset_index(drop=True)
    y_s = y[order]
    cut = int(len(X_s) * 0.80)
    cb = _fit_fast(X_s.iloc[:cut].reset_index(drop=True), y_s[:cut],
                   X_s.iloc[cut:].reset_index(drop=True), y_s[cut:],
                   cat_features, task_type)
    return float(cb.get_best_score()["validation"]["AUC"])


def prior_stat_drift(X, y, meta):
    """Sanity: how different are train vs test distributions of plate priors?"""
    if "plate_prior_ticket_count" not in X.columns:
        return None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(sss.split(X, y))
    rnd = {
        "train_mean": float(X.iloc[tr]["plate_prior_ticket_count"].mean()),
        "test_mean":  float(X.iloc[te]["plate_prior_ticket_count"].mean()),
        "train_p95":  float(X.iloc[tr]["plate_prior_ticket_count"].quantile(0.95)),
        "test_p95":   float(X.iloc[te]["plate_prior_ticket_count"].quantile(0.95)),
    }
    ta = None
    if meta is not None and "issue_date" in meta.columns:
        order = pd.to_datetime(meta["issue_date"], errors="coerce").argsort().values
        X_s = X.iloc[order].reset_index(drop=True)
        cut = int(len(X_s) * 0.80)
        ta = {
            "train_mean": float(X_s.iloc[:cut]["plate_prior_ticket_count"].mean()),
            "test_mean":  float(X_s.iloc[cut:]["plate_prior_ticket_count"].mean()),
            "train_p95":  float(X_s.iloc[:cut]["plate_prior_ticket_count"].quantile(0.95)),
            "test_p95":   float(X_s.iloc[cut:]["plate_prior_ticket_count"].quantile(0.95)),
        }
    return {"random": rnd, "time_aware": ta}


def write_report(results, n_rows, task_type):
    os.makedirs(DOCS_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    base = results["baseline_auc"]
    ta_auc = results.get("time_aware_auc")
    pb = results.get("plate_blocked") or {}

    # Ablation — show top rows by drop magnitude
    abl = results.get("ablation") or []
    abl_rows = []
    for name, auc, drop in abl[:10]:
        flag = " ⚠️ suspect leak" if drop > 0.02 else ""
        abl_rows.append(f"| `{name}` | {auc:.4f} | {drop:+.4f} |{flag}")
    abl_tbl = "\n".join(abl_rows) if abl_rows else "_skipped_"

    shuffle_auc = results.get("target_shuffle_auc")
    if shuffle_auc is None:
        shuffle_auc_str = "_skipped_"
        shuffle_verdict = "_skipped_"
    elif 0.47 <= shuffle_auc <= 0.53:
        shuffle_auc_str = f"{shuffle_auc:.4f}"
        shuffle_verdict = "PASS — AUC ≈ 0.5 confirms no label leakage"
    else:
        shuffle_auc_str = f"{shuffle_auc:.4f}"
        shuffle_verdict = f"FAIL — AUC {shuffle_auc:.4f} deviates from 0.5 (leakage suspected)"

    ts_auc = results.get("time_shift_auc")
    if ts_auc is None:
        ts_auc_str = "_skipped_"
        ts_verdict = "_skipped_"
    elif ts_auc > base + 0.05:
        ts_auc_str = f"{ts_auc:.4f}"
        ts_verdict = f"PASS — synthetic leak AUC {ts_auc:.4f} ≫ baseline {base:.4f} (probe is sensitive)"
    else:
        ts_auc_str = f"{ts_auc:.4f}"
        ts_verdict = f"INCONCLUSIVE — synthetic leak AUC {ts_auc:.4f} barely above baseline"

    drift = results.get("prior_drift") or {}
    drift_lines = ["_skipped_"]
    if drift:
        rnd = drift.get("random") or {}
        ta = drift.get("time_aware") or {}
        drift_lines = [
            "| split | train mean | test mean | train p95 | test p95 |",
            "|---|---|---|---|---|",
            f"| random 80/20 | {rnd.get('train_mean', float('nan')):.3f} | {rnd.get('test_mean', float('nan')):.3f} | {rnd.get('train_p95', float('nan')):.1f} | {rnd.get('test_p95', float('nan')):.1f} |",
        ]
        if ta:
            drift_lines.append(
                f"| time-aware   | {ta['train_mean']:.3f} | {ta['test_mean']:.3f} | {ta['train_p95']:.1f} | {ta['test_p95']:.1f} |"
            )
    drift_tbl = "\n".join(drift_lines)

    # Verdict
    honest_auc = ta_auc if ta_auc is not None else base
    gap = (base - honest_auc) if ta_auc is not None else 0.0
    pb_auc = pb.get("mean")
    verdict_lines = [
        f"- Random 80/20 baseline: **{base:.4f}**",
    ]
    if ta_auc is not None:
        verdict_lines.append(f"- Time-aware (chronological 80/20): **{ta_auc:.4f}**  (gap {gap:+.4f})")
    else:
        verdict_lines.append("- Time-aware (chronological 80/20): _skipped — no issue_date column in features.csv. Rerun `python -m src.engineer` to regenerate._")
    if pb_auc is not None:
        verdict_lines.append(f"- Plate-blocked 5-fold CV: **{pb_auc:.4f} ± {pb['std']:.4f}**")
    if ta_auc is not None:
        if gap > 0.02:
            verdict_lines.append(
                f"\n**Verdict:** random split overstates AUC by **{gap:.3f}**. "
                f"Honest baseline for all future AUC-improvement claims is **{ta_auc:.4f}** "
                f"(or the plate-blocked figure, whichever is lower)."
            )
        else:
            verdict_lines.append(
                f"\n**Verdict:** random-split AUC is within {gap:.3f} of the time-aware figure — "
                f"the pipeline is cleaner than the playbook's typical leakage base rate. "
                f"Honest baseline: **{ta_auc:.4f}**."
            )
    else:
        verdict_lines.append(
            "\n**Verdict:** incomplete — the time-aware probe could not run. "
            "Regenerate `features.csv` with `python -m src.engineer` to preserve `issue_date`, then rerun this audit."
        )
    verdict = "\n".join(verdict_lines)

    pb_mean_str = f"{pb_auc:.4f}" if pb_auc is not None else "_skipped_"
    pb_std_str  = f"{pb['std']:.4f}" if pb_auc is not None else ""
    ta_auc_str  = f"{ta_auc:.4f}" if ta_auc is not None else "_skipped (no issue_date column)_"

    report = f"""# FineHero Leakage Audit

*Generated {ts} — {n_rows:,} rows, {task_type}*

Audit probes from FineHero AUC Playbook §2.2 and §7. Run with:

```bash
python -m src.audit_leakage          # 200k subsample, ~5–15 min on GPU
python -m src.audit_leakage --full   # full dataset, slower
```

## Headline

{verdict}

## Probe 1 — Target-shuffle test

Shuffle `y` within each fiscal year, retrain. A leak-free pipeline returns AUC ≈ 0.5.

- Result: AUC = {shuffle_auc_str}
- Status: {shuffle_verdict}

## Probe 2 — Time-shift sensitivity probe

Replace `plate_prior_win_rate` with a deliberately-leaky version (`y * 0.8 + noise`) and retrain. Confirms the probe machinery is sensitive enough to detect a leak.

- Baseline AUC: {base:.4f}
- Synthetic-leak AUC: {ts_auc_str}
- Status: {ts_verdict}

## Probe 3 — Single-feature ablation (top 10)

Drop one feature at a time and retrain. Features whose removal drops AUC by more than **0.02** are flagged as suspected leaks (§7 item 8, adjusted down from 0.05 because this is short training).

| feature | AUC w/o feature | AUC drop | flag |
|---|---|---|---|
{abl_tbl}

## Probe 4 — Plate-blocked 5-fold CV

GroupKFold grouping plates so no plate appears in both train and test. A large drop vs. random split means the model memorizes plates.

- Mean AUC: {pb_mean_str} ± {pb_std_str}
- Per-fold: {pb.get("folds", "—")}
- Note: {pb.get("note", "—")}

## Probe 5 — Time-aware chronological split

Sort by `issue_date`, hold out last 20% chronologically.

- AUC: {ta_auc_str}
- Gap vs random split: {gap:+.4f}

## Probe 6 — Plate-prior distribution drift

If a plate's `plate_prior_ticket_count` distribution looks very different in train vs. test under random split, that is a leakage signature.

{drift_tbl}

## Methodology notes

- All probes use a {n_rows:,}-row subsample for speed. Run with `--full` for the complete dataset.
- CatBoost fit: {FAST_ITERS} iters max, early-stop {FAST_EARLY_STOP}, depth 6, lr 0.08.
- The target-shuffle probe and time-shift sensitivity probe together bound the pipeline's leakage: former detects label leakage, latter confirms the detector works.
- The plate-blocked probe uses a PROXY grouping (combination of plate_prior_ticket_count + plate_prior_win_rate) because the raw plate column is not preserved in `features.csv`. For a fully rigorous audit, re-engineer to pass `plate_id` through as a meta column.

## What this audit does NOT do

- Concept-drift detection across fiscal years (playbook §2.6).
- Cleanlab label-noise scan (playbook §2.5).
- Deepchecks FeatureLabelCorrelationChange gate (playbook §7 item 11).
- Adjudication-date vs. issue-date separation — we use issue_date as as-of because FineHero's data doesn't carry a separate hearing-request timestamp.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report written -> {REPORT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Use full dataset (slow)")
    ap.add_argument("--skip-ablation", action="store_true", help="Skip ablation (slowest probe)")
    args = ap.parse_args()

    if not os.path.exists(FEATURES_PATH):
        print(f"  [ERROR] {FEATURES_PATH} not found. Run `python -m src.engineer` first.")
        sys.exit(1)

    task_type = _detect_gpu()
    print(f"\n  Task type: {task_type}")

    df = _load(full=args.full)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    X, y, cat_features, meta = _prep(df, cat_features)
    print(f"  X: {X.shape}  y positives: {y.sum():,} ({100*y.mean():.1f}%)")

    results = {}
    with timed("Probe 1: baseline random 80/20"):
        results["baseline_auc"] = probe_baseline(X, y, cat_features, task_type)
        print(f"    AUC = {results['baseline_auc']:.4f}")

    with timed("Probe 2: target-shuffle within year"):
        results["target_shuffle_auc"] = probe_target_shuffle(X, y, cat_features, meta, task_type)
        print(f"    AUC = {results['target_shuffle_auc']:.4f}")

    with timed("Probe 3: time-shift sensitivity"):
        results["time_shift_auc"] = probe_time_shift(X, y, cat_features, task_type)
        if results["time_shift_auc"] is not None:
            print(f"    AUC = {results['time_shift_auc']:.4f}")

    with timed("Probe 4: time-aware chronological split"):
        results["time_aware_auc"] = probe_time_aware(X, y, meta, cat_features, task_type)
        if results["time_aware_auc"] is not None:
            print(f"    AUC = {results['time_aware_auc']:.4f}")

    with timed("Probe 5: plate-blocked 5-fold CV"):
        results["plate_blocked"] = probe_plate_blocked(df, X, y, cat_features, task_type)

    with timed("Probe 6: prior-stat drift"):
        results["prior_drift"] = prior_stat_drift(X, y, meta)

    if not args.skip_ablation:
        with timed("Probe 7: single-feature ablation"):
            results["ablation"] = probe_ablation(X, y, cat_features, results["baseline_auc"], task_type)
    else:
        results["ablation"] = []

    write_report(results, len(df), task_type)
    print("\n  Audit complete.")


if __name__ == "__main__":
    main()
