# test_batch_with_fix.py
#
# Re-runs the 998-row live-API batch from test_live_nyc_tickets.py twice:
#   - Variant A (broken):  rolling priors stay NaN, as predict.py does today
#   - Variant B (fixed):   rolling priors filled with training-set population means
#                          (poor man's stand-in for actually computing them)
#
# Reports AUC, ECE, F1, precision, recall, calibration table for each variant
# and a head-to-head accuracy table at the F1 threshold.

import os
import sys
import json
import joblib
import requests
import io
import re
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_live_nyc_tickets as t

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")


def expected_calibration_error(probs, y, n_bins=15):
    order = np.argsort(probs)
    p_sorted = probs[order]
    y_sorted = y[order]
    edges = np.linspace(0, len(p_sorted), n_bins + 1, dtype=int)
    n = len(p_sorted)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        ece += ((hi - lo) / n) * abs(p_sorted[lo:hi].mean() - y_sorted[lo:hi].mean())
    return float(ece)


def metrics(probs, y, threshold):
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    ece = expected_calibration_error(probs, y)
    pred = (probs >= threshold).astype(int)
    if pred.sum() == 0:
        prec = float("nan"); rec = 0.0; f1 = 0.0
    elif pred.sum() == len(pred):
        prec = float(y.mean()); rec = 1.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    else:
        prec = precision_score(y, pred)
        rec  = recall_score(y, pred)
        f1   = f1_score(y, pred)
    return {"auc": auc, "ece": ece, "precision": prec, "recall": rec, "f1": f1,
            "pct_dispute": pred.mean() * 100,
            "median_prob": float(np.median(probs)) * 100}


def bootstrap_metrics(probs, y, threshold, *, n_boot=1000, seed=42):
    """Bootstrap-resample the batch n_boot times; return mean and 95% CI for
    AUC, ECE, F1, precision, recall.

    Per Pernot 2023 (arXiv:2306.05180) and the "Classifier Calibration at
    Scale" 2026 paper, ECE estimates have nontrivial variance below n=1000
    and isotonic stratification can introduce additional aleatoric variance
    in the metric itself. So our 998-row headline numbers should always be
    reported with a confidence interval, not as a single point.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, eces, f1s, precs, recs = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = probs[idx]
        if len(np.unique(yb)) < 2:
            continue  # degenerate sample
        m = metrics(pb, yb, threshold)
        aucs.append(m["auc"]); eces.append(m["ece"])
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    def summarize(vals):
        a = np.asarray([v for v in vals if not np.isnan(v)])
        if len(a) == 0:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"mean": float(a.mean()),
                "lo":   float(np.percentile(a, 2.5)),
                "hi":   float(np.percentile(a, 97.5))}

    return {"auc":       summarize(aucs),
            "ece":       summarize(eces),
            "f1":        summarize(f1s),
            "precision": summarize(precs),
            "recall":    summarize(recs),
            "n_boot":    int(len(aucs))}


def calibration_table(probs, y):
    bins = [0, 0.05, 0.15, 0.30, 0.50, 0.85, 1.01]
    df = pd.DataFrame({"p": probs, "y": y})
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["p"] >= lo) & (df["p"] < hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})",
                     "n": int(m.sum()),
                     "mean_pred": df.loc[m, "p"].mean() * 100,
                     "empirical": df.loc[m, "y"].mean() * 100})
    return rows


def main():
    print(" Loading model + calibrator + threshold + plate history...")
    model = CatBoostClassifier(); model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
    cat_features = joblib.load(os.path.join(MODELS_DIR, "cat_features.joblib"))
    plate_history = joblib.load(os.path.join(MODELS_DIR, "plate_history_map.joblib"))
    cal = joblib.load(os.path.join(MODELS_DIR, "isotonic_calibrator.joblib"))["calibrator"]
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        fnames = json.load(f)["feature_names"]
    with open(os.path.join(MODELS_DIR, "dispute_threshold.json")) as f:
        cfg = json.load(f)
    threshold = cfg["default_threshold"]; policy = cfg["default_policy"]
    print(f"   threshold={threshold:.3f} (policy: {policy})")

    # Read the SAME artifact production uses, instead of recomputing locally.
    # Older versions of this script recomputed means inline and hardcoded
    # issuer_bayes_default = 0.18, which diverged from the production
    # base-rate fallback (~0.236). See docs/rolling_prior_investigation.md §12
    # for the verdict-flipping consequence on the borderline winner 9070812812.
    rolling_means_path = os.path.join(MODELS_DIR, "rolling_prior_means.joblib")
    if os.path.exists(rolling_means_path):
        artifact = joblib.load(rolling_means_path)
        prior_means = artifact["rolling_prior_means"]
        days_since_means = artifact.get("days_since_defaults", {})
        issuer_bayes_default = artifact.get("issuer_bayes_default", 0.18)
        print(f"   loaded rolling_prior_means.joblib: "
              f"{len(prior_means)} rolling priors, "
              f"issuer_bayes_default = {issuer_bayes_default:.4f}")
    else:
        # Fallback: recompute from features.csv if the artifact is missing.
        print("   [WARN] rolling_prior_means.joblib missing — recomputing from features.csv")
        use_cols = lambda c: "prior_" in c and ("30D" in c or "90D" in c or "365D" in c)
        fcsv = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                           usecols=use_cols, low_memory=False)
        prior_means = {c: float(fcsv[c].mean()) for c in fcsv.columns}
        days_since_means = {"days_since_plate_last_ticket": 365.0,
                            "days_since_plate_last_win": 365.0,
                            "days_since_issuer_last_ticket": 30.0}
        # Compute the same base-rate fallback build_rolling_prior_means.py uses
        issuer_bayes_default = float(
            pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                        usecols=["won"], low_memory=False)["won"].mean()
        )
    print(f"   {len(prior_means)} rolling-prior features cached\n")

    # ----- 1. Fetch & dedupe -----
    print(" Fetching outcomes (training-matching filter)...")
    where = ("violation_status IS NOT NULL AND "
             "violation_status NOT IN ('OUTSTANDING','IN PROCESS','HEARING PENDING','HEARING ADJOURNMENT')")
    r = requests.get("https://data.cityofnewyork.us/resource/nc67-uf89.csv",
                     params={"$limit": 1000, "$where": where, "$order": "issue_date DESC"},
                     timeout=60)
    r.raise_for_status()
    out = pd.read_csv(io.StringIO(r.text), low_memory=False)
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="coerce")
    WIN = re.compile("DISMISS|NOT GUILTY|NOT LIABLE", re.IGNORECASE)
    out["won"] = out["violation_status"].astype(str).apply(lambda s: 1 if WIN.search(s) else 0)
    print(f"   got {len(out)} rows, win rate {out['won'].mean()*100:.1f}%")

    fresh = t.dedupe_against_training(out)

    # ----- 2. pvqr enrichment -----
    fresh["fy"] = fresh["issue_date"].apply(t.fiscal_year)
    pvqr_frames = []
    for fy, group in fresh.groupby("fy"):
        if fy not in t.PVQR_FISCAL_ENDPOINTS:
            continue
        ss = group["summons_number"].astype(int).tolist()
        pv = t.fetch_pvqr_for_summons(ss, fy)
        if not pv.empty:
            pvqr_frames.append(pv)
    pvqr = pd.concat(pvqr_frames, ignore_index=True) if pvqr_frames else pd.DataFrame()
    if not pvqr.empty:
        fresh["summons_str"] = fresh["summons_number"].astype(str)
        pvqr["summons_str"] = pvqr["summons_number"].astype(str)
        pv_join = pvqr.drop(columns=[c for c in pvqr.columns if c in fresh.columns and c != "summons_str"])
        joined = fresh.merge(pv_join, on="summons_str", how="left")
    else:
        joined = fresh.copy()

    n_pvqr = joined["from_hours_in_effect"].notna().sum() if "from_hours_in_effect" in joined.columns else 0
    print(f"   pvqr hit rate: {n_pvqr}/{len(joined)} ({n_pvqr/len(joined)*100:.1f}%)\n")

    # ----- 3. Build feature rows -----
    print(" Building feature rows...")
    rows = []
    for _, row in joined.iterrows():
        feat, _ = t.build_feature_row(row, plate_history, fnames)
        rows.append(feat)
    df_broken = pd.DataFrame(rows)
    df_fixed  = df_broken.copy()

    # Apply the fix: fill rolling priors with population means (only if NaN)
    for c, m in prior_means.items():
        if c in df_fixed.columns:
            df_fixed[c] = df_fixed[c].fillna(m)
    for c, m in days_since_means.items():
        if c in df_fixed.columns:
            df_fixed[c] = df_fixed[c].fillna(m)
    if "issuer_bayes_rate" in df_fixed.columns:
        df_fixed["issuer_bayes_rate"] = df_fixed["issuer_bayes_rate"].fillna(issuer_bayes_default)

    # Categorical string normalization (both)
    for d in (df_broken, df_fixed):
        for c in cat_features:
            if c in d.columns:
                d[c] = d[c].fillna("UNKNOWN").astype(str).replace(
                    {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    y = joined["won"].values
    print(f"   {len(df_broken):,} rows, {y.mean()*100:.1f}% wins\n")

    # ----- 4. Score both variants -----
    print(" Scoring (broken)...")
    raw_b = model.predict_proba(df_broken)[:, 1]
    cal_b = cal.predict(raw_b)

    print(" Scoring (fixed)...")
    raw_f = model.predict_proba(df_fixed)[:, 1]
    cal_f = cal.predict(raw_f)

    # ----- 5. Metrics -----
    print("\n" + "=" * 70)
    print(" Metrics on calibrated probabilities (training-matching filter)")
    print("=" * 70)

    mb = metrics(cal_b, y, threshold)
    mf = metrics(cal_f, y, threshold)

    print(f" {'metric':<22} {'broken':>14} {'fixed':>14} {'delta':>10}")
    print(" " + "-" * 64)
    print(f" {'AUC':<22} {mb['auc']:>14.4f} {mf['auc']:>14.4f} {mf['auc']-mb['auc']:>+10.4f}")
    print(f" {'ECE':<22} {mb['ece']:>14.4f} {mf['ece']:>14.4f} {mf['ece']-mb['ece']:>+10.4f}")
    print(f" {'precision':<22} {mb['precision']*100:>13.1f}% {mf['precision']*100:>13.1f}% "
          f"{(mf['precision']-mb['precision'])*100:>+9.1f}")
    print(f" {'recall':<22} {mb['recall']*100:>13.1f}% {mf['recall']*100:>13.1f}% "
          f"{(mf['recall']-mb['recall'])*100:>+9.1f}")
    print(f" {'F1':<22} {mb['f1']:>14.3f} {mf['f1']:>14.3f} {mf['f1']-mb['f1']:>+10.3f}")
    print(f" {'% disputed':<22} {mb['pct_dispute']:>13.1f}% {mf['pct_dispute']:>13.1f}% "
          f"{mf['pct_dispute']-mb['pct_dispute']:>+9.1f}")
    print(f" {'median calibrated':<22} {mb['median_prob']:>13.2f}% {mf['median_prob']:>13.2f}% "
          f"{mf['median_prob']-mb['median_prob']:>+9.2f}")

    # ----- 6. Calibration tables -----
    print(f"\n Calibration table: BROKEN")
    print(f"   {'bin':<14} {'n':>5} {'mean pred':>10} {'empirical':>10}")
    for r in calibration_table(cal_b, y):
        print(f"   {r['bin']:<14} {r['n']:>5} {r['mean_pred']:>9.1f}% {r['empirical']:>9.1f}%")

    print(f"\n Calibration table: FIXED")
    print(f"   {'bin':<14} {'n':>5} {'mean pred':>10} {'empirical':>10}")
    for r in calibration_table(cal_f, y):
        print(f"   {r['bin']:<14} {r['n']:>5} {r['mean_pred']:>9.1f}% {r['empirical']:>9.1f}%")

    # ----- 7. Confusion at threshold -----
    print(f"\n Verdict accuracy at threshold {threshold:.2f}:")
    print(f"   {'variant':<10} {'TP':>6} {'FN':>6} {'FP':>6} {'TN':>6} {'acc':>7}")
    for label, p in (("broken", cal_b), ("fixed", cal_f)):
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        acc = (tp + tn) / len(y)
        print(f"   {label:<10} {tp:>6} {fn:>6} {fp:>6} {tn:>6} {acc*100:>6.1f}%")

    # ----- 8. Bootstrap 95% CIs (so future readers know precision of headline numbers) -----
    print("\n Bootstrap 95% CIs (1000 resamples):")
    print(f"   {'metric':<10} {'broken (mean [lo, hi])':<32} {'fixed (mean [lo, hi])':<32}")
    print("   " + "-" * 74)
    boot_b = bootstrap_metrics(cal_b, y, threshold)
    boot_f = bootstrap_metrics(cal_f, y, threshold)
    for key in ("auc", "ece", "f1", "precision", "recall"):
        b = boot_b[key]; f = boot_f[key]
        scale = 100 if key in ("precision", "recall") else 1
        unit = "%" if key in ("precision", "recall") else ""
        print(f"   {key:<10} "
              f"{b['mean']*scale:6.3f}{unit} [{b['lo']*scale:6.3f}, {b['hi']*scale:6.3f}]{unit:<5} "
              f"{f['mean']*scale:6.3f}{unit} [{f['lo']*scale:6.3f}, {f['hi']*scale:6.3f}]{unit}")


if __name__ == "__main__":
    main()
