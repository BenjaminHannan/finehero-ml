# test_masked_vs_others.py
#
# Three-way comparison on the 998-row live batch:
#   A) Broken: original model, no rolling-prior fallback (production before fix)
#   B) Stand-in: original model + population-mean fallback (current production)
#   C) Masked: masked-retrained model + masked calibrator + masked threshold

import sys
import os
import json
import joblib
import requests
import io
import re
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict
import test_live_nyc_tickets as t

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
DATA   = os.path.join(ROOT, "data")


def ece(p, y, n_bins=15):
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    e = np.linspace(0, len(p), n_bins + 1, dtype=int)
    return float(sum(((hi - lo) / len(p)) * abs(ps[lo:hi].mean() - ys[lo:hi].mean())
                     for lo, hi in zip(e[:-1], e[1:]) if hi > lo))


def metrics(p, y, t):
    pred = (p >= t).astype(int)
    base = {"auc": roc_auc_score(y, p), "ece": ece(p, y),
            "median_p": float(np.median(p))}
    if pred.sum() == 0:
        base.update({"prec": float("nan"), "rec": 0.0, "f1": 0.0, "pct": 0.0,
                     "tp": 0, "fn": int(y.sum()), "fp": 0, "tn": int((1 - y).sum())})
    else:
        base.update({"prec": float(precision_score(y, pred)),
                     "rec":  float(recall_score(y, pred)),
                     "f1":   float(f1_score(y, pred)),
                     "pct":  float(pred.mean() * 100),
                     "tp": int(((pred == 1) & (y == 1)).sum()),
                     "fn": int(((pred == 0) & (y == 1)).sum()),
                     "fp": int(((pred == 1) & (y == 0)).sum()),
                     "tn": int(((pred == 0) & (y == 0)).sum())})
    return base


def main():
    predict.MODELS_DIR = MODELS
    predict.DATA_DIR = DATA
    predict.PLATE_HISTORY_PATH = os.path.join(MODELS, "plate_history_map.joblib")
    predict.ROLLING_MEANS_PATH = os.path.join(MODELS, "rolling_prior_means.joblib")
    predict.THRESHOLD_PATH = os.path.join(MODELS, "dispute_threshold.json")

    print(" Loading models, calibrators, thresholds...")
    m_orig = CatBoostClassifier(); m_orig.load_model(os.path.join(MODELS, "catboost_model.cbm"))
    m_mask = CatBoostClassifier(); m_mask.load_model(os.path.join(MODELS, "catboost_model_masked.cbm"))
    cat_features = joblib.load(os.path.join(MODELS, "cat_features.joblib"))
    plate_history = joblib.load(os.path.join(MODELS, "plate_history_map.joblib"))
    cal_orig = joblib.load(os.path.join(MODELS, "isotonic_calibrator.joblib"))["calibrator"]
    cal_mask = joblib.load(os.path.join(MODELS, "isotonic_calibrator_masked.joblib"))["calibrator"]
    with open(os.path.join(MODELS, "metadata.json")) as f:
        fnames = json.load(f)["feature_names"]
    with open(os.path.join(MODELS, "dispute_threshold.json")) as f:
        thr_orig = json.load(f)["default_threshold"]
    with open(os.path.join(MODELS, "dispute_threshold_masked.json")) as f:
        thr_mask = json.load(f)["default_threshold"]
    fb = predict._load_rolling_prior_means()

    print(" Fetching live batch...")
    where = ("violation_status IS NOT NULL AND violation_status NOT IN "
             "('OUTSTANDING','IN PROCESS','HEARING PENDING','HEARING ADJOURNMENT')")
    out = pd.read_csv(io.StringIO(requests.get(
        "https://data.cityofnewyork.us/resource/nc67-uf89.csv",
        params={"$limit": 1000, "$where": where, "$order": "issue_date DESC"},
        timeout=60).text), low_memory=False)
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="coerce")
    WIN = re.compile("DISMISS|NOT GUILTY|NOT LIABLE", re.IGNORECASE)
    out["won"] = out["violation_status"].astype(str).apply(lambda s: 1 if WIN.search(s) else 0)
    fresh = t.dedupe_against_training(out)
    fresh["fy"] = fresh["issue_date"].apply(t.fiscal_year)
    pvqr_frames = []
    for fy, group in fresh.groupby("fy"):
        if fy not in t.PVQR_FISCAL_ENDPOINTS:
            continue
        pv = t.fetch_pvqr_for_summons(group["summons_number"].astype(int).tolist(), fy)
        if not pv.empty:
            pvqr_frames.append(pv)
    pvqr = pd.concat(pvqr_frames, ignore_index=True) if pvqr_frames else pd.DataFrame()
    if not pvqr.empty:
        fresh["summons_str"] = fresh["summons_number"].astype(str)
        pvqr["summons_str"] = pvqr["summons_number"].astype(str)
        pv_join = pvqr.drop(columns=[c for c in pvqr.columns
                                     if c in fresh.columns and c != "summons_str"])
        joined = fresh.merge(pv_join, on="summons_str", how="left")
    else:
        joined = fresh.copy()
    print(f"   {len(joined)} rows, {joined['won'].mean()*100:.1f}% wins\n")

    rows = []
    for _, row in joined.iterrows():
        feat, _ = t.build_feature_row(row, plate_history, fnames)
        rows.append(feat)
    df_no_fb   = pd.DataFrame(rows)
    df_with_fb = df_no_fb.copy()
    predict._apply_rolling_prior_fallback(df_with_fb, fb)
    for d in (df_no_fb, df_with_fb):
        for c in cat_features:
            if c in d.columns:
                d[c] = d[c].fillna("UNKNOWN").astype(str).replace(
                    {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})

    y = joined["won"].values

    raw_A = m_orig.predict_proba(df_no_fb)[:, 1]
    cal_A = cal_orig.predict(raw_A)
    raw_B = m_orig.predict_proba(df_with_fb)[:, 1]
    cal_B = cal_orig.predict(raw_B)
    raw_C = m_mask.predict_proba(df_no_fb)[:, 1]
    cal_C = cal_mask.predict(raw_C)

    mA = metrics(cal_A, y, thr_orig)
    mB = metrics(cal_B, y, thr_orig)
    mC = metrics(cal_C, y, thr_mask)

    print(f" {'':<10} {'A) broken':>11} {'B) standin':>11} {'C) masked':>11}")
    print(" " + "-" * 46)
    print(f" {'AUC':<10} {mA['auc']:>11.4f} {mB['auc']:>11.4f} {mC['auc']:>11.4f}")
    print(f" {'ECE':<10} {mA['ece']:>11.4f} {mB['ece']:>11.4f} {mC['ece']:>11.4f}")
    print(f" {'F1':<10} {mA['f1']:>11.3f} {mB['f1']:>11.3f} {mC['f1']:>11.3f}")
    print(f" {'prec':<10} {mA['prec']*100:>10.1f}% {mB['prec']*100:>10.1f}% {mC['prec']*100:>10.1f}%")
    print(f" {'recall':<10} {mA['rec']*100:>10.1f}% {mB['rec']*100:>10.1f}% {mC['rec']*100:>10.1f}%")
    print(f" {'% disp':<10} {mA['pct']:>10.1f}% {mB['pct']:>10.1f}% {mC['pct']:>10.1f}%")
    print(f" {'median p':<10} {mA['median_p']*100:>10.2f}% {mB['median_p']*100:>10.2f}% "
          f"{mC['median_p']*100:>10.2f}%")
    print(f" {'thr':<10} {thr_orig*100:>10.1f}% {thr_orig*100:>10.1f}% {thr_mask*100:>10.1f}%")
    print()
    for label, m in [("A) BROKEN (orig, no FB)", mA),
                     ("B) STAND-IN (orig + FB)", mB),
                     ("C) MASKED RETRAIN", mC)]:
        print(f" {label:<28} TP={m['tp']:>3} FN={m['fn']:>3} FP={m['fp']:>3} TN={m['tn']:>3}")

    print(f"\n Calibration table on live batch — MASKED (variant C):")
    bins = [0, 0.05, 0.15, 0.30, 0.50, 0.85, 1.01]
    df_cal = pd.DataFrame({"p": cal_C, "y": y})
    print(f"   {'bin':<14} {'n':>5} {'mean pred':>10} {'empirical':>10}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df_cal["p"] >= lo) & (df_cal["p"] < hi)
        if mask.sum() == 0:
            continue
        print(f"   [{lo:.2f},{hi:.2f})    {int(mask.sum()):>5} "
              f"{df_cal.loc[mask, 'p'].mean()*100:>9.1f}% "
              f"{df_cal.loc[mask, 'y'].mean()*100:>9.1f}%")


if __name__ == "__main__":
    main()
