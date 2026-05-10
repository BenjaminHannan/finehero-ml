# test_holdout_anchor.py
#
# Score the model on the saved 200k held-out test set as an in-distribution
# anchor for the synthetic-ticket experiment. Reports:
#   - Overall AUC on this sample (sanity vs metadata.json)
#   - Calibration table: predicted-prob deciles vs actual empirical win rate
#   - 10 sampled real tickets (5 wins, 5 losses) with predicted probs
#
# Compared to test_hypothetical_tickets.py, which probes ranking on
# hand-crafted scenarios, this script tells us where the real-ticket
# probability distribution actually sits — so we know how compressed the
# synthetic-ticket numbers are due to OOD effects.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

MODELS_DIR = r"C:\Users\benja\Downloads\finehero-ml\models"
DATA_DIR   = r"C:\Users\benja\Downloads\finehero-ml\data"
CATBOOST_PATH = os.path.join(MODELS_DIR, "catboost_model.cbm")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.joblib")

DISPUTE_THRESHOLD = 0.40
SEED = 42


def main():
    print(" Loading model + held-out test set...")
    model = CatBoostClassifier()
    model.load_model(CATBOOST_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    X_test, y_test, cat_features = joblib.load(TEST_SET_PATH)
    print(f" Model: {meta['model_type']}  metadata AUC {meta['auc_score']:.4f}")
    print(f" Test set: {len(X_test):,} rows, {y_test.mean()*100:.1f}% wins\n")

    print(" Scoring full test set...")
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    preds = (probs >= DISPUTE_THRESHOLD).astype(int)
    acc = (preds == y_test.values).mean()
    print(f" Re-computed AUC: {auc:.4f}  (metadata: {meta['auc_score']:.4f})")
    print(f" Accuracy @ {DISPUTE_THRESHOLD}: {acc*100:.2f}%\n")

    # --- Calibration table ---
    print(" Calibration (predicted-prob bin -> empirical win rate):")
    print(f" {'Bin':<14} {'Count':>8} {'Mean pred':>10} {'Empirical':>10} {'Gap':>8}")
    print(" " + "-" * 54)
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.01]
    df = pd.DataFrame({"prob": probs, "y": y_test.values})
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["prob"] >= lo) & (df["prob"] < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = df.loc[mask, "prob"].mean()
        emp = df.loc[mask, "y"].mean()
        gap = mean_p - emp
        bin_label = f"[{lo:.2f},{hi:.2f})"
        print(f" {bin_label:<14} {n:>8,} {mean_p*100:>9.1f}% {emp*100:>9.1f}% {gap*100:>+7.1f}")
    print()

    # --- Sample 5 wins + 5 losses ---
    rng = np.random.default_rng(SEED)
    win_idx = np.where(y_test.values == 1)[0]
    lose_idx = np.where(y_test.values == 0)[0]
    sampled = np.concatenate([
        rng.choice(win_idx, 5, replace=False),
        rng.choice(lose_idx, 5, replace=False),
    ])

    print(" 10 sampled held-out tickets:")
    print(f" {'#':<3} {'Actual':<8} {'Pred':>7} {'Verdict':<8} {'Correct':<8} "
          f"{'viol':<6} {'precinct':<10} {'within_hours':<14} {'kw_present'}")
    print(" " + "-" * 100)
    correct_count = 0
    for i, idx in enumerate(sampled, 1):
        row = X_test.iloc[idx]
        actual = "WON" if y_test.values[idx] == 1 else "LOST"
        p = probs[idx]
        verdict = "DISPUTE" if p >= DISPUTE_THRESHOLD else "PAY"
        if (verdict == "DISPUTE") == (actual == "WON"):
            correct_count += 1
            mark = "OK"
        else:
            mark = "MISS"
        kws = [k.replace("kw_", "") for k in
               ("kw_meter", "kw_hydrant", "kw_bus_stop", "kw_sign", "kw_blocking", "kw_expired")
               if k in row.index and row[k] == 1.0]
        kw_str = ",".join(kws) if kws else "(none)"
        within = row.get("within_posted_hours", np.nan)
        within_s = "?" if pd.isna(within) else ("yes" if within == 1.0 else "no")
        print(f" {i:<3} {actual:<8} {p*100:>6.1f}% {verdict:<8} {mark:<8} "
              f"{str(row.get('violation_code', '?'))[:6]:<6} "
              f"{str(row.get('precinct', '?'))[:10]:<10} "
              f"{within_s:<14} {kw_str}")
    print(" " + "-" * 100)
    print(f" {correct_count}/10 correct on sample\n")

    # --- Headline comparison ---
    print(" Headline comparison vs synthetic tickets:")
    print(f"   Real held-out tickets: prob range {probs.min()*100:.1f}% – {probs.max()*100:.1f}%, "
          f"median {np.median(probs)*100:.1f}%")
    print(f"   Real win-rate among predicted DISPUTE (>={DISPUTE_THRESHOLD}): "
          f"{y_test.values[probs >= DISPUTE_THRESHOLD].mean()*100:.1f}%")
    print(f"   Real loss-rate among predicted PAY (<{DISPUTE_THRESHOLD}): "
          f"{(1 - y_test.values[probs < DISPUTE_THRESHOLD].mean())*100:.1f}%")
    n_dispute = int((probs >= DISPUTE_THRESHOLD).sum())
    print(f"   {n_dispute:,} of {len(probs):,} ({n_dispute/len(probs)*100:.1f}%) "
          f"real tickets cross the {DISPUTE_THRESHOLD} threshold.")
    print(f"   Synthetic tickets: 0/7 crossed the threshold (max was 22.5%).")


if __name__ == "__main__":
    main()
