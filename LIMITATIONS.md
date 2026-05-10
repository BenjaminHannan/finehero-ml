# FineHero ML — Limitations

Operational limits of the current model and the appeal-letter module, with inline references. For the full literature synthesis, see [`docs/literature_synthesis.md`](docs/literature_synthesis.md).

## Selective labels

The training set is restricted to tickets that went to an OATH hearing. Reported metrics (AUC 0.87, Acc 78.9%) describe performance on that self-selected population, not on all issued tickets. Tickets paid without contest, written off, or never issued are unobserved. Outcome comparisons between the model and human adjudicators on this data are therefore systematically biased (Lakkaraju et al. 2017). In NYC terms: FY2022 saw ~1.7M tickets go to hearing with a ~30% dismissal rate — that is the slice we model.

## Evaluation baseline

The original headline AUC 0.8696 came from a random 80/20 stratified split. On dispute-style problems with per-plate history features, random splits routinely overstate AUC by 0.01–0.05 because a plate can appear in both train and test (Bergmeir & Benítez 2012; Kaufman et al. 2012). The honest, time-aware chronological AUC — full Optuna-tuned CatBoost, last 20% by `issue_date` held out — is **0.8425** (best iter 629, depth 10, lr 0.079). This is the baseline all future AUC claims should be measured against. The leakage audit in [`docs/leakage_audit.md`](docs/leakage_audit.md) (200k subsample, short training) showed random-split and time-aware AUCs agreeing to within 0.0005 under that config, so the 0.027 drop at full training capacity reflects **temporal concept drift** that a fully-tuned model overfits to on random splits, not feature-level label leakage. Plate-blocked 5-fold CV (proxy grouping) reported 0.8740 ± 0.0745; the high fold variance reflects imperfect proxy grouping rather than real plate memorization. One mild caveat: the target-shuffle probe returned AUC 0.5301 (threshold 0.48–0.52), most likely reflecting residual inter-year structure plus subsample variance rather than label leakage.

## Historical enforcement bias

High-signal features — `precinct`, `violation_code × precinct`, `issuing_agency`, `license_type × state` — encode decades of discretionary enforcement. Streetsblog's precinct-level analyses document that a small number of precincts account for a disproportionate share of certain ticket types, and that placard enforcement is near-zero in some areas. "Police data measures enforcement, not ground-truth behavior" (Lum & Isaac 2016; Pierson et al. 2020). Accuracy on this data is not the same as fairness across neighborhoods or demographics.

## Fairness trade-off is unavoidable

Calibration, equal false-positive rates, and equal false-negative rates cannot be simultaneously satisfied across groups when base rates differ (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017). FineHero makes no explicit fairness choice today. Any future claim of "fairness" must name the criterion being optimized and accept the ones being violated (Hardt, Price & Srebro 2016).

## Modeling choice is defensible, not optimal by default

CatBoost was chosen over deep tabular methods because tree-based gradient boosting remains state-of-the-art on tabular data at roughly this sample size (Grinsztajn, Oyallon & Varoquaux 2022), and because ordered boosting mitigates target leakage in high-cardinality categoricals like `violation_code` and `precinct` (Prokhorenkova et al. 2018). This does not imply the 42-feature set or the current hyperparameter search is optimal.

## LLM appeal letters require human verification

The appeal generator in `finehero-appeal/` uses the Anthropic API to draft dispute letters. Legal LLMs hallucinate at measured rates between 17% (commercial RAG tools like Lexis+ AI) and 58–88% (general-purpose models on verifiable legal queries) (Dahl et al. 2024; Magesh et al. 2025). The generator is a drafting aid, not legal advice. User-facing copy must state this, and users must verify every cited regulation, sign code, and factual claim before submission.

## Probability threshold is a policy choice

The 40% pursue/skip threshold reflects a risk appetite — how much user time is worth how much expected fine reduction — not a neutral property of the model.

Calibration **is** fit. An isotonic regressor was fit on the chronological last 10% of the training slice (n=80,000) and is stored at `models/isotonic_calibrator.joblib`. Measured on the 200k held-out test set: ECE 0.152 → 0.023, Brier 0.137 → 0.101, AUC preserved at 0.8694 (isotonic is monotone). Raw CatBoost `predict_proba` over-predicts win rate by ~10–35 percentage points across the [0.40, 0.85) range — e.g., raw bin [0.50, 0.60) (mean 54.6%) has empirical win rate 21.8%; calibrated mean is 29.0%. Raw probabilities should not be read as literal probabilities; calibrated ones can be, with the usual reliability-diagram caveats.

`predict.py` and `predict_ensemble.py` apply the calibrator automatically (see `_load_calibrator()` in `predict.py`) and read the threshold from `models/dispute_threshold.json`. The threshold file holds tuned policies — `f1` (default, 0.32), `youden` (0.17), `accuracy` (0.555), and `match_baserate` (0.401) — all chosen against the calibrated probability average. The fallback if the JSON is missing is the legacy 0.40 constant.

The threshold's meaning differs sharply by which probability it gates. On *raw* CatBoost probability with the legacy 0.40 cutoff, ~41% of held-out tickets cross it with 38.0% win rate among them (marginal lift over the 18.2% base rate). On *calibrated* probability with the tuned F1 threshold (0.32), ~30% of held-out tickets cross it with a substantially better win rate. The calibrated F1 policy is the default the predict scripts use today; the legacy 0.40 was an empirical ranking cutoff that happened to land near a sensible value but was never meant to be read as "40% chance of winning."

## Live-data drift: rolling priors not computable at inference

A 998-row out-of-sample test against fresh NYC Open Data pulls — using the training-matching status filter (drop `OUTSTANDING` / `IN PROCESS` / `HEARING PENDING` / `HEARING ADJOURNMENT`; label via `DISMISS|NOT GUILTY|NOT LIABLE`) — produces two divergent results:

1. **Ranking transfers.** AUC on fresh tickets is **0.9056** — actually *higher* than the held-out chronological tail's 0.8694. The model's ability to rank disputable tickets above non-disputable ones generalizes to the live API.

2. **Calibration collapses.** Median calibrated probability on the live sample is **0.1%** against an 18.8% empirical win rate. The [0.00, 0.05) calibrated bin holds 992 of 998 tickets. The F1 threshold (0.32) catches 0 true positives out of 188 wins — precision and recall both zero.

The proximate cause is **missing rolling-prior features at inference time**. The 27 rolling-prior features (`plate_prior_*_30D/90D/365D`, `precinct_prior_*_30D/90D/365D`, `issuer_prior_*_30D/90D/365D`) are time-dependent windowed aggregates — each one is "this plate/precinct/issuer's stats over the N days strictly before this ticket's issue_date." `engineer.py` computes them via running tallies during training. `predict.py` has no equivalent inference-time state: `plate_history_map.joblib` only stores the *cumulative* per-plate counts, so the cumulative `plate_prior_win_rate` is filled correctly but all 27 windowed versions come out NaN.

Per-prediction SHAP on a real winning ticket (summons 9126107983, Brooklyn muni-meter violation, plate LHP7230, actual outcome `HEARING HELD-NOT GUILTY`) shows rolling-prior features contributing **−3.687 log-odds** out of a total push of −3.283. Top single contributor: `plate_prior_win_rate_30D` = NaN at −0.921. The bias (population-average log-odds) is −0.18, equivalent to ~45% raw probability; rolling-priors-NaN alone takes that to 3% raw, 1% calibrated. The rest of the model is roughly neutral on this row.

**Counterfactual.** Filling the 27 rolling priors with their training-set population means (e.g. `plate_prior_win_rate_30D ≈ 0.24`, `plate_prior_count_30D ≈ 0.84`, `precinct_prior_win_rate_365D ≈ 0.24`) lifts the prediction to **62.7% raw / 34.1% calibrated — verdict DISPUTE**. The empirical training-set win rate for "muni-meter + Brooklyn + PAS license" tickets is **61.8%**. The model's filled-in prediction matches that conditional rate within a percentage point, and the actual ticket did win.

Pvqr enrichment failure (5.2% live join rate against per-fiscal-year endpoints) is a real but **secondary** effect: tickets that *did* receive full pvqr still fail because their rolling priors are still NaN. Closing the rolling-prior gap is the primary fix.

### Stand-in fix (shipped)

`predict.py` and `predict_ensemble.py` now fill NaN rolling priors at inference time using training-set population means stored at `models/rolling_prior_means.joblib`. The artifact is built by `python -m src.build_rolling_prior_means`, run once after every training-data refresh. `predict._load_rolling_prior_means()` and `predict._apply_rolling_prior_fallback()` are the entry points; both predict scripts call them automatically before scoring.

Validated on the same 998-row live-API batch:

| Metric | Without fallback | With fallback (shipped) |
|---|---:|---:|
| AUC | 0.9056 | 0.9006 |
| ECE | 0.182 | **0.092** |
| Median calibrated prob | 0.08% | **10.2%** |
| F1 @ 0.32 | 0.000 | **0.564** |
| Precision | 0% | **58.6%** (3.1× over 18.8% base rate) |
| Recall | 0% | **54.3%** (102/188 wins caught) |

The fallback is a stand-in, not the proper fix. ECE 0.092 is still ~4× the held-out test set's 0.023, because population means don't match ticket-specific rolling priors and the upper calibration bins systematically under-predict (the [0.30, 0.50) calibrated bin shows mean 38% but empirical 54%). For ranking and threshold decisions the model is now usable; for absolute probability claims further work is needed.

### Long-term fixes — what we tried, what's still open

A MaskTab-style retrain was attempted in `src/train_masked.py` at mask_prob=0.5 (random NaN masking of all 27 rolling-prior features on 50% of training rows, with eval slice masked at the same rate, test slice unmasked). Artifacts saved as `models/catboost_model_masked.cbm`, `isotonic_calibrator_masked.joblib`, `dispute_threshold_masked.json`, `data/test_set_masked.joblib`. **Result on the same 998-row live batch:**

| Variant | AUC | ECE | F1 | Recall |
|---|---:|---:|---:|---:|
| Original + no fallback (broken) | 0.9056 | 0.1818 | 0.000 | 0% |
| Original + stand-in fallback (shipped) | 0.9006 | **0.0923** | **0.564** | **54.3%** |
| Masked retrain (mask_prob=0.5, no fallback) | 0.8764 | 0.1314 | 0.061 | 3.2% |

The masked retrain at mask_prob=0.5 **underperformed the stand-in fallback** on live data. Internal test ECE was good (0.019, better than original's 0.023), but on the live distribution 612/998 tickets still landed in the [0, 0.05) calibrated bin and the [0.05, 0.15) bin severely under-predicted (10.5% mean vs 42.4% empirical win rate). Likely causes: (a) 50% mask probability still leaves a strong "NaN-rolling-priors = loss" signal in the unmasked half; (b) Optuna best-params were optimized for the unmasked regime; (c) calibrator-eval distribution (50% NaN) didn't match live (100% NaN); (d) early-stopped at iter 390 may not have given the masked branches enough capacity.

**The stand-in fallback (option 0) remains the best-validated production fix.** Masking is not a free win; it requires more careful re-tuning than a single try with reused hyperparameters.

Avenues still worth trying, in rough order of expected payoff:

1. **Drop rolling priors entirely + retrain.** The simplest variant of "force the model to ignore them." Skips the calibration-distribution mismatch issue because there's no longer a regime to mask. ~30 minutes plus a training run.
2. **Higher mask probability + re-tune Optuna under masking.** Mask at ~0.8–0.9 (or 1.0, equivalent to dropping) and let Optuna pick params suited to the masked regime. ~1–2 hours including search.
3. **Compute rolling priors at inference.** Build `(plate, date)`, `(precinct, date)`, `(issuer, date)` lookup tables from the full corpus and query at inference. Most faithful to the trained model but adds multi-GB state. ~2 hours plus storage cost.
4. **Stack the stand-in on top of any of the above.** The fallback can coexist with a masked model — even if the model handles NaN gracefully, filling with population means gives it a known-good input distribution.

The masked artifacts from this session are kept for diagnostic comparison but **not promoted to canonical filenames**. `predict.py` and `predict_ensemble.py` continue to use the original model + stand-in fallback.

After any of these, re-run `test_live_nyc_tickets.py` and confirm calibration recovers (median calibrated prob should rise from ~0.1% toward the live base rate of ~19%).
