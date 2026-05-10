# FineHero ML — Project Reference

A single-document overview of the FineHero parking-ticket dispute-prediction system: what it is, how it's built, how to run it, what its known failure modes are, and how to extend it without breaking anything. This file is the canonical entry point. The other docs (`CLAUDE.md`, `LIMITATIONS.md`, `docs/*.md`) are still authoritative on their specific topics, but everything important is consolidated here.

---

## 1. What this is

FineHero ML predicts the probability that a New York City parking ticket will be successfully disputed at an OATH (Office of Administrative Trials and Hearings) hearing. Given a ticket and the plate's history, it outputs a calibrated win-probability and a DISPUTE/PAY recommendation.

The system has two parts:

- **`finehero-ml/`** (the model) — CatBoost gradient-boosting classifier trained on ~1M historical NYC tickets pulled from NYC Open Data. Augmented by LightGBM and XGBoost for ensemble use. ~95 features spanning temporal patterns, financial characteristics, weather, officer-level metadata, vehicle info, plate history, and rolling priors per (plate / precinct / issuer) × time window.
- **`finehero-appeal/`** (the letter generator) — separate pipeline that uses the Anthropic Claude API (`claude-sonnet-4-6`) to analyze user-submitted evidence (photos, PDFs, documents), select from 15+ NYC-specific dispute strategies, and produce a professionally formatted PDF appeal letter via ReportLab.

Headline numbers (chronological holdout, time-aware split):

- AUC: **0.8694** (CatBoost alone) / **0.8742** (XGBoost) / **0.8739** (best rank-blend)
- Calibration ECE: **0.023** (after isotonic calibration; 0.152 before)
- Test base rate: 18.2% wins
- Default DISPUTE/PAY threshold: **0.32** (F1-optimal on calibrated probabilities)

Live-API performance (998 fresh tickets pulled from the same Socrata endpoints, training-matching filter, deduped):

- AUC: **0.9006** (after the rolling-prior fallback shipped in this codebase)
- F1: **0.564** (vs 0.000 before the fallback)
- Recall: **54.3%** of actual winners caught at the 0.32 threshold
- Precision: **58.6%** — when the model says DISPUTE, the driver wins ~59% of the time, a 3.1× lift over the 18.8% live base rate

Reaching those live numbers required diagnosing a real production bug. See §11.

---

## 2. Quick start

```bash
pip install -r requirements.txt

# Train (interactive: prompts for row count)
python pipeline.py

# Single-ticket inference (uses calibrator + tuned threshold automatically)
python predict.py

# Ensemble inference (CatBoost + LightGBM + XGBoost rank-blended)
python predict_ensemble.py

# Build the rolling-prior fallback artifact (run after every retrain)
python -m src.build_rolling_prior_means

# Run the as-of prior tests
pytest tests/

# Live-API spot check
python test_live_nyc_tickets.py

# 998-row live-batch validation
python test_batch_with_fix.py
```

`finehero-appeal/` has its own `requirements.txt` and runs separately.

---

## 3. Repo layout

```
finehero-ml/
├── pipeline.py                  # orchestrator: fetch → engineer → train → evaluate
├── pipeline_ensemble.py         # same, for the LGB/XGB/CB ensemble path
├── pipeline_ui.py               # interactive UI wrapper
├── predict.py                   # single-ticket CatBoost inference (calibrator + threshold + rolling-prior fallback)
├── predict_ensemble.py          # ensemble inference (rank-blend across three models)
├── requirements.txt
│
├── src/
│   ├── fetch_data.py            # pulls the two NYC Open Data endpoints + weather
│   ├── engineer.py              # feature engineering. ~40 base features + cross + cyclical + rolling priors
│   ├── train.py                 # CatBoost training (Optuna search, time-aware split)
│   ├── train_lgb.py             # LightGBM training
│   ├── train_xgb.py             # XGBoost training
│   ├── train_masked.py          # diagnostic: MaskTab-style retrain (see §11.4)
│   ├── ensemble.py              # rank-percentile blending logic
│   ├── evaluate.py              # metrics + saves models/metadata.json
│   ├── audit_leakage.py         # leakage probes (random vs time-aware AUC, target shuffle, plate-blocked CV)
│   ├── cleanlab_scan.py         # label-noise scan
│   └── build_rolling_prior_means.py  # builds models/rolling_prior_means.joblib
│
├── finehero-appeal/             # separate appeal-letter service
│   ├── appeal.py
│   ├── requirements.txt
│   └── src/
│
├── docs/
│   ├── auc_progress.md          # running scoreboard of AUC changes
│   ├── leakage_audit.md         # leakage probe results
│   ├── literature_synthesis.md  # cited references
│   └── rolling_prior_investigation.md  # full narrative of §11 (this section in expanded form)
│
├── tests/
│   └── test_asof_priors.py      # enforces strict-past-only for any aggregate feature
│
├── models/                      # trained artifacts (created by pipeline.py + the build_* scripts in src/)
│   ├── catboost_model.cbm
│   ├── lgb_model.txt
│   ├── xgb_model.json
│   ├── lr_model.joblib
│   ├── isotonic_calibrator.joblib    # ECE 0.152→0.023 on test set
│   ├── dispute_threshold.json        # F1=0.32 (default), youden, accuracy, match_baserate
│   ├── rolling_prior_means.joblib    # the rolling-prior fallback (see §11)
│   ├── plate_history_map.joblib      # cumulative per-plate history
│   ├── cat_features.joblib
│   ├── metadata.json
│   ├── best_params.joblib            # Optuna-tuned CatBoost params
│   └── *_masked.*                    # diagnostic artifacts from train_masked.py
│
├── data/                        # cached datasets and intermediates
│   ├── violations_raw.csv       # joined nc67-uf89 + pvqr (training source)
│   ├── outcomes_raw.csv         # nc67-uf89 cache
│   ├── pvqr_raw.csv             # pvqr-* caches per fiscal year
│   ├── features.csv             # post-engineering training data
│   ├── test_set.joblib          # X_test, y_test, feature_names — held out from training
│   ├── adjudications_raw.csv
│   ├── weather_cache.csv
│   └── raw_categoricals.csv
│
├── FineHero_Methodology.docx    # academic-style methodology writeup
├── PROJECT.md                   # this file
├── CLAUDE.md                    # ground rules + Claude-agent instructions
├── LIMITATIONS.md               # operational caveats with numbers
└── README.md                    # (if present, defers to PROJECT.md)
```

---

## 4. Data sources

Two NYC Open Data Socrata endpoints (free, no API key):

- **`nc67-uf89`** — Open Parking & Camera Violations. Provides ticket *outcomes*: dismissed, paid, guilty, reduction granted, hearing held. This is the label source.
- **`pvqr-7yc4`** — Parking Violations Issued (current fiscal year). Plus the per-fiscal-year endpoints `7mxj-7a6y` (FY2022), `869v-vr48` (FY2023), `8zf9-spf8` (FY2024), `m5vz-tzqv` (FY2025). Each covers ~1 fiscal year of pre-hearing metadata: issuing officer, street, vehicle details, posted hours-in-effect, feet from curb, legal code.

**Acquisition strategy** (in `src/fetch_data.py`):

- Both endpoints fetched in 50,000-row paginated requests.
- pvqr is fetched at 2× the target row count (across 4 fiscal years, ~250k each) to maximize summons-level join coverage with outcomes (since pvqr is split by FY and outcomes are not).
- All API responses cached to disk (`outcomes_raw.csv`, `pvqr_raw.csv`) so reruns skip network fetches.
- Joined on `summons_number` via left join with outcomes as the base, written to `violations_raw.csv`.
- Rows with `violation_status` of `OUTSTANDING` or `IN PROCESS` are dropped — no outcome yet.

**Weather**: Open-Meteo Historical Archive (free, rate-limited, no API key). Cached per NYC borough.

Final merged training corpus: ~1,000,000 rows.

**Important:** the per-FY pvqr coverage matters for live serving. Training join rate is 30–50%. Live join rate (when querying for a freshly-fetched batch) is closer to 5%. This isn't the dominant production failure mode, but it's a real distribution gap. See §11.5.

---

## 5. Feature engineering

All features must be **strictly pre-hearing** — information available at issue time. The `tests/test_asof_priors.py` suite enforces this for any aggregate feature.

The 95 model features (after `engineer.py`):

### 5.1 Temporal (11 features)

- `hour_of_offense` — parsed from NYC's non-standard `"0156P"` format via multi-format regex
- `day_of_week`, `month`, `is_weekend`, `is_holiday`
- `days_since_ticket` — captures temporal drift in enforcement
- Cyclical encodings: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`

### 5.2 Financial (1 feature)

- `fine_amount` — pre-hearing
- **Dropped at source**: `penalty_amount`, `reduction_amount`, `payment_amount`, `amount_due` (all post-hearing, never feature-engineered)

### 5.3 Weather (5 features)

- `precipitation`, `visibility`, `wind_speed`, `weather_code`, `is_bad_weather`
- `is_bad_weather` flags: rain >2mm, fog (codes 45/48), snow (≥71)
- Sourced from Open-Meteo *archive*, not forecast — genuinely available the day of the ticket

### 5.4 Keyword flags (6 features)

Regex-matched against `violation_description` (preferred) or `violation_code` text:

- `kw_meter`, `kw_hydrant`, `kw_bus_stop`, `kw_sign`, `kw_blocking`, `kw_expired`

### 5.5 pvqr-only rich features (9 features, only when join hits)

- `within_posted_hours` — handles overnight ranges like 10PM–6AM. **One of the strongest dismissal signals** when populated.
- `feet_from_curb`, `house_number`, `vehicle_year`
- `from_hour`, `to_hour`, `hours_kind_from`, `hours_kind_to`, `hours_in_effect_all`
- `first_observed_filled`

### 5.6 Core categoricals (passed as raw strings to CatBoost's `cat_features`)

CatBoost builds ordered target statistics internally — leakage-safe, no manual encoding pipeline.

- `violation_code`, `precinct`, `county`, `issuing_agency`, `license_type`, `state`
- `issuer_code`, `issuer_command`, `issuer_squad` (high-cardinality officer fields)
- `street_name` (very high-cardinality; CatBoost smooths rare values automatically)
- `vehicle_make`, `vehicle_color`, `vehicle_body_type`
- `violation_legal_code`, `law_section`, `sub_division`
- `violation_in_front_of_or_opposite`, `days_parking_in_effect`

### 5.7 Cross-features (string-concatenated, also categorical)

- `viol_x_precinct` — violation × precinct (some violations dismissed in some precincts but not others)
- `viol_x_license` — violation × license type (commercial vs passenger have very different outcomes)
- `hour_x_dow` — hour bin × day of week
- `format_x_viol`, `summons_format`, `format_x_missing_plate`
- `agency_x_command`

### 5.8 Plate history (cumulative; 2 features)

Leave-one-out chronological encoding. Computed via `plate_history_map.joblib` at inference time.

- `plate_prior_ticket_count` — number of *strictly prior* tickets for this plate
- `plate_prior_win_rate` — historical win rate, smoothed with `PLATE_SMOOTH_K=10` toward the global mean

### 5.9 Rolling priors (27 features) — **see §11 for the operational gotcha**

Three entity types × three windows × three statistics:

- Entities: `plate`, `precinct`, `issuer`
- Windows: 30D, 90D, 365D (strictly before this ticket's issue_date)
- Statistics: `wins`, `count`, `win_rate`

Plus 3 `days_since_*` features and `issuer_bayes_rate` (smoothed posterior win rate per issuer).

These features are time-dependent windowed aggregates that `engineer.py` builds via running tallies during training. **At inference, `predict.py` cannot recompute them on the fly** — it has no `(entity, date)` lookup tables. The `models/rolling_prior_means.joblib` artifact (built by `python -m src.build_rolling_prior_means`) holds population-mean fallbacks that the inference scripts apply automatically. See §11 for the full story.

### 5.10 Missing-field flags (10 features)

`is_missing_vehicle_make`, `is_missing_vehicle_body_type`, `is_missing_street_name`, `is_missing_issuer_command`, `is_missing_issuer_squad`, `is_missing_violation_legal_code`, `is_missing_days_parking_in_effect`, `is_missing_vehicle_year`, `is_missing_plate`, `missing_fields_count`.

---

## 6. Leakage prevention

Treated as the most important property of the model. Six mechanisms:

1. **Post-hearing financial columns dropped at source.** `penalty_amount`, `reduction_amount`, `payment_amount`, `amount_due` never reach the engineering pipeline.
2. **`OUTSTANDING` / `IN PROCESS` rows filtered out.** Tickets without resolved outcomes don't enter training.
3. **Plate history is leave-one-out chronological.** Current ticket's outcome is never in its own features. Smoothing constant `PLATE_SMOOTH_K=10` toward the chronological-80% training mean.
4. **Train/test split happens before any target encoding.** No test-set outcomes leak into training stats.
5. **Weather is archive (past), not forecast.** The Open-Meteo Historical Archive returns weather as recorded for that date.
6. **CatBoost ordered target statistics handle within-fold encoding leakage.** The model's permuted-order computation prevents future-row leakage during target encoding for high-cardinality categoricals.

If you add a new aggregate feature: add an as-of test in `tests/test_asof_priors.py` and confirm the random-vs-time AUC gap stays small via `src/audit_leakage.py`.

The leakage audit (`docs/leakage_audit.md`) shows random-split and time-aware AUCs agreeing within 0.0005 on a 200k subsample, so the ~0.027 drop at full-training capacity vs random splits reflects **temporal concept drift** rather than feature-level leakage.

---

## 7. Model: CatBoost (with rationale)

Why CatBoost as the headline model:

- **Native categorical support.** Ordered target statistics on high-cardinality columns (`street_name`, `issuer_code`) without a preprocessing pipeline. Strictly better than manual ordinal encoding for thousands-of-unique-values columns.
- **GPU acceleration.** RTX 5070 Ti (Blackwell sm_120, CUDA 13.1), reduces training from ~45 minutes (AutoGluon CPU stacking) to ~5–10 minutes.
- **Native NaN handling.** No imputer needed for the main model. Avoids imputation-induced bias on sparse pvqr columns.
- **Class imbalance.** `auto_class_weights='Balanced'` handles the ~18% win rate without manual weight tuning. Note: this inflates raw predicted probabilities — calibration is required for literal probability claims (§9).
- **Production-friendliness.** A single `.cbm` file is the only artifact at inference time. No encoder pipeline chain to serialize.

Alternatives considered:

| Model | AUC | Why not |
|---|---:|---|
| AutoGluon `best_quality` preset | 0.8578 | ~45-min training, 2GB+ artifact, neural-net stacking dominates time |
| XGBoost | 0.8743 | No native cats; requires ordinal encoding; runs as ensemble member instead |
| LightGBM | 0.8716 | GPU support requires separately compiled binary; runs as ensemble member |
| Logistic Regression | ~0.74–0.76 | Retained as interpretable baseline |

---

## 8. Training & validation

- **Stratified 80/20 train/test split**, but for FineHero the headline split is **chronological** (last 20% by `issue_date` is the test set). See `src/train.py:140`. This produces honest time-aware AUC, not the inflated random-split number.
- **Eval slice**: chronological last 10% of training, used for early stopping during Optuna and the final fit.
- **Test set touched once**, in `evaluate.py`. Saved to `data/test_set.joblib` for downstream calibration and threshold tuning.
- `use_best_model=True` — checkpoint with highest eval AUC wins.
- **Optuna**: TPE sampler, 80 trials × 800-iter cap (each), 100-round early stopping. Search space includes `depth`, `learning_rate`, `l2_leaf_reg`, `border_count`, `bagging_temperature`, `leaf_estimation_iterations`, `grow_policy`. `best_params.joblib` is consumed by the final 5000-iter run with 100-round early stopping.

Default best params (after Optuna):

```python
{
    "depth": 6,
    "learning_rate": 0.1267,
    "l2_leaf_reg": 7.378,
    "border_count": 194,
    "bagging_temperature": 0.022,
    "leaf_estimation_iterations": 8,
    "grow_policy": "SymmetricTree",
}
```

To run Optuna off and use sane defaults (~5-min training), set `USE_OPTUNA = False` in `train.py`.

---

## 9. Calibration and threshold

### Calibration

Raw CatBoost `predict_proba` is **not** well-calibrated. `auto_class_weights='Balanced'` reweights the loss function, which inflates predicted probabilities. Empirically on the 200k held-out test set:

| Bin | Count | Raw mean | Empirical |
|---|---:|---:|---:|
| [0.30, 0.40) | 17,696 | 35.0% | 13.6% |
| [0.50, 0.60) | 14,536 | 54.6% | 21.8% |
| [0.70, 0.85) | 22,471 | 77.8% | 43.5% |
| [0.85, 1.00) | 16,200 | 92.1% | 73.4% |

Raw ECE: 0.1519.

`models/isotonic_calibrator.joblib` is fit on the chronological last 10% of training (n=80,000). Applying it brings calibration substantially closer to the diagonal:

| Bin | Calibrated mean | Empirical |
|---|---:|---:|
| [0.30, 0.40) raw | 13.1% | 13.6% |
| [0.50, 0.60) raw | 29.0% | 21.8% |
| [0.70, 0.85) raw | 57.2% | 43.5% |
| [0.85, 1.00) raw | 86.9% | 73.4% |

Calibrated ECE: **0.0234**. Brier: 0.137 raw → 0.101 calibrated. AUC preserved (isotonic is monotone).

`predict.py` and `predict_ensemble.py` apply the calibrator automatically via `_load_calibrator()`. The output prints both raw and calibrated probabilities for transparency, and uses calibrated for the verdict.

**Calibration drift across the test set:** the chronological tail's newest 10% has calibrated ECE of **0.052**, vs 0.023 across the full test. The calibrator has a shelf life. Refit periodically as new training data lands.

### Threshold

`models/dispute_threshold.json` holds four tuned policies, all chosen against the calibrated probability average:

| Policy | Threshold | F1 | Lift over base rate |
|---|---:|---:|---:|
| **f1** (default) | 0.32 | 0.579 | 2.81× |
| youden | 0.17 | 0.537 | 2.17× |
| accuracy | 0.555 | 0.543 | 3.51× |
| match_baserate | 0.40 | 0.580 | 2.88× |

Default is `f1`. `predict._load_threshold()` reads the JSON automatically. The legacy hardcoded `0.40` constant remains in code as a fallback if the JSON is missing.

Out-of-sample threshold tuning (chronological half-and-half within the test set) showed an overfit gap of +0.012 F1 — the existing thresholds are nearly indistinguishable from re-tuned ones. The threshold is robust; the calibrator is the part that drifts.

---

## 10. Honest evaluation

Three numbers worth tracking for any reporting:

1. **Time-aware AUC on the 200k chronological test set.** This is the headline. **0.8694** for CatBoost alone, **0.8742** for XGBoost. The blends top out around 0.874.
2. **Calibrated test ECE.** 0.0234. Treat as the in-distribution upper bound on what's achievable; live data will be higher.
3. **Live-API F1.** 0.564 with the rolling-prior fallback shipped in this codebase. This is the operational metric — what the model actually delivers when serving fresh tickets.

**Do not quote random-split numbers as the result.** Random splits run 0.01–0.05 higher than time-aware because plates appear in both folds.

Selective-labels caveat: training is OATH-hearing tickets only (~30% of all issued). Reported metrics describe that population. Tickets paid without contest, written off, or never issued are unobserved.

---

## 11. The rolling-prior gotcha (production-critical)

This section describes the dominant production failure mode discovered in May 2026. It's the reason the model can score live tickets correctly today; without the patch documented here, it doesn't.

### 11.1 The bug

Of the 95 model features, 27 are **rolling priors** — windowed aggregates over plate / precinct / issuer history (see §5.9). `engineer.py` computes them by running tallies as it walks the training corpus chronologically. They're cheap to compute in batch but require state that doesn't exist at inference time.

`predict.py` and `predict_ensemble.py` have no inference-time mechanism to compute them. The cumulative `plate_history_map.joblib` covers `plate_prior_win_rate` and `plate_prior_ticket_count` (the non-windowed versions) but provides nothing for the 30D / 90D / 365D windowed columns. So at inference, all 27 rolling-prior features come back as `NaN`.

The model was trained on data where rolling priors were always populated. The rare cases where they were `NaN` (a plate's first-ever ticket, a precinct with no recent activity) correlated with weak tickets. So the model learned, accurately within the training distribution: **"rolling priors NaN ⇒ losing ticket"** — and weighted that signal heavily.

At inference, **every** new ticket has all 27 rolling priors as NaN. The model applies the strong "NaN = loss" signal it learned, and predictions collapse toward zero.

### 11.2 Evidence

**SHAP attribution on a known winning ticket** (summons 9126107983, Brooklyn meter violation, plate LHP7230, actual outcome `HEARING HELD-NOT GUILTY`):

| Feature family | Total log-odds contribution |
|---|---:|
| **Rolling priors (27 features)** | **−3.687** |
| Core categoricals (14) | +0.538 |
| Plate-level non-rolling (4) | +0.155 |
| Temporal (11) | +0.044 |
| pvqr-only (9) | 0.000 |
| Missing flags (10) | 0.000 |
| Keyword flags (6) | 0.000 |
| Issuer-level non-rolling (6) | −0.071 |

Top single contributor: `plate_prior_win_rate_30D = NaN` at −0.921 log-odds. The bias term (population-average log-odds) is −0.182, equivalent to ~45% raw probability. Rolling-priors-NaN alone takes that to 3% raw, 1% calibrated.

**Distribution-level evidence on a 998-row live-API batch** (training-matching status filter, deduped against training summons):

| Metric | Without fix | With fix |
|---|---:|---:|
| AUC | 0.9056 | 0.9006 |
| ECE | 0.182 | 0.092 |
| Median calibrated probability | 0.08% | 10.2% |
| F1 | 0.000 | **0.564** |
| Precision | 0% | 58.6% |
| Recall (wins caught) | 0/188 | **102/188** |

AUC barely moves. The bug is approximately a constant additive shift in log-odds, so *ranking* is preserved (AUC = 0.90 throughout). The bug destroys *absolute* probabilities and therefore threshold-based decisions.

**Spot checks on four real tickets with known outcomes:**

| Ticket | Description | Outcome | Without fix | With fix |
|---|---|---|---|---|
| 9126107983 | BK meter, $35 | WON | 1.1% PAY ❌ | 48.9% DISPUTE ✓ |
| 9070812812 | BK meter, $35 | WON | 1.1% PAY ❌ | 36.4% DISPUTE ✓ |
| 9109263937 | QN no-standing, $115 | LOST | 1.1% PAY ✓ | 28.7% PAY ✓ |
| 9093590236 | MN expired-insp, $65 | LOST | 0.2% PAY ✓ | 7.6% PAY ✓ |

Without the fix: 2/4. The two correct verdicts are right-by-accident — the constant ~1% just happens to fall below threshold for everything, so all losers are correctly classified PAY by side effect.

With the fix: 4/4, with sensible gradation:
- Brooklyn meter winners → ~36–49% (above threshold)
- Queens no-standing loser → ~29% (close to threshold, borderline)
- Manhattan expired-inspection loser → ~8% (far below threshold, strongest loss)

### 11.3 The shipped fix

A static-mean fallback at inference. Three components:

**`src/build_rolling_prior_means.py`** — one-shot artifact builder. Scans `features.csv`, computes population means and medians for the 27 rolling-prior columns, defensible defaults for the three `days_since_*` columns, and the training base rate (used as the principled `issuer_bayes_default` for unseen issuers — `engineer.py` would assign exactly this for a `count=0` issuer; the ticket-weighted mean of `issuer_bayes_rate` is biased by high-volume issuers and produces a verdict regression on borderline winners). Output: `models/rolling_prior_means.joblib`.

**`predict._load_rolling_prior_means()`** — loads the artifact, returns `None` if missing.

**`predict._apply_rolling_prior_fallback(df, fallback)`** — fills NaN cells in the 27 rolling-prior columns + 3 `days_since_*` + `issuer_bayes_rate` with their stored defaults. Operates in-place, returns a count for diagnostic logging.

Wired into both `predict.py` and `predict_ensemble.py`. The latter applies the fallback to all three model rows (CatBoost, LGB, XGB) since they all share the same dependency.

Run after every training-data refresh:

```bash
python -m src.build_rolling_prior_means
```

The fallback is a stand-in, not a complete fix. ECE on live data is 0.092 (vs 0.023 internal). Upper calibration bins systematically under-predict because population means don't match ticket-specific rolling priors. For ranking and threshold decisions the model is now usable; for absolute probability claims further work is owed.

### 11.4 What didn't work

A MaskTab-style retrain was attempted in `src/train_masked.py`. The idea: randomly NaN out the rolling-prior block on 50% of training rows, forcing the model to learn a viable inference path when they're absent. Mask the eval slice the same way for early-stopping consistency. Leave the test slice unmasked so reported AUC stays comparable.

Internal results looked good:

| Metric | Original | Masked (unmasked test) | Masked (fully-masked test) |
|---|---:|---:|---:|
| Test AUC | 0.8694 | 0.8699 | 0.8662 |
| Test ECE (calibrated) | 0.0234 | **0.0191** | **0.0197** |

The masked model had **better** internal calibration than the original, and AUC was preserved within 0.0036 even when scored with all rolling priors masked.

**But on the 998-row live batch:**

| Metric | Original + fallback (shipped) | Masked retrain (no fallback) |
|---|---:|---:|
| AUC | 0.9006 | 0.8764 |
| ECE | 0.0923 | 0.1314 |
| F1 | 0.564 | 0.061 |
| Recall | 54.3% | 3.2% |

The masked retrain underperformed the stand-in fallback on every operational metric. F1 0.061 vs 0.564 — only 6 of 188 winners caught.

Three plausible reasons, none yet validated by follow-up experiments:

1. **Mask probability 0.5 was too low.** Half of training rows still had real rolling priors. The model could still lean on them for that half and apparently never properly learned the no-rolling-priors regime.
2. **Hyperparameters were reused from the unmasked Optuna run.** Under masking, the optimal depth, learning rate, and L2 are likely different.
3. **Calibrator-distribution mismatch.** The calibrator was fit on a 50%-masked eval slice. Live data is 100% masked. The raw-probability distributions don't match.

Masked artifacts (`*_masked.*`) are kept as diagnostic for future MaskTab attempts but are not promoted. Production continues to use the original model + stand-in fallback.

### 11.5 What's still owed

Three avenues for closing the live-vs-internal calibration gap (still 0.092 vs 0.023):

1. **Drop rolling priors entirely + retrain.** Equivalent to mask_prob=1.0. Cleanest variant. ~30 minutes of code change plus a training run.
2. **Higher mask probability + re-tune Optuna under masking.** Mask at 0.8–0.9 and let Optuna pick params suited to the masked regime. ~1–2 hours including search.
3. **Compute rolling priors at inference.** Build `(plate, date)`, `(precinct, date)`, `(issuer, date)` lookup tables from the full corpus, save as joblib, query at inference. Most faithful to the trained model but adds multi-GB state. ~2 hours plus storage cost.

A reasonable order of operations for a future engineer: try (1) first because it's the simplest test of "the model just shouldn't see these features." If it matches or beats the stand-in, ship it. If not, escalate to (3).

The full chronological narrative — including how the bug was discovered, the dead-ends, the iteration on `issuer_bayes_default`, and the lessons — lives in [`docs/rolling_prior_investigation.md`](docs/rolling_prior_investigation.md).

---

## 12. The appeal letter system (`finehero-appeal/`)

A separate module that takes ticket data and user-submitted evidence and produces a professionally formatted PDF appeal letter.

Pipeline:

1. **Evidence analysis.** Photos base64'd to `claude-sonnet-4-6` vision API; PDFs converted to page images via PyMuPDF and analyzed the same way; text documents extracted directly.
2. **Strategy selection.** Structured-output call to `claude-sonnet-4-6` picks from 15+ NYC-specific dispute grounds: sign not posted/obscured, meter malfunction, vehicle not present, ticket outside hours-in-effect, incorrect plate/make data on ticket, officer not authorized for this violation type, first-time offender, etc.
3. **Letter generation.** Full appeal letter from `claude-sonnet-4-6` incorporating ticket data + evidence analysis + selected legal grounds.
4. **PDF assembly.** ReportLab combines a styled cover, body letter, and exhibit pages.

Estimated cost: $0.08–0.15 per letter (vision calls dominate; strategy + letter generation ~$0.02–0.04 combined).

**The appeal letters are drafts, not legal advice.** Legal LLMs hallucinate at measured rates between 17% (commercial RAG tools like Lexis+ AI) and 58–88% (general-purpose models on verifiable legal queries). The generator is a drafting aid. User-facing copy must say so, and users must verify every cited regulation, sign code, and factual claim before submission.

---

## 13. Operational ground rules

These are the invariants the codebase enforces. Anyone changing the model or the predict scripts should keep them.

1. **Honest AUC = time-aware AUC.** Headline 0.8694, time-aware. Random-split numbers are not the result.
2. **No future-information features.** As-of priors only; new aggregates need a test in `tests/test_asof_priors.py`.
3. **Selective-labels caveat.** Training is OATH-hearing tickets only (~30% of issued). Metrics describe that population.
4. **Fairness is not solved.** `precinct`, `violation_code × precinct`, `issuing_agency` encode enforcement bias. Don't claim fairness without naming a criterion.
5. **Appeal letters are drafts, not legal advice.** User-facing copy must say so.
6. **Calibration and threshold both auto-applied.** `predict.py` and `predict_ensemble.py` load `models/isotonic_calibrator.joblib` (ECE 0.152→0.023 on the 200k holdout) and `models/dispute_threshold.json` (default policy `f1` → 0.32). New scripts that score tickets should follow the same pattern via `predict._load_calibrator()` and `predict._load_threshold()`. Raw probabilities are reported alongside calibrated ones for transparency but are not used for the verdict. See §9.
7. **Rolling-prior fallback shipped; canonical model unchanged.** The 27 rolling-prior features (`plate_prior_*_30D/90D/365D`, `precinct_prior_*_*`, `issuer_prior_*_*`) are time-dependent windowed aggregates `engineer.py` builds via running tallies during training but `predict.py` has no inference-time equivalent for. SHAP on a confirmed winning ticket showed they contribute **−3.7 log-odds** alone — the entire prediction collapse. Patched at inference: `predict.py` and `predict_ensemble.py` load `models/rolling_prior_means.joblib` (built by `python -m src.build_rolling_prior_means`) and fill NaN rolling priors with training-set population means before scoring. On the 998-row live batch this lifts F1 from 0.0 to 0.564, recall from 0% to 54%, ECE from 0.18 to 0.09. A MaskTab-style retrain at mask_prob=0.5 was attempted but underperformed the stand-in. The stand-in remains the best-validated production fix. See §11.

---

## 14. Limitations (operational caveats)

These should be honored when reporting results to users or making product claims.

### Selective labels

The training set is restricted to tickets that went to an OATH hearing. Reported metrics (AUC 0.87, accuracy 78.9%) describe performance on that self-selected population, not on all issued tickets. Tickets paid without contest, written off, or never issued are unobserved. Outcome comparisons between the model and human adjudicators on this data are therefore systematically biased (Lakkaraju et al. 2017).

### Time-aware vs random-split AUC

Random splits routinely overstate AUC by 0.01–0.05 on dispute-style problems with per-plate history features, because a plate can appear in both train and test (Bergmeir & Benítez 2012; Kaufman et al. 2012). The honest, time-aware chronological AUC is what gets cited. The leakage audit confirmed random-split and time-aware AUCs agree within 0.0005 on a 200k subsample, so the gap at full training capacity reflects temporal concept drift, not feature-level label leakage.

### Historical enforcement bias

High-signal features — `precinct`, `violation_code × precinct`, `issuing_agency`, `license_type × state` — encode decades of discretionary enforcement. Streetsblog's precinct-level analyses document that small numbers of precincts account for disproportionate shares of certain ticket types, and that placard enforcement is near-zero in some areas. "Police data measures enforcement, not ground-truth behavior" (Lum & Isaac 2016; Pierson et al. 2020). Accuracy on this data is not the same as fairness across neighborhoods or demographics.

### Fairness trade-off is unavoidable

Calibration, equal false-positive rates, and equal false-negative rates cannot be simultaneously satisfied across groups when base rates differ (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017). FineHero makes no explicit fairness choice today. Any future claim of "fairness" must name the criterion being optimized and accept the ones being violated (Hardt, Price & Srebro 2016).

### Modeling choice is defensible, not optimal by default

CatBoost was chosen over deep tabular methods because tree-based gradient boosting remains state-of-the-art on tabular data at this sample size (Grinsztajn, Oyallon & Varoquaux 2022), and because ordered boosting mitigates target leakage in high-cardinality categoricals (Prokhorenkova et al. 2018). This does not imply the 95-feature set or the current hyperparameter search is optimal. Newer tabular foundation models (TabPFN-2.5) outperform tuned tree-based models on benchmarks up to ~50k rows; for FineHero's 1M-row corpus they're an open question worth investigating.

### LLM appeal letters require human verification

The appeal generator uses the Anthropic API to draft dispute letters. Legal LLMs hallucinate at measured rates between 17% (commercial RAG) and 58–88% (general-purpose) on verifiable legal queries (Dahl et al. 2024; Magesh et al. 2025). The generator is a drafting aid, not legal advice. User-facing copy must state this; users must verify every citation before submission.

### Live-data drift

See §11 for the full story. Key facts:
- Held-out test ECE: 0.023.
- Live-API ECE (with the shipped fallback): 0.092.
- Live-API ECE (without the fallback): 0.18.
- Median live calibrated probability without fallback: 0.1%, against an 18.8% empirical win rate.
- Cause: rolling priors not computable at inference. Stand-in fallback is shipped; proper fix (drop features and retrain, or compute rolling priors at inference) is open work.

### Calibration drift across the test set

The newest 10% of the test set has calibrated ECE of 0.052, vs 0.023 across the full test. The calibrator has a shelf life. Refit periodically as new training data lands.

---

## 15. Tech stack

- Python 3.11+
- CatBoost 1.2+ (gradient boosting, native categorical support, GPU)
- LightGBM, XGBoost (ensemble members)
- Optuna 3.5+ (TPE sampler, 80 trials)
- pandas 2.0+, NumPy 1.26+
- scikit-learn 1.4+ (LogisticRegression baseline, IsotonicRegression, OrdinalEncoder, StratifiedShuffleSplit)
- joblib (model serialization)
- requests (HTTP for NYC Open Data + Open-Meteo)
- PyTorch 2.12 nightly (GPU detection only)
- Hardware: NVIDIA RTX 5070 Ti, Blackwell sm_120, CUDA 13.1

Appeal module adds:
- anthropic (Claude API)
- ReportLab (PDF assembly)
- PyMuPDF / fitz (PDF parsing)
- Pillow (image processing)

---

## 16. When changing the model

A short checklist for anyone modifying the training pipeline:

1. **Update `docs/auc_progress.md`** with the new honest (time-aware) AUC and what changed.
2. **If you add a feature**, add or update the leakage probe in `src/audit_leakage.py` and confirm random-vs-time gap stays small. If the feature is an aggregate, add an as-of test in `tests/test_asof_priors.py`.
3. **If the new feature requires inference-time state** (anything time-windowed, lookup-keyed, etc.), build the inference-time path *immediately*. Don't defer it. The rolling-prior bug existed because the feature was easy to compute in batch and hard to compute at inference, and the inference-time path was deferred. Don't repeat that.
4. **If you change the threshold or calibration**, refit by running `python -m src.build_calibrator_and_threshold`. This loads the trained CatBoost, scores the chronological eval slice, fits a new isotonic regressor, and tunes the four threshold policies. Writes `models/isotonic_calibrator.joblib` and `models/dispute_threshold.json` to the canonical paths. Update `LIMITATIONS.md` §"Probability threshold" and §9 of this file with the new ECE/Brier numbers. (Note: the currently-shipped artifact was historically fit on ensemble `prob_avg`, not single CatBoost; the new builder operates on the single-CatBoost path. If you need an ensemble-scope calibrator, extend the builder accordingly.)
5. **If you retrain**, also rebuild the rolling-prior means: `python -m src.build_rolling_prior_means`. Verify on the live batch via `python test_batch_with_fix.py` — F1 should be ≥ 0.5 if the fallback is functional.
6. **If you alter `predict.py`** or `predict_ensemble.py`, run the four spot-check tickets (9126107983, 9070812812 — both winners; 9109263937, 9093590236 — both losers) to confirm verdicts match. The patched pipeline should produce 4/4 correct.

---

## 17. Reproducing key validation tests

The investigation that produced this codebase left several runnable scripts at the repo root. Each is self-contained and reproduces a specific piece of evidence:

```bash
# Hand-crafted synthetic tickets — the original probe
python test_hypothetical_tickets.py

# Full 200k held-out scoring + raw vs calibrated comparison
python test_holdout_anchor.py
python test_calibrated_comparison.py

# Out-of-sample threshold tuning (chronological half-and-half)
python test_threshold_oos.py

# Live API single-ticket scoring (4 spot-check tickets)
python test_live_nyc_tickets.py

# Live API 998-row batch with the rolling-prior fallback applied
python test_batch_with_fix.py

# Three-way comparison: broken / shipped fix / masked retrain
python test_masked_vs_others.py
```

Numbers will drift slightly between live-API runs because NYC Open Data updates daily, but qualitative findings should reproduce.

---

## 18. Documentation map

For specific topics, refer to:

| Topic | Document |
|---|---|
| Project summary + ground rules + Claude-agent instructions | `CLAUDE.md` |
| Operational caveats with numbers | `LIMITATIONS.md` |
| Full chronological investigation narrative (~620 lines) | `docs/rolling_prior_investigation.md` |
| AUC scoreboard (running history of changes) | `docs/auc_progress.md` |
| Leakage audit results | `docs/leakage_audit.md` |
| Cited references | `docs/literature_synthesis.md` |
| Academic-style methodology writeup | `FineHero_Methodology.docx` |
| **Everything in one place (this document)** | `PROJECT.md` |

When the source documents disagree, this file is the more recent canonical reference.

---

*Last consolidated: May 2026. Reflects the state of the codebase after the rolling-prior investigation and shipped fix.*
