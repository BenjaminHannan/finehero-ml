# FineHero ML

NYC parking-ticket dispute-outcome predictor. Given a ticket + plate history, predicts the probability the contestant wins at an OATH hearing. Companion `finehero-appeal/` module drafts dispute letters via the Anthropic API.

Full methodology is in `FineHero_Methodology.docx` — this file is the working summary.

## Layout

- `pipeline.py` — orchestrates fetch → engineer → train → evaluate (CatBoost is the headline model).
- `pipeline_ensemble.py` / `predict_ensemble.py` — LGB + XGB + CatBoost ensemble path.
- `pipeline_ui.py` — interactive UI wrapper.
- `predict.py` — single-ticket inference using the saved CatBoost model.
- `src/`
  - `fetch_data.py` — pulls from NYC Open Data (synthetic fallback if offline).
  - `engineer.py` — feature engineering. **As-of priors only** — any per-plate / per-precinct aggregate must use strictly past data (see `tests/test_asof_priors.py`).
  - `train.py`, `train_lgb.py`, `train_xgb.py` — per-model trainers.
  - `ensemble.py` — stacking / averaging.
  - `evaluate.py` — metrics + saves model metadata.
  - `audit_leakage.py` — leakage probes (random vs. time-aware split, target shuffle, plate-blocked CV).
  - `cleanlab_scan.py` — label-noise scan.
- `finehero-appeal/` — separate appeal-letter service. Has its own `requirements.txt`.
- `docs/` — `auc_progress.md` (running scoreboard), `leakage_audit.md`, `literature_synthesis.md`.
- `LIMITATIONS.md` — required reading before changing the model or making claims about it.
- `tests/` — pytest. Currently covers as-of prior correctness.

## Data sources

Two NYC Open Data Socrata endpoints, free, no API key:

- **`nc67-uf89`** — Open Parking & Camera Violations. Provides outcomes (dismissed, paid, guilty, reduction granted, hearing held). This is the label source.
- **`pvqr-7yc4`** — Parking Violations Issued. Provides rich pre-hearing metadata (issuing officer, street, vehicle, hours-in-effect, feet from curb, legal code). Covers ~1 fiscal year, so only ~30–50% of outcome rows match — fetched at 2× target row count to maximize join coverage.

Both fetched in 50k-row pages, cached to `outcomes_raw.csv` / `pvqr_raw.csv`, joined on `summons_number` (left join with outcomes as base) into `violations_raw.csv`. Rows with `violation_status` of `OUTSTANDING` or `IN PROCESS` are dropped — no outcome yet.

Weather is from **Open-Meteo Historical Archive** (free, no key, rate-limited). Cached per borough.

## Feature engineering

All features must be **strictly pre-hearing** — information available at issue time. ~40 features across:

- **Temporal** — `hour_of_offense` (parsed from NYC's non-standard `0156P` format), `day_of_week`, `month`, `is_weekend`, `is_holiday`, `days_since_ticket`.
- **Financial** — `fine_amount`. Post-hearing financials (`penalty_amount`, `reduction_amount`, `payment_amount`, `amount_due`) are **dropped at source** — never feature-engineered.
- **Weather** — `precipitation`, `visibility`, `wind_speed`, `weather_code`, `is_bad_weather` (rain >2mm, fog 45/48, snow ≥71). Archive data, not forecasts.
- **Keyword flags** from `violation_description`: `kw_meter`, `kw_hydrant`, `kw_bus_stop`, `kw_sign`, `kw_blocking`, `kw_expired`.
- **pvqr-only rich features** (only set when join hits): `within_posted_hours` (one of the strongest dismissal signals — handles overnight ranges like 10PM–6AM), `feet_from_curb`, `house_number`, `vehicle_year`, `from_hour`, `to_hour`.
- **Categoricals** passed as raw strings to CatBoost's `cat_features` (no manual ordinal encoding): `violation_code`, `precinct`, `county`, `issuing_agency`, `license_type`, `state`, `issuer_code`, `street_name`, `vehicle_make/color/body_type`, `violation_legal_code`, `law_section`, `sub_division`.
- **Cross-features** (string-concatenated, also categorical): `viol_x_precinct`, `viol_x_license`, `hour_x_dow`.
- **Plate history** — leave-one-out chronological encoding. `plate_prior_ticket_count` and `plate_prior_win_rate` (smoothed toward global mean with `PLATE_SMOOTH_K=10`). Only strictly prior tickets count.

## Leakage prevention

Treat this as the most important property of the model. Mechanisms:

1. Post-hearing financial columns dropped at source.
2. `OUTSTANDING` / `IN PROCESS` rows filtered out.
3. Plate history is **LOO chronological** — current ticket's outcome is never in its own features.
4. Train/test split happens **before** any target encoding — no test-set outcomes leak into training stats.
5. Weather is archive (past), not forecast.
6. CatBoost's ordered target statistics handle within-fold encoding leakage automatically.

If you add a new aggregate feature: add an as-of test in `tests/test_asof_priors.py` and confirm the random-vs-time AUC gap stays small via `src/audit_leakage.py`.

## Model: CatBoost (with rationale)

- **Native categorical support** — ordered target statistics on high-cardinality cols (`street_name`, `issuer_code`) without a preprocessing pipeline.
- **GPU acceleration** — RTX 5070 Ti (Blackwell sm_120, CUDA 13.1), ~5–10 min training vs ~45 min for AutoGluon CPU.
- **Native NaN handling** — no imputer needed; avoids imputation bias on sparse pvqr columns.
- **Class imbalance** — `auto_class_weights='Balanced'`.
- **Single artifact** — one `.cbm` file at inference, no encoder chain to serialize.

Alternatives considered: AutoGluon `best_quality` (AUC 0.8578, ~45 min, 2GB+, rejected); XGBoost (no native cats); LightGBM (no GPU in pip wheel); Logistic Regression kept as baseline (~0.74–0.76 AUC).

## Training & validation

- Stratified 80/20 train/test split (`StratifiedShuffleSplit`, seed=42).
- A further 10% of train carved off as eval slice (seed=7) for early stopping. Used during Optuna and final fit, never in test.
- Test set touched **once** in `evaluate.py`.
- `use_best_model=True` — checkpoint with highest eval AUC wins.
- **Optuna** TPE sampler, 20 trials, 800-iter cap with 100-round early stopping. Search space: `depth` ∈ [6,10], `learning_rate` log-uniform [0.02,0.12], `l2_leaf_reg` log-uniform [1,10], `border_count` ∈ [128,254]. `best_params.joblib` is consumed by the final 5000-iter run. Toggle off via `USE_OPTUNA = False` for ~5 min sane defaults (depth=8, lr=0.05, l2=3.0).

## Honest evaluation

- Stratified random split is the methodology default but **not the headline number**.
- Headline = **chronological holdout** (last 20% by `issue_date`). Current honest AUC: **0.8662** (Optuna-tuned CatBoost). Random-split numbers can be 0.01–0.05 higher because plates appear in both folds — never quote them as the result.
- Metadata (AUC, accuracy, feature list, timestamp, row count) is written to `models/metadata.json` after every `evaluate.py` run; `predict.py` consumes it.

## Appeal letter system (`finehero-appeal/`)

Pipeline:

1. **Evidence analysis** — photos base64'd to `claude-sonnet-4-6` vision; PDFs converted page-by-page via PyMuPDF; text docs read directly.
2. **Strategy selection** — structured-output call picks from 15+ NYC-specific dispute grounds (sign not posted/obscured, meter malfunction, vehicle not present, ticket outside hours-in-effect, incorrect plate/make data, officer not authorized, first-time offender).
3. **Letter generation** — full letter from `claude-sonnet-4-6` incorporating ticket data + evidence + grounds.
4. **PDF assembly** — ReportLab combines styled cover, body letter, and exhibit pages.

Estimated cost: $0.08–0.15/letter (vision calls dominate; strategy + letter generation ~$0.02–0.04 combined).

## Ground rules

1. **Honest AUC = time-aware AUC.** Headline 0.8662, time-aware. Random-split numbers are not the result.
2. **No future-information features.** As-of priors only; new aggregates need a test.
3. **Selective-labels caveat.** Training is OATH-hearing tickets only (~30% of issued). Metrics describe that population.
4. **Fairness is not solved.** `precinct`, `violation_code × precinct`, `issuing_agency` encode enforcement bias. Don't claim fairness without naming a criterion.
5. **Appeal letters are drafts, not legal advice.** User-facing copy must say so.
6. **Calibration and threshold both auto-applied.** `predict.py` and `predict_ensemble.py` load `models/isotonic_calibrator.joblib` (ECE 0.152→0.023 on the 200k holdout) and `models/dispute_threshold.json` (default policy `f1` → 0.32). New scripts that score tickets should follow the same pattern via `predict._load_calibrator()` and `predict._load_threshold()`. Raw probabilities are reported alongside calibrated ones for transparency but are not used for the verdict. See LIMITATIONS.md §"Probability threshold".

7. **Rolling-prior fallback shipped; canonical model unchanged.** The 27 rolling-prior features (`plate_prior_*_30D/90D/365D`, `precinct_prior_*_*`, `issuer_prior_*_*`) are time-dependent windowed aggregates `engineer.py` builds via running tallies during training but `predict.py` has no inference-time equivalent for. SHAP on a confirmed winning ticket showed they contribute **−3.7 log-odds** alone — the entire prediction collapse. **Patched at inference**: `predict.py` and `predict_ensemble.py` load `models/rolling_prior_means.joblib` (built by `python -m src.build_rolling_prior_means`) and fill NaN rolling priors with training-set population means before scoring. On the 998-row live batch this lifts F1 from 0.0 to **0.564**, recall from 0% to 54%, ECE from 0.18 to 0.09. A MaskTab-style retrain at mask_prob=0.5 (`src/train_masked.py`, artifacts have `_masked` suffix) was attempted as a "proper" fix but **underperformed the stand-in on live data** (F1 0.061, recall 3.2%). The stand-in remains the best-validated production fix; alternative retrain strategies (drop features, higher mask prob, full Optuna re-tune under masking) are listed in LIMITATIONS.md §"Live-data drift" but not yet validated.

## Tech stack

Python 3.11+, CatBoost 1.2+, Optuna 3.5+, pandas 2.0+, NumPy 1.26+, scikit-learn 1.4+ (baseline LR), joblib, requests, PyTorch 2.12 nightly (GPU detection only). Appeal module adds `anthropic`, ReportLab, PyMuPDF (`fitz`), Pillow.

## Running

```
pip install -r requirements.txt
python pipeline.py            # interactive row-count prompt
python predict.py             # single-ticket inference
pytest tests/                 # as-of prior tests
```

`finehero-appeal/` has its own deps and runs separately.

## When changing the model

- Update `docs/auc_progress.md` with the new honest (time-aware) AUC and what changed.
- If you add a feature, add or update the leakage probe in `src/audit_leakage.py` and confirm random-vs-time gap stays small.
- If you change the threshold or calibration, update LIMITATIONS.md §"Probability threshold".
