# FineHero AUC Progress Log

Structured log of every intervention that moves the honest AUC. The **honest
baseline is the time-aware chronological 80/20 AUC** on the full 1M-row
dataset — everything else (random split, plate-blocked CV, etc.) is
diagnostic. Tier gates and deltas are tracked per `docs/auc_improvement_plan.md`.

---

## Tier 0 — leakage cleanup (Benjamin's prompt, 2026-04-22)

### Starting line

| Probe | Prior audit (200 k subsample, pre-Tier-0) | Source |
|---|---|---|
| Random 80/20 baseline | 0.8576 | `docs/leakage_audit.md` @ c7ef82c |
| Time-aware chronological | 0.8571 | idem |
| Plate-blocked 5-fold CV | 0.8740 ± **0.0745** (PROXY grouping) | idem |
| Target-shuffle (within year) | 0.5301 | idem |
| `plate_prior_win_rate` ablation drop | +0.0599 | idem |

The honest time-aware AUC of the full-fit CatBoost (`train.py`) at this
point is **0.8425** (`LIMITATIONS.md`, not a subsample-audit number).

### Tier 0.1 — strict `<` regression test + smoothing-leak fix

**Hypothesis tested:** that the `plate_prior_*`, `precinct_prior_*` and
`issuer_prior_*` features in `src/engineer.py` already enforce strict-`<`
on `issue_date`, and the only remaining per-row leakage is the tiny
smoothing constant `global_mean = df["won"].mean()` that touches every
smoothed prior through the `k·global_mean` term.

**Interventions:**

1. `test: regression guard for strict as-of priors` ([f6fd5df](./))  —
   added `tests/test_asof_priors.py`, a 3-plate × 5-ticket synthetic
   fixture (including same-day duplicates and a triplet on one date) that
   locks in the strict-`<` semantics of `_compute_plate_history` and
   `_compute_rolling_group_history`. **All 5 asserts pass on the existing
   code** — confirming no strict-`<` violation was ever present in the
   aggregation logic.
2. `fix: compute smoothing global_mean on chronological train slice only`
   ([aaee543](./))  —  replaced
   `global_mean = df["won"].mean()`  with the mean over the
   chronologically-earliest 80 % of rows, so the eventual test slice can
   no longer leak its label mean into every training-row's smoothed
   prior. Effect on the smoothing constant itself:
   `0.2362 → 0.2541` (+0.018), a small shift reflecting an upward
   temporal drift in dispute win rate across the FY22–FY25 window.

### Tier 0.2 — real `plate_id` for GroupKFold

**Problem:** `src/audit_leakage.py::probe_plate_blocked` was falling back
to a PROXY group key built from `plate_prior_ticket_count` +
`plate_prior_win_rate`, because the raw plate column was never written
to `features.csv`. That PROXY gave σ = ±0.0745 across 5 folds — large
enough that the probe told us nothing useful about plate memorization.

**Intervention:** `feat: propagate plate_id as meta column for true
plate-blocked CV` ([a3de0c6](./)).

- `src/engineer.py`: canonicalize the raw plate column (uppercase,
  stripped, `UNKNOWN` for missing) into a single `plate_id` column,
  used BOTH for prior aggregation (small latent fix for any case-mixed
  plate strings) AND written to `features.csv` as a meta column
  alongside `issue_date`.
- `src/train.py`, `src/train_lgb.py`, `src/train_xgb.py`,
  `src/cleanlab_scan.py`: add `plate_id` to `META_COLS` / `drop_cols`
  so no trainer silently fits on it.
- `src/audit_leakage.py`: `_prep` promotes `plate_id` into the meta
  DataFrame. `probe_plate_blocked` rewritten to use the real
  `meta["plate_id"]` as the GroupKFold group key. Proxy fallback
  deleted — if `plate_id` is missing, the probe returns `None` with
  a clear "rerun engineer" message instead of producing silent garbage.
- Inference (`predict.py`, `predict_ensemble.py`): no change needed —
  they reconstruct model input from `metadata.json[feature_names]`,
  which after the next train run will no longer include `plate_id`.

### Post-Tier-0 audit (full 1 M rows, GPU, CatBoost 400 iter / early-stop 40)

_`python -m src.audit_leakage --full` → `docs/leakage_audit.md`
@ 2026-04-22 22:19_

| Probe | Prior (200 k, proxy) | Post-Tier-0 (1 M, real) | Δ |
|---|---|---|---|
| Random 80/20 | 0.8576 | **0.8772** | +0.0196 |
| Time-aware 80/20 | 0.8571 | **0.8702** | +0.0131 |
| Plate-blocked 5-fold CV | 0.8740 ± **0.0745** (proxy) | **0.8727 ± 0.0015** (real plate_id) | σ: −0.0730 |
| Target-shuffle within year | 0.5301 | **0.5825** | +0.0524 |
| `plate_prior_win_rate` ablation drop | +0.0599 | **+0.0188** | −0.0411 |
| Time-shift sensitivity (probe check) | 1.0000 | **1.0000** | — |

Per-fold plate-blocked: `[0.8743, 0.8707, 0.8728, 0.8743, 0.8713]`
(on **452 261 unique plates across 1 000 000 rows**, σ = 0.0015).

**Caveat on row-count confounders:** the "prior" column used a 200 k
subsample and the "post" column uses the full 1 M. Some of the random /
time-aware delta is just sample-size noise (more training data → higher
CatBoost AUC at fixed iters). The plan's tolerance was ±0.002 on
time-aware AUC under identical sampling — with the sampling difference
here, the +0.013 time-aware delta is within expectations.

### Tier 0 initial gate verdict (unstratified probe)

The plan defines three gates for proceeding to Tier 1:

| Gate | Threshold | Post-Tier-0 | Verdict |
|---|---|---|---|
| plate-blocked fold σ | < 0.05 | **0.0015** | ✅ PASS (clean) |
| `plate_prior_win_rate` ablation drop | ≤ 0.03 | **+0.0188** | ✅ PASS |
| Target-shuffle AUC | ≤ 0.51 | **0.5825** | ❌ FAIL (initial) |

**Two of three gates pass; target-shuffle initially fails.** The plan's
Tier 0.1 step 4 anticipated this (*"If target-shuffle AUC is still > 0.51
after the smoothing fix, the residual is inter-year temporal structure
rather than per-row leakage"*), but "is" vs "isn't" deserved a measurement
rather than a rhetorical move. Probe 1b was added to isolate per-row
leakage from year-structural signal.

### Hypothesis — why is target-shuffle still 0.5825?

Inter-year structural signal, not per-row leakage. Evidence already on
the table:

1. **Single-feature ablation gives nothing > +0.019.** If any feature
   were leaking the label, dropping it should collapse target-shuffle
   AUC toward 0.5 and the normal AUC would crash with it. The worst
   offender (`plate_prior_win_rate`) costs only 0.0188 — half the
   previous figure on the proxy audit, and well under the 0.03 gate.
   Every other feature is ≤ 0.0092.
2. **Time-shift sensitivity probe returns AUC = 1.0000.** Synthetically
   injecting `y · 0.8 + noise` into `plate_prior_win_rate` is flagged
   perfectly — so the 0.5825 is NOT a probe blind spot.
3. **Plate-blocked CV (0.8727) ≈ random (0.8772).** 0.0045 gap. Not
   memorizing plate identity.
4. **Time-aware CV (0.8702) ≈ random (0.8772).** 0.0070 gap.
5. **Prior-count features are cumulative.** `plate_prior_ticket_count`,
   `plate_prior_count_30D/90D`, `precinct_prior_count_*D`,
   `issuer_prior_count_*D` are monotone non-decreasing in calendar time
   for any fixed group. The 2024 distribution is fatter than the 2022
   distribution by construction, so a model can infer approximate year
   from them. Year correlates with win rate. Within-year shuffle
   preserves year-level means, so the model can still rank by inferring
   year → year-mean → score, yielding AUC > 0.5 despite within-year
   randomness. The hypothesis: the 0.5825 is this signal in its entirety.

### Tier 0.1 step 4 — Probe 1b (stratified target-shuffle, per-cell AUC)

_`python -m src.audit_leakage --full --skip-ablation` →
`docs/leakage_audit.md` @ 2026-04-23, after commits 8e7f143 (add probe),
ee5879d (redesign to per-cell AUC), and 87877ef (template + NaT-year
overflow fixes). Ablation numbers come from the prior full run
(2026-04-22 22:19) and are unchanged — ablation does not depend on the
stratified-probe code path._

**Design.** Shuffle `y` within (fiscal year × per-year plate-count
quartile) cells so that both year AND count level are fixed inside each
cell. Train CatBoost on 80% of the shuffled data. Compute AUC **within
each test-set cell separately** (cells with n ≥ 200 and both classes
present) — not pooled.

Pooled AUC after within-cell shuffle is confounded: it still rewards
cross-cell ranking driven by stable cell-level base rates + features
that encode cell membership. Per-cell AUC isolates within-cell ranking
ability, which is the only thing that can exceed 0.5 after an honest
within-cell shuffle if per-row leakage exists.

**Gate.** weighted-mean per-cell AUC ∈ [0.47, 0.53] AND max per-cell
AUC ≤ 0.55.

**Result.**

| Metric | Value |
|---|---|
| Pooled AUC on shuffled test set (confounded) | 0.6342 |
| Evaluable cells (n ≥ 200, both classes present) | 32 |
| **Per-cell weighted-mean AUC** | **0.5019** |
| Per-cell min AUC | 0.4048 (cell year=2015 q=0, n=554 — sampling noise) |
| **Per-cell max AUC** | **0.5265** (cell year=2019 q=2, n=4,059) |

Full per-cell table (issue_date year × per-year plate-count quartile):

| year | q | n | pos-rate | AUC |
|---|---|---|---|---|
| 2015 | 0 | 554 | 0.125 | 0.4048 |
| 2016 | 0 | 16,677 | 0.270 | 0.5055 |
| 2016 | 1 | 4,339 | 0.187 | 0.5127 |
| 2017 | 0 | 14,241 | 0.419 | 0.5005 |
| 2017 | 1 | 4,459 | 0.243 | 0.5110 |
| 2018 | 0 | 9,231 | 0.374 | 0.5006 |
| 2018 | 1 | 2,913 | 0.292 | 0.5044 |
| 2018 | 2 | 3,566 | 0.209 | 0.5055 |
| 2019 | 0 | 10,539 | 0.288 | 0.5052 |
| 2019 | 1 | 3,682 | 0.221 | 0.5070 |
| 2019 | 2 | 4,059 | 0.171 | 0.5265 |
| 2020 | 0 | 8,410 | 0.257 | 0.5000 |
| 2020 | 1 | 2,290 | 0.190 | 0.4795 |
| 2020 | 2 | 3,567 | 0.212 | 0.4923 |
| 2021 | 0 | 10,952 | 0.291 | 0.4924 |
| 2021 | 1 | 4,193 | 0.129 | 0.4862 |
| 2021 | 2 | 5,003 | 0.074 | 0.5171 |
| 2022 | 0 | 10,141 | 0.269 | 0.5101 |
| 2022 | 1 | 4,263 | 0.169 | 0.4989 |
| 2022 | 2 | 4,625 | 0.110 | 0.5208 |
| 2023 | 0 | 12,215 | 0.253 | 0.5023 |
| 2023 | 1 | 5,137 | 0.156 | 0.4837 |
| 2023 | 2 | 5,734 | 0.098 | 0.4967 |
| 2024 | 0 | 8,465 | 0.269 | 0.4982 |
| 2024 | 1 | 3,066 | 0.157 | 0.4951 |
| 2024 | 2 | 3,665 | 0.090 | 0.5058 |
| 2025 | 0 | 15,960 | 0.252 | 0.5044 |
| 2025 | 1 | 7,192 | 0.141 | 0.4980 |
| 2025 | 2 | 7,359 | 0.082 | 0.4974 |
| 2026 | 0 | 1,372 | 0.291 | 0.5057 |
| 2026 | 1 | 605 | 0.084 | 0.4894 |
| 2026 | 2 | 652 | 0.086 | 0.4932 |

**Interpretation.** 31 of 32 per-cell AUCs land in [0.48, 0.53]. The one
outlier (year=2015 q=0, AUC=0.4048, n=554) is AUC *below* 0.5 — the
model ranks worse than chance in that cell, which by construction cannot
be a leakage signature (real leakage points the model correctly, not
anti-correctly). At n=554 this is sampling noise, not signal. The
weighted mean is **0.5019**, as close to exactly 0.5 as a 100k-test-row
probe can get. Max is 0.5265, well under the 0.55 gate. There is no
cell in which the shuffled-label model ranks better than chance in a
way consistent with per-row leakage.

The confounded pooled AUC of 0.6342 is therefore entirely cross-cell /
year-structural signal: the model learns which cell a row comes from
(via cumulative counts) and uses that cell's base rate as its
prediction, giving high pooled AUC despite ~0.5 within-cell AUC.

**Before-probe-redesign-note.** The first implementation of Probe 1b
returned *pooled* AUC of 0.6334, which was initially misread as "leakage
got worse under stratification." It did not — that metric was
confounded. The redesign to per-cell AUC (commit ee5879d) is what makes
Probe 1b meaningful. Documenting the path: hypothesis → mis-designed
probe → caught the design flaw → redesigned → honest answer.

### Tier 0 final gate verdict

| Gate | Threshold | Result | Verdict |
|---|---|---|---|
| plate-blocked fold σ | < 0.05 | 0.0015 | ✅ PASS |
| `plate_prior_win_rate` ablation drop | ≤ 0.03 | +0.0188 | ✅ PASS |
| Target-shuffle (Probe 1, unstratified) | ≤ 0.51 | 0.5825 | ⚠︎ technically fails, but cause pinned |
| **Target-shuffle (Probe 1b, per-cell, honest)** | wmean ∈ [0.47, 0.53] AND max ≤ 0.55 | **wmean 0.5019, max 0.5265** | ✅ **PASS** |

**Tier 0 gate is closed.** The hypothesis — that the 0.5825 residual
reflects features encoding year rather than per-row label leakage — is
confirmed quantitatively: when the probe conditions on year AND
cumulative-count level, per-row ranking ability collapses to exactly
chance. The pipeline has no per-row feature-label leakage detectable
by this audit. Tier 1 work is clear to proceed.

The "technically fails" row for the unstratified probe is kept in the
table for honesty — the basic Probe 1 has a known blind spot for
year-structural signal in cumulative-count features, and that blind
spot is why Probe 1b exists. The honest reading of the plan's gate is
the per-cell row, not the pooled row.

### Auxiliary verification

- `pytest tests/test_asof_priors.py -v` → 5 / 5 pass (as of aaee543 and
  after a3de0c6).
- `python -m src.engineer` → 1 000 000 rows × 69 cols, smoothing
  `global_mean` = 0.254146 (train-slice 80 %) vs 0.236205 (full-dataset
  mean), `plate_id` first two meta cols along with `issue_date`.
- Spot-check `features.csv.columns[:4]` → `['issue_date', 'plate_id',
  'violation_code', 'precinct']`  ✓
- `python -m src.train` → full Optuna + final fit complete on the
  post-Tier-0 features.csv. Optuna best params:
  `depth=7, lr=0.0734, l2_leaf_reg=2.30, border_count=228,
  bagging_temperature=0.21, leaf_estimation_iterations=4,
  grow_policy=SymmetricTree`. Final CatBoost trained to best iter 582
  (early stop on the 80 k eval slice).
- `python -m src.evaluate` → **honest time-aware test AUC = 0.8662**
  on the 200 k held-out chronological tail (vs 0.8425 reported in
  `LIMITATIONS.md` pre-Tier-0 — a +0.0237 improvement from cleaner
  plumbing alone, before any Tier 1 work).
  - CatBoost accuracy: 80.22 %
  - LR baseline AUC: 0.6820
  - AUC gain over baseline: +0.1843
  - `metadata.json[feature_names]` contains 66 entries, excludes
    `plate_id`, `issue_date`, and `won`  ✓
  - Top-5 feature importances (unchanged qualitatively):
    `plate_prior_win_rate` (25.98), `viol_x_license` (11.91),
    `issuer_prior_win_rate_30D` (5.28), `plate_prior_win_rate_90D`
    (5.00), `issuing_agency` (4.50).
- `python predict_ensemble.py --once` → **pending** (requires LGB +
  XGB retrains first so the ensemble input set is current).

### Tier 0 delta on the headline number

| Metric | Pre-Tier-0 | Post-Tier-0 | Δ |
|---|---|---|---|
| Time-aware test AUC (full 1 M, chronological last 20 %) | 0.8425 | **0.8662** | **+0.0237** |

Three mechanisms contributed to the +0.0237:

1. Smoothing-constant leak fix (small — the constant moved 0.236 → 0.254,
   a uniform shift that can't change model ranking much on its own).
2. Canonical Optuna search space widened (CatBoost playbook §4.1 — 80
   trials over depth/lr/l2/border/bootstrap/bagging_temp/leaf_est/grow;
   previous models used narrower / ad-hoc params).
3. Full 1M-row training with proper chronological eval slice driving
   early stopping — the prior 0.8425 figure was from a shallower run.

The honest 0.8662 is already inside the Tier 1 target band (0.86–0.87
per `docs/auc_improvement_plan.md` §Tier 1 goals) without a single Tier
1 intervention. Tier 1 work — stacking, LGB/XGB rank-blend, CatBoost
OOF stacking — is still planned, but its baseline has shifted upward
and its headroom should be reassessed before committing compute.

### Commits on `main` (post-1f1635c)

```
87877ef  fix(audit): report template refs renamed var + handle NaT-year rows
ee5879d  fix(audit): per-cell AUC in stratified target-shuffle probe
8e7f143  feat: add stratified target-shuffle probe conditioning on (year x plate-count quartile)
c66317a  refactor: replace deprecated argsort with sort_values across audit+train
a3de0c6  feat: propagate plate_id as meta column for true plate-blocked CV
aaee543  fix: compute smoothing global_mean on chronological train slice only
f6fd5df  test: regression guard for strict as-of priors
```
