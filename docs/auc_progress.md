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

### Tier 0 gate verdict

The plan defines three gates for proceeding to Tier 1:

| Gate | Threshold | Post-Tier-0 | Verdict |
|---|---|---|---|
| plate-blocked fold σ | < 0.05 | **0.0015** | ✅ PASS (clean) |
| `plate_prior_win_rate` ablation drop | ≤ 0.03 | **+0.0188** | ✅ PASS |
| Target-shuffle AUC | ≤ 0.51 | **0.5825** | ❌ FAIL |

**Two of three gates pass. Target-shuffle does not.** Per the plan's
"If any of the three is not met: stop, report, discuss with Benjamin"
rule, Tier 1 work is paused pending your decision.

### Analysis — why is target-shuffle still 0.5825?

The evidence strongly suggests this is **inter-year structural signal,
not per-row leakage**:

1. **Single-feature ablation gives nothing > +0.019.** If any feature
   were leaking the label, dropping it should collapse target-shuffle
   AUC toward 0.5 and the normal AUC would crash with it. The worst
   offender (`plate_prior_win_rate`) costs only 0.0188 — half the
   previous figure on the proxy audit, and well under the 0.03
   gate. Every other feature is ≤ 0.0092.
2. **Time-shift sensitivity probe returns AUC = 1.0000.** The probe
   machinery *would* catch a real feature-level leak — we synthetically
   injected one (`y · 0.8 + noise` into `plate_prior_win_rate`) and
   the probe flagged it perfectly. So the 0.5825 isn't a probe blind
   spot.
3. **Plate-blocked CV (0.8727) ≈ random (0.8772).** 0.0045 gap. The
   model is not memorizing plate identity — any plate-level signal is
   captured by the priors, which generalize to held-out plates.
4. **Time-aware CV (0.8702) ≈ random (0.8772).** 0.0070 gap. The model
   transfers across time, which is the actual concern for production.
5. **Prior-count features are cumulative.** `plate_prior_ticket_count`,
   `plate_prior_count_30D/90D`, `precinct_prior_count_*D`,
   `issuer_prior_count_*D` — all are monotone non-decreasing in calendar
   time for any fixed group. Their full-dataset distribution in 2024 is
   fatter than in 2022 by construction, and the model can infer
   approximate year from them. Year correlates with win rate (concept
   drift documented in `LIMITATIONS.md`). Target-shuffle within year
   preserves year-level means, so the model can still rank across years
   by inferring year → year-mean → score, yielding AUC > 0.5 despite
   within-year label randomness.

This matches the plan's own anticipation (Tier 0.1 step 4: *"If
target-shuffle AUC is still > 0.51 after the smoothing fix, the residual
is inter-year temporal structure rather than per-row leakage — document
that conclusion with numbers and move on to 0.2."*).

### Recommendation (for discussion)

Three candidates, in ascending invasiveness:

**Option A — Accept 0.5825 as inter-year signal; proceed to Tier 1.**
Justification: every other leakage signal is clean, the time-shift
probe works, and the target-shuffle result is mechanistically
explainable. Time-aware AUC 0.8702 is the number that matters for
production and has improved materially. Cost: the target-shuffle
gate is technically failing, which is uncomfortable.

**Option B — Adjust the target-shuffle probe to also stratify by
prior-count quartile (or replace with a Friedman-style permutation
test that conditions on quartile bins of cumulative prior counts).**
This would isolate per-row label leakage from year-structural signal.
Cost: 1 extra probe, maybe 30 min of work. If the stratified version
returns ≈ 0.5, it would confirm Option A's interpretation and pass
the gate.

**Option C — Add year-stationary variants of the prior counts (e.g.,
`plate_count_last_365D / plate_total_lifetime_count`, or log-scaled
and z-scored per year) and rerun the audit.** This addresses the
root cause but changes feature semantics and should be treated as
Tier 1 work, not Tier 0.

My recommendation is **Option B** if you want the gate to honestly
close before Tier 1, or **Option A** if you're willing to treat the
plate-blocked σ and ablation-drop passes as sufficient evidence of
no per-row leakage (they are strong evidence).

### Auxiliary verification (completed regardless of gate outcome)

- `pytest tests/test_asof_priors.py -v` → 5 / 5 pass (as of aaee543 and
  after a3de0c6).
- `python -m src.engineer` → 1 000 000 rows × 69 cols, smoothing
  `global_mean` = 0.254146 (train-slice 80 %) vs 0.236205 (full-dataset
  mean), `plate_id` first two meta cols along with `issue_date`.
- Spot-check `features.csv.columns[:4]` → `['issue_date', 'plate_id',
  'violation_code', 'precinct']`  ✓
- `python -m src.train` → **pending re-run**; will confirm
  `metadata.json[feature_names]` excludes `plate_id`.
- `python predict_ensemble.py --once` → **pending re-run**.

### Commits on `main` (post-1f1635c)

```
a3de0c6  feat: propagate plate_id as meta column for true plate-blocked CV
aaee543  fix: compute smoothing global_mean on chronological train slice only
f6fd5df  test: regression guard for strict as-of priors
```
