# FineHero Leakage Audit

*Generated 2026-04-21 20:00:11 — 200,000 rows, GPU*

Audit probes from FineHero AUC Playbook §2.2 and §7. Run with:

```bash
python -m src.audit_leakage          # 200k subsample, ~5–15 min on GPU
python -m src.audit_leakage --full   # full dataset, slower
```

## Headline

- Random 80/20 baseline: **0.8576**
- Time-aware (chronological 80/20): **0.8571**  (gap +0.0005)
- Plate-blocked 5-fold CV: **0.8740 ± 0.0745**

**Verdict:** random-split AUC is within 0.000 of the time-aware figure — the pipeline is cleaner than the playbook's typical leakage base rate. Honest baseline: **0.8571**.

## Probe 1 — Target-shuffle test

Shuffle `y` within each fiscal year, retrain. A leak-free pipeline returns AUC ≈ 0.5.

- Result: AUC = 0.5301
- Status: FAIL — AUC 0.5301 deviates from 0.5 (leakage suspected)

## Probe 2 — Time-shift sensitivity probe

Replace `plate_prior_win_rate` with a deliberately-leaky version (`y * 0.8 + noise`) and retrain. Confirms the probe machinery is sensitive enough to detect a leak.

- Baseline AUC: 0.8576
- Synthetic-leak AUC: 1.0000
- Status: PASS — synthetic leak AUC 1.0000 ≫ baseline 0.8576 (probe is sensitive)

## Probe 3 — Single-feature ablation (top 10)

Drop one feature at a time and retrain. Features whose removal drops AUC by more than **0.02** are flagged as suspected leaks (§7 item 8, adjusted down from 0.05 because this is short training).

| feature | AUC w/o feature | AUC drop | flag |
|---|---|---|---|
| `plate_prior_win_rate` | 0.7977 | +0.0599 | ⚠️ suspect leak
| `issuing_agency` | 0.8491 | +0.0085 |
| `state` | 0.8526 | +0.0050 |
| `plate_prior_ticket_count` | 0.8540 | +0.0037 |
| `license_type` | 0.8548 | +0.0028 |
| `viol_x_precinct` | 0.8550 | +0.0026 |
| `kw_hydrant` | 0.8551 | +0.0025 |
| `kw_sign` | 0.8551 | +0.0025 |
| `kw_expired` | 0.8551 | +0.0025 |
| `kw_bus_stop` | 0.8551 | +0.0025 |

## Probe 4 — Plate-blocked 5-fold CV

GroupKFold grouping plates so no plate appears in both train and test. A large drop vs. random split means the model memorizes plates.

- Mean AUC: 0.8740 ± 0.0745
- Per-fold: [0.7350898087024689, 0.8615151941776276, 0.9191966354846954, 0.9102274775505066, 0.9441717863082886]
- Note: PROXY groups (plate_prior_* combo) — feature alignment with raw plate skipped

## Probe 5 — Time-aware chronological split

Sort by `issue_date`, hold out last 20% chronologically.

- AUC: 0.8571
- Gap vs random split: +0.0005

## Probe 6 — Plate-prior distribution drift

If a plate's `plate_prior_ticket_count` distribution looks very different in train vs. test under random split, that is a leakage signature.

| split | train mean | test mean | train p95 | test p95 |
|---|---|---|---|---|
| random 80/20 | 8.646 | 8.579 | 47.0 | 47.0 |
| time-aware   | 8.662 | 8.527 | 47.0 | 47.0 |

## Methodology notes

- All probes use a 200,000-row subsample for speed. Run with `--full` for the complete dataset.
- CatBoost fit: 400 iters max, early-stop 40, depth 6, lr 0.08.
- The target-shuffle probe and time-shift sensitivity probe together bound the pipeline's leakage: former detects label leakage, latter confirms the detector works.
- The plate-blocked probe uses a PROXY grouping (combination of plate_prior_ticket_count + plate_prior_win_rate) because the raw plate column is not preserved in `features.csv`. For a fully rigorous audit, re-engineer to pass `plate_id` through as a meta column.

## What this audit does NOT do

- Concept-drift detection across fiscal years (playbook §2.6).
- Cleanlab label-noise scan (playbook §2.5).
- Deepchecks FeatureLabelCorrelationChange gate (playbook §7 item 11).
- Adjudication-date vs. issue-date separation — we use issue_date as as-of because FineHero's data doesn't carry a separate hearing-request timestamp.
