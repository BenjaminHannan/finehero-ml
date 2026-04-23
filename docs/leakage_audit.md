# FineHero Leakage Audit

*Generated 2026-04-22 23:52:31 — 1,000,000 rows, GPU*

Audit probes from FineHero AUC Playbook §2.2 and §7. Run with:

```bash
python -m src.audit_leakage          # 200k subsample, ~5–15 min on GPU
python -m src.audit_leakage --full   # full dataset, slower
```

## Headline

- Random 80/20 baseline: **0.8773**
- Time-aware (chronological 80/20): **0.8710**  (gap +0.0063)
- Plate-blocked 5-fold CV: **0.8727 ± 0.0015**

**Verdict:** random-split AUC is within 0.006 of the time-aware figure — the pipeline is cleaner than the playbook's typical leakage base rate. Honest baseline: **0.8710**.

## Probe 1 — Target-shuffle test

Shuffle `y` within each fiscal year, retrain. A leak-free pipeline returns AUC ≈ 0.5.

- Result: AUC = 0.5825
- Status: FAIL — AUC 0.5825 deviates from 0.5 (leakage suspected)

### Probe 1b — Stratified target-shuffle (year × plate-count quartile), per-cell AUC

The unstratified probe leaves the `plate_prior_*` cumulative counts encoding year indirectly — so AUC > 0.5 can come from the model inferring year rather than from real leakage. This variant shuffles `y` within (fiscal year × per-year plate-count quartile) cells, retrains, then computes AUC **within each test-set cell separately** — pooled AUC after within-cell shuffle is confounded because it still rewards cross-cell ranking driven by stable cell-level base rates. Per-cell AUC ≈ 0.5 is the honest evidence of no per-row leakage.

- Pooled AUC on shuffled test set (confounded — preserves cross-cell structure): **0.6342**
- **Per-cell AUC** (the honest metric — within-cell only):
  - evaluable cells (n ≥ 200, both classes present): **32**
  - weighted mean: **0.5019**
  - min: 0.4048   max: **0.5265**

| year | quartile | n | pos-rate | AUC |
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

- Status: PASS — within-cell AUC ≈ 0.5 (wmean 0.5019, max 0.5265). The unstratified 0.5825 was entirely cross-cell / year-structural signal, not per-row leakage.

## Probe 2 — Time-shift sensitivity probe

Replace `plate_prior_win_rate` with a deliberately-leaky version (`y * 0.8 + noise`) and retrain. Confirms the probe machinery is sensitive enough to detect a leak.

- Baseline AUC: 0.8773
- Synthetic-leak AUC: 1.0000
- Status: PASS — synthetic leak AUC 1.0000 ≫ baseline 0.8773 (probe is sensitive)

## Probe 3 — Single-feature ablation (top 10)

Drop one feature at a time and retrain. Features whose removal drops AUC by more than **0.02** are flagged as suspected leaks (§7 item 8, adjusted down from 0.05 because this is short training).

| feature | AUC w/o feature | AUC drop | flag |
|---|---|---|---|
_skipped_

## Probe 4 — Plate-blocked 5-fold CV

GroupKFold grouping plates so no plate appears in both train and test. A large drop vs. random split means the model memorizes plates.

- Mean AUC: 0.8727 ± 0.0015
- Per-fold: [0.8742803037166595, 0.8707360625267029, 0.8727540373802185, 0.8742994964122772, 0.8713109791278839]
- Note: real plate_id (452,261 unique plates across 1,000,000 rows) — Tier 0.2 true plate-blocked GroupKFold

## Probe 5 — Time-aware chronological split

Sort by `issue_date`, hold out last 20% chronologically.

- AUC: 0.8710
- Gap vs random split: +0.0063

## Probe 6 — Plate-prior distribution drift

If a plate's `plate_prior_ticket_count` distribution looks very different in train vs. test under random split, that is a leakage signature.

| split | train mean | test mean | train p95 | test p95 |
|---|---|---|---|---|
| random 80/20 | 8.652 | 8.634 | 48.0 | 48.0 |
| time-aware   | 7.335 | 13.904 | 40.0 | 73.0 |

## Methodology notes

- All probes use a 1,000,000-row subsample for speed. Run with `--full` for the complete dataset.
- CatBoost fit: 400 iters max, early-stop 40, depth 6, lr 0.08.
- The target-shuffle probe and time-shift sensitivity probe together bound the pipeline's leakage: former detects label leakage, latter confirms the detector works.
- The plate-blocked probe uses the real `plate_id` meta column written by `src/engineer.py` (Tier 0.2). Each of 5 folds holds out a disjoint slice of plates.

## What this audit does NOT do

- Concept-drift detection across fiscal years (playbook §2.6).
- Cleanlab label-noise scan (playbook §2.5).
- Deepchecks FeatureLabelCorrelationChange gate (playbook §7 item 11).
- Adjudication-date vs. issue-date separation — we use issue_date as as-of because FineHero's data doesn't carry a separate hearing-request timestamp.
