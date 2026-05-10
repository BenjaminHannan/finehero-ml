# The Rolling-Prior Investigation

A postmortem-style narrative of how we discovered, diagnosed, and patched the inference-time bug that was causing FineHero's CatBoost model to recommend PAY on tickets it had actually learned to identify as wins. Written so a future maintainer can reconstruct what was tried, what landed, and what's still open.

This document is meant to be read sequentially. It does not replace `LIMITATIONS.md` (which holds the canonical operational caveats) or `CLAUDE.md` (which holds the ground rules) — both contain the conclusions in compact form. This file holds the *path*.

---

## 0. Starting state

We started with a model that looked, by every internal metric, ready for production:

- **CatBoost classifier** trained on 1,000,000 NYC parking-ticket rows. 95 features (94 + 1 label `won`), 80/20 chronological split (last 20% by `issue_date` is held out as the test set, 200k rows). Eval slice for early stopping = chronological last 10% of the training portion.
- **Internal performance:** test AUC = **0.8694**, accuracy = **77.6%** at the (then) hardcoded 0.40 threshold.
- **Companion ensemble:** LightGBM (AUC 0.8716) and XGBoost (AUC 0.8743) trained on the same `features.csv`. Rank-percentile blends top out around 0.874.
- **Calibration:** an isotonic regressor at `models/isotonic_calibrator.joblib`, fit on the chronological last 10% of training (n=80,000), bringing ECE from 0.152 raw to **0.023 calibrated** on the 200k test set. AUC preserved (isotonic is monotone).
- **Threshold tuning:** `models/dispute_threshold.json` with four policies tuned against `prob_avg_calibrated`: F1 (0.32, default), Youden (0.17), accuracy (0.555), match-baserate (0.40). Test base rate 18.2%.
- **Inference scripts:** `predict.py` (single-ticket CatBoost lookup) and `predict_ensemble.py` (rank-blend of all three). Both pulled `data/violations_raw.csv` and scored a randomly-sampled disputed row.

The `LIMITATIONS.md` already flagged the right operational caveats: selective labels, fairness not solved, time-aware AUC is the headline. What it didn't say — at the start of this investigation — was that the predict scripts weren't actually applying the calibrator or reading the tuned threshold. They used `prob = model.predict_proba(...)[0, 1]` and a hardcoded `DISPUTE_THRESHOLD = 0.40`. So the *probabilities* shown to users were raw CatBoost output (badly miscalibrated), and the *threshold* was the legacy 0.40 (which in retrospect catches everything in the loose middle of the distribution, with marginal precision).

The investigation began with what looked like a simple question: how does the model do on tickets that aren't in `violations_raw.csv`?

---

## 1. The synthetic-ticket experiment

The first probe was the most artificial: hand-construct seven hypothetical tickets covering scenarios where the methodology gives a clear directional prediction. Score them. See if the predictions move in the expected direction.

The seven scenarios, with their predicted-direction labels:

| # | Scenario | Expected | Methodology reason |
|---|---|---|---|
| 1 | Outside posted hours, sign violation | HIGH | `within_posted_hours = 0` is the methodology's #1 dismissal signal |
| 2 | Sign violation in heavy snow | HIGH | `kw_sign + is_bad_weather` — sign violations frequently dismissed; bad weather supports "obscured" defenses |
| 3 | Blocking driveway, clear day | LOW | `kw_blocking` violations rarely dismissed; safety-critical |
| 4 | Bus stop violation, repeat-offender plate | LOW | `kw_bus_stop` + plate prior win rate well below baseline |
| 5 | Meter violation, light drizzle | MID-HIGH | `kw_meter` — methodology says meters are "frequently dismissed" |
| 6 | Hydrant violation, clear conditions | LOW | `kw_hydrant` — almost never dismissed |
| 7 | Holiday late-night sign violation | HIGH | `is_holiday + kw_sign + late hour`; ambiguous posting at night |

The build was straightforward: construct a feature dict matching the 95-name schema, leave anything I didn't have an obvious value for as NaN (CatBoost handles missing natively), score through `predict_proba`. The 27 rolling-prior columns and the three `days_since_*` columns were left NaN — there's no obvious value to fill them with for a hypothetical plate.

Results were the first surprise:

| # | Scenario | Expected | Predicted (raw) | Bucket-match? |
|---|---|---|---:|---|
| 1 | outside_posted_hours | HIGH | 18.6% | ✗ |
| 2 | sign_violation_in_snow | HIGH | 9.9% | ✗ |
| 3 | blocking_driveway | LOW | 1.4% | ✓ |
| 4 | bus_stop_repeat | LOW | 0.7% | ✓ |
| 5 | meter_violation_drizzle | MID-HIGH | 22.5% | ✗ |
| 6 | hydrant_clear | LOW | 11.9% | ✓ |
| 7 | holiday_late_night | HIGH | 9.2% | ✗ |

Three of seven scenarios landed in the predicted bucket. All three correct ones were the LOW-expected scenarios. Every HIGH-expected scenario came back well below threshold — including the methodology's #1 strongest signal (`within_posted_hours = 0`), which only got 18.6%.

But the *ranking* was roughly right: sort by predicted probability and you get `meter (22.5%) > outside_hours (18.6%) > hydrant (11.9%) > sign+snow (9.9%) > holiday+sign (9.2%) > blocking (1.4%) > bus_stop (0.7%)`. The two clear losses were correctly at the bottom; meter and outside-hours were at the top. The model knew the right order; it just had everything compressed downward.

I attributed the compression to two things at the time:
1. The OATH base rate is ~30%, and the model was trained with `auto_class_weights='Balanced'` — but the raw probabilities still reflect the underlying class distribution. Most synthetic tickets in the 5–25% range was consistent with "ambiguous-to-slightly-losing" in calibrated terms.
2. Synthetic tickets had `is_missing_*` flags firing because I had no issuer/legal-code data. Training rows with that pattern were under-represented and likely associated with weaker tickets.

What I missed at this stage — and what would matter enormously later — is that all 27 rolling-prior features were also NaN for every synthetic ticket. I treated this as one of many minor missing-value issues. It wasn't; it was the dominant signal. But identifying that required real held-out tickets, not synthetic ones.

The output script was saved as `test_hypothetical_tickets.py`. It still runs and reproduces the same numbers.

---

## 2. The held-out anchor and the calibration discovery

The second probe was the obvious counterfactual to the synthetic experiment: how does the model do on the 200k *real* held-out test set, which it has never seen? `data/test_set.joblib` was right there.

Loading and scoring the full test set immediately reproduced the metadata's claimed 0.8694 AUC. Accuracy at the legacy 0.40 threshold was 71.94%. The calibration table was the real surprise:

| Predicted bin | Count | Mean predicted | Empirical |
|---|---:|---:|---:|
| [0.00, 0.05) | 58,391 | 1.2% | 0.3% |
| [0.05, 0.10) | 10,264 | 7.3% | 2.2% |
| [0.10, 0.20) | 15,662 | 15.1% | 5.6% |
| [0.20, 0.30) | 15,797 | 25.1% | 9.6% |
| [0.30, 0.40) | 17,696 | 35.0% | 13.6% |
| [0.40, 0.50) | 17,461 | 45.0% | 17.8% |
| [0.50, 0.60) | 14,536 | 54.6% | 21.8% |
| [0.60, 0.70) | 11,522 | 65.1% | 28.8% |
| [0.70, 0.85) | 22,471 | 77.8% | 43.5% |
| [0.85, 1.00) | 16,200 | 92.1% | 73.4% |

The raw probabilities were systematically inflated across every bin. Rule-of-thumb: a well-calibrated mid-bin should have predicted mean ≈ empirical rate. Here the model predicted 65% on tickets that actually won 28.8% of the time. This is the cost of `auto_class_weights='Balanced'` — it makes the loss function reweight the minority class, which inflates predicted probabilities.

My initial conclusion was: the threshold of 0.40 is a *ranking cutoff*, not a probability claim, and the user-facing "win probability: X%" is misleading. We needed isotonic calibration.

Then I noticed something I'd missed: `models/isotonic_calibrator.joblib` already existed. Inspecting it:

```python
{
    "calibrator": IsotonicRegression(...),
    "fitted_on": "X_ev (chronological last 10% of train slice)",
    "n_calibration": 80000,
    "auc_test_raw":  0.8737,
    "auc_test_cal":  0.8735,
    "ece_test_raw":  0.1519,
    "ece_test_cal":  0.0234,
    "brier_test_raw": 0.1367,
    "brier_test_cal": 0.1012,
}
```

The calibrator was fit, validated, and sitting in `models/`. It was just never wired into `predict.py`. Applying it to the test set probabilities collapses the calibration error dramatically:

| Bin | Count | Raw mean | **Calibrated** | **Empirical** |
|---|---:|---:|---:|---:|
| [0.30, 0.40) | 17,696 | 35.0% | **13.1%** | 13.6% |
| [0.40, 0.50) | 17,461 | 45.0% | **20.0%** | 17.8% |
| [0.50, 0.60) | 14,536 | 54.6% | **29.0%** | 21.8% |
| [0.70, 0.85) | 22,471 | 77.8% | **57.2%** | 43.5% |
| [0.85, 1.00) | 16,200 | 92.1% | **86.9%** | 73.4% |

After calibration, predicted columns track empirical within ~5 percentage points across most of the range. ECE: 0.1519 → 0.0234.

`models/dispute_threshold.json` told the same story: it was tuned on `prob_avg_calibrated`, not on raw. The whole infrastructure was built around calibration; the inference code just hadn't picked up the artifacts.

This was the first wiring fix:

1. Add `_load_calibrator()` and `_load_threshold()` helpers to `predict.py`.
2. After `model.predict_proba(...)`, pass through the calibrator if available; print both raw and calibrated numbers.
3. Replace the hardcoded 0.40 with the threshold from JSON.
4. Mirror the change in `predict_ensemble.py` (which has its own three-model rank-blend path).

That fix shipped. The synthetic-ticket numbers got smaller after calibration (e.g., outside_posted_hours: 18.6% raw → 7.0% calibrated; meter_drizzle: 22.5% → 7.6%) because the raw values were inflated and isotonic pulled them down. The ranking was preserved. None of the seven synthetic tickets crossed the new 0.32 (F1) threshold.

At the time I read this as: synthetic tickets are out-of-distribution and that's why they're compressed. The held-out test set (which the model was *trained* to be calibrated against) was clearly fine. The picture would change once I tried real *new* tickets from the live API — but I didn't know that yet.

---

## 3. Threshold drift and a small overfitting check

Before pushing on the live API, one more piece of due diligence: the `dispute_threshold.json` thresholds were tuned on the full 200k test set, which is also the set we report metrics on. Were they overfit to that fold?

The trick that made this checkable: per `src/train.py:140`, the test set is *chronological* (last 20% by `issue_date`). So we can split it in half by date — first 100k as a tuning slice, last 100k as a report slice — and re-tune.

Results in `test_threshold_oos.py`:

| Slice | AUC | ECE (raw / cal) |
|---|---:|---:|
| Older half (tune, 100k) | 0.8697 | – |
| Newer half (report, 100k) | 0.8695 | 0.168 / **0.052** |

AUC barely moved between halves (no temporal drift in ranking quality). But the calibrated ECE on the newest 10% of the corpus was **0.052** vs the full test set's 0.023 — calibration was already drifting on data the calibrator had been fit close to. This wasn't the live-API issue we'd find later; it was a more local effect that will recur whenever the calibrator sits stale.

Threshold-wise, re-tuning each policy on the older half and evaluating on the newer half gave nearly identical F1 to the existing thresholds:

| Policy | Existing | Tune-half | Shift | Report-slice F1 |
|---|---:|---:|---:|---:|
| f1 | 0.32 | 0.40 | +0.08 | 0.574 (oos) vs 0.573 (existing) |
| accuracy | 0.555 | 0.62 | +0.07 | – |
| youden | 0.17 | 0.22 | +0.05 | – |

The thresholds drifted up slightly when re-tuned on the older half alone, but the empirical performance gap was 0.001 F1 — essentially noise. The from-fold concern was real but tiny in magnitude.

The honest takeaway from this section was that the calibrator has a *shelf life*. ECE doubles between the start and end of the chronological test set. The threshold tuning is robust enough that it isn't worth re-tuning, but the calibrator is something to refit whenever new training data lands.

This finding made it into LIMITATIONS.md but is not the main story of this document. The next probe is.

---

## 4. The live API: first attempt and the filter trap

The setup so far had been: held-out tickets the model has never seen are scored well. The model has 0.87 AUC and ECE 0.023 on a 200k random sample. The synthetic experiment was a curiosity.

The next experiment was supposed to confirm that the model also works on tickets pulled fresh from the NYC Open Data API today — the same data source the training set came from, but rows the training set didn't include.

**`test_live_nyc_tickets.py` v1.** Fetch the most recent 500 tickets that have a hearing-decided outcome:

```python
where = "violation_status IN ('HEARING HELD-NOT GUILTY','HEARING HELD-GUILTY')"
r = requests.get(OPEN_PARKING_URL,
                 params={"$limit": 500, "$where": where, "$order": "issue_date DESC"})
```

Dedupe against `violations_raw.csv` by summons_number, attempt pvqr enrichment per fiscal year, build features the same way `predict.py` does for a single row, score them all.

Results:
- 500 fresh tickets, 261 won (52.2%), 239 lost
- pvqr enrichment hit rate: **8.6%** (43 of 500). Most rich features (`within_posted_hours`, `feet_from_curb`, etc.) were therefore NaN
- **AUC: 0.5621.** Barely better than random.
- Median calibrated probability: ~1%. The F1 threshold caught essentially nothing.

The model that got 0.87 AUC on the held-out test set got 0.56 on freshly-fetched tickets. That's a catastrophic gap that would normally indicate a serious model failure.

But there were two suspicious things about the test:

1. **The win rate was 52%, not 18%.** The held-out test set's base rate is 18.2%; the live sample's was 52%. That's a 35-point distribution shift on the *outcome*. Why?
2. **pvqr enrichment was 8.6%.** Training claimed a 30–50% join rate.

The first thing turned out to be the bigger issue. The training set's win rate was 18.2% because the model was trained on **all** outcome types: HEARING HELD-NOT GUILTY (won), HEARING HELD-GUILTY (lost), HEARING HELD-GUILTY REDUCTION (lost in training's regex), ADMIN REDUCTION (lost), APPEAL AFFIRMED (lost), and so on. My filter `IN ('HEARING HELD-NOT GUILTY', 'HEARING HELD-GUILTY')` excluded the ~62% of training-population statuses that are REDUCTIONS / ADMIN / APPEAL outcomes — the ones training counted as losses but I excluded entirely.

Verifying this directly: filter the training set's `violations_raw.csv` to just the same two statuses I used live, and the win rate jumps to **37.5%**. So the training distribution in that filter is 37.5% wins, but my live sample (with the same filter) is 52%. There's still a real distribution shift — but it's 15 percentage points, not 35.

The pvqr-coverage gap was real but not the dominant issue. Even tickets that *did* get pvqr enrichment had AUC 0.522 (worse than the unenriched 0.555). Whatever was breaking the model on live data wasn't fixed by enriching with pvqr.

---

## 5. The filter correction — and the new mystery

Re-running with a training-matching filter: drop only the genuine "no outcome yet" statuses (`OUTSTANDING`, `IN PROCESS`, `HEARING PENDING`, `HEARING ADJOURNMENT`), and label `won=1` if `violation_status` matches `DISMISS|NOT GUILTY|NOT LIABLE` (the same regex `engineer.py` uses).

Results on a 998-row pull (after dedupe):
- Win rate: **18.8%** (matches training's 18.2% within rounding)
- pvqr enrichment hit rate: **5.2%**
- **AUC: 0.9056** — actually higher than the held-out test set's 0.8694
- Median calibrated probability: **0.1%**
- F1 at 0.32 threshold: **0.000**
- Confusion: TP=0, FN=188, FP=2, TN=808

This is a strange combination: AUC 0.90 (the model is *ranking* very well) but calibration is so destroyed that the threshold catches zero true positives out of 188 actual wins. The model knows which tickets are more likely to win — it just has every single prediction crushed near zero.

Calibration table on the live batch:

| Bin | Count | Mean predicted | Empirical |
|---|---:|---:|---:|
| [0.00, 0.05) | 992 | 0.5% | 18.8% |
| [0.05, 0.15) | 2 | 9.2% | 0.0% |
| [0.15, 0.30) | 2 | 25.3% | 100.0% |
| [0.30, 0.50) | 1 | 40.1% | 0.0% |
| [0.50, 0.85) | 1 | 52.9% | 0.0% |

99.4% of the live tickets sit in the bottom calibrated bin (0–5%), but 18.8% of those tickets actually win. The model is saying "this won't win" with 0.5% confidence on a population that actually wins 19% of the time.

At this point I had two hypotheses, both of which I'd later have to refine:

- **H1: pvqr coverage drop.** Training's 30–50% pvqr join rate was much higher than live's 5%. With `within_posted_hours` and `feet_from_curb` mostly missing, the model's strongest known signal is gone, and it defaults toward losses.
- **H2: Concept drift.** Maybe NYC's enforcement or adjudicator behavior has shifted enough that the model is no longer aligned with current outcomes.

Neither hypothesis turned out to be the dominant explanation. But H1 felt right at the time, and the first version of the LIMITATIONS update blamed pvqr coverage. I'll come back to this.

---

## 6. Spot checks: ticket 9249319368 and 9126107983

The aggregate numbers were strange enough that I wanted to look at individual tickets. Pick a couple, run them through `predict.py`'s feature builder, look at what the model said and why.

**Ticket 9249319368.** Manhattan precinct 18 (Midtown North), Mercedes-Benz SUV on Broadway, $65 muni-meter violation issued at 7:39 AM, posted hours 7 AM – 6 PM. Within posted hours = yes. pvqr enriched (one of the lucky 5%, hit on the FY2026 endpoint `pvqr-7yc4`). Plate not in training (new). Status: outstanding (no hearing yet).

Score:
- Raw probability: 2.8%
- Calibrated probability: **0.9%**
- Verdict: **Pay this one**

Operationally this looked defensible. Within posted hours, normal commercial corridor, normal enforcement, no obvious procedural defense. Even at uncalibrated face value, "low chance of winning" is what intuition says. We had no actual outcome to validate against.

**Ticket 9126107983.** Brooklyn precinct 66 (Borough Park), 2024 Toyota SUV on 13th Avenue, $35 muni-meter violation issued at 10:56 AM, posted hours 8 AM – 7 PM. Within posted hours = yes. pvqr enriched (FY2024 endpoint `8zf9-spf8`). Plate not in training. Status: **HEARING HELD-NOT GUILTY** — the driver took it to a hearing and the ticket was dismissed.

Score:
- Raw probability: 3.0%
- Calibrated probability: **1.1%**
- Verdict: **Pay this one** — wrong

Two near-identical tickets (Manhattan vs Brooklyn meter, both within posted hours, both with normal enforcement profiles) both scored ~1% calibrated. One pending, one actually won. The model said "essentially 0% chance of winning" on a ticket that actually won.

This was the first direct evidence of the model failing on a real ticket with a known outcome. From here the question shifted from "is calibration drifting?" to "*why* is the model putting 1% on a winning ticket?"

The pvqr-coverage hypothesis didn't fit. Both these tickets were pvqr-enriched. `within_posted_hours` was correctly populated. `feet_from_curb` was set. The strongest pvqr signals were available. The compression was happening anyway.

---

## 7. The diagnosis: SHAP attribution

CatBoost has built-in TreeShap support: `model.get_feature_importance(pool, type='ShapValues')` returns per-feature contributions for each row, plus a bias term that makes them sum to the raw log-odds output.

Running this on ticket 9126107983 (the Brooklyn meter ticket that won) gave a clean answer.

- **Raw log-odds: −3.465** (= 3.0% probability)
- **Bias: −0.182** (= 45.5% base probability)
- **Sum of contributions: −3.283** (the bias-relative shift)

So the model started at a 45% base rate (the bias is the global average log-odds) and then features pushed it down by 3.3 log-odds to land at 3% raw.

Top 10 features by absolute SHAP:

| Feature | Value | SHAP (log-odds) |
|---|---|---:|
| `plate_prior_win_rate_30D` | NaN | **−0.921** |
| `viol_x_license` | "FAIL TO DSPLY MUNI METER R..." | +0.499 |
| `plate_prior_win_rate` | 0.250 | +0.406 |
| `issuer_prior_win_rate_365D` | NaN | **−0.383** |
| `issuer_prior_count_30D` | NaN | **−0.339** |
| `issuer_prior_win_rate_30D` | NaN | **−0.322** |
| `summons_format` | "UNKNOWN" | −0.273 |
| `precinct_prior_win_rate_30D` | NaN | **−0.247** |
| `plate_prior_win_rate_365D` | NaN | **−0.246** |
| `issuer_prior_win_rate_90D` | NaN | **−0.234** |

Eight of the top ten contributors are rolling-prior features whose values are NaN. Four are plate-level rolling priors; four are issuer-level. All NaN.

Aggregated by feature family:

| Family | Total log-odds contribution |
|---|---:|
| **Rolling priors (27 features)** | **−3.687** |
| Core categoricals (14) | +0.538 |
| Plate-level non-rolling (4) | +0.155 |
| Temporal (11) | +0.044 |
| pvqr-only (9) | 0.000 |
| Missing flags (10) | 0.000 |
| Keyword flags (6) | 0.000 |
| Issuer-level non-rolling (6) | −0.071 |
| Weather (4) | −0.071 |

The 27 rolling-prior features alone contributed −3.687 log-odds. Everything else summed to almost neutral (+0.595 from positive contributors, mildly negative pulls cancel). The rolling-prior block is the entire prediction collapse.

What are these features? `engineer.py` builds them via running tallies during training. For each ticket, "rolling prior X for entity Y over window W" is "stats over Y's tickets that occurred in the W days strictly before this ticket's `issue_date`." Three entity types (`plate`, `precinct`, `issuer`) × three windows (30D, 90D, 365D) × three statistics (`wins`, `count`, `win_rate`) = 27 features.

These features are time-dependent. Computing them at inference for a *new* ticket requires either:

- A precomputed lookup keyed by `(entity, date)` for every entity and every date, or
- A streaming infrastructure that maintains running tallies per entity as new tickets arrive.

`predict.py` had **neither**. The only thing close was `models/plate_history_map.joblib`, which holds *cumulative* per-plate stats (no time window). That populated `plate_prior_win_rate` and `plate_prior_ticket_count` correctly, but the windowed versions came out NaN.

In training, the windowed features were always populated (engineer.py computes them in batch). The rare cases where they were NaN — say, the very first ticket for a plate — were a small minority and tended to correlate with weak tickets. The model learned, accurately within the training distribution, that "rolling priors NaN = loss-leaning."

At inference, the live distribution flips: **every** new ticket has all 27 rolling priors as NaN. The model applies the strong "NaN = loss" signal it learned, and predictions collapse.

This explained, all at once, every previous puzzling result:

- The synthetic-ticket experiment in section 1: I left rolling priors NaN. Same root cause.
- The 998-row live-API test in section 5: every fresh ticket has NaN rolling priors. Same root cause.
- The 5% pvqr coverage finding was a *secondary* effect — even tickets that did get pvqr couldn't fix the rolling-prior collapse, because rolling priors aren't in either NYC endpoint.

The pvqr framing in the LIMITATIONS update was wrong. It was the most visible distribution gap (5% vs 30–50%), but it wasn't the dominant one.

---

## 8. The counterfactual: what does the model say if rolling priors are filled?

If the diagnosis is right, filling the rolling priors with reasonable values should flip ticket 9126107983's prediction. The simplest "reasonable values" are the training-set population means: `plate_prior_win_rate_30D ≈ 0.24`, `plate_prior_count_30D ≈ 0.84`, `precinct_prior_win_rate_365D ≈ 0.24`, etc.

The numbers fell out cleanly:

| Variant | Raw | Calibrated | Verdict |
|---|---:|---:|---|
| Original (rolling priors NaN) | 3.0% | 1.1% | PAY |
| + rolling priors filled with population means | **62.7%** | **34.1%** | **DISPUTE** |
| + days_since/issuer_bayes also filled | 73.1% | 48.9% | DISPUTE |

The same model on the same features produced a 49% calibrated probability (DISPUTE) once the inference-time fill matched what training-time engineer.py would have produced. Threshold-flipping move from PAY to DISPUTE on a ticket that actually won.

The validation that this isn't an arbitrary lift: the empirical training-set win rate for "violation = FAIL TO DSPLY MUNI METER RECPT, county = K (Kings/Brooklyn), license_type = PAS" is **61.8%** across 21,685 training rows. The model with rolling priors filled said 62.7% raw. They match within a percentage point — meaning the model has correctly learned the conditional rate for that ticket profile, and the rolling-prior collapse was simply hiding it.

This was the moment the diagnosis crystallized. The model is genuinely good. The inference pipeline has a bug. The fix is to make the inference-time feature row look like the training-time feature row.

---

## 9. Spot-check expansion: four tickets

One ticket isn't enough. Three more spot-check candidates, mixing wins and losses:

**Ticket 9070812812** (Brooklyn meter, Montague St, $35, May 2023, won)
- Original: 3.0% raw / 1.1% calibrated → PAY (wrong)
- Filled: 64.0% raw / 36.4% calibrated → DISPUTE (correct)

**Ticket 9109263937** (Queens no-standing, 43rd Ave, $115, Feb 2024, lost)
- Original: 3.2% raw / 1.1% calibrated → PAY (correct, by accident)
- Filled: 52.8% raw / 28.7% calibrated → PAY (correct, *because* 28.7% is below the 32% threshold)

**Ticket 9093590236** (Manhattan expired-inspection sticker, Prince St, $65, Nov 2023, lost)
- Original: 1.0% raw / 0.2% calibrated → PAY (correct, by accident)
- Filled: 23.4% raw / 7.6% calibrated → PAY (correct, *much* further below threshold)

Combined scorecard:

| Ticket | Type | Outcome | Original | Filled | Filled cal % |
|---|---|---|---|---|---:|
| 9126107983 | BK meter | WON | PAY ❌ | DISPUTE ✅ | 36.4% |
| 9070812812 | BK meter | WON | PAY ❌ | DISPUTE ✅ | 36.4% |
| 9109263937 | QN no-standing | LOST | PAY ✅ | PAY ✅ | 28.7% |
| 9093590236 | MN expired-insp | LOST | PAY ✅ | PAY ✅ | 7.6% |

Original pipeline: 2/4 correct, both right-by-accident on losers (the constant ~1% just happens to land below threshold for everything).

Filled pipeline: 4/4 correct. And the magnitudes correctly graded the strength of dispute:
- Brooklyn meter winners: ~36%
- Queens no-standing loser: ~29% (close to threshold — borderline case)
- Manhattan expired-inspection loser: ~8% (far from threshold — strongest loss case)

The methodology says meter violations are "frequently dismissed" and inspection-sticker violations are nearly impossible to dispute (you either have a valid sticker or you don't). The filled-pipeline magnitudes match that ordering. The original pipeline collapsed all four to ~1% and lost the ranking entirely — which contradicts the AUC=0.90 finding from section 5. Resolution: AUC measures *relative* ranking, and the rolling-prior NaN signal is approximately a constant additive shift in log-odds, so it preserves order across rows. The constant shift collapses *absolute* probability into a narrow band, but doesn't fundamentally break ranking. That's why AUC stayed at 0.90 while every prediction sat at ~0.5%.

Four spot-checks aren't a definitive validation. The next step had to be distribution-level.

---

## 10. Distribution-level validation: 998-row batch with population-mean fill

`test_batch_with_fix.py` re-fetched the same training-matching 998-row live batch from section 5, scored each ticket twice (once with rolling priors NaN, once with population means filled), and reported AUC, ECE, F1, precision, recall, calibration table.

| Metric | Broken | Filled |
|---|---:|---:|
| AUC | 0.9056 | 0.9006 |
| **ECE** | **0.1818** | **0.0923** |
| **F1** | **0.000** | **0.564** |
| Precision (P[won \| DISPUTE]) | 0% | **58.6%** |
| Recall (P[DISPUTE \| won]) | 0% | **54.3%** |
| % disputed | 0.2% | 17.4% |
| Median calibrated probability | 0.08% | 10.2% |
| Accuracy at threshold | 81.0% | 84.2% |

Key takeaways:

1. **AUC barely moves** (0.906 → 0.901). Confirms the diagnosis: the rolling-prior NaN is a roughly constant additive log-odds shift, so ranking is preserved.
2. **F1 jumps from 0 to 0.564.** The broken pipeline never made a confident DISPUTE recommendation in 998 tickets. The fixed pipeline disputes 17.4% of tickets, and **58.6% of those actually win** — a 3.1× lift over the 18.8% base rate.
3. **ECE halves** (0.18 → 0.09). Still high (held-out test was 0.023), but the calibration table shows the predicted bins now monotonically track empirical win rate:

| Bin | Count | Predicted (filled) | Empirical |
|---|---:|---:|---:|
| [0.00, 0.05) | 111 | 4.5% | 0.0% |
| [0.05, 0.15) | 500 | 8.8% | 2.0% |
| [0.15, 0.30) | 141 | 23.3% | 29.8% |
| [0.30, 0.50) | 229 | 37.9% | 54.1% |
| [0.50, 0.85) | 17 | 57.2% | 70.6% |

The shape is roughly right — predicted increases as empirical increases. The upper bins systematically *under*-predict (predicts 38% on a population that wins 54%). That's because the population-mean fill is a stand-in: it doesn't match ticket-specific rolling priors. A ticket where the actual plate has a strong dispute history would score higher than the population-mean fill suggests. The isotonic calibrator, fit on training data with proper rolling priors, has less information per row than its training-time inputs assumed.

But the operational claim — model is now usable for individual ticket decisions — was solid.

---

## 11. The production fix

The proof-of-concept fix in `test_batch_with_fix.py` worked. The next step was to wire it into the actual inference scripts so it ran in production for every ticket.

The design:

**(a) `src/build_rolling_prior_means.py`** — one-shot script that scans `features.csv` and produces `models/rolling_prior_means.joblib`. Computes:

- Population mean and median for each of the 27 rolling-prior columns.
- Population mean for the three `days_since_*` columns, with overrideable defaults (365 days for plate-level, 30 days for issuer-level — matching the typical activity profile).
- A defensible default for `issuer_bayes_rate` (more on this in a moment).
- The training-set base rate (mean of `won` across all 1M rows = 0.236).

The artifact is built once after every training-data refresh. A future maintainer should run `python -m src.build_rolling_prior_means` whenever the model retrains.

**(b) `predict._load_rolling_prior_means()`** — loads the artifact, returns None if missing. `predict._apply_rolling_prior_fallback(df, fallback)` — fills NaN cells in the listed columns with their stored values. The function operates on the feature DataFrame in-place, returns a count for diagnostic logging.

**(c) Wire into `predict.py`** — after building the per-ticket feature row but before categorical-string normalization, call the fallback. Print a warning if the artifact is missing. Continue with calibration and threshold logic unchanged.

**(d) Wire into `predict_ensemble.py`** — apply the fallback to each of the three model rows (CatBoost, LightGBM, XGBoost) before encoder transforms. All three trained on the same `features.csv`, so all three have the same rolling-prior dependency.

That's the whole design. ~80 lines of code total, no new dependencies, no retraining required.

---

## 12. The `issuer_bayes_default` subtlety

The first version of `build_rolling_prior_means.py` used the training-set mean of the `issuer_bayes_rate` column as its fallback for unseen issuers. That mean was **0.2658**.

This sounded reasonable but turned out to be wrong, in a small but verdict-flipping way. Re-running the four-ticket spot check with the artifact-driven fallback:

| Ticket | Outcome | Verdict (training-mean issuer_bayes) |
|---|---|---|
| 9126107983 | WON | DISPUTE ✓ |
| 9070812812 | WON | **PAY** ✗ — 31.84% calibrated, just below 32% threshold |
| 9109263937 | LOST | PAY ✓ |
| 9093590236 | LOST | PAY ✓ |

3/4. The borderline winner flipped back to PAY because `issuer_bayes_rate = 0.2658` (the artifact's value) pushed the model's prediction down vs the 0.18 I'd been using in the proof-of-concept tests.

Why was 0.27 the wrong number? `engineer.py` defines `issuer_bayes_rate` as `(wins + alpha × global_mean) / (count + alpha)` — i.e., a smoothed posterior win rate per issuer. For an issuer with `count = 0` (an issuer never seen in training), this collapses to `global_mean` — the training base rate. That base rate, computed across all training rows, is **0.2362**.

The 0.27 figure comes from a different quantity: the *ticket-weighted* mean of `issuer_bayes_rate` across all training rows. Issuers with many tickets contribute more to that mean, and high-volume issuers happen to have systematically different `issuer_bayes_rate` from the base rate. So 0.27 = "average issuer_bayes_rate weighted by ticket count" ≠ 0.236 = "global win rate."

The principled fallback for a *new* (unseen) issuer is what `engineer.py` would assign: the base rate. Switching the artifact builder to use `df["won"].mean()` instead of `df["issuer_bayes_rate"].mean()` produced `issuer_bayes_default = 0.2362`, and the spot check returned to 4/4:

| Ticket | Outcome | Verdict (base-rate issuer_bayes) |
|---|---|---|
| 9126107983 | WON | DISPUTE ✓ (cal 48.9%) |
| 9070812812 | WON | **DISPUTE ✓** (cal 36.4%) |
| 9109263937 | LOST | PAY ✓ (cal 28.7%) |
| 9093590236 | LOST | PAY ✓ (cal 7.6%) |

The 998-row distribution-level numbers matched the proof-of-concept exactly: F1 0.564, ECE 0.092, AUC 0.9006, 102/188 wins caught at 58.6% precision.

This is the kind of thing that doesn't show up in aggregate metrics but matters at the boundary. Borderline cases sit close to the threshold and are sensitive to small shifts in any feature. Using the wrong fallback for `issuer_bayes_rate` could have been "fine enough" on aggregate (F1 wouldn't have moved much) but produces a verdict regression on a known winning ticket — which is the thing the user actually feels.

The fix shipped with `issuer_bayes_default = base_rate`. The artifact also stores both the `issuer_bayes_means` (the wrong-but-sometimes-useful number) and `issuer_bayes_default` (the right one) so future maintainers can see the choice.

---

## 13. End-to-end verification

After wiring the fix into `predict.py` and `predict_ensemble.py`, the verification was straightforward: run the same 998-row batch through the *patched* pipeline (rather than through the proof-of-concept inline fill) and confirm the numbers match.

They did exactly:

| Metric | PoC (test_batch_with_fix.py) | Patched (test_masked_vs_others.py B-column) |
|---|---:|---:|
| AUC | 0.9006 | 0.9006 |
| ECE | 0.0923 | 0.0923 |
| F1 | 0.564 | 0.564 |
| Precision | 58.6% | 58.6% |
| Recall | 54.3% | 54.3% |
| TP / FN / FP / TN | 102 / 86 / 72 / 738 | 102 / 86 / 72 / 738 |

Identical to the basis point. The integration is clean.

The four-ticket spot check through the patched pipeline:

| Ticket | Outcome | Raw | Calibrated | Verdict | Correct? |
|---|---|---:|---:|---|---|
| 9126107983 | WON | 73.08% | 48.87% | DISPUTE | ✓ |
| 9070812812 | WON | 64.04% | 36.38% | DISPUTE | ✓ |
| 9109263937 | LOST | 52.77% | 28.66% | PAY | ✓ |
| 9093590236 | LOST | 23.39% | 7.61% | PAY | ✓ |

4/4. The production fix shipped.

---

## 14. The MaskTab attempt: trying for the proper fix

The shipped fix is a stand-in. ECE is 0.092 vs the held-out test set's 0.023 — about 4× worse. Upper bins under-predict by 15+ percentage points. Population-mean fill doesn't capture ticket-specific rolling priors.

The proper fix has three options, each requiring a retrain. Listed in LIMITATIONS.md from cheapest to most expensive:

1. **MaskTab-style training** — randomly NaN out the rolling-prior block on a fraction of training rows so the model learns to perform without them.
2. **Drop the features entirely** — equivalent to mask probability 1.0; cleanest pipeline, lose any signal the features carry when present.
3. **Compute rolling priors at inference** — build `(entity, date)` lookup tables from the full corpus; query at inference. Most faithful but adds multi-GB state.

Option 1 has the best theoretical story: the model retains the ability to use rolling priors when they're available (training rows where they aren't masked) but also learns a viable inference path when they're absent (the masked half). At deployment, where rolling priors are always NaN, the model uses the "absent" path.

I built this in `src/train_masked.py`. The implementation:

- Reuse `train.py`'s loading and chronological-split logic.
- After splitting train/test (test stays unmasked) and carving the eval slice, mask the rolling-prior block on each row independently with probability 0.5. Block-level masking: when a row is selected, all 27 rolling-prior columns go to NaN simultaneously (matches the inference scenario where they're all NaN together).
- Apply the same masking to the eval slice (so early-stopping criterion reflects the deployed regime).
- Reuse `models/best_params.joblib` from the unmasked training. Skip Optuna re-tuning (each Optuna trial is 5–10 minutes; 80 trials is too long).
- Final fit at 5000-iter cap with 100-round early stopping.
- Save artifacts with `_masked` suffix (`catboost_model_masked.cbm`, `isotonic_calibrator_masked.joblib`, `dispute_threshold_masked.json`, `data/test_set_masked.joblib`) so the original artifacts stay intact for A/B comparison.
- Refit the calibrator on the masked-eval slice's outputs.
- Re-tune the threshold on calibrated outputs from the *fully-masked* test set (the deployed regime).

Training ran for 390 iterations before early stopping (vs the original's longer training). Best eval AUC during training: 0.8613. Total wall-clock: ~5 minutes on the RTX 5070 Ti.

The internal results were genuinely encouraging:

| Metric | Original | Masked (unmasked test) | Masked (fully-masked test) |
|---|---:|---:|---:|
| Test AUC | 0.8694 | **0.8699** | **0.8662** |
| AUC gap from masking | – | – | 0.0036 |
| Test ECE (calibrated) | 0.0234 | **0.0191** | **0.0197** |
| Brier (calibrated) | 0.1012 | 0.0993 | 0.1001 |
| F1-optimal threshold (masked test) | – | – | 0.31 |
| F1 at threshold (masked test) | – | – | 0.5843 |

The masked model had **better** calibration than the original on the held-out test set, AUC was preserved within 0.0036 even when scoring with all rolling priors masked, and the F1-optimal threshold barely moved (0.32 → 0.31).

This looked like a clean win on internal metrics. The expectation going into the live-batch validation was: ECE on live should drop from 0.092 (stand-in fallback) to ~0.03 (matching internal test ECE).

---

## 15. The MaskTab attempt: the negative result

`test_masked_vs_others.py` ran the same 998-row live batch through three pipelines:

A. Original model, no fallback (the broken state pre-investigation).
B. Original model + stand-in fallback (the shipped production fix).
C. Masked model, no fallback (the masked retrain expected to make B obsolete).

| Metric | A) Broken | B) Stand-in (shipped) | C) Masked retrain |
|---|---:|---:|---:|
| AUC | 0.9056 | **0.9006** | 0.8764 |
| ECE | 0.1818 | **0.0923** | 0.1314 |
| F1 | 0.000 | **0.564** | 0.061 |
| Precision | 0% | 58.6% | 60.0% |
| Recall | 0% | **54.3%** | 3.2% |
| % disputed | 0.2% | 17.4% | 1.0% |
| Median calibrated | 0.08% | 10.19% | 1.61% |
| Wins caught (TP / FN) | 0 / 188 | **102 / 86** | 6 / 182 |

The masked retrain underperformed the stand-in fallback on every operational metric. F1 0.061 vs 0.564, recall 3% vs 54%. AUC was *worse* than the broken pipeline. ECE was halfway between broken and stand-in.

The calibration table on the masked variant for the live batch:

| Bin | Count | Predicted (masked) | Empirical |
|---|---:|---:|---:|
| [0.00, 0.05) | 612 | 0.6% | 2.0% |
| [0.05, 0.15) | 283 | 10.5% | 42.4% |
| [0.15, 0.30) | 93 | 21.8% | 53.8% |
| [0.30, 0.50) | 6 | 38.9% | 66.7% |
| [0.50, 0.85) | 4 | 66.8% | 50.0% |

Better than the broken pipeline (which had 992/998 in the bottom bin) but worse than the stand-in (which had 111/998 in the bottom bin). The [0.05, 0.15) bin under-predicts catastrophically: 10.5% mean vs 42.4% empirical. The masked model's "no-rolling-priors" branch is not well-calibrated to the live distribution.

Why didn't this work? Three plausible contributors, each of which I'd want to test if I were continuing:

1. **Mask probability 0.5 was too low.** Half the training rows still have real rolling priors. The model can still learn signals from the unmasked half — and apparently learned them in a way that doesn't transfer when *every* inference-time row is masked. A higher mask probability (0.8–0.9, or 1.0 = drop the features entirely) would force the model to rely less on rolling priors during training.
2. **Reused hyperparameters from the unmasked Optuna run.** `best_params.joblib` was tuned on the unmasked-distribution training. Under masking, the optimal depth, learning rate, regularization, and growth policy are likely different. A separate Optuna search with masking enabled would address this — but adds 30+ minutes.
3. **Calibrator-distribution mismatch.** The calibrator was fit on the masked eval slice (50% NaN). Live data is 100% NaN. The raw-probability distributions are different, so the isotonic mapping might be suboptimal at inference. Refitting the calibrator on a fully-masked slice (eval with rolling priors zeroed out, not 50%-zeroed) would address this.

Note that the *internal* test ECE (0.019) was better than the original. The masked model's calibration is good when the *test distribution* matches the *eval distribution* it was calibrated on. It's the live distribution shift that's still misaligned. The chronological test slice has its own quirks but it's much closer to training than the live API is.

The honest takeaway: **MaskTab is not a free win**. The literature framing (e.g., "force the model to handle missing features") works in theory; in practice, parameter choices matter a lot, and a single try at mask_prob=0.5 with reused hyperparameters can produce a regression. My earlier prediction in LIMITATIONS.md ("Expected ECE recovery: 0.092 → ~0.03 via MaskTab") was a confident projection that didn't survive empirical contact. That claim was retracted in the docs.

The masked artifacts were kept for diagnostic comparison but not promoted to canonical filenames. Production continues to use `catboost_model.cbm` + `rolling_prior_means.joblib` + `isotonic_calibrator.joblib` + `dispute_threshold.json` (default policy F1 = 0.32). The masked variants live alongside as `_masked.*` files.

---

## 16. What's actually in the codebase now

After the investigation, the following changes ship:

**New files**

- `src/build_rolling_prior_means.py` — one-shot artifact builder. Run after every training-data refresh.
- `src/train_masked.py` — MaskTab-style retrain driver (kept for future experiments; not currently producing a promoted model).
- `models/rolling_prior_means.joblib` — population-mean fallback artifact.
- `models/catboost_model_masked.cbm`, `models/isotonic_calibrator_masked.joblib`, `models/dispute_threshold_masked.json`, `data/test_set_masked.joblib` — masked-retrain artifacts (diagnostic only).

**Modified files**

- `predict.py` — added `_load_calibrator()`, `_load_threshold()`, `_load_rolling_prior_means()`, `_apply_rolling_prior_fallback()`. Wired all four into `predict_ticket()`. The function now applies isotonic calibration after raw scoring, reads the threshold from JSON, and fills NaN rolling priors before scoring. Output prints both raw and calibrated probabilities.
- `predict_ensemble.py` — same calibrator/threshold/fallback wiring, applied to all three model rows.
- `LIMITATIONS.md` — §"Probability threshold" rewritten to reflect that calibration is fit and auto-applied. New §"Live-data drift" (renamed from "Live-API drift" to "Live-data drift: rolling priors not computable at inference") with the diagnosis, shipped fix, and the negative MaskTab result.
- `CLAUDE.md` — ground rules #6 (calibration) and #7 (rolling-prior fallback) updated to reflect the new state.

**Diagnostic / one-off scripts left at the repo root**

- `test_hypothetical_tickets.py` — synthetic ticket scoring (the original probe).
- `test_holdout_anchor.py` — full 200k held-out scoring + calibration table.
- `test_calibrated_comparison.py` — shows raw vs calibrated on synthetic and held-out side-by-side.
- `test_threshold_oos.py` — chronological half-and-half threshold tuning.
- `test_live_nyc_tickets.py` — fresh-API scoring (still uses the old "rolling priors stay NaN" path; useful as a baseline reproducer).
- `test_batch_with_fix.py` — 998-row batch with broken-vs-PoC-fill comparison (predates the wired-up production fix).
- `test_masked_vs_others.py` — three-way comparison (broken / shipped / masked).

These can be deleted, kept, or moved to a `tools/` directory. They were how the investigation got done.

---

## 17. Validation summary

The shipped fix has been validated at three scales:

**Single-ticket spot check (4 tickets, 2 outcomes):**

| Ticket | Outcome | Production fix verdict | Correct? |
|---|---|---|---|
| 9126107983 | WON | DISPUTE | ✓ |
| 9070812812 | WON | DISPUTE | ✓ |
| 9109263937 | LOST | PAY | ✓ |
| 9093590236 | LOST | PAY | ✓ |

4/4 vs 2/4 for the broken pipeline. The two correct answers from the broken pipeline were both right-by-accident on losers (the constant ~1% just happens to be below the threshold).

**Distribution-level validation (998 fresh tickets, training-matching filter):**

| Metric | Before fix | After fix |
|---|---:|---:|
| AUC | 0.9056 | 0.9006 |
| ECE | 0.182 | 0.092 |
| F1 | 0.000 | 0.564 |
| Precision | 0% | 58.6% |
| Recall | 0% | 54.3% |
| Wins caught (TP / FN) | 0 / 188 | 102 / 86 |
| Lift over base rate | 0× | 3.1× |

**Failed alternative (MaskTab retrain at mask_prob=0.5):**

| Metric | Stand-in (shipped) | Masked retrain |
|---|---:|---:|
| AUC | 0.9006 | 0.8764 |
| ECE | 0.092 | 0.131 |
| F1 | 0.564 | 0.061 |
| Recall | 54.3% | 3.2% |

The stand-in is the empirically validated production choice.

---

## 18. Lessons

A few things worth carrying forward to similar investigations:

**Time-dependent features are operationally fragile.** Features like `plate_prior_win_rate_30D` that require running tallies are inexpensive to compute in batch on a fixed training corpus and expensive to compute correctly at inference. The model learns to lean on them when they're cheap, then fails when they're hard. If you add a feature that requires state at inference, build the inference-time path immediately — don't defer it. If you can't build the inference-time path, consider whether the feature is worth its production cost.

**Calibration is not the same as ranking.** AUC stayed at 0.90 on the live batch through every variant — broken, fixed, masked. The model's *ability to order tickets by win likelihood* survived everything. What broke was the absolute probability claim. For ranking use cases ("of my N pending tickets, which is most worth disputing?"), the model worked the whole time. For threshold-based decisions ("does this ticket have >32% chance of winning?"), the model needed the fallback. These two use cases have different reliability profiles and probably deserve different documentation.

**Internal test sets can mislead about production performance.** The held-out chronological test set is by construction the closest available proxy for production. But "closest available" can still miss real production behavior. The original model had ECE 0.023 on internal test and 0.18 on the live API. The masked model had ECE 0.019 on internal test and 0.13 on the live API. The internal-test number isn't predictive of the live-API number unless every distribution-shift dimension is monitored.

**Spot checks reveal what aggregate metrics hide.** The aggregate AUC of 0.90 on the live batch looked acceptable. The individual ticket review (1% on a winning ticket) revealed the calibration disaster. Both signals were available; only the spot check made the failure feel real enough to investigate. When a model has high AUC and bad calibration simultaneously, aggregate metrics can lull you into thinking it's working.

**SHAP attribution is the cleanest way to find this kind of bug.** Per-prediction SHAP values produced an unambiguous answer in one query: `rolling priors contribute -3.687 log-odds, everything else is roughly neutral`. Without SHAP, the diagnosis would have been a much longer process of feature ablation.

**MaskTab-style fixes need careful tuning to land.** Random-mask training works in principle but is sensitive to mask probability, hyperparameter choice, and calibrator-distribution alignment. The headline finding from the literature ("force the model to handle missing features") is true but operationally underspecified. A working implementation needs at minimum: a thoughtful mask probability matched to the inference distribution, hyperparameters re-tuned under masking, and a calibrator fit on the deployed regime. Any of these omitted (and we omitted all three) and the outcome is a regression vs the simpler stand-in fix.

**A stand-in fix can be the right answer.** I expected to write this document with MaskTab as the headline solution and the population-mean fallback as a temporary patch. The numbers said otherwise. The patch is operationally adequate (3× lift, F1 0.56) and required no retraining. The "proper" fix needed at least one more iteration of retraining experiments — and even after that, the patch might still be competitive. Stand-ins aren't always inferior; sometimes they're load-bearing.

**Some claims should be promoted from "assumed" to "verified."** The original LIMITATIONS.md said calibration was "not verified" when in fact `isotonic_calibrator.joblib` was already fit and producing ECE 0.023 on the held-out test. That kind of stale claim sits in a repo silently misleading readers until someone audits it. Worth a periodic doc audit against the artifacts.

---

## 19. Open questions

What's not yet known:

- **Would a higher mask probability work?** The 0.5 mask was a guess. 0.8 or 1.0 might force the model to actually learn the no-rolling-priors regime; I haven't tested.
- **Would dropping rolling priors entirely + retraining beat the stand-in?** Cleanest experiment, hasn't been run.
- **Would a re-tuned Optuna pass under masking change the picture?** Yes, probably, but at 30+ minutes per try. Worth scoping if anyone wants to commit time to it.
- **What's the live-batch performance on a model retrained without rolling priors at all?** Equivalent to mask_prob=1.0; the masked model's deployed-regime test ECE (0.0197) suggests it could match or beat the stand-in.
- **Why does the live distribution under-predict in the upper calibrated bins?** The bin [0.30, 0.50) shows mean 38% but empirical 54%. The stand-in fallback has less information than ticket-specific rolling priors would, so the model is less confident than the true conditional rate. The proper fix is computing rolling priors at inference (option 3 in LIMITATIONS.md), but it's the most expensive option and hasn't been built.

The investigation in this document covers the bug discovery, diagnosis, fix, and one failed alternative. It doesn't cover those open questions — they're the work that's still owed if someone wants live-API ECE to match the held-out test ECE.

---

## 20. Reproducing this investigation

If you want to reproduce any specific finding:

```bash
# Synthetic-ticket experiment (section 1)
python test_hypothetical_tickets.py

# Held-out test calibration (section 2)
python test_holdout_anchor.py
python test_calibrated_comparison.py

# Threshold OOS check (section 3)
python test_threshold_oos.py

# Live-API single tickets (section 6)
# (see the script for the four spot-check summons numbers)
python test_live_nyc_tickets.py

# Distribution-level validation (section 10, 13)
python test_batch_with_fix.py

# Three-way masked comparison (section 15)
python test_masked_vs_others.py

# Rebuild the production artifact
python -m src.build_rolling_prior_means

# Re-run the masked training (section 14)
python -m src.train_masked
```

All scripts read from `data/violations_raw.csv`, `data/features.csv`, and the `models/` directory. Live-API tests pull fresh data from NYC Open Data over HTTP. Numbers will drift slightly between runs because the API changes daily, but the qualitative picture should reproduce.

---

*This investigation took place over a single working session in May 2026. The diagnosis was unblocked by a single SHAP query that took under a minute to run; the production fix was 80 lines of code; the unsuccessful retrain was about an hour of training plus validation. The bulk of the time went to building intuition and validating each finding before moving to the next. The codebase was good. The investigation was about confirming what was already there and finding the one thing that wasn't.*
