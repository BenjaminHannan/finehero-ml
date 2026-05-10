# test_hypothetical_tickets.py
#
# Score a small set of HAND-CRAFTED hypothetical tickets through the trained
# CatBoost model. None of these tickets exist in violations_raw.csv — they are
# synthetic scenarios chosen because the FineHero methodology gives a clear
# directional prediction for each.
#
# For each scenario we print:
#   - the model's win probability
#   - the methodology's predicted direction (HIGH / LOW / MID)
#   - whether the model agrees
#
# This is a sanity check, not a calibration test. We want the model to
# rank scenarios in roughly the order the methodology claims.

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Use the trained model from the parent (non-worktree) repo
MODELS_DIR = r"C:\Users\benja\Downloads\finehero-ml\models"
CATBOOST_PATH      = os.path.join(MODELS_DIR, "catboost_model.cbm")
METADATA_PATH      = os.path.join(MODELS_DIR, "metadata.json")
CAT_FEATURES_PATH  = os.path.join(MODELS_DIR, "cat_features.joblib")
PLATE_HISTORY_PATH = os.path.join(MODELS_DIR, "plate_history_map.joblib")

DISPUTE_THRESHOLD = 0.40


def cyclical(values, base, period):
    """Return (sin, cos) cyclical encoding, or (0,0) if value missing."""
    out = {}
    for label, v in values.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or v < 0:
            out[f"{label}_sin"] = 0.0
            out[f"{label}_cos"] = 0.0
        else:
            two_pi = 2.0 * np.pi
            out[f"{label}_sin"] = float(np.sin(two_pi * v / period))
            out[f"{label}_cos"] = float(np.cos(two_pi * v / period))
    return out


def build_row(*, vcode, precinct, county, agency, license_type, state,
              street_name, vehicle_make, vehicle_body_type, vehicle_year,
              fine_amount, hour, day_of_week, month, is_holiday,
              from_hour, to_hour, days_in_effect,
              feet_from_curb, side,
              precip_mm, visibility_m, wind_mps, is_bad_weather,
              kw_flags,
              plate_prior_count, plate_prior_win_rate,
              precinct_global_win_rate=0.30,
              issuer_global_win_rate=0.30):
    """Construct a feature dict matching the trained model's schema.

    Anything we don't have a sensible value for is left NaN — CatBoost
    handles missing values natively for numerics, and 'UNKNOWN' for
    categoricals matches the training-time fallback.
    """
    # within_posted_hours: handles overnight ranges
    within = np.nan
    if hour is not None and from_hour is not None and to_hour is not None:
        if from_hour <= to_hour:
            within = float(from_hour <= hour <= to_hour)
        else:
            within = float(hour >= from_hour or hour <= to_hour)

    cyc = {}
    cyc.update(cyclical({"hour": hour}, base=0, period=24.0))
    cyc.update(cyclical({"dow": day_of_week}, base=0, period=7.0))
    cyc.update(cyclical({"month": month}, base=0, period=12.0))

    row = {
        # Core categoricals
        "violation_code": str(vcode),
        "precinct": str(precinct),
        "county": str(county),
        "issuing_agency": str(agency),
        "license_type": str(license_type),
        "state": str(state),
        "issuer_code": "UNKNOWN",
        "issuer_command": "UNKNOWN",
        "issuer_squad": "UNKNOWN",
        "street_name": str(street_name).upper(),
        "vehicle_make": str(vehicle_make).upper(),
        "vehicle_body_type": str(vehicle_body_type).upper(),
        "violation_legal_code": "UNKNOWN",
        "law_section": "UNKNOWN",
        "sub_division": "UNKNOWN",
        "violation_in_front_of_or_opposite": str(side).upper(),
        "days_parking_in_effect": str(days_in_effect).upper(),
        # Cross-features
        "viol_x_precinct": f"{vcode}_{precinct}",
        "viol_x_license": f"{vcode}_{license_type}",
        "hour_x_dow": f"{int(hour) if hour is not None else -1}_{int(day_of_week) if day_of_week is not None else -1}",
        "summons_format": "UNKNOWN",
        "format_x_viol": "UNKNOWN",
        "hours_kind_from": "UNKNOWN",
        "hours_kind_to": "UNKNOWN",
        "agency_x_command": f"{agency}_UNKNOWN",
        "format_x_missing_plate": "UNKNOWN",
        # Temporal numerics
        "hour_of_offense": float(hour) if hour is not None else np.nan,
        "day_of_week": float(day_of_week) if day_of_week is not None else np.nan,
        "month": float(month) if month is not None else np.nan,
        "is_holiday": float(is_holiday),
        # Cyclical
        **cyc,
        # Financial
        "fine_amount": float(fine_amount),
        # pvqr numerics
        "feet_from_curb": float(feet_from_curb) if feet_from_curb is not None else np.nan,
        "vehicle_year": float(vehicle_year) if vehicle_year is not None else np.nan,
        "from_hour": float(from_hour) if from_hour is not None else np.nan,
        "to_hour": float(to_hour) if to_hour is not None else np.nan,
        "within_posted_hours": within,
        "hours_in_effect_all": np.nan,
        "first_observed_filled": np.nan,
        # Weather
        "precipitation": float(precip_mm),
        "visibility": float(visibility_m),
        "wind_speed": float(wind_mps),
        "is_bad_weather": float(is_bad_weather),
        # Plate priors
        "plate_prior_ticket_count": float(plate_prior_count),
        "plate_prior_win_rate": float(plate_prior_win_rate),
        "issuer_bayes_rate": float(issuer_global_win_rate),
        "days_since_plate_last_ticket": np.nan,
        "days_since_plate_last_win": np.nan,
        "days_since_issuer_last_ticket": np.nan,
        # Keyword flags
        **{f"kw_{k}": float(kw_flags.get(k, 0)) for k in
           ("meter", "hydrant", "bus_stop", "sign", "blocking", "expired")},
        # 30/90/365D priors (unknown for synthetic plates → NaN)
        **{f"{prefix}_prior_{stat}_{w}D": np.nan
           for prefix in ("plate", "precinct", "issuer")
           for stat in ("wins", "count", "win_rate")
           for w in (30, 90, 365)},
        # Missing-field flags
        "is_missing_vehicle_make": 0.0,
        "is_missing_vehicle_body_type": 0.0,
        "is_missing_street_name": 0.0,
        "is_missing_issuer_command": 1.0,  # we don't have one
        "is_missing_issuer_squad": 1.0,
        "is_missing_violation_legal_code": 1.0,
        "is_missing_days_parking_in_effect": 0.0,
        "is_missing_vehicle_year": 0.0,
        "is_missing_plate": 0.0,
        "missing_fields_count": 3.0,
    }
    return row


def make_scenarios():
    """Each scenario is (label, expected_direction, methodology_reason, row_dict)."""
    base_kw = {"meter": 0, "hydrant": 0, "bus_stop": 0, "sign": 0, "blocking": 0, "expired": 0}

    common = dict(
        county="MANHATTAN",
        agency="TRAFFIC",
        license_type="PAS",
        state="NY",
        street_name="BROADWAY",
        vehicle_make="HONDA",
        vehicle_body_type="SUBN",
        vehicle_year=2020,
        feet_from_curb=12,
        side="F",
        days_in_effect="BBBBBBB",
    )

    scenarios = []

    # 1. Strong WIN: ticket issued OUTSIDE posted hours-in-effect.
    #    Methodology: within_posted_hours == 0 is one of the strongest dismissal signals.
    scenarios.append((
        "outside_posted_hours",
        "HIGH",
        "within_posted_hours=0 — methodology calls this 'one of the strongest dismissal signals'",
        build_row(
            **common,
            vcode="21", precinct="14",
            fine_amount=65,
            hour=23, day_of_week=2, month=6, is_holiday=0,
            from_hour=8, to_hour=18,            # posted 8AM-6PM, ticket at 11PM
            precip_mm=0, visibility_m=10000, wind_mps=2, is_bad_weather=0,
            kw_flags={**base_kw, "sign": 1},
            plate_prior_count=2, plate_prior_win_rate=0.35,
        ),
    ))

    # 2. Strong WIN: sign-related violation in bad weather, fresh plate.
    scenarios.append((
        "sign_violation_in_snow",
        "HIGH",
        "kw_sign=1 + is_bad_weather=1 — sign violations are routinely dismissed; bad weather supports 'signage obscured'",
        build_row(
            **common,
            vcode="38", precinct="20",
            fine_amount=115,
            hour=10, day_of_week=2, month=1, is_holiday=0,
            from_hour=7, to_hour=19,
            precip_mm=8.5, visibility_m=1200, wind_mps=8, is_bad_weather=1,
            kw_flags={**base_kw, "sign": 1},
            plate_prior_count=0, plate_prior_win_rate=0.30,
        ),
    ))

    # 3. Strong LOSS: blocking driveway, clear day, within posted hours.
    scenarios.append((
        "blocking_driveway_clear_day",
        "LOW",
        "kw_blocking=1, weather clear, within posted hours — blocking violations rarely get dismissed",
        build_row(
            **common,
            vcode="74", precinct="13",
            fine_amount=95,
            hour=14, day_of_week=2, month=5, is_holiday=0,
            from_hour=0, to_hour=24,            # always in effect
            precip_mm=0, visibility_m=10000, wind_mps=2, is_bad_weather=0,
            kw_flags={**base_kw, "blocking": 1},
            plate_prior_count=8, plate_prior_win_rate=0.10,   # repeat loser
        ),
    ))

    # 4. Strong LOSS: bus stop violation, clean weather, repeat offender plate.
    scenarios.append((
        "bus_stop_repeat_offender",
        "LOW",
        "kw_bus_stop=1 in posted hours + plate prior win rate well below 0.30",
        build_row(
            **common,
            vcode="51", precinct="19",
            fine_amount=115,
            hour=8, day_of_week=1, month=9, is_holiday=0,
            from_hour=7, to_hour=19,
            precip_mm=0, visibility_m=10000, wind_mps=3, is_bad_weather=0,
            kw_flags={**base_kw, "bus_stop": 1},
            plate_prior_count=15, plate_prior_win_rate=0.05,
        ),
    ))

    # 5. MID-HIGH: meter violation, slight rain, plate has decent history.
    scenarios.append((
        "meter_violation_drizzle",
        "MID-HIGH",
        "kw_meter=1 — methodology says meters are 'frequently dismissed', but conditions otherwise normal",
        build_row(
            **common,
            vcode="37", precinct="14",
            fine_amount=65,
            hour=12, day_of_week=3, month=4, is_holiday=0,
            from_hour=9, to_hour=19,
            precip_mm=1.5, visibility_m=8000, wind_mps=4, is_bad_weather=0,
            kw_flags={**base_kw, "meter": 1},
            plate_prior_count=3, plate_prior_win_rate=0.45,
        ),
    ))

    # 6. Strong LOSS: hydrant violation, clear conditions.
    scenarios.append((
        "hydrant_clear_conditions",
        "LOW",
        "kw_hydrant=1 — hydrant violations rarely dismissed; safety-critical",
        build_row(
            **common,
            vcode="40", precinct="13",
            fine_amount=115,
            hour=15, day_of_week=4, month=7, is_holiday=0,
            from_hour=0, to_hour=24,
            precip_mm=0, visibility_m=10000, wind_mps=2, is_bad_weather=0,
            kw_flags={**base_kw, "hydrant": 1},
            plate_prior_count=1, plate_prior_win_rate=0.30,
        ),
    ))

    # 7. EDGE CASE: late-night ticket on holiday, sign violation.
    scenarios.append((
        "holiday_late_night_sign",
        "HIGH",
        "is_holiday=1 + kw_sign=1 + late-night when posting rules are ambiguous",
        build_row(
            **common,
            vcode="38", precinct="01",
            fine_amount=115,
            hour=23, day_of_week=6, month=12, is_holiday=1,
            from_hour=8, to_hour=20,
            precip_mm=0, visibility_m=9000, wind_mps=1, is_bad_weather=0,
            kw_flags={**base_kw, "sign": 1},
            plate_prior_count=0, plate_prior_win_rate=0.30,
        ),
    ))

    return scenarios


def main():
    print(" Loading model + metadata...")
    model = CatBoostClassifier()
    model.load_model(CATBOOST_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    print(f" Model: {meta['model_type']}  AUC {meta['auc_score']:.4f}  ({meta['row_count']:,} rows)")
    print(f" Threshold: {DISPUTE_THRESHOLD}\n")

    scenarios = make_scenarios()
    rows = []
    for label, _, _, row in scenarios:
        rows.append({f: row.get(f, np.nan) for f in feature_names})
    df = pd.DataFrame(rows)
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str).replace(
                {"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    probs = model.predict_proba(df)[:, 1]

    print(f" {'#':<3} {'Scenario':<32} {'Expected':<10} {'Predicted':<10} {'Verdict':<8} {'Match'}")
    print(" " + "-" * 88)
    matches = 0
    for i, ((label, expected, reason, _), p) in enumerate(zip(scenarios, probs), 1):
        verdict = "DISPUTE" if p >= DISPUTE_THRESHOLD else "PAY"
        # Match logic
        if expected == "HIGH":
            ok = p >= 0.45
        elif expected == "LOW":
            ok = p <= 0.30
        elif expected == "MID-HIGH":
            ok = 0.30 <= p <= 0.65
        else:
            ok = True
        matches += int(ok)
        mark = "OK" if ok else "MISS"
        print(f" {i:<3} {label:<32} {expected:<10} {p*100:>6.1f}%   {verdict:<8} {mark}")
    print(" " + "-" * 88)
    print(f" {matches}/{len(scenarios)} scenarios moved in the methodology-predicted direction.\n")

    print(" Per-scenario reasoning:")
    for i, ((label, expected, reason, _), p) in enumerate(zip(scenarios, probs), 1):
        print(f"  {i}. {label}  ({p*100:.1f}%)")
        print(f"     why expected {expected}: {reason}")


if __name__ == "__main__":
    main()
