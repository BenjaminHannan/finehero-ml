# fetch_data.py - Downloads NYC Open Data parking datasets and joins them:
#   1. Open Parking & Camera Violations (nc67-uf89) — has ticket outcome
#   2. Parking Violations Issued — FY2022, FY2023, FY2024, FY2025
#      These years are old enough that outcomes are fully adjudicated in nc67.
#      FY2026 (pvqr-7yc4) is skipped — too recent, most tickets still outstanding.
# All datasets are cached to disk so reruns skip network fetches.

import io
import os
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

OPEN_PARKING_URL = "https://data.cityofnewyork.us/resource/nc67-uf89.csv"

# Fiscal years with good adjudication coverage in nc67-uf89.
# FY runs July 1 -> June 30. FY2025 ended June 2025 so outcomes are mostly resolved.
PVQR_FISCAL_YEARS = {
    "FY2022": "7mxj-7a6y",
    "FY2023": "869v-vr48",
    "FY2024": "8zf9-spf8",
    "FY2025": "m5vz-tzqv",
}

VIOL_PATH      = os.path.join(DATA_DIR, "violations_raw.csv")
OUTCOMES_CACHE = os.path.join(DATA_DIR, "outcomes_raw.csv")
PVQR_CACHE     = os.path.join(DATA_DIR, "pvqr_raw.csv")
ADJ_PATH       = os.path.join(DATA_DIR, "adjudications_raw.csv")

DEFAULT_TARGET_ROWS = 500_000
PAGE_SIZE = 50_000

# Rows fetched per fiscal year. 4 years x 250k = 1M rows total -> maximises join coverage.
PVQR_ROWS_PER_YEAR = 250_000

# Rich feature columns we want from pvqr (all pre-hearing, no leakage).
PVQR_KEEP = [
    "summons_number",
    "issuer_code", "issuer_command", "issuer_squad",
    "street_name", "house_number", "intersecting_street",
    "vehicle_make", "vehicle_year", "vehicle_color", "vehicle_body_type",
    "violation_legal_code", "violation_description",
    "from_hours_in_effect", "to_hours_in_effect",
    "feet_from_curb", "violation_in_front_of_or_opposite",
    "law_section", "sub_division",
    "days_parking_in_effect",
    "violation_precinct",
]


def fetch_endpoint(url: str, target: int, label: str, extra_params: dict = None) -> pd.DataFrame:
    frames = []
    offset = 0
    print(f"  Fetching {label} (target {target:,} rows)...")
    while offset < target:
        limit = min(PAGE_SIZE, target - offset)
        params = {"$limit": limit, "$offset": offset}
        if extra_params:
            params.update(extra_params)
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if chunk.empty:
                print(f"    {label}: empty page at offset {offset:,}, stopping")
                break
            frames.append(chunk)
            print(f"    {label}: {offset + len(chunk):,} rows fetched")
            offset += len(chunk)
            if len(chunk) < limit:
                break
            time.sleep(0.3)
        except Exception as exc:
            print(f"  [WARN] API error for {label} at offset {offset:,}: {exc}")
            break

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def make_synthetic_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    issue_dates = pd.date_range("2018-01-01", periods=n, freq="30min")
    vtimes = [
        f"{h:02d}:{m:02d}{'A' if h < 12 else 'P'}"
        for h, m in zip(rng.integers(0, 23, n), rng.integers(0, 59, n))
    ]
    statuses = rng.choice(
        ["Paid", "Dismissed", "Reduction Granted", "Guilty", "Hearing Held - Guilty"],
        n,
        p=[0.50, 0.18, 0.05, 0.22, 0.05],
    )
    return pd.DataFrame({
        "summons_number": [f"S{i:010d}" for i in range(n)],
        "issue_date": rng.choice(issue_dates.strftime("%Y-%m-%dT%H:%M:%S"), n),
        "violation_time": vtimes,
        "violation": rng.integers(1, 99, n).astype(str),
        "fine_amount": rng.choice([35.0, 45.0, 65.0, 115.0, 180.0], n),
        "penalty_amount": rng.choice([0.0, 10.0, 25.0, 60.0], n),
        "reduction_amount": rng.choice([0.0, 15.0, 30.0], n),
        "payment_amount": rng.choice([0.0, 65.0, 115.0], n),
        "amount_due": rng.choice([0.0, 65.0, 115.0, 180.0], n),
        "precinct": rng.integers(1, 123, n).astype(str),
        "county": rng.choice(["NY", "BX", "BK", "QN", "ST"], n),
        "issuing_agency": rng.choice(["TRAFFIC", "POLICE", "SANITATION", "DOT"], n),
        "violation_status": statuses,
        "license_type": rng.choice(["PAS", "COM", "TRK", "OMS"], n),
        "plate": [f"ABC{rng.integers(1000,9999)}" for _ in range(n)],
        "state": rng.choice(["NY", "NJ", "CT", "PA", "FL"], n),
    })


def _migrate_old_cache() -> None:
    """
    If violations_raw.csv exists but has no pvqr columns (old format),
    move it to outcomes_raw.csv so the new fetch_all() flow can reuse it.
    """
    if os.path.exists(VIOL_PATH) and not os.path.exists(OUTCOMES_CACHE):
        try:
            head = pd.read_csv(VIOL_PATH, nrows=1)
            if "issuer_code" not in head.columns:
                print(f"  Migrating old {VIOL_PATH} -> {OUTCOMES_CACHE}")
                os.rename(VIOL_PATH, OUTCOMES_CACHE)
        except Exception as exc:
            print(f"  [WARN] Could not inspect old cache: {exc}")


def _fetch_outcomes(target: int) -> pd.DataFrame:
    if os.path.exists(OUTCOMES_CACHE):
        df = pd.read_csv(OUTCOMES_CACHE, low_memory=False)
        if "violation_status" in df.columns and len(df) >= 1000:
            print(f"  Using cached outcomes: {len(df):,} rows -> {OUTCOMES_CACHE}")
            return df

    df = fetch_endpoint(
        OPEN_PARKING_URL, target, "outcomes (nc67-uf89)",
        extra_params={"$where": "violation_status != 'OUTSTANDING'"},
    )
    if df.empty or len(df) < 1000:
        print("  Outcomes API unavailable — generating 50,000 synthetic rows.")
        df = make_synthetic_data(50_000)
    df.to_csv(OUTCOMES_CACHE, index=False)
    return df


def _fetch_pvqr_multiyear(rows_per_year: int) -> pd.DataFrame:
    """Fetch pvqr data from multiple fiscal years and stack them.

    Using FY2022-FY2025 gives 4 years of coverage, dramatically improving
    the join match rate with nc67-uf89 compared to using only one fiscal year.
    """
    if os.path.exists(PVQR_CACHE):
        df = pd.read_csv(PVQR_CACHE, low_memory=False)
        if "summons_number" in df.columns and len(df) >= 1000:
            print(f"  Using cached pvqr: {len(df):,} rows -> {PVQR_CACHE}")
            return df

    select_clause = ",".join(PVQR_KEEP)
    frames = []

    for fy_label, dataset_id in PVQR_FISCAL_YEARS.items():
        url = f"https://data.cityofnewyork.us/resource/{dataset_id}.csv"
        chunk = fetch_endpoint(
            url, rows_per_year, f"pvqr {fy_label} ({dataset_id})",
            extra_params={"$select": select_clause},
        )
        if not chunk.empty:
            chunk["_fy"] = fy_label
            frames.append(chunk)
            print(f"    {fy_label}: {len(chunk):,} rows")
        else:
            print(f"    [WARN] {fy_label} returned no data — skipping")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(PVQR_CACHE, index=False)
    print(f"  Combined pvqr: {len(combined):,} rows across {len(frames)} fiscal year(s) -> {PVQR_CACHE}")
    return combined


def _join(outcomes: pd.DataFrame, pvqr: pd.DataFrame) -> pd.DataFrame:
    if pvqr.empty:
        print("  [WARN] pvqr dataset empty — proceeding with outcomes only")
        return outcomes

    outcomes["summons_number"] = outcomes["summons_number"].astype(str).str.strip()
    pvqr["summons_number"] = pvqr["summons_number"].astype(str).str.strip()

    # Keep only pvqr fields that actually came back from the API
    keep = ["summons_number"] + [c for c in PVQR_KEEP[1:] if c in pvqr.columns]
    pvqr_slim = pvqr[keep].drop_duplicates(subset="summons_number", keep="first")

    merged = outcomes.merge(pvqr_slim, on="summons_number", how="left", suffixes=("", "_pvqr"))

    if "issuer_code" in merged.columns:
        match = merged["issuer_code"].notna().sum()
        rate  = match / len(merged) * 100 if len(merged) else 0
        print(f"  Join match rate: {match:,}/{len(merged):,} ({rate:.1f}%)")

        if "_fy" in merged.columns:
            for fy in sorted(merged["_fy"].dropna().unique()):
                n = (merged["_fy"] == fy).sum()
                print(f"    {fy}: {n:,} matched rows")

    return merged


def fetch_all(target_rows: int = DEFAULT_TARGET_ROWS) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    _migrate_old_cache()

    outcomes = _fetch_outcomes(target_rows)
    print(f"  Outcomes: {len(outcomes):,} rows")
    if "violation_status" in outcomes.columns:
        unique_statuses = sorted(outcomes['violation_status'].dropna().unique().tolist())[:10]
        print(f"  Sample statuses: {unique_statuses}")

    pvqr = _fetch_pvqr_multiyear(PVQR_ROWS_PER_YEAR)
    print(f"  PVQR total: {len(pvqr):,} rows")

    merged = _join(outcomes, pvqr)

    merged.to_csv(VIOL_PATH, index=False)
    print(f"  Saved {len(merged):,} joined rows -> {VIOL_PATH}")

    adj = merged[["summons_number", "violation_status"]].copy()
    adj.to_csv(ADJ_PATH, index=False)
    print(f"  Adjudications stub saved -> {ADJ_PATH}")


if __name__ == "__main__":
    fetch_all()
