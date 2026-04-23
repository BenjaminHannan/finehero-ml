"""Regression tests: strict-< (as-of) semantics for prior-history features.

Per docs/auc_improvement_plan.md Tier 0.1, we must guarantee that:

    1. plate_prior_ticket_count[i] counts rows in the same plate whose
       issue_date is strictly earlier than row i's issue_date.
    2. plate_prior_wins[i] counts wins in the same plate from strictly
       earlier dates only.
    3. Same-day duplicates do NOT see each other (same plate, same date =>
       they share the same prior stats, computed from strictly-earlier dates).
    4. _compute_rolling_group_history with closed='left' likewise excludes
       the current row's timestamp, so rows with no strictly-earlier same-
       group row get 0 for both wins_*D and count_*D.

These are regression tests. If any assert fires, a leakage path has been
reintroduced and Tier 0 must stop until it is fixed.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `src.engineer` importable when pytest is invoked from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.engineer import (  # noqa: E402
    PLATE_SMOOTH_K,
    _compute_plate_history,
    _compute_rolling_group_history,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixture: 3 plates x 5 tickets with same-day duplicates.
# ---------------------------------------------------------------------------
#
# Plate A: two tickets on 2024-01-01 (1 win + 1 loss), one on 2024-01-15 (win),
#          two on 2024-02-01 (1 loss + 1 win).
# Plate B: one on 01-10 (loss), two on 01-20 (1 win + 1 loss), one on 02-05
#          (win), one on 03-01 (win).
# Plate C: three on 02-10 (2 wins + 1 loss), two on 03-15 (1 loss + 1 win).
#
# Row order is deliberately NOT sorted globally by date so we also exercise
# the function's internal sorting.
@pytest.fixture
def synthetic_priors_df() -> pd.DataFrame:
    rows = [
        # plate, issue_date, won
        ("A", "2024-01-01", 1),
        ("A", "2024-01-01", 0),
        ("A", "2024-01-15", 1),
        ("A", "2024-02-01", 0),
        ("A", "2024-02-01", 1),
        ("B", "2024-01-10", 0),
        ("B", "2024-01-20", 1),
        ("B", "2024-01-20", 0),
        ("B", "2024-02-05", 1),
        ("B", "2024-03-01", 1),
        ("C", "2024-02-10", 1),
        ("C", "2024-02-10", 1),
        ("C", "2024-02-10", 0),
        ("C", "2024-03-15", 0),
        ("C", "2024-03-15", 1),
    ]
    df = pd.DataFrame(rows, columns=["plate_id", "issue_date", "won"])
    return df


# ---------------------------------------------------------------------------
# _compute_plate_history: flat plate priors (ticket_count + win_rate)
# ---------------------------------------------------------------------------

def test_plate_prior_ticket_count_strict_less_than(synthetic_priors_df):
    """Row i's plate_prior_ticket_count == #rows in same plate with strictly earlier date."""
    df = synthetic_priors_df.copy()
    feats, _ = _compute_plate_history(
        df, plate_col="plate_id", date_col="issue_date", global_mean=0.22
    )
    got = feats["plate_prior_ticket_count"].tolist()

    # Hand-computed expectations (see module docstring for fixture layout):
    # Plate A rows: 0, 0 (same-day twin), 2, 3, 3 (same-day twin)
    # Plate B rows: 0, 1, 1 (same-day twin), 3, 4
    # Plate C rows: 0, 0, 0 (triplet), 3, 3 (same-day twin)
    expected = [0, 0, 2, 3, 3, 0, 1, 1, 3, 4, 0, 0, 0, 3, 3]
    assert got == expected, f"ticket_count mismatch:\n got={got}\n exp={expected}"


def test_plate_prior_wins_strict_less_than(synthetic_priors_df):
    """plate_prior_wins must reflect wins on STRICTLY earlier dates only.

    We recover the raw prior_wins value from the smoothed win_rate + count:
        win_rate = (wins + k*global_mean) / (count + k)
    => wins = win_rate * (count + k) - k*global_mean
    """
    df = synthetic_priors_df.copy()
    gmean = 0.3  # arbitrary non-trivial smoothing constant
    feats, _ = _compute_plate_history(
        df, plate_col="plate_id", date_col="issue_date", global_mean=gmean
    )
    count = feats["plate_prior_ticket_count"].to_numpy(dtype=float)
    rate = feats["plate_prior_win_rate"].to_numpy(dtype=float)
    wins_reconstructed = rate * (count + PLATE_SMOOTH_K) - PLATE_SMOOTH_K * gmean

    # Hand-computed expected prior_wins per row:
    # Plate A: 0, 0 (same-day twin), 1 (row-0 won), 2, 2 (same-day twin)
    # Plate B: 0, 0 (row-0 lost), 0 (same-day twin), 1 (row at 01-20 won), 2
    # Plate C: 0, 0, 0 (triplet), 2 (rows 0,1 won), 2 (same-day twin)
    expected = np.array(
        [0, 0, 1, 2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 2, 2],
        dtype=float,
    )
    np.testing.assert_allclose(
        wins_reconstructed,
        expected,
        atol=1e-9,
        err_msg=f"reconstructed prior_wins wrong:\n got={wins_reconstructed}\n exp={expected}",
    )


def test_same_day_plate_duplicates_do_not_see_each_other(synthetic_priors_df):
    """Explicit guard: two tickets with identical (plate, date) share priors
    computed from strictly-earlier rows only."""
    df = synthetic_priors_df.copy()
    feats, _ = _compute_plate_history(
        df, plate_col="plate_id", date_col="issue_date", global_mean=0.22
    )

    # Plate A rows 0,1 are both 2024-01-01 -> identical priors
    assert feats.loc[0, "plate_prior_ticket_count"] == feats.loc[1, "plate_prior_ticket_count"]
    assert feats.loc[0, "plate_prior_win_rate"]     == feats.loc[1, "plate_prior_win_rate"]

    # Plate A rows 3,4 are both 2024-02-01 -> identical priors
    assert feats.loc[3, "plate_prior_ticket_count"] == feats.loc[4, "plate_prior_ticket_count"]
    assert feats.loc[3, "plate_prior_win_rate"]     == feats.loc[4, "plate_prior_win_rate"]

    # Plate B rows 6,7 are both 2024-01-20 -> identical priors
    assert feats.loc[6, "plate_prior_ticket_count"] == feats.loc[7, "plate_prior_ticket_count"]

    # Plate C rows 10,11,12 are all 2024-02-10 -> must all be 0
    # (no strictly earlier row exists for plate C)
    assert feats.loc[10, "plate_prior_ticket_count"] == 0
    assert feats.loc[11, "plate_prior_ticket_count"] == 0
    assert feats.loc[12, "plate_prior_ticket_count"] == 0


# ---------------------------------------------------------------------------
# _compute_rolling_group_history: 30D/90D rolling wins/count per group
# ---------------------------------------------------------------------------

def test_rolling_group_history_strict_left(synthetic_priors_df):
    """Rolling 30D/90D with closed='left' excludes the current row's timestamp,
    so same-day duplicates see 0 from each other."""
    df = synthetic_priors_df.copy()
    out = _compute_rolling_group_history(
        df, group_col="plate_id", date_col="issue_date",
        global_mean=0.25, windows=("30D", "90D"), prefix="plate",
    )

    cnt30 = out["plate_prior_count_30D"].to_numpy(dtype=float)
    win30 = out["plate_prior_wins_30D"].to_numpy(dtype=float)
    cnt90 = out["plate_prior_count_90D"].to_numpy(dtype=float)
    win90 = out["plate_prior_wins_90D"].to_numpy(dtype=float)

    # --- 30D window (closed='left': [t-30D, t)) ---
    # Plate A:
    #   row 0, 01-01: no earlier -> 0/0
    #   row 1, 01-01: same day, excluded -> 0/0
    #   row 2, 01-15: window [12-16, 01-15); rows 0,1 at 01-01 -> cnt=2, wins=1+0=1
    #   row 3, 02-01: window [01-02, 02-01); row 2 at 01-15 in window; rows 0,1 at 01-01 excluded
    #                 -> cnt=1, wins=1
    #   row 4, 02-01: same as row 3 -> cnt=1, wins=1
    # Plate B:
    #   row 5, 01-10: no earlier -> 0/0
    #   row 6, 01-20: window [12-21, 01-20); row 5 at 01-10 -> cnt=1, wins=0
    #   row 7, 01-20: same day as row 6 -> cnt=1, wins=0
    #   row 8, 02-05: window [01-06, 02-05); rows 5,6,7 all in window -> cnt=3, wins=0+1+0=1
    #   row 9, 03-01: window [01-31, 03-01); row 8 at 02-05 only -> cnt=1, wins=1
    # Plate C:
    #   rows 10,11,12 at 02-10: no earlier same-group row -> 0/0 each
    #   rows 13,14 at 03-15: window [02-14, 03-15); rows at 02-10 excluded -> 0/0
    expected_cnt30 = [0, 0, 2, 1, 1, 0, 1, 1, 3, 1, 0, 0, 0, 0, 0]
    expected_win30 = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(cnt30, expected_cnt30, atol=1e-9)
    np.testing.assert_allclose(win30, expected_win30, atol=1e-9)

    # --- 90D window (closed='left': [t-90D, t)) ---
    # 90 days back from 03-15 = ~12-16 prior year; from 02-05 = ~11-07 prior year.
    # So all prior rows in the fixture fall inside the 90D window.
    # Plate A:
    #   row 0: 0/0   row 1: 0/0 (same-day twin)
    #   row 2, 01-15: rows 0,1 -> cnt=2, wins=1
    #   row 3, 02-01: rows 0,1,2 -> cnt=3, wins=2
    #   row 4, 02-01: same-day twin -> cnt=3, wins=2
    # Plate B:
    #   row 5: 0/0
    #   row 6, 01-20: row 5 -> cnt=1, wins=0
    #   row 7: same-day twin -> cnt=1, wins=0
    #   row 8, 02-05: rows 5,6,7 -> cnt=3, wins=1
    #   row 9, 03-01: rows 5,6,7,8 -> cnt=4, wins=2
    # Plate C:
    #   rows 10,11,12 at 02-10: no earlier -> 0/0 each
    #   rows 13,14 at 03-15: rows 10,11,12 at 02-10 (33 days earlier, inside 90D) ->
    #       cnt=3, wins=1+1+0=2
    expected_cnt90 = [0, 0, 2, 3, 3, 0, 1, 1, 3, 4, 0, 0, 0, 3, 3]
    expected_win90 = [0, 0, 1, 2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 2, 2]
    np.testing.assert_allclose(cnt90, expected_cnt90, atol=1e-9)
    np.testing.assert_allclose(win90, expected_win90, atol=1e-9)


def test_rolling_group_first_row_per_group_is_zero(synthetic_priors_df):
    """Any row with no strictly-earlier same-group row must get 0/0 for every window."""
    df = synthetic_priors_df.copy()
    out = _compute_rolling_group_history(
        df, group_col="plate_id", date_col="issue_date",
        global_mean=0.25, windows=("30D", "90D"), prefix="plate",
    )

    # Indices with no strictly-earlier same-plate row:
    #   Plate A: rows 0, 1 (both 2024-01-01, first date for plate A)
    #   Plate B: row 5 (2024-01-10, first for plate B)
    #   Plate C: rows 10, 11, 12 (all 2024-02-10, first for plate C)
    firsts = [0, 1, 5, 10, 11, 12]
    for i in firsts:
        for col in (
            "plate_prior_count_30D", "plate_prior_wins_30D",
            "plate_prior_count_90D", "plate_prior_wins_90D",
        ):
            assert out.loc[i, col] == 0, f"row {i} col {col} expected 0, got {out.loc[i, col]}"
