from __future__ import annotations

import pandas as pd
import pytest

from nba_impact.models.five_year_spm_feature_research import (
    CANDIDATE_GROUPS,
    enforce_same_season_stabilization,
    pool_annual_candidates,
)


def _annual_row(player: int, season: int, value: float) -> dict:
    row = {
        "PLAYER_ID": player,
        "Window_End": season,
        "OffPoss": 100.0,
        "DefPoss": 100.0,
        "deflections_p100": value,
        "charges_drawn_p100": value,
        "contested_2pt_p100": value,
        "contested_3pt_p100": value,
        "contested_3pt_share": value,
        "def_loose_balls_recovered_p100": value,
    }
    for group in CANDIDATE_GROUPS.values():
        for feature in group["features"]:
            row[feature] = value
    return row


def test_same_season_contract_removes_historical_cross_season_fill() -> None:
    annual = pd.DataFrame(
        [_annual_row(1, 2017, 9.0), _annual_row(1, 2018, 2.0)]
    )

    clean = enforce_same_season_stabilization(annual)

    assert clean.loc[clean["Window_End"].eq(2017), "deflections_p100"].item() == 0.0
    assert clean.loc[clean["Window_End"].eq(2018), "deflections_p100"].item() == 2.0


def test_five_year_pool_only_uses_rows_inside_the_explicit_window() -> None:
    annual = pd.DataFrame(
        [_annual_row(1, season, float(season - 2013)) for season in range(2014, 2020)]
    )

    pooled = pool_annual_candidates(annual, window_ends=(2018, 2019))
    feature = "shot_making_points_above_expected_p100_eb"

    assert pooled.loc[pooled["Window_End"].eq(2018), feature].item() == pytest.approx(3.0)
    assert pooled.loc[pooled["Window_End"].eq(2019), feature].item() == pytest.approx(4.0)


def test_season_2027_is_rejected() -> None:
    with pytest.raises(ValueError, match="2027"):
        enforce_same_season_stabilization(pd.DataFrame([_annual_row(1, 2027, 1.0)]))
