from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.possession_context import (
    CAUSAL_FEATURE_COLUMNS,
    compute_possession_start_context,
)


def _possessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "possession_id": "0000000001:001", "game_id": "0000000001",
                "possession_number": 1, "season_start": 2023, "season_end": 2024,
                "season_label": "2023-24", "season_type": "regular", "game_date": "2023-10-24",
                "period": 1, "start_order_number": 1, "start_seconds_elapsed": 4.0,
                "offense_team_id": 10, "defense_team_id": 20, "home_team_id": 10,
                "away_team_id": 20, "offense_is_home": True, "points": 2.0,
                "home_points": 2.0, "away_points": 0.0, "lineup_ready": True,
            },
            {
                "possession_id": "0000000001:002", "game_id": "0000000001",
                "possession_number": 2, "season_start": 2023, "season_end": 2024,
                "season_label": "2023-24", "season_type": "regular", "game_date": "2023-10-24",
                "period": 1, "start_order_number": 2, "start_seconds_elapsed": 15.0,
                "offense_team_id": 20, "defense_team_id": 10, "home_team_id": 10,
                "away_team_id": 20, "offense_is_home": False, "points": 3.0,
                "home_points": 0.0, "away_points": 3.0, "lineup_ready": True,
            },
            {
                "possession_id": "0000000001:003", "game_id": "0000000001",
                "possession_number": 3, "season_start": 2023, "season_end": 2024,
                "season_label": "2023-24", "season_type": "regular", "game_date": "2023-10-24",
                "period": 5, "start_order_number": 3, "start_seconds_elapsed": 2887.0,
                "offense_team_id": 10, "defense_team_id": 20, "home_team_id": 10,
                "away_team_id": 20, "offense_is_home": True, "points": 0.0,
                "home_points": 0.0, "away_points": 0.0, "lineup_ready": False,
            },
        ]
    )


def test_possession_context_uses_only_prior_possession_outcomes() -> None:
    original, quality = compute_possession_start_context(_possessions())
    assert quality["passed"]
    assert set(CAUSAL_FEATURE_COLUMNS).issubset(original.columns)
    first, second, overtime = original.itertuples(index=False)
    assert first.is_first_possession
    assert first.offense_score_diff_start == 0.0
    assert second.previous_possession_points == 2.0
    assert second.offense_score_diff_start == -2.0
    assert overtime.is_overtime
    assert overtime.seconds_remaining_period_start == 293.0

    revised_source = _possessions()
    revised_source.loc[revised_source["possession_number"].eq(2), "points"] = 1.0
    revised_source.loc[revised_source["possession_number"].eq(2), "away_points"] = 1.0
    revised, _ = compute_possession_start_context(revised_source)
    # The second possession's own outcome cannot change its own start context.
    assert np.allclose(
        original.loc[original["possession_number"].eq(2), CAUSAL_FEATURE_COLUMNS].select_dtypes(include="number"),
        revised.loc[revised["possession_number"].eq(2), CAUSAL_FEATURE_COLUMNS].select_dtypes(include="number"),
    )
    # The third possession can use the completed second possession, which is causal.
    assert revised.loc[revised["possession_number"].eq(3), "previous_possession_points"].item() == 1.0


def test_possession_context_rejects_nonconserved_points() -> None:
    source = _possessions()
    source.loc[0, "home_points"] = 1.0
    try:
        compute_possession_start_context(source)
    except ValueError as exc:
        assert "do not sum" in str(exc)
    else:
        raise AssertionError("Nonconserved possession points were accepted.")
