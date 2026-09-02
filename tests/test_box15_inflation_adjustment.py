from __future__ import annotations

import numpy as np
import pandas as pd

from research.run_box15_inflation_adjustment import (
    apply_inflation_adjustment,
    season_opponent_context,
)


def test_inflation_adjustment_uses_defense_faced_and_league_fallback() -> None:
    row = {
        "home_team_id": 10,
        "away_team_id": 20,
        "home_possessions": 100,
        "away_possessions": 100,
        "home_points": 110,
        "away_points": 100,
        "home_technical_points_excluded": 0,
        "away_technical_points_excluded": 0,
    }
    row.update({f"home_player_{slot}": slot for slot in range(1, 6)})
    row.update({f"away_player_{slot}": slot + 5 for slot in range(1, 6)})
    player_context, season_row = season_opponent_context(pd.DataFrame([row]), season=2020)
    season_context = pd.DataFrame([season_row])
    features = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "Window_End": 2020, "OffPoss": 100, "PTS_p100": 25.0},
            {"PLAYER_ID": 99, "Window_End": 2020, "OffPoss": 20, "PTS_p100": 25.0},
        ]
    )
    adjusted, quality = apply_inflation_adjustment(
        features,
        player_context,
        season_context,
        reference_ortg=120.0,
    )
    assert np.isclose(adjusted.loc[0, "PTS_p100"], 25.0 * 120.0 / 110.0)
    assert np.isclose(adjusted.loc[1, "PTS_p100"], 25.0 * 120.0 / 105.0)
    assert quality["matched_rows"] == 1
    assert np.isclose(quality["offensive_possession_match_rate"], 100 / 120)
