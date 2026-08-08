from __future__ import annotations

import pandas as pd

from nba_impact.models.win_probability_ablation import build_pregame_elo, make_elo_features


def test_elo_updates_only_after_the_full_date() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "season_start": [2024, 2024, 2024],
            "home_team_id": [1, 3, 1], "away_team_id": [2, 4, 2],
            "home_win": [True, False, True],
        }
    )
    result = build_pregame_elo(games).set_index("game_id")
    assert result.loc["g1", "pregame_home_elo"] == 1500
    assert result.loc["g2", "pregame_home_elo"] == 1500
    assert result.loc["g3", "pregame_home_elo"] > 1500
    assert result.loc["g3", "pregame_away_elo"] < 1500


def test_elo_strength_decays_toward_zero_with_time_remaining() -> None:
    states = pd.DataFrame(
        {
            "home_score_diff_after": [0.0, 0.0], "regulation_seconds_remaining": [2880.0, 0.0],
            "seconds_remaining_period": [720.0, 0.0], "seconds_elapsed_game": [0.0, 2880.0],
            "is_overtime": [False, False], "pregame_elo_diff": [0.5, 0.5],
        }
    )
    features = make_elo_features(states)
    assert features.loc[0, "pregame_elo_remaining"] == 0.5
    assert features.loc[1, "pregame_elo_remaining"] == 0.0
