from __future__ import annotations

import pandas as pd

from nba_impact.models.win_probability_possession import (
    build_possession_start_states,
    make_possession_features,
)


def test_possession_start_score_excludes_current_and_future_outcomes() -> None:
    possessions = pd.DataFrame(
        {
            "possession_id": ["g:1", "g:2", "g:3"],
            "game_id": ["g", "g", "g"],
            "possession_number": [1, 2, 3],
            "season_label": ["2024-25"] * 3,
            "season_type": ["regular"] * 3,
            "period": [1, 1, 1],
            "start_order_number": [1, 2, 3],
            "start_seconds_elapsed": [0.0, 20.0, 40.0],
            "offense_is_home": [True, False, True],
            "points": [2.0, 3.0, 4.0],
            "home_points": [2.0, 0.0, 4.0],
            "away_points": [0.0, 3.0, 0.0],
        }
    )
    games = pd.DataFrame(
        {"game_id": ["g"], "home_win": [True], "home_score": [6], "away_score": [3]}
    )
    states = build_possession_start_states(possessions, games)
    assert states["home_score_diff_before"].tolist() == [0.0, 2.0, -1.0]

    changed = possessions.copy()
    changed.loc[2, "points"] = 40.0
    changed.loc[2, "home_points"] = 40.0
    changed_games = games.assign(home_score=42)
    changed_states = build_possession_start_states(changed, changed_games)
    assert changed_states.loc[:2, "home_score_diff_before"].tolist() == [0.0, 2.0, -1.0]


def test_possession_feature_sign_and_late_pressure_are_causal() -> None:
    states = pd.DataFrame(
        {
            "home_score_diff_after": [0.0, 0.0],
            "regulation_seconds_remaining": [2880.0, 0.0],
            "seconds_remaining_period": [720.0, 0.0],
            "seconds_elapsed_game": [0.0, 2880.0],
            "is_overtime": [False, False],
            "pregame_elo_diff": [0.0, 0.0],
            "pregame_starter_net_diff": [0.0, 0.0],
            "pregame_rolling_margin_diff": [0.0, 0.0],
            "pregame_rest_advantage_days": [0.0, 0.0],
            "offense_is_home": [True, False],
        }
    )
    features = make_possession_features(states, time_interactions=True)
    assert features["home_possession"].tolist() == [1.0, -1.0]
    assert abs(features.loc[1, "possession_time_pressure"]) > abs(
        features.loc[0, "possession_time_pressure"]
    )
