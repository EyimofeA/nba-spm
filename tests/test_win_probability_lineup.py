from __future__ import annotations

import pandas as pd

from nba_impact.models.win_probability_lineup import build_starter_strength, make_lineup_features


def test_starter_strength_uses_only_prior_season_ratings_and_centered_missing_value() -> None:
    rows = []
    for side, start_id in (("home", 1), ("away", 6)):
        for player_id in range(start_id, start_id + 5):
            rows.append(
                {
                    "game_id": "g1",
                    "season_label": "2025-26",
                    "season_start": 2025,
                    "team_side": side,
                    "player_id": player_id,
                    "starter": True,
                }
            )
    player_games = pd.DataFrame(rows)
    ratings = pd.DataFrame(
        {
            "prior_season_end": [2025] * 9,
            "player_id": list(range(1, 10)),
            "offense_per_100": [1.0] * 9,
            "defense_per_100": [0.5] * 9,
            "net_per_100": [1.5] * 9,
        }
    )
    strength = build_starter_strength(player_games, ratings).iloc[0]
    assert strength["starter_rating_coverage"] == 0.9
    assert strength["starter_net_per_100_home"] == 7.5
    assert strength["starter_net_per_100_away"] == 6.0
    assert strength["pregame_starter_net_diff"] == 1.5


def test_lineup_strength_decays_out_of_in_game_probability() -> None:
    states = pd.DataFrame(
        {
            "home_score_diff_after": [0.0, 0.0],
            "regulation_seconds_remaining": [2880.0, 0.0],
            "seconds_remaining_period": [720.0, 0.0],
            "seconds_elapsed_game": [0.0, 2880.0],
            "is_overtime": [False, False],
            "pregame_elo_diff": [0.0, 0.0],
            "pregame_starter_net_diff": [5.0, 5.0],
        }
    )
    features = make_lineup_features(states)
    assert features.loc[0, "pregame_starter_remaining"] == 5.0
    assert features.loc[1, "pregame_starter_remaining"] == 0.0
