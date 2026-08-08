from __future__ import annotations

import pandas as pd

from nba_impact.models.win_probability import make_features, sample_game_states


def test_sample_states_keeps_one_per_time_bucket_and_terminal() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["g:1", "g:2", "g:3", "g:4"],
            "game_id": ["g"] * 4,
            "actionId": [1, 2, 3, 4],
            "seconds_elapsed_game": [0.0, 10.0, 35.0, 40.0],
            "is_terminal_event": [False, False, False, True],
        }
    )
    result = sample_game_states(frame, interval_seconds=30)
    assert result["event_id"].tolist() == ["g:1", "g:2", "g:4"]


def test_win_probability_features_use_only_current_state() -> None:
    states = pd.DataFrame(
        {
            "home_score_diff_after": [0.0, 6.0],
            "regulation_seconds_remaining": [2880.0, 120.0],
            "seconds_remaining_period": [720.0, 120.0],
            "seconds_elapsed_game": [0.0, 2760.0],
            "is_overtime": [False, False],
        }
    )
    features = make_features(states)
    assert features.loc[0, "score_time_pressure"] == 0.0
    assert features.loc[1, "score_time_pressure"] > features.loc[1, "home_score_diff"] / 3
