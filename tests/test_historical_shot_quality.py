from __future__ import annotations

import pandas as pd

from nba_impact.models.historical_shot_quality import attach_pbp_context


def test_context_match_does_not_use_make_miss_outcome() -> None:
    shots = pd.DataFrame(
        {
            "GAME_ID": [1],
            "PERIOD": [1],
            "GAME_CLOCK": ["1:40"],
            "player_id": [10],
            "PTS_TYPE": [2],
            "SHOT_RESULT": ["made"],
        }
    )
    pbp = pd.DataFrame(
        {
            "game_id": [1],
            "period": [1],
            "clock_seconds": [99.0],
            "player_id": [10],
            "points_type": [2],
            "shot_result": ["missed"],
            "assisted": [0.0],
            "fast_break": [1.0],
            "x": [5.0],
            "y": [10.0],
        }
    )

    matched, quality = attach_pbp_context(shots, pbp)

    assert matched.loc[0, "fast_break"] == 1.0
    assert matched.loc[0, "x"] == 5.0
    assert quality["pbp_match_rate"] == 1.0
