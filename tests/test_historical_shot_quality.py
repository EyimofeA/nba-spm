from __future__ import annotations

import pandas as pd

from nba_impact.models.historical_shot_quality import attach_pbp_context, attach_player_heights


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


def test_height_difference_uses_player_id_and_collapses_team_rows(tmp_path) -> None:
    shots = pd.DataFrame(
        {"player_id": [10], "CLOSEST_DEFENDER_PLAYER_ID": [20]}
    )
    pd.DataFrame(
        {
            "PLAYER_ID": [10, 10, 20],
            "PLAYER_HEIGHT_INCHES": [78.0, 78.0, 82.0],
        }
    ).to_parquet(tmp_path / "players.parquet", index=False)

    output, quality = attach_player_heights(shots, tmp_path / "players.parquet")

    assert output.loc[0, "height_difference_inches"] == -4.0
    assert quality["height_source_players"] == 2
    assert quality["height_difference_coverage"] == 1.0
