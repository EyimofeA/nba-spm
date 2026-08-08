from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.rapm_lineup_policy import (
    build_fractional_current_design,
    fractional_segment_weights,
)


def test_fractional_weights_use_time_then_same_clock_action_fallback() -> None:
    segments = pd.DataFrame(
        {
            "possession_id": ["a", "a", "b", "b"],
            "segment_number": [1, 2, 1, 2],
            "start_seconds_elapsed": [10.0, 20.0, 30.0, 30.0],
            "possession_end_seconds_elapsed": [40.0, 40.0, 30.0, 30.0],
            "action_count": [1, 1, 1, 3],
        }
    )
    weights = fractional_segment_weights(segments)
    assert np.allclose(weights.iloc[:2], [1 / 3, 2 / 3])
    assert np.allclose(weights.iloc[2:], [1 / 4, 3 / 4])


def test_fractional_design_splits_player_exposure(tmp_path) -> None:
    possessions = pd.DataFrame(
        {
            "possession_id": ["g:1"], "game_id": ["g"], "possession_number": [1],
            "season_type": ["regular"], "season_end": [2026], "game_date": ["2026-01-01"],
            "period": [1], "offense_is_home": [True], "points": [2.0],
            "end_seconds_elapsed": [30.0],
        }
    )
    segments = pd.DataFrame(
        {
            "possession_id": ["g:1", "g:1"], "segment_number": [1, 2],
            "start_seconds_elapsed": [10.0, 20.0], "action_count": [1, 1],
            **{f"home_player_{i}": [i, i + 20] for i in range(1, 6)},
            **{f"away_player_{i}": [i + 10, i + 30] for i in range(1, 6)},
        }
    )
    possession_path = tmp_path / "possessions.parquet"
    segment_path = tmp_path / "segments.parquet"
    possessions.to_parquet(possession_path, index=False)
    segments.to_parquet(segment_path, index=False)
    design = build_fractional_current_design(possession_path, segment_path)
    player_index = {int(player): index for index, player in enumerate(design.players)}
    assert design.X[0, player_index[1]] == 0.5
    assert design.X[0, player_index[21]] == 0.5
    assert np.isclose(design.X[0, : len(design.players)].sum(), 5.0)
