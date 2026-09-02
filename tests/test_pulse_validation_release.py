from __future__ import annotations

import pandas as pd

from research.build_pulse_validation_release import metric_rows, paired_bootstrap


def _games(candidate: str, offset: float) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate": [candidate] * 4,
        "rating_season": [2020, 2020, 2021, 2021],
        "test_season": [2021, 2021, 2022, 2022],
        "game_id": ["1", "2", "3", "4"],
        "actual_margin": [1.0, -1.0, 2.0, -2.0],
        "predicted_margin": [1.0 + offset, -1.0 + offset, 2.0 + offset, -2.0 + offset],
    })


def test_equal_season_metrics_and_paired_bootstrap() -> None:
    left = _games("left", 0.0)
    right = _games("right", 1.0)
    metrics = metric_rows(pd.concat([left, right]), "test")
    assert len(metrics) == 4
    result = paired_bootstrap(pd.concat([left, right]), "left", "right", scope="test", draws=100)
    assert result["mean_mse_delta_left_minus_right"] < 0
    assert result["games"] == 4
    assert result["seasons"] == 2
