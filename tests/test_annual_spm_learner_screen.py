import numpy as np
import pandas as pd

from research.run_annual_spm_learner_screen import (
    _net_diagnostics,
    _prune_features,
    _weighted_metrics,
)


def test_correlation_pruning_keeps_box15_control() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 1, 2],
            "Season": [2019, 2019, 2020, 2020],
            "PTS_p100": [1.0, 2.0, 1.5, 2.5],
            "PTS_p100_relative": [2.0, 4.0, 3.0, 5.0],
            "other": [0.0, 1.0, 2.0, 0.0],
        }
    )
    kept = _prune_features(
        frame,
        ("PTS_p100_relative", "PTS_p100", "other"),
        threshold=0.95,
    )
    assert "PTS_p100" in kept
    assert "PTS_p100_relative" not in kept
    assert "other" in kept


def test_weighted_metrics_use_target_on_prediction_slope() -> None:
    metrics = _weighted_metrics(
        np.array([0.0, 2.0, 4.0]),
        np.array([0.0, 1.0, 2.0]),
        np.ones(3),
    )
    assert metrics["weighted_rmse"] == np.sqrt(5.0 / 3.0)
    assert metrics["weighted_correlation"] == 1.0
    assert metrics["calibration_slope"] == 2.0


def test_net_diagnostics_add_offense_and_defense() -> None:
    rows = []
    for side, target, prediction in (
        ("offense", [1.0, 2.0], [0.5, 1.5]),
        ("defense", [0.5, -0.5], [0.25, -0.25]),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "PLAYER_ID": [1, 2],
                    "Season": [2022, 2022],
                    f"target_{side}": target,
                    "sample_weight": [1.0, 1.0],
                    "side": side,
                    "arm": "audited_all",
                    "learner": "ridge",
                    "phase": "diagnostic",
                    "prediction": prediction,
                }
            )
        )
    folds, summary = _net_diagnostics(
        pd.concat(rows, ignore_index=True),
        {"offense": "ridge", "defense": "ridge"},
        {"offense": "ridge", "defense": "ridge"},
    )
    assert list(folds["candidate"]) == ["frozen_rich_winner"]
    assert summary.iloc[0]["folds"] == 1
