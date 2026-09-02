import numpy as np
import pandas as pd

from nba_impact.models.canonical_pulse import game_metrics


def test_game_metrics_reports_readable_units() -> None:
    games = pd.DataFrame(
        {"actual_margin": [2.0, -2.0], "predicted_margin": [1.0, -1.0]}
    )
    metrics = game_metrics(games)
    assert metrics["mse"] == 1.0
    assert metrics["rmse"] == 1.0
    assert np.isclose(metrics["correlation"], 1.0)
    assert np.isclose(metrics["calibration_slope"], 2.0)
