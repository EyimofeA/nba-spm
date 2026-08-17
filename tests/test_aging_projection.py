import numpy as np
import pandas as pd

from nba_impact.models.aging_projection import _design, _weighted_metrics


def test_aging_designs_are_finite_and_distinct() -> None:
    frame = pd.DataFrame(
        {
            "AGE": [21.0, 27.0, 35.0],
            "MIN": [400.0, 1800.0, 2800.0],
            "filtered_offense": [-1.0, 0.5, 3.0],
        }
    )
    linear = _design(frame, "linear_age", "offense")
    spline = _design(frame, "spline_age_impact", "offense")
    assert linear.shape == (3, 1)
    assert spline.shape[1] > linear.shape[1]
    assert np.isfinite(spline).all()


def test_weighted_metrics_report_true_r2() -> None:
    actual = np.array([-1.0, 1.0, 3.0])
    predicted = np.array([-1.0, 1.0, 3.0])
    metrics = _weighted_metrics(actual, predicted, np.ones(3))
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["correlation"] == 1.0
