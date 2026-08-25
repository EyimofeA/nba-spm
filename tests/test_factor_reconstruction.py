import numpy as np

from nba_impact.models.factor_reconstruction import (
    fit_weighted_ridge,
    predict_weighted_ridge,
    weighted_metrics,
)


def test_weighted_ridge_recovers_linear_signal():
    x = np.random.default_rng(7).normal(size=(20, 3))
    y = 2.0 + x @ np.array([0.5, -1.0, 2.0])
    weights = np.linspace(1.0, 2.0, len(x))
    model = fit_weighted_ridge(x, y, weights, alpha=0.0)
    prediction = predict_weighted_ridge(model, x)
    assert np.allclose(prediction, y)
    assert weighted_metrics(y, prediction, weights)["weighted_rmse"] < 1e-10


def test_regularized_prediction_shape_and_metrics():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    weights = np.ones(4)
    model = fit_weighted_ridge(x, y, weights, alpha=10.0)
    prediction = predict_weighted_ridge(model, x)
    metrics = weighted_metrics(y, prediction, weights)
    assert prediction.shape == y.shape
    assert set(metrics) == {"weighted_rmse", "weighted_correlation", "weighted_r2"}
