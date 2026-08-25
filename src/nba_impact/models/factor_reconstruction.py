"""Held-out linear reconstruction of points RAPM from factor RAPMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedRidge:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> WeightedRidge:
    """Fit a weighted, standardized ridge with an unpenalized intercept."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("features, target, and weights have incompatible shapes")
    if alpha < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite and alpha must be nonnegative")
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and positive")
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
    z = (x - mean) / scale
    intercept = float(np.average(y, weights=w))
    root_weight = np.sqrt(w)
    weighted_x = z * root_weight[:, None]
    weighted_y = (y - intercept) * root_weight
    lhs = weighted_x.T @ weighted_x + alpha * np.eye(z.shape[1])
    coefficients = np.linalg.solve(lhs, weighted_x.T @ weighted_y)
    return WeightedRidge(mean, scale, coefficients, intercept, float(alpha))


def predict_weighted_ridge(model: WeightedRidge, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model.coefficients):
        raise ValueError("feature count does not match the fitted model")
    return model.intercept + ((x - model.mean) / model.scale) @ model.coefficients


def weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    w = np.asarray(weights, dtype=float)
    error = y - p
    mean_y = float(np.average(y, weights=w))
    denominator = float(np.sum(w * (y - mean_y) ** 2))
    numerator = float(np.sum(w * error**2))
    covariance = float(np.average((y - mean_y) * (p - np.average(p, weights=w)), weights=w))
    variance_y = float(np.average((y - mean_y) ** 2, weights=w))
    variance_p = float(np.average((p - np.average(p, weights=w)) ** 2, weights=w))
    correlation = covariance / np.sqrt(variance_y * variance_p) if variance_p > 0 else 0.0
    return {
        "weighted_rmse": float(np.sqrt(numerator / w.sum())),
        "weighted_correlation": float(correlation),
        "weighted_r2": float(1.0 - numerator / denominator) if denominator > 0 else 0.0,
    }
