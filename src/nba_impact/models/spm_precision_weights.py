"""Precision diagnostics and bounded weights for noisy RAPM labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import load_npz


def recentered_variance_diagonal(
    covariance: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return diag Var(beta - weighted_mean(beta))."""
    covariance = np.asarray(covariance, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if covariance.shape != (len(weights), len(weights)):
        raise ValueError("Covariance and centering weights have incompatible shapes.")
    if not np.isfinite(covariance).all() or not np.isfinite(weights).all():
        raise ValueError("Covariance and centering weights must be finite.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Centering weights must have positive total exposure.")
    covariance_weight = covariance @ weights
    centered = (
        np.diag(covariance)
        - 2.0 * covariance_weight / total
        + float(weights @ covariance_weight) / total**2
    )
    return np.clip(centered, 0.0, None)


def analytic_ridge_label_variance(
    matrix_dir: str | Path,
    *,
    lambda_offense: float = 3000.0,
    lambda_defense: float = 3000.0,
    lambda_home: float = 300.0,
) -> pd.DataFrame:
    """Estimate five-year RAPM label variance from stored sufficient statistics.

    This is the homoskedastic ridge covariance diagnostic. It is cheaper than
    game-cluster covariance and must not be presented as publication uncertainty.
    """
    root = Path(matrix_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    xtx = load_npz(root / "train_xtx.npz").toarray().astype(np.float64)
    xty = np.load(root / "train_xty_centered.npy").astype(np.float64)
    players = np.load(root / "player_ids.npy").astype(np.int64)
    off_possessions = np.load(root / "train_off_possessions.npy").astype(np.float64)
    def_possessions = np.load(root / "train_def_possessions.npy").astype(np.float64)
    n_players = len(players)
    expected_shape = (2 * n_players + 1, 2 * n_players + 1)
    if xtx.shape != expected_shape or xty.shape != (expected_shape[0],):
        raise ValueError("Stored RAPM matrix dimensions do not match player IDs.")
    penalty = np.concatenate(
        [
            np.full(n_players, lambda_offense),
            np.full(n_players, lambda_defense),
            np.asarray([lambda_home]),
        ]
    )
    system = xtx.copy()
    system.flat[:: system.shape[0] + 1] += penalty
    factor = cho_factor(system, lower=True, check_finite=False)
    inverse = cho_solve(factor, np.eye(system.shape[0]), check_finite=False)
    beta = cho_solve(factor, xty, check_finite=False)
    residual_ss = (
        float(manifest["train"]["centered_y_sum_squares"])
        - 2.0 * float(beta @ xty)
        + float(beta @ xtx @ beta)
    )
    effective_df = float(np.sum(xtx * inverse.T))
    residual_df = float(manifest["train"]["possession_rows"]) - effective_df
    if residual_ss <= 0 or residual_df <= 0:
        raise ValueError("Stored RAPM system produced invalid residual variance.")
    sigma_squared = residual_ss / residual_df
    covariance = sigma_squared * (inverse @ xtx @ inverse)
    off_covariance = covariance[:n_players, :n_players]
    def_covariance = covariance[n_players : 2 * n_players, n_players : 2 * n_players]
    off_variance = recentered_variance_diagonal(off_covariance, off_possessions)
    def_variance = recentered_variance_diagonal(def_covariance, def_possessions)
    scale = 10_000.0
    return pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Window_End": int(manifest["metadata"]["window_end"]),
            "label_variance_offense": off_variance * scale,
            "label_variance_defense": def_variance * scale,
            "variance_method": "analytic_homoskedastic_ridge_diagnostic",
            "residual_variance_points_per_possession": sigma_squared,
            "effective_degrees_of_freedom": effective_df,
        }
    )


def bounded_inverse_variance_weights(
    variance: pd.Series,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.Series:
    """Convert positive label variances to finite, mean-one precision weights."""
    values = pd.to_numeric(variance, errors="raise").astype(float)
    if values.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Label variances must be positive and finite.")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Weight quantiles must satisfy 0 <= lower < upper <= 1.")
    low, high = values.quantile([lower_quantile, upper_quantile]).to_numpy()
    precision = 1.0 / values.clip(lower=low, upper=high)
    return precision / precision.mean()
