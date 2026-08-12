"""Precision-aware empirical-Bayes centers for RAPM research comparisons.

This implements the frozen specification, not a new production model.  The
prior mean is a cross-fitted SPM prediction in RAPM coefficient units.  The
side-specific prior variance is calibrated only from earlier-fold residuals
after removing RAPM measurement variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags

from nba_impact.models.rapm import RapmConfig, RapmDesign, _penalty
from nba_impact.models.rapm_uncertainty import _recenter, _solve


@dataclass(frozen=True)
class PriorPrecision:
    side: str
    tau_squared: float
    residual_scale_squared: float
    mean_label_variance: float
    status: str


def calibrate_prior_precision(
    calibration: pd.DataFrame,
    *,
    side: str,
    label_column: str,
    prior_column: str,
    label_variance_column: str,
    minimum_rows: int = 50,
) -> PriorPrecision:
    """Robustly estimate latent prior error after removing label noise.

    Inputs must contain only *earlier* cross-fitted folds. Values are in RAPM
    coefficient units, not points per 100. A median/MAD scale limits one noisy
    historical player or season from setting the global side precision.
    """
    values = calibration[[label_column, prior_column, label_variance_column]].copy()
    values = values.apply(pd.to_numeric, errors="coerce").dropna()
    values = values.loc[values[label_variance_column].ge(0)]
    if len(values) < minimum_rows:
        return PriorPrecision(side, float("nan"), float("nan"), float("nan"), "insufficient_earlier_fold_rows")
    residual = (values[label_column] - values[prior_column]).to_numpy(dtype=float)
    center = float(np.median(residual))
    # A nonzero systematic residual is a mean-calibration failure, not a width
    # parameter to hide. The caller must either correct the cross-fitted mean or
    # reject the candidate.
    mad_scale = float(1.4826 * np.median(np.abs(residual - center)))
    robust_residual_variance = mad_scale**2
    mean_label_variance = float(values[label_variance_column].mean())
    tau_squared = max(0.0, robust_residual_variance - mean_label_variance)
    status = "identified" if tau_squared > 0 else "boundary_zero"
    return PriorPrecision(
        side=side,
        tau_squared=tau_squared,
        residual_scale_squared=robust_residual_variance,
        mean_label_variance=mean_label_variance,
        status=status,
    )


def fit_precision_aware_center(
    design: RapmDesign,
    config: RapmConfig,
    center: np.ndarray,
    *,
    sigma_squared: float,
    offense_precision: PriorPrecision,
    defense_precision: PriorPrecision,
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit SSE RAPM with side-specific EB penalties around an SPM mean.

    Under the existing unstandardized SSE loss, the coherent penalty is
    ``sigma_squared / tau_squared``. A zero tau is a degenerate prior and is
    intentionally rejected rather than replaced with an arbitrary huge number.
    """
    if sigma_squared <= 0 or not np.isfinite(sigma_squared):
        raise ValueError("sigma_squared must be finite and positive.")
    if offense_precision.status != "identified" or defense_precision.status != "identified":
        raise ValueError("Prior precision is not identified on earlier folds.")
    if center.shape != (design.X.shape[1],):
        raise ValueError("Prior center shape must match RAPM design.")
    X = design.X if row_mask is None else design.X[row_mask]
    y = design.y if row_mask is None else design.y[row_mask]
    n_players = len(design.players)
    penalty = _penalty(config, n_players)
    # Replace only player-block penalties. Home court retains its fixed ridge
    # penalty and zero center.
    penalty[:n_players] = sigma_squared / offense_precision.tau_squared
    penalty[n_players : 2 * n_players] = sigma_squared / defense_precision.tau_squared
    intercept = float(y.mean())
    lhs = (X.T @ X).tocsr() + diags(penalty, format="csr")
    rhs = np.asarray(X.T @ (y - intercept)).ravel() + penalty * center
    raw_beta = _solve(lhs, rhs)
    off_counts = np.asarray(X[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    beta, intercept, _ = _recenter(
        raw_beta, off_counts, def_counts, n_players, intercept=intercept
    )
    return beta, intercept, penalty
