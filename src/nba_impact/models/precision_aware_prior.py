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

from nba_impact.models.prior_informed_rapm import (
    _fit_prior_only_nuisance,
    _record_candidate,
    build_prior_center,
    paired_confirmation_bootstrap,
)
from nba_impact.models.rapm import RapmConfig, RapmDesign, _penalty
from nba_impact.models.rapm import build_design, fit_coefficients, fit_coefficients_with_center
from nba_impact.models.rapm_uncertainty import game_cluster_sandwich
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


def run_precision_aware_prior_comparison(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    calibration: pd.DataFrame,
    config: RapmConfig,
    *,
    test_seasons: tuple[int, ...],
    train_window: int,
    selection_seasons: tuple[int, ...],
    diagnostic_seasons: tuple[int, ...],
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260812,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the four preregistered models on identical future game rows.

    ``calibration`` is an earlier-fold-only panel in coefficient units with
    columns ``window_end``, ``offense_label``, ``offense_prior``,
    ``offense_label_variance`` and defensive equivalents.  The runner rejects
    any fold whose precision calibration is not strictly older than its RAPM
    training window.
    """
    if set(selection_seasons) & set(diagnostic_seasons):
        raise ValueError("Selection and diagnostic seasons must be disjoint.")
    if set(test_seasons) != set(selection_seasons) | set(diagnostic_seasons):
        raise ValueError("Test seasons must equal selection plus diagnostic seasons.")
    design = build_design(frame, include_home=config.include_home)
    rows: list[dict] = []
    games: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    for test_season in test_seasons:
        train_seasons = tuple(range(test_season - train_window, test_season))
        train_mask = np.isin(design.seasons, train_seasons)
        test_mask = design.seasons == test_season
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Missing train/test possessions for {test_season}.")
        prior_end = train_seasons[-1]
        center, coverage = build_prior_center(
            design, priors, prior_window_end=prior_end,
            train_mask=train_mask, test_mask=test_mask,
        )
        # Calibration may use earlier *training* windows but never the scored
        # season or a future window. It is not a test-set tuning input.
        earlier = calibration.loc[calibration["window_end"].lt(prior_end)]
        off_precision = calibrate_prior_precision(
            earlier, side="offense", label_column="offense_label",
            prior_column="offense_prior", label_variance_column="offense_label_variance",
        )
        def_precision = calibrate_prior_precision(
            earlier, side="defense", label_column="defense_label",
            prior_column="defense_prior", label_variance_column="defense_label_variance",
        )
        calibration_rows.append({
            "test_season": test_season, "calibration_latest_window": int(earlier["window_end"].max()) if not earlier.empty else None,
            **{f"offense_{k}": v for k, v in off_precision.__dict__.items()},
            **{f"defense_{k}": v for k, v in def_precision.__dict__.items()}, **coverage,
        })
        fitted: dict[str, tuple[np.ndarray, float]] = {
            "zero_prior": fit_coefficients(design, config, train_mask),
            "fixed_center_prior": fit_coefficients_with_center(design, config, center, row_mask=train_mask),
            "statistical_prior_only": _fit_prior_only_nuisance(design, center, train_mask),
        }
        residual = design.y[train_mask] - np.asarray(design.X[train_mask] @ fitted["fixed_center_prior"][0]).ravel() - fitted["fixed_center_prior"][1]
        sigma_squared = float(np.mean(residual**2))
        if off_precision.status == "identified" and def_precision.status == "identified":
            fitted["precision_aware_side_specific_prior"] = fit_precision_aware_center(
                design, config, center, sigma_squared=sigma_squared,
                offense_precision=off_precision, defense_precision=def_precision,
                row_mask=train_mask,
            )[:2]
        for candidate, (beta, intercept) in fitted.items():
            metric, game = _record_candidate(
                design, beta, intercept, candidate=candidate, test_season=test_season,
                train_seasons=train_seasons, prior_window_end=prior_end,
                train_mask=train_mask, test_mask=test_mask,
            )
            metric["sigma_squared"] = sigma_squared
            rows.append(metric); games.append(game)
    folds = pd.DataFrame(rows)
    game_frame = pd.concat(games, ignore_index=True)
    comparison = paired_confirmation_bootstrap(
        game_frame, selected_candidate="precision_aware_side_specific_prior",
        confirmation_test_seasons=diagnostic_seasons,
        repetitions=bootstrap_repetitions, seed=bootstrap_seed,
    ) if (folds["candidate"] == "precision_aware_side_specific_prior").any() else {"status": "invalid_precision_calibration"}
    return folds, pd.DataFrame(calibration_rows), comparison


def build_prior_calibration_panel(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    config: RapmConfig,
    *,
    window_ends: tuple[int, ...],
    window_length: int = 3,
) -> pd.DataFrame:
    """Create earlier-fold RAPM labels with analytic measurement variance.

    This is a calibration artifact only. It uses the normal-RAPM point estimator
    and its CR0 game-cluster covariance as label-noise diagnostics; publication
    uncertainty remains the whole-game bootstrap.
    """
    outputs: list[pd.DataFrame] = []
    for end in window_ends:
        seasons = tuple(range(end - window_length + 1, end + 1))
        subset = frame.loc[frame["season"].isin(seasons)].copy()
        if set(seasons) != set(int(value) for value in subset["season"].unique()):
            raise ValueError(f"Calibration window {end} lacks required seasons.")
        design = build_design(subset, include_home=config.include_home)
        covariance, beta, _ = game_cluster_sandwich(design, config)
        n_players = len(design.players)
        labels = pd.DataFrame(
            {
                "PLAYER_ID": design.players,
                "window_end": end,
                "offense_label": beta[:n_players] * 100.0,
                "defense_label": -beta[n_players : 2 * n_players] * 100.0,
                "offense_label_variance": np.clip(np.diag(covariance)[:n_players] * 10000.0, 0.0, None),
                "defense_label_variance": np.clip(np.diag(covariance)[n_players : 2 * n_players] * 10000.0, 0.0, None),
            }
        )
        prior = priors.loc[priors["Window_End"].eq(end), [
            "PLAYER_ID", "prior_offense_per_100", "prior_defense_per_100"
        ]].rename(columns={
            "prior_offense_per_100": "offense_prior",
            "prior_defense_per_100": "defense_prior",
        })
        outputs.append(labels.merge(prior, on="PLAYER_ID", how="inner", validate="one_to_one"))
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "window_end"]).any():
        raise ValueError("Calibration panel must have unique player-window keys.")
    return result
