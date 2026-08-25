"""Cross-fitted empirical score-margin adjustment for possession RAPM."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve

from nba_impact.models.rapm import build_design


@dataclass(frozen=True)
class RubberbandSpec:
    name: str
    time_buckets: int
    margin_clip: float | None


@dataclass(frozen=True)
class RubberbandFit:
    spec: RubberbandSpec
    coefficients: np.ndarray
    covariance: np.ndarray | None
    rows: int
    games: int


def annotate_score_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pre-possession offense margin and actual six-minute game time."""
    required = {
        "gameid",
        "period",
        "num",
        "pts",
        "home_poss",
        "start_seconds_elapsed",
        "season",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rubber-band input is missing columns: {missing}")
    result = frame.copy()
    result["_source_order"] = np.arange(len(result), dtype=np.int64)
    result = result.sort_values(["gameid", "period", "num"], kind="stable")
    home_poss = result["home_poss"].astype(bool).to_numpy()
    points = pd.to_numeric(result["pts"], errors="raise").to_numpy(dtype=float)
    result["_home_points"] = np.where(home_poss, points, 0.0)
    result["_away_points"] = np.where(home_poss, 0.0, points)
    grouped = result.groupby("gameid", sort=False)
    home_before = grouped["_home_points"].cumsum() - result["_home_points"]
    away_before = grouped["_away_points"].cumsum() - result["_away_points"]
    home_margin = home_before - away_before
    result["offense_margin_before"] = np.where(home_poss, home_margin, -home_margin)
    elapsed = pd.to_numeric(result["start_seconds_elapsed"], errors="raise")
    if elapsed.isna().any() or (elapsed < 0).any():
        raise ValueError("Actual possession start time must be finite and non-negative.")
    result["regulation_seconds_elapsed"] = elapsed.clip(lower=0, upper=2880)
    result["six_minute_bucket"] = np.minimum(
        (result["regulation_seconds_elapsed"] // 360).astype(int), 7
    )
    result["regulation"] = result["period"].astype(int) <= 4
    return (
        result.sort_values("_source_order", kind="stable")
        .drop(columns=["_source_order", "_home_points", "_away_points"])
        .reset_index(drop=True)
    )


def deterministic_game_fold(game_id: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("At least two game folds are required.")
    digest = hashlib.sha256(str(game_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _solve_fold(
    design,
    train_mask: np.ndarray,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_home: float,
) -> tuple[np.ndarray, float]:
    X_train = design.X[train_mask]
    y_train = design.y[train_mask]
    intercept = float(y_train.mean())
    penalties = np.concatenate(
        [
            np.full(len(design.players), lambda_off, dtype=float),
            np.full(len(design.players), lambda_def, dtype=float),
            np.asarray([lambda_home], dtype=float),
        ]
    )
    lhs = (X_train.T @ X_train).tocsr() + diags(penalties, format="csr")
    rhs = np.asarray(X_train.T @ (y_train - intercept)).ravel()
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta, dtype=float)
    diagonal = np.asarray(X_train.power(2).sum(axis=0)).ravel()
    n_players = len(design.players)
    off_mean = float(np.average(beta[:n_players], weights=diagonal[:n_players]))
    def_mean = float(
        np.average(
            beta[n_players : 2 * n_players],
            weights=diagonal[n_players : 2 * n_players],
        )
    )
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    return beta, intercept + 5.0 * (off_mean + def_mean)


def cross_fitted_lineup_residuals(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> pd.DataFrame:
    """Predict every possession from a model that did not fit that game."""
    if frame["season"].nunique() != 1:
        raise ValueError("Cross-fitting expects exactly one season at a time.")
    design = build_design(frame, include_home=True)
    game_folds = np.asarray(
        [deterministic_game_fold(game_id, folds) for game_id in design.game_ids],
        dtype=np.int8,
    )
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold in range(folds):
        test_mask = game_folds == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Game fold {fold} has no train or test rows.")
        beta, intercept = _solve_fold(
            design,
            train_mask,
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
        )
        predictions[test_mask] = np.asarray(design.X[test_mask] @ beta).ravel() + intercept
    if not np.isfinite(predictions).all():
        raise AssertionError("Every possession must receive one out-of-fold prediction.")
    result = frame[
        [
            "gameid",
            "season",
            "period",
            "num",
            "offense_margin_before",
            "regulation_seconds_elapsed",
            "six_minute_bucket",
            "regulation",
            "pts",
        ]
    ].copy()
    result["game_fold"] = game_folds
    result["lineup_expected_points"] = predictions
    result["lineup_residual_points"] = result["pts"].to_numpy(dtype=float) - predictions
    return result


def _time_bucket(frame: pd.DataFrame, buckets: int) -> np.ndarray:
    if buckets not in {1, 4, 8}:
        raise ValueError("time_buckets must be 1, 4, or 8.")
    six_minute = frame["six_minute_bucket"].to_numpy(dtype=int)
    if buckets == 1:
        return np.zeros(len(frame), dtype=int)
    if buckets == 4:
        return six_minute // 2
    return six_minute


def rubberband_design(frame: pd.DataFrame, spec: RubberbandSpec) -> np.ndarray:
    """Build time intercepts and time-specific signed-margin slopes."""
    buckets = _time_bucket(frame, spec.time_buckets)
    margin = frame["offense_margin_before"].to_numpy(dtype=float)
    if spec.margin_clip is not None:
        margin = np.clip(margin, -spec.margin_clip, spec.margin_clip)
    one_hot = np.eye(spec.time_buckets, dtype=float)[buckets]
    return np.column_stack([one_hot, one_hot * margin[:, None]])


def fit_rubberband(
    frame: pd.DataFrame,
    spec: RubberbandSpec,
    *,
    cluster_covariance: bool = False,
) -> RubberbandFit:
    """Fit the score effect after out-of-fold lineup residualization."""
    clean = frame.loc[frame["regulation"].astype(bool)].reset_index(drop=True)
    X = rubberband_design(clean, spec)
    y = clean["lineup_residual_points"].to_numpy(dtype=float)
    xtx = X.T @ X
    beta = np.linalg.solve(xtx + np.eye(X.shape[1]) * 1e-10, X.T @ y)
    covariance = None
    if cluster_covariance:
        residual = y - X @ beta
        meat = np.zeros_like(xtx)
        for indices in clean.groupby("gameid", sort=False).indices.values():
            score = X[indices].T @ residual[indices]
            meat += np.outer(score, score)
        bread = np.linalg.pinv(xtx)
        groups = clean["gameid"].nunique()
        correction = groups / (groups - 1) * (len(clean) - 1) / (len(clean) - X.shape[1])
        covariance = correction * bread @ meat @ bread
    return RubberbandFit(
        spec=spec,
        coefficients=beta,
        covariance=covariance,
        rows=len(clean),
        games=clean["gameid"].nunique(),
    )


def predict_rubberband(fit: RubberbandFit, frame: pd.DataFrame) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    regulation = frame["regulation"].astype(bool).to_numpy()
    if regulation.any():
        prediction[regulation] = rubberband_design(
            frame.loc[regulation], fit.spec
        ) @ fit.coefficients
    return prediction


def score_adjustment(
    frame: pd.DataFrame,
    baseline_fit: RubberbandFit,
    candidate_fit: RubberbandFit,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    actual = frame["lineup_residual_points"].to_numpy(dtype=float)
    baseline = predict_rubberband(baseline_fit, frame)
    candidate = predict_rubberband(candidate_fit, frame)
    baseline_error = actual - baseline
    candidate_error = actual - candidate
    rows = pd.DataFrame(
        {
            "gameid": frame["gameid"].astype(str).to_numpy(),
            "baseline_squared_error": baseline_error**2,
            "candidate_squared_error": candidate_error**2,
        }
    )
    games = rows.groupby("gameid", as_index=False).agg(
        rows=("baseline_squared_error", "size"),
        baseline_squared_error=("baseline_squared_error", "sum"),
        candidate_squared_error=("candidate_squared_error", "sum"),
    )
    return (
        {
            "rows": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "baseline_rmse": float(math.sqrt(np.mean(baseline_error**2))),
            "candidate_rmse": float(math.sqrt(np.mean(candidate_error**2))),
            "mean_squared_error_improvement": float(
                np.mean(baseline_error**2 - candidate_error**2)
            ),
            "residual_correlation": float(np.corrcoef(actual, candidate)[0, 1]),
        },
        games,
    )


def paired_game_bootstrap(
    game_losses: pd.DataFrame,
    *,
    draws: int = 2000,
    seed: int = 20260825,
) -> dict[str, float | int]:
    """Resample fixed game-level sufficient statistics, not full RAPM fits."""
    rng = np.random.default_rng(seed)
    values = (
        game_losses["baseline_squared_error"].to_numpy(dtype=float)
        - game_losses["candidate_squared_error"].to_numpy(dtype=float)
    )
    rows = game_losses["rows"].to_numpy(dtype=float)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(game_losses), len(game_losses))
        estimates[draw] = values[indices].sum() / rows[indices].sum()
    return {
        "draws": int(draws),
        "seed": int(seed),
        "mean_mse_improvement": float(estimates.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_improvement": float(np.mean(estimates > 0)),
    }


def coefficient_table(fit: RubberbandFit) -> pd.DataFrame:
    """Return interpretable slope estimates in points per 100."""
    buckets = fit.spec.time_buckets
    slope = fit.coefficients[buckets:]
    if fit.covariance is None:
        standard_error = np.full(buckets, np.nan)
    else:
        standard_error = np.sqrt(np.diag(fit.covariance)[buckets:])
    width_minutes = 48 / buckets
    return pd.DataFrame(
        {
            "time_bucket": np.arange(buckets, dtype=int),
            "minutes_elapsed_start": np.arange(buckets) * width_minutes,
            "minutes_elapsed_end": (np.arange(buckets) + 1) * width_minutes,
            "slope_points_per_100_per_margin_point": 100.0 * slope,
            "standard_error": 100.0 * standard_error,
            "lower_95": 100.0 * (slope - 1.96 * standard_error),
            "upper_95": 100.0 * (slope + 1.96 * standard_error),
        }
    )
