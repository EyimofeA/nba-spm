"""JE-style categorical score-state terms inside possession RAPM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve

from nba_impact.models.rapm import RapmConfig, RapmDesign


@dataclass(frozen=True)
class ScoreStateRapmFit:
    player_coefficients: np.ndarray
    state_coefficients: np.ndarray
    intercept: float
    state_values: np.ndarray
    state_counts: np.ndarray
    bucket_width: int
    rows: int


@dataclass(frozen=True)
class ClockMarginRapmFit:
    """Player RAPM with jointly fitted actual-clock score-margin slopes."""

    player_coefficients: np.ndarray
    context_coefficients: np.ndarray
    context_mean: float
    intercept: float
    time_buckets: int
    margin_clip: float
    rows: int


def clock_margin_design(
    frame: pd.DataFrame,
    *,
    time_buckets: int = 8,
    margin_clip: float = 15.0,
) -> csr_matrix:
    """Encode signed score margin inside actual-clock buckets.

    The clipped margin is scaled to [-1, 1], which puts its ridge penalty on a
    scale comparable to the signed home indicator. Overtime receives zeros.
    """
    required = {"offense_margin_before", "six_minute_bucket", "regulation"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Clock-margin input is missing columns: {missing}")
    if time_buckets not in {1, 4, 8}:
        raise ValueError("time_buckets must be 1, 4, or 8")
    if margin_clip <= 0:
        raise ValueError("margin_clip must be positive")
    six_minute = frame["six_minute_bucket"].to_numpy(dtype=int)
    if time_buckets == 1:
        bucket = np.zeros(len(frame), dtype=int)
    elif time_buckets == 4:
        bucket = six_minute // 2
    else:
        bucket = six_minute
    regulation = frame["regulation"].astype(bool).to_numpy()
    margin = np.clip(
        frame["offense_margin_before"].to_numpy(dtype=float),
        -margin_clip,
        margin_clip,
    ) / margin_clip
    rows = np.flatnonzero(regulation)
    return csr_matrix(
        (margin[rows], (rows, bucket[rows])),
        shape=(len(frame), time_buckets),
    )


def fit_clock_margin_rapm(
    design: RapmDesign,
    frame: pd.DataFrame,
    config: RapmConfig,
    *,
    time_buckets: int = 8,
    margin_clip: float = 15.0,
    context_penalty: float = 300.0,
    row_mask: np.ndarray | None = None,
) -> ClockMarginRapmFit:
    """Fit player, home, and actual-clock margin terms in one ridge system."""
    if context_penalty < 0:
        raise ValueError("context_penalty must be non-negative")
    if len(frame) != design.X.shape[0]:
        raise ValueError("Clock-margin rows must match the RAPM design")
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    if mask.shape != (design.X.shape[0],) or not mask.any():
        raise ValueError("row_mask must select at least one design row")
    context = clock_margin_design(
        frame, time_buckets=time_buckets, margin_clip=margin_clip
    )
    player_design = design.X[mask]
    context_train = context[mask]
    matrix = hstack([player_design, context_train], format="csr")
    y = design.y[mask]
    intercept = float(y.mean())
    n_players = len(design.players)
    penalties = np.concatenate(
        [
            np.full(n_players, config.lambda_off, dtype=float),
            np.full(n_players, config.lambda_def, dtype=float),
            np.asarray([config.lambda_home], dtype=float)
            if config.include_home
            else np.empty(0, dtype=float),
            np.full(time_buckets, context_penalty, dtype=float),
        ]
    )
    lhs = (matrix.T @ matrix).tocsr() + diags(penalties, format="csr")
    rhs = np.asarray(matrix.T @ (y - intercept)).ravel()
    try:
        coefficients, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        coefficients, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        coefficients = spsolve(lhs.tocsc(), rhs)
    coefficients = np.asarray(coefficients, dtype=float)
    player_coefficients = coefficients[: design.X.shape[1]].copy()
    context_coefficients = coefficients[design.X.shape[1] :].copy()

    off_counts = np.asarray(player_design[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(
        player_design[:, n_players : 2 * n_players].sum(axis=0)
    ).ravel()
    off_mean = float(np.average(player_coefficients[:n_players], weights=off_counts))
    def_mean = float(
        np.average(
            player_coefficients[n_players : 2 * n_players], weights=def_counts
        )
    )
    player_coefficients[:n_players] -= off_mean
    player_coefficients[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)

    # Define the player-only prediction at average score context while keeping
    # the joint fitted values exactly unchanged.
    context_mean = float(np.asarray(context_train @ context_coefficients).mean())
    intercept += context_mean
    return ClockMarginRapmFit(
        player_coefficients=player_coefficients,
        context_coefficients=context_coefficients,
        context_mean=context_mean,
        intercept=intercept,
        time_buckets=time_buckets,
        margin_clip=margin_clip,
        rows=int(mask.sum()),
    )


def predict_clock_margin_rapm(
    fit: ClockMarginRapmFit,
    design: RapmDesign,
    frame: pd.DataFrame,
    *,
    row_mask: np.ndarray | None = None,
    include_context: bool = True,
) -> np.ndarray:
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    prediction = (
        np.asarray(design.X[mask] @ fit.player_coefficients).ravel() + fit.intercept
    )
    if include_context:
        context = clock_margin_design(
            frame,
            time_buckets=fit.time_buckets,
            margin_clip=fit.margin_clip,
        )
        prediction += (
            np.asarray(context[mask] @ fit.context_coefficients).ravel()
            - fit.context_mean
        )
    return prediction


def clock_margin_curve(fit: ClockMarginRapmFit) -> pd.DataFrame:
    """Return each clock bucket's fitted effect for margins -30 through 30."""
    rows: list[dict[str, float | int]] = []
    for bucket, coefficient in enumerate(fit.context_coefficients):
        for margin in range(-30, 31):
            effect = coefficient * np.clip(
                margin / fit.margin_clip, -1.0, 1.0
            )
            rows.append(
                {
                    "time_bucket": bucket,
                    "minutes_elapsed_start": bucket * 48 / fit.time_buckets,
                    "minutes_elapsed_end": (bucket + 1) * 48 / fit.time_buckets,
                    "offense_margin_before": margin,
                    "effect_points_per_100_vs_tie": 100.0 * effect,
                }
            )
    return pd.DataFrame(rows)


def annotate_offense_margin_before(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the offense's score lead before every possession."""
    required = {"gameid", "period", "num", "pts", "home_poss"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Score-state input is missing columns: {missing}")
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
    return (
        result.sort_values("_source_order", kind="stable")
        .drop(columns=["_source_order", "_home_points", "_away_points"])
        .reset_index(drop=True)
    )


def score_state_indices(
    margins: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    bucket_width: int = 1,
) -> np.ndarray:
    """Map integer margins to centered, top-coded score buckets."""
    if minimum >= maximum:
        raise ValueError("minimum must be smaller than maximum")
    if bucket_width < 1:
        raise ValueError("bucket_width must be positive")
    if minimum % bucket_width or maximum % bucket_width:
        raise ValueError("minimum and maximum must be multiples of bucket_width")
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Score margins must be finite.")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        raise ValueError("Score margins must be integer-valued.")
    bucketed = (
        np.floor((rounded + bucket_width / 2.0) / bucket_width).astype(int)
        * bucket_width
    )
    return (
        np.clip(bucketed, minimum, maximum) - minimum
    ) // bucket_width


def fit_score_state_rapm(
    design: RapmDesign,
    margins: np.ndarray,
    config: RapmConfig,
    *,
    minimum: int = -57,
    maximum: int = 57,
    bucket_width: int = 1,
    state_penalty: float = 1.0,
    row_mask: np.ndarray | None = None,
) -> ScoreStateRapmFit:
    """Fit player and exact score-margin indicators in one ridge system.

    Player penalties remain the project's frozen RAPM penalties. Score-state
    indicators use the alpha=1 penalty in the public reproduction of JE's
    method. The fitted state block is weighted-mean centered so removing it
    yields an average-context player-only prediction without changing the joint
    model's fitted values.
    """
    if state_penalty < 0:
        raise ValueError("state_penalty must be non-negative")
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    if mask.shape != (design.X.shape[0],) or not mask.any():
        raise ValueError("row_mask must select at least one design row")
    all_indices = score_state_indices(
        margins,
        minimum=minimum,
        maximum=maximum,
        bucket_width=bucket_width,
    )
    indices = all_indices[mask]
    rows = np.arange(mask.sum(), dtype=np.int64)
    state_count = (maximum - minimum) // bucket_width + 1
    state_design = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, indices)),
        shape=(len(rows), state_count),
    )
    player_design = design.X[mask]
    matrix = hstack([player_design, state_design], format="csr")
    y = design.y[mask]
    intercept = float(y.mean())
    n_players = len(design.players)
    penalties = np.concatenate(
        [
            np.full(n_players, config.lambda_off, dtype=float),
            np.full(n_players, config.lambda_def, dtype=float),
            np.asarray([config.lambda_home], dtype=float)
            if config.include_home
            else np.empty(0, dtype=float),
            np.full(state_count, state_penalty, dtype=float),
        ]
    )
    lhs = (matrix.T @ matrix).tocsr() + diags(penalties, format="csr")
    rhs = np.asarray(matrix.T @ (y - intercept)).ravel()
    try:
        coefficients, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        coefficients, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        coefficients = spsolve(lhs.tocsc(), rhs)
    coefficients = np.asarray(coefficients, dtype=float)

    player_coefficients = coefficients[: design.X.shape[1]].copy()
    state_coefficients = coefficients[design.X.shape[1] :].copy()

    # Preserve the established offense/defense level convention.
    off_counts = np.asarray(player_design[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(
        player_design[:, n_players : 2 * n_players].sum(axis=0)
    ).ravel()
    off_mean = float(np.average(player_coefficients[:n_players], weights=off_counts))
    def_mean = float(
        np.average(
            player_coefficients[n_players : 2 * n_players], weights=def_counts
        )
    )
    player_coefficients[:n_players] -= off_mean
    player_coefficients[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)

    state_counts = np.bincount(indices, minlength=state_count).astype(np.int64)
    state_mean = float(np.average(state_coefficients, weights=state_counts))
    state_coefficients -= state_mean
    intercept += state_mean
    return ScoreStateRapmFit(
        player_coefficients=player_coefficients,
        state_coefficients=state_coefficients,
        intercept=intercept,
        state_values=np.arange(minimum, maximum + 1, bucket_width, dtype=int),
        state_counts=state_counts,
        bucket_width=bucket_width,
        rows=int(mask.sum()),
    )


def predict_score_state_rapm(
    fit: ScoreStateRapmFit,
    design: RapmDesign,
    margins: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
    include_score_state: bool = True,
) -> np.ndarray:
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    prediction = (
        np.asarray(design.X[mask] @ fit.player_coefficients).ravel() + fit.intercept
    )
    if include_score_state:
        indices = score_state_indices(
            np.asarray(margins)[mask],
            minimum=int(fit.state_values[0]),
            maximum=int(fit.state_values[-1]),
            bucket_width=fit.bucket_width,
        )
        prediction += fit.state_coefficients[indices]
    return prediction


def score_state_curve(fit: ScoreStateRapmFit) -> pd.DataFrame:
    """Return effects relative to a tied score, in points per 100."""
    tie = int(np.flatnonzero(fit.state_values == 0)[0])
    effect = 100.0 * (fit.state_coefficients - fit.state_coefficients[tie])
    return pd.DataFrame(
        {
            "margin": fit.state_values,
            "effect_points_per_100_vs_tie": effect,
            "possessions": fit.state_counts,
            "state": np.where(
                fit.state_values < 0,
                "Trailing",
                np.where(fit.state_values > 0, "Leading", "Tied"),
            ),
        }
    )
