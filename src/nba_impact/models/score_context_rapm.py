"""Joint nuisance-context controls for possession RAPM research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve
from sklearn.preprocessing import SplineTransformer

from nba_impact.models.rapm import RapmConfig, RapmDesign
from nba_impact.models.age_adjusted_rapm import AgeDesign


@dataclass(frozen=True)
class ContextRapmFit:
    player_coefficients: np.ndarray
    context_coefficients: np.ndarray
    context_column_means: np.ndarray
    context_mean: float
    intercept: float
    rows: int


def signed_score_bucket_design(
    margins: np.ndarray,
) -> tuple[csr_matrix, tuple[str, ...]]:
    """Encode separate leading and trailing score-magnitude buckets.

    A tied score is the all-zero reference. Boundaries are 1-5, 6-10,
    11-15, 16-20, and 21-plus points on each side.
    """
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Score margins must be finite")
    if not np.allclose(values, np.round(values)):
        raise ValueError("Score margins must be integer-valued")
    magnitude = np.abs(values)
    band = np.select(
        [
            (magnitude >= 1) & (magnitude <= 5),
            (magnitude >= 6) & (magnitude <= 10),
            (magnitude >= 11) & (magnitude <= 15),
            (magnitude >= 16) & (magnitude <= 20),
            magnitude >= 21,
        ],
        [0, 1, 2, 3, 4],
        default=-1,
    ).astype(int)
    active = band >= 0
    side = (values > 0).astype(int)  # trailing columns first, leading second
    columns = band + 5 * side
    rows = np.flatnonzero(active)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns[rows])),
        shape=(len(values), 10),
    )
    labels = tuple(
        f"{side_name}_{band_name}"
        for side_name in ("trailing", "leading")
        for band_name in ("1_5", "6_10", "11_15", "16_20", "21_plus")
    )
    return matrix, labels


def clipped_linear_score_design(
    margins: np.ndarray,
    *,
    clip: float,
) -> tuple[csr_matrix, tuple[str, ...]]:
    if clip <= 0:
        raise ValueError("clip must be positive")
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Score margins must be finite")
    scaled = np.clip(values, -clip, clip) / clip
    return csr_matrix(scaled[:, None]), (f"linear_clip_{clip:g}",)


def spline_score_design(
    margins: np.ndarray,
    *,
    train_mask: np.ndarray,
    clip: float,
    n_knots: int,
    degree: int = 3,
) -> tuple[csr_matrix, tuple[str, ...]]:
    """Fit a training-only B-spline basis and make a tie-relative design."""
    if clip <= 0:
        raise ValueError("clip must be positive")
    if n_knots < 3:
        raise ValueError("n_knots must be at least three")
    values = np.asarray(margins, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != values.shape or not mask.any():
        raise ValueError("train_mask must select score-margin rows")
    clipped = np.clip(values, -clip, clip)[:, None]
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots="quantile",
        extrapolation="constant",
        include_bias=False,
    )
    transformer.fit(clipped[mask])
    transformed = transformer.transform(clipped)
    tie = transformer.transform(np.asarray([[0.0]]))[0]
    transformed -= tie
    labels = tuple(f"spline_{index + 1}" for index in range(transformed.shape[1]))
    return csr_matrix(transformed), labels


def spline_age_design(
    age_design: AgeDesign,
    *,
    n_knots: int,
    degree: int = 3,
) -> tuple[csr_matrix, tuple[str, ...]]:
    """Convert categorical lineup ages into smooth offense/defense bases."""
    if n_knots < 3:
        raise ValueError("n_knots must be at least three")
    ages = age_design.ages.astype(float)
    full_range = np.arange(int(ages.min()), int(ages.max()) + 1, dtype=float)[:, None]
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots="uniform",
        extrapolation="constant",
        include_bias=False,
    )
    transformer.fit(full_range)
    basis = transformer.transform(ages[:, None])
    reference = transformer.transform(np.asarray([[float(age_design.reference_age)]]))[0]
    relative = basis - reference
    count = len(age_design.ages)
    offense = age_design.X[:, :count] @ relative
    defense = age_design.X[:, count:] @ relative
    matrix = csr_matrix(np.column_stack([offense, defense]))
    labels = tuple(
        f"{side}_age_spline_{index + 1}"
        for side in ("offense", "defense")
        for index in range(relative.shape[1])
    )
    return matrix, labels


def fit_context_rapm(
    design: RapmDesign,
    context: csr_matrix,
    config: RapmConfig,
    *,
    context_penalty: float | np.ndarray,
    row_mask: np.ndarray,
) -> ContextRapmFit:
    """Fit player, home, and supplied context columns in one ridge system."""
    context_penalties = np.asarray(context_penalty, dtype=float)
    if context_penalties.ndim == 0:
        context_penalties = np.full(context.shape[1], float(context_penalties))
    if context_penalties.shape != (context.shape[1],):
        raise ValueError("context_penalty must be scalar or match context columns")
    if (context_penalties < 0).any():
        raise ValueError("context_penalty must be non-negative")
    if context.shape[0] != design.X.shape[0] or context.shape[1] < 1:
        raise ValueError("context must align with the RAPM design and have columns")
    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != (design.X.shape[0],) or not mask.any():
        raise ValueError("row_mask must select at least one design row")
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
            context_penalties,
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
    player = coefficients[: design.X.shape[1]].copy()
    nuisance = coefficients[design.X.shape[1] :].copy()

    off_counts = np.asarray(player_design[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(
        player_design[:, n_players : 2 * n_players].sum(axis=0)
    ).ravel()
    off_mean = float(np.average(player[:n_players], weights=off_counts))
    def_mean = float(
        np.average(player[n_players : 2 * n_players], weights=def_counts)
    )
    player[:n_players] -= off_mean
    player[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)

    context_column_means = np.asarray(context_train.mean(axis=0)).ravel()
    context_mean = float(context_column_means @ nuisance)
    intercept += context_mean
    return ContextRapmFit(
        player_coefficients=player,
        context_coefficients=nuisance,
        context_column_means=context_column_means,
        context_mean=context_mean,
        intercept=intercept,
        rows=int(mask.sum()),
    )


def predict_context_rapm(
    fit: ContextRapmFit,
    design: RapmDesign,
    context: csr_matrix,
    *,
    row_mask: np.ndarray,
    include_context: bool,
    include_columns: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.asarray(row_mask, dtype=bool)
    prediction = (
        np.asarray(design.X[mask] @ fit.player_coefficients).ravel() + fit.intercept
    )
    if include_context:
        selected = (
            np.ones(context.shape[1], dtype=bool)
            if include_columns is None
            else np.asarray(include_columns, dtype=bool)
        )
        if selected.shape != (context.shape[1],):
            raise ValueError("include_columns must match context columns")
        prediction += (
            np.asarray(
                context[mask][:, selected] @ fit.context_coefficients[selected]
            ).ravel()
            - float(
                fit.context_column_means[selected]
                @ fit.context_coefficients[selected]
            )
        )
    return prediction
