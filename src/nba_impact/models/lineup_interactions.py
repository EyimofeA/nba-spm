"""Regularized residual interaction layers for 2- through 5-player units."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS


@dataclass(frozen=True)
class InteractionFit:
    order: int
    penalty: float
    minimum_exposure: int
    combinations: np.ndarray
    coefficients: np.ndarray
    fit_iterations: int
    fit_stop_code: int


def offense_defense_lineups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted offense and defense player IDs for each possession."""
    away = frame.loc[:, AWAY_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home = frame.loc[:, HOME_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home_offense = frame["home_poss"].to_numpy(dtype=bool, copy=False)
    offense = np.where(home_offense[:, None], home, away)
    defense = np.where(home_offense[:, None], away, home)
    return np.sort(offense, axis=1), np.sort(defense, axis=1)


def aggregate_lineup_rows(
    offense: np.ndarray,
    defense: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse identical matchup rows exactly for squared-error fitting."""
    keys = np.column_stack([offense, defense])
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.bincount(inverse, weights=np.asarray(target, dtype=float), minlength=len(unique))
    means = sums / counts
    return unique, inverse, counts.astype(float), means


def _combination_rows(lineups: np.ndarray, order: int) -> np.ndarray:
    patterns = tuple(itertools.combinations(range(5), order))
    return np.concatenate([lineups[:, pattern] for pattern in patterns], axis=0)


def build_combination_vocabulary(
    matchup_keys: np.ndarray,
    row_exposure: np.ndarray,
    *,
    order: int,
    minimum_exposure: int,
) -> np.ndarray:
    """Keep units reaching the same exposure floor on either side combined."""
    vocabulary, _ = combination_vocabulary_exposure(
        matchup_keys,
        row_exposure,
        order=order,
        minimum_exposure=minimum_exposure,
    )
    return vocabulary


def combination_vocabulary_exposure(
    matchup_keys: np.ndarray,
    row_exposure: np.ndarray,
    *,
    order: int,
    minimum_exposure: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained units and their combined offense/defense exposure."""
    if order not in {2, 3, 4, 5}:
        raise ValueError("order must be one of 2, 3, 4, or 5.")
    if minimum_exposure < 1:
        raise ValueError("minimum_exposure must be positive.")
    patterns = tuple(itertools.combinations(range(5), order))
    offense = matchup_keys[:, :5]
    defense = matchup_keys[:, 5:]
    combinations = np.vstack(
        [_combination_rows(offense, order), _combination_rows(defense, order)]
    )
    # _combination_rows is pattern-major, so exposure must also be pattern-major.
    exposures = np.tile(np.asarray(row_exposure, dtype=float), 2 * len(patterns))
    unique, inverse = np.unique(combinations, axis=0, return_inverse=True)
    total = np.bincount(inverse, weights=exposures, minlength=len(unique))
    keep = total >= minimum_exposure
    return unique[keep], total[keep]


def build_interaction_design(
    matchup_keys: np.ndarray,
    combinations: np.ndarray,
    *,
    order: int,
) -> csr_matrix:
    """Build offense and points-allowed defense unit columns."""
    if combinations.ndim != 2 or combinations.shape[1] != order:
        raise ValueError("Combination vocabulary shape does not match order.")
    lookup = {tuple(row): index for index, row in enumerate(combinations.tolist())}
    patterns = tuple(itertools.combinations(range(5), order))
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    n_rows = len(matchup_keys)
    n_combinations = len(combinations)
    row_ids = np.arange(n_rows, dtype=np.int64)
    for side, offset in ((matchup_keys[:, :5], 0), (matchup_keys[:, 5:], n_combinations)):
        for pattern in patterns:
            columns = np.fromiter(
                (lookup.get(tuple(row), -1) for row in side[:, pattern]),
                dtype=np.int64,
                count=n_rows,
            )
            keep = columns >= 0
            if keep.any():
                row_parts.append(row_ids[keep])
                column_parts.append(columns[keep] + offset)
    if not row_parts:
        return csr_matrix((n_rows, 2 * n_combinations), dtype=np.float64)
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(n_rows, 2 * n_combinations),
    )


def fit_interaction_layer(
    offense: np.ndarray,
    defense: np.ndarray,
    residual: np.ndarray,
    *,
    order: int,
    penalty: float,
    minimum_exposure: int,
    maximum_iterations: int = 500,
) -> InteractionFit:
    """Fit one interaction order to player-RAPM residuals with ridge LSMR."""
    if penalty <= 0:
        raise ValueError("penalty must be positive.")
    keys, _, counts, means = aggregate_lineup_rows(offense, defense, residual)
    vocabulary = build_combination_vocabulary(
        keys, counts, order=order, minimum_exposure=minimum_exposure
    )
    if len(vocabulary) == 0:
        raise ValueError("No combinations satisfy the exposure floor.")
    design = build_interaction_design(keys, vocabulary, order=order)
    root_weight = np.sqrt(counts)
    weighted = design.multiply(root_weight[:, None]).tocsr()
    target = means * root_weight
    root_penalty = math.sqrt(penalty)
    n_rows, n_columns = weighted.shape

    def matvec(values: np.ndarray) -> np.ndarray:
        return np.concatenate([weighted @ values, root_penalty * values])

    def rmatvec(values: np.ndarray) -> np.ndarray:
        return weighted.T @ values[:n_rows] + root_penalty * values[n_rows:]

    operator = LinearOperator(
        (n_rows + n_columns, n_columns),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    response = np.concatenate([target, np.zeros(n_columns, dtype=float)])
    solution = lsmr(
        operator,
        response,
        atol=1e-7,
        btol=1e-7,
        maxiter=maximum_iterations,
    )
    return InteractionFit(
        order=order,
        penalty=float(penalty),
        minimum_exposure=int(minimum_exposure),
        combinations=vocabulary,
        coefficients=np.asarray(solution[0]),
        fit_iterations=int(solution[2]),
        fit_stop_code=int(solution[1]),
    )


def predict_interaction_layer(
    fit: InteractionFit,
    offense: np.ndarray,
    defense: np.ndarray,
) -> np.ndarray:
    """Predict residual points per possession; unseen units contribute zero."""
    keys, inverse, _, _ = aggregate_lineup_rows(
        offense, defense, np.zeros(len(offense), dtype=float)
    )
    design = build_interaction_design(keys, fit.combinations, order=fit.order)
    return np.asarray(design @ fit.coefficients)[inverse]


def game_margin_metrics(
    frame: pd.DataFrame,
    predicted_points: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Score possession predictions as final home margin on exact games."""
    sign = np.where(frame["home_poss"].to_numpy(dtype=bool), 1.0, -1.0)
    rows = pd.DataFrame(
        {
            "gameid": frame["gameid"].astype(str).to_numpy(),
            "actual": sign * frame["pts"].to_numpy(dtype=float),
            "predicted": sign * np.asarray(predicted_points, dtype=float),
        }
    )
    games = rows.groupby("gameid", as_index=False).agg(
        actual_margin=("actual", "sum"), predicted_margin=("predicted", "sum")
    )
    error = games["actual_margin"] - games["predicted_margin"]
    predicted_variance = float(np.var(games["predicted_margin"], ddof=0))
    calibration_slope = (
        float(
            np.cov(
                games["actual_margin"], games["predicted_margin"], ddof=0
            )[0, 1]
            / predicted_variance
        )
        if predicted_variance > 0
        else float("nan")
    )
    metrics = {
        "games": int(len(games)),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": float(
            games[["actual_margin", "predicted_margin"]].corr().iloc[0, 1]
        ),
        "actual_margin_sd": float(games["actual_margin"].std(ddof=0)),
        "predicted_margin_sd": float(games["predicted_margin"].std(ddof=0)),
        "calibration_intercept": float(
            games["actual_margin"].mean()
            - calibration_slope * games["predicted_margin"].mean()
        ),
        "calibration_slope": calibration_slope,
    }
    return metrics, games
