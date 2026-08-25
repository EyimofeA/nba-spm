"""Standalone pair, trio, four-player, and five-player RAPM models.

Each model contains only k-player offensive units, k-player defensive units,
and one signed home-offense column. Individual player columns are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import LinearOperator, lsmr

from nba_impact.models.lineup_interactions import (
    build_interaction_design,
    combination_vocabulary_exposure,
)


@dataclass(frozen=True)
class UnitRapmFit:
    order: int
    unit_penalty: float
    home_penalty: float
    minimum_exposure: int
    combinations: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fit_iterations: int
    fit_stop_code: int
    penalty_strategy: str = "hard_floor"


def _aggregate_rows(
    offense: np.ndarray,
    defense: np.ndarray,
    home_offense: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse exact offense, defense, and venue rows for weighted fitting."""
    signed_home = np.where(np.asarray(home_offense, dtype=bool), 1, -1)
    keys = np.column_stack([offense, defense, signed_home])
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.bincount(
        inverse, weights=np.asarray(target, dtype=float), minlength=len(unique)
    )
    means = sums / counts
    return unique[:, :10], unique[:, 10].astype(float), inverse, counts.astype(float), means


def _design_with_home(
    matchup_keys: np.ndarray,
    signed_home: np.ndarray,
    combinations: np.ndarray,
    *,
    order: int,
) -> csr_matrix:
    units = build_interaction_design(matchup_keys, combinations, order=order)
    return hstack(
        [units, csr_matrix(np.asarray(signed_home, dtype=float)[:, None])],
        format="csr",
    )


def fit_unit_rapm(
    offense: np.ndarray,
    defense: np.ndarray,
    home_offense: np.ndarray,
    points: np.ndarray,
    *,
    order: int,
    unit_penalty: float,
    minimum_exposure: int,
    home_penalty: float = 300.0,
    maximum_iterations: int = 500,
    penalty_strategy: str = "hard_floor",
) -> UnitRapmFit:
    """Fit raw possession points using k-player units and no player columns."""
    if order not in {2, 3, 4, 5}:
        raise ValueError("order must be one of 2, 3, 4, or 5.")
    if unit_penalty <= 0 or home_penalty <= 0:
        raise ValueError("Ridge penalties must be positive.")

    keys, signed_home, _, counts, means = _aggregate_rows(
        offense, defense, home_offense, points
    )
    vocabulary_floor = minimum_exposure if penalty_strategy == "hard_floor" else max(
        5, minimum_exposure // 100
    )
    vocabulary, exposure = combination_vocabulary_exposure(
        keys,
        counts,
        order=order,
        minimum_exposure=vocabulary_floor,
    )
    if len(vocabulary) == 0:
        raise ValueError("No units satisfy the exposure floor.")
    design = _design_with_home(
        keys, signed_home, vocabulary, order=order
    )
    intercept = float(np.average(means, weights=counts))
    root_weight = np.sqrt(counts)
    weighted = design.multiply(root_weight[:, None]).tocsr()
    target = (means - intercept) * root_weight
    n_rows, n_columns = weighted.shape
    if penalty_strategy == "hard_floor":
        unit_penalties = np.full(2 * len(vocabulary), float(unit_penalty))
    elif penalty_strategy == "exposure_buckets":
        ratio = exposure / float(minimum_exposure)
        multiplier = np.select(
            [ratio < 0.1, ratio < 0.5, ratio < 1.0],
            [100.0, 10.0, 3.0],
            default=1.0,
        )
        unit_penalties = np.tile(float(unit_penalty) * multiplier, 2)
    elif penalty_strategy == "inverse_exposure":
        multiplier = np.clip(minimum_exposure / exposure, 1.0, 100.0)
        unit_penalties = np.tile(float(unit_penalty) * multiplier, 2)
    else:
        raise ValueError(
            "penalty_strategy must be hard_floor, exposure_buckets, or inverse_exposure."
        )
    penalty = np.concatenate(
        [
            unit_penalties,
            np.array([float(home_penalty)]),
        ]
    )
    root_penalty = np.sqrt(penalty)

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
    return UnitRapmFit(
        order=order,
        unit_penalty=float(unit_penalty),
        home_penalty=float(home_penalty),
        minimum_exposure=int(minimum_exposure),
        combinations=vocabulary,
        coefficients=np.asarray(solution[0]),
        intercept=intercept,
        fit_iterations=int(solution[2]),
        fit_stop_code=int(solution[1]),
        penalty_strategy=penalty_strategy,
    )


def predict_unit_rapm(
    fit: UnitRapmFit,
    offense: np.ndarray,
    defense: np.ndarray,
    home_offense: np.ndarray,
) -> np.ndarray:
    """Predict points per possession; unseen units receive zero coefficients."""
    keys, signed_home, inverse, _, _ = _aggregate_rows(
        offense,
        defense,
        home_offense,
        np.zeros(len(offense), dtype=float),
    )
    design = _design_with_home(
        keys, signed_home, fit.combinations, order=fit.order
    )
    return fit.intercept + np.asarray(design @ fit.coefficients).ravel()[inverse]


def unit_slot_coverage(
    fit: UnitRapmFit,
    offense: np.ndarray,
    defense: np.ndarray,
) -> float:
    """Share of possible offensive and defensive unit slots seen in training."""
    keys = np.column_stack([offense, defense])
    design = build_interaction_design(keys, fit.combinations, order=fit.order)
    possible = 2 * math.comb(5, fit.order) * len(keys)
    return float(design.nnz / possible) if possible else float("nan")
