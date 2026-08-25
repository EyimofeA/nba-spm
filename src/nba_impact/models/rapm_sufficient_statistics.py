"""Persist RAPM sufficient statistics for reproducible penalty research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, load_npz, save_npz
from scipy.sparse.linalg import cg, spsolve
from scipy.stats import norm

from nba_impact.data.manifest import write_json_atomic
from nba_impact.models.rapm import RapmDesign, build_design


COMPONENTS = ("offense", "defense", "home")


@dataclass(frozen=True)
class StoredRidgeSolution:
    """Generalized-ridge solution in raw and published coordinates."""

    beta: np.ndarray
    intercept: float
    players: np.ndarray
    raw_beta: np.ndarray
    solver: str


def _season_adjusted_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    season_means = frame.groupby("season")["pts"].mean().sort_index()
    observed = tuple(int(value) for value in season_means.index)
    if not observed:
        raise ValueError("Sufficient statistics require at least one season.")
    overall_mean = float(frame["pts"].mean())
    adjusted = frame.copy()
    adjusted["pts"] = adjusted["pts"] - adjusted["season"].map(season_means) + overall_mean
    metadata = {
        "seasons": list(observed),
        "overall_points_per_possession": overall_mean,
        "season_points_per_possession": {
            str(int(season)): float(value) for season, value in season_means.items()
        },
    }
    return adjusted, metadata


def season_adjusted_design(frame: pd.DataFrame) -> tuple[RapmDesign, dict]:
    """Build the exact rolling-RAPM design after removing season environments."""
    adjusted, metadata = _season_adjusted_frame(frame)
    return build_design(adjusted, include_home=True), metadata


def _combined_design(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame | None,
) -> tuple[RapmDesign, int, dict]:
    adjusted_train, environment = _season_adjusted_frame(train_frame)
    if evaluation_frame is None:
        return build_design(adjusted_train, include_home=True), len(train_frame), environment
    combined = pd.concat([adjusted_train, evaluation_frame], ignore_index=True)
    return build_design(combined, include_home=True), len(train_frame), environment


def _game_aggregates(
    design: RapmDesign,
    evaluation_frame: pd.DataFrame,
    train_rows: int,
    train_off_counts: np.ndarray,
    train_def_counts: np.ndarray,
) -> tuple[csr_matrix, dict[str, np.ndarray]]:
    evaluation = design.X[train_rows:].tocsr()
    game_codes, game_ids = pd.factorize(
        evaluation_frame["gameid"].astype(str), sort=True
    )
    sign = np.where(
        evaluation_frame["home_poss"].to_numpy(dtype=bool), 1.0, -1.0
    )
    aggregator = csr_matrix(
        (
            sign,
            (game_codes, np.arange(len(evaluation_frame), dtype=np.int64)),
        ),
        shape=(len(game_ids), len(evaluation_frame)),
    )
    count_aggregator = csr_matrix(
        (
            np.ones(len(evaluation_frame), dtype=np.float64),
            (game_codes, np.arange(len(evaluation_frame), dtype=np.int64)),
        ),
        shape=(len(game_ids), len(evaluation_frame)),
    )
    game_design = (aggregator @ evaluation).tocsr()
    actual_margin = np.asarray(
        aggregator @ evaluation_frame["pts"].to_numpy(dtype=np.float64)
    ).ravel()
    intercept_multiplier = np.asarray(
        aggregator @ np.ones(len(evaluation_frame), dtype=np.float64)
    ).ravel()

    n_players = len(design.players)
    known = (train_off_counts + train_def_counts) > 0
    player_entries = evaluation[:, : 2 * n_players].tocoo()
    unknown_entry = (~known[player_entries.col % n_players]).astype(np.int16)
    unknown_by_possession = np.bincount(
        player_entries.row,
        weights=unknown_entry,
        minlength=len(evaluation_frame),
    )
    unknown_slots = np.asarray(count_aggregator @ unknown_by_possession).ravel()
    arrays = {
        "game_ids": np.asarray(game_ids, dtype=str),
        "actual_margin": actual_margin,
        "intercept_multiplier": intercept_multiplier,
        "unknown_player_slots": unknown_slots,
    }
    return game_design, arrays


def store_lambda_research_matrices(
    train_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    evaluation_frame: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    """Store train cross-products and optional next-season game aggregates.

    The possession-level design is intentionally not persisted. ``X'X`` and
    centered ``X'y`` are sufficient for any diagonal offense/defense/home ridge
    penalty. Game-level evaluation aggregates reproduce held-out margin scoring.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    design, train_rows, environment = _combined_design(train_frame, evaluation_frame)
    X_train = design.X[:train_rows].tocsr()
    y_train = design.y[:train_rows]
    base_intercept = float(y_train.mean())
    centered_y = y_train - base_intercept
    xtx = (X_train.T @ X_train).tocsr()
    xty_centered = np.asarray(X_train.T @ centered_y).ravel()
    n_players = len(design.players)
    off_counts = np.asarray(X_train[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(
        X_train[:, n_players : 2 * n_players].sum(axis=0)
    ).ravel()

    save_npz(output / "train_xtx.npz", xtx, compressed=True)
    np.save(output / "train_xty_centered.npy", xty_centered)
    np.save(output / "player_ids.npy", design.players)
    np.save(output / "train_off_possessions.npy", off_counts)
    np.save(output / "train_def_possessions.npy", def_counts)

    evaluation_summary: dict[str, object] = {"status": "not_available"}
    if evaluation_frame is not None:
        game_design, arrays = _game_aggregates(
            design,
            evaluation_frame,
            train_rows,
            off_counts,
            def_counts,
        )
        save_npz(output / "evaluation_game_design.npz", game_design, compressed=True)
        for name, values in arrays.items():
            np.save(output / f"evaluation_{name}.npy", values)
        evaluation_summary = {
            "status": "complete",
            "season": int(pd.to_numeric(evaluation_frame["season"]).unique()[0]),
            "games": int(len(arrays["game_ids"])),
            "possession_rows": int(len(evaluation_frame)),
            "game_design_shape": list(game_design.shape),
            "games_with_unknown_players": int(
                np.count_nonzero(arrays["unknown_player_slots"])
            ),
        }

    manifest = {
        "contract_version": "rapm_lambda_sufficient_statistics_v1",
        "train": {
            "seasons": environment["seasons"],
            "possession_rows": int(train_rows),
            "players_in_train_or_evaluation": n_players,
            "matrix_shape": list(xtx.shape),
            "matrix_nonzero": int(xtx.nnz),
            "base_intercept": base_intercept,
            "centered_y_sum_squares": float(centered_y @ centered_y),
            "season_environment": environment,
        },
        "columns": {
            "offense": [0, n_players],
            "points_allowed_defense": [n_players, 2 * n_players],
            "home": 2 * n_players,
            "player_order": "player_ids.npy",
            "published_defense_sign": "negative coefficient",
        },
        "stored": {
            "train_xtx": "train_xtx.npz",
            "train_xty_centered": "train_xty_centered.npy",
            "train_off_possessions": "train_off_possessions.npy",
            "train_def_possessions": "train_def_possessions.npy",
            "evaluation_game_design": (
                "evaluation_game_design.npz"
                if evaluation_frame is not None
                else None
            ),
        },
        "evaluation": evaluation_summary,
        "metadata": dict(metadata or {}),
        "forbidden_interpretation": (
            "These matrices support penalty research. They are not a selected "
            "lambda, a validated model, or Season 2027 confirmation."
        ),
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def solve_stored_ridge(
    matrix_dir: str | Path,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_home: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve one ridge configuration from stored sufficient statistics."""
    root = Path(matrix_dir).resolve()
    players = _load_train_bundle(str(root))[3]
    penalty = diagonal_penalty_matrix(
        len(players),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=lambda_home,
    )
    solution = solve_stored_generalized_ridge(root, penalty)
    return solution.beta, solution.intercept, solution.players


def diagonal_penalty_matrix(
    n_players: int,
    *,
    lambda_off: float | np.ndarray,
    lambda_def: float | np.ndarray,
    lambda_home: float,
) -> csr_matrix:
    """Build a diagonal penalty, optionally with one precision per player."""
    off = np.broadcast_to(np.asarray(lambda_off, dtype=np.float64), (n_players,))
    defense = np.broadcast_to(np.asarray(lambda_def, dtype=np.float64), (n_players,))
    values = np.concatenate([off, defense, np.asarray([lambda_home])])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Ridge precisions must be finite and nonnegative.")
    return diags(values, format="csr")


def bivariate_penalty_matrix(
    n_players: int,
    *,
    lambda_off: float | np.ndarray,
    lambda_def: float | np.ndarray,
    lambda_home: float,
    published_prior_correlation: float,
) -> csr_matrix:
    """Build player-level OFF/DEF prior precision with full 2x2 blocks.

    ``published_prior_correlation`` is the Gaussian prior correlation between
    positive-good offense and positive-good defense. Stored defensive
    coefficients are points allowed, so the coefficient-space cross term has
    the opposite sign transformation from the published defense coordinate.
    """
    correlation = float(published_prior_correlation)
    if not np.isfinite(correlation) or abs(correlation) >= 1.0:
        raise ValueError("Prior correlation must be finite and strictly inside (-1, 1).")
    off = np.broadcast_to(np.asarray(lambda_off, dtype=np.float64), (n_players,))
    defense = np.broadcast_to(np.asarray(lambda_def, dtype=np.float64), (n_players,))
    if (
        not np.isfinite(off).all()
        or not np.isfinite(defense).all()
        or (off < 0).any()
        or (defense < 0).any()
        or not np.isfinite(lambda_home)
        or lambda_home < 0
    ):
        raise ValueError("Bivariate ridge precisions must be finite and nonnegative.")
    scale = 1.0 / (1.0 - correlation**2)
    off_diagonal = off * scale
    def_diagonal = defense * scale
    # P_pub has -rho*sqrt(lambda_off*lambda_def). D_pub=-D_points_allowed,
    # so transforming P_pub to coefficient coordinates makes this positive.
    cross = correlation * np.sqrt(off * defense) * scale
    player_index = np.arange(n_players, dtype=np.int64)
    rows = np.concatenate(
        [player_index, n_players + player_index, player_index, n_players + player_index]
    )
    columns = np.concatenate(
        [player_index, n_players + player_index, n_players + player_index, player_index]
    )
    values = np.concatenate([off_diagonal, def_diagonal, cross, cross])
    rows = np.concatenate([rows, np.asarray([2 * n_players])])
    columns = np.concatenate([columns, np.asarray([2 * n_players])])
    values = np.concatenate([values, np.asarray([lambda_home], dtype=np.float64)])
    return coo_matrix(
        (values, (rows, columns)),
        shape=(2 * n_players + 1, 2 * n_players + 1),
    ).tocsr()


def solve_stored_generalized_ridge(
    matrix_dir: str | Path,
    penalty: csr_matrix,
) -> StoredRidgeSolution:
    """Solve a zero-centered generalized-ridge model from stored statistics."""
    root = Path(matrix_dir).resolve()
    manifest, xtx, rhs, players, off_counts, def_counts = _load_train_bundle(
        str(root)
    )
    penalty = penalty.tocsr().astype(np.float64)
    if penalty.shape != xtx.shape:
        raise ValueError(f"Penalty shape {penalty.shape} does not match {xtx.shape}.")
    asymmetry = penalty - penalty.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-10:
        raise ValueError("Generalized ridge penalty must be symmetric.")
    if not np.isfinite(penalty.data).all():
        raise ValueError("Generalized ridge penalty contains non-finite values.")
    lhs = (xtx + penalty).tocsr()
    try:
        raw_beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        raw_beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    solver = "conjugate_gradient"
    if info != 0:
        raw_beta = spsolve(lhs.tocsc(), rhs)
        solver = "sparse_direct_fallback"
    raw_beta = np.asarray(raw_beta)
    beta = raw_beta.copy()
    off_mean = float(np.average(beta[: len(players)], weights=off_counts))
    def_mean = float(
        np.average(beta[len(players) : 2 * len(players)], weights=def_counts)
    )
    beta[: len(players)] -= off_mean
    beta[len(players) : 2 * len(players)] -= def_mean
    intercept = float(manifest["train"]["base_intercept"]) + 5.0 * (
        off_mean + def_mean
    )
    return StoredRidgeSolution(
        beta=beta,
        intercept=intercept,
        players=players,
        raw_beta=raw_beta,
        solver=solver,
    )


def stored_training_diagnostics(
    matrix_dir: str | Path,
    solution: StoredRidgeSolution,
    penalty: csr_matrix,
    *,
    probes: int,
    seed: int,
    return_inverse_diagonal: bool = False,
) -> dict[str, float | int | np.ndarray]:
    """Estimate effective degrees of freedom and GCV from training statistics.

    A deterministic Hutchinson estimator avoids forming the dense inverse. The
    returned GCV is diagnostic candidate-generation evidence, not confirmation.
    """
    if probes < 1:
        raise ValueError("At least one trace probe is required.")
    root = Path(matrix_dir).resolve()
    manifest, xtx, rhs, players, _, _ = _load_train_bundle(str(root))
    penalty = penalty.tocsr().astype(np.float64)
    if penalty.shape != xtx.shape:
        raise ValueError("Penalty shape does not match stored training statistics.")
    lhs = (xtx + penalty).tocsr()
    beta = solution.raw_beta
    y_square = float(manifest["train"]["centered_y_sum_squares"])
    residual_ss = float(y_square - 2.0 * beta @ rhs + beta @ (xtx @ beta))
    residual_ss = max(residual_ss, 0.0)
    rng = np.random.default_rng(seed)
    inverse_diagonal = np.zeros(len(beta), dtype=np.float64)
    trace_penalty_inverse = 0.0
    for _ in range(probes):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=len(beta))
        try:
            inverse_probe, info = cg(lhs, probe, rtol=1e-5, maxiter=10_000)
        except TypeError:
            inverse_probe, info = cg(lhs, probe, tol=1e-5, maxiter=10_000)
        if info != 0:
            inverse_probe = spsolve(lhs.tocsc(), probe)
        inverse_probe = np.asarray(inverse_probe)
        inverse_diagonal += probe * inverse_probe
        trace_penalty_inverse += float(inverse_probe @ (penalty @ probe))
    inverse_diagonal /= probes
    parameter_count = len(beta)
    effective_df = float(parameter_count - trace_penalty_inverse / probes)
    rows = int(manifest["train"]["possession_rows"])
    denominator = max(1.0 - effective_df / rows, np.finfo(float).eps)
    gcv = float((residual_ss / rows) / denominator**2)
    residual_variance = float(residual_ss / max(rows - effective_df, 1.0))
    result: dict[str, float | int | np.ndarray] = {
        "rows": rows,
        "parameters": parameter_count,
        "probes": probes,
        "effective_df": effective_df,
        "residual_sum_squares": residual_ss,
        "residual_variance": residual_variance,
        "gcv": gcv,
    }
    if return_inverse_diagonal:
        result["inverse_diagonal"] = inverse_diagonal
    return result


def stored_homoskedastic_ridge_intervals(
    matrix_dir: str | Path,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_home: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute fixed-window analytic ridge intervals from sufficient statistics.

    This is the homoskedastic sampling covariance ``sigma2 A^-1 X'X A^-1``.
    It is cheaper than a whole-game bootstrap and does not capture clustered
    game shocks, model bias, or window-selection uncertainty.
    """
    root = Path(matrix_dir).resolve()
    manifest, xtx, rhs, players, off_counts, def_counts = _load_train_bundle(
        str(root)
    )
    penalty = diagonal_penalty_matrix(
        len(players),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=lambda_home,
    )
    solution = solve_stored_generalized_ridge(root, penalty)
    penalty_values = penalty.diagonal()
    lhs = xtx.toarray()
    lhs[np.diag_indices_from(lhs)] += penalty_values
    factor = cho_factor(lhs, lower=True, check_finite=False)
    inverse = cho_solve(factor, np.eye(len(lhs)), check_finite=False)
    inverse = (inverse + inverse.T) / 2.0

    centered_y_ss = float(manifest["train"]["centered_y_sum_squares"])
    raw_beta = solution.raw_beta
    residual_ss = float(
        centered_y_ss - 2.0 * raw_beta @ rhs + raw_beta @ (xtx @ raw_beta)
    )
    effective_df = float(len(raw_beta) - penalty_values @ np.diag(inverse))
    rows = int(manifest["train"]["possession_rows"])
    residual_variance = max(residual_ss, 0.0) / max(rows - effective_df, 1.0)
    weighted_inverse = inverse * np.sqrt(penalty_values)[None, :]

    def covariance_with(vector: np.ndarray) -> np.ndarray:
        transformed = weighted_inverse.T @ vector
        return residual_variance * (
            inverse @ vector - weighted_inverse @ transformed
        )

    def variance_of(vector: np.ndarray) -> float:
        transformed = weighted_inverse.T @ vector
        return float(
            residual_variance
            * (vector @ (inverse @ vector) - transformed @ transformed)
        )

    raw_variance = residual_variance * (
        np.diag(inverse) - np.square(weighted_inverse).sum(axis=1)
    )
    n = len(players)
    off_weight = np.zeros(len(raw_beta))
    def_weight = np.zeros(len(raw_beta))
    off_weight[:n] = off_counts / off_counts.sum()
    def_weight[n : 2 * n] = def_counts / def_counts.sum()
    cov_off_mean = covariance_with(off_weight)
    cov_def_mean = covariance_with(def_weight)
    var_off_mean = variance_of(off_weight)
    var_def_mean = variance_of(def_weight)
    cov_means = float(off_weight @ cov_def_mean)
    raw_cross = residual_variance * (
        inverse[np.arange(n), n + np.arange(n)]
        - np.sum(weighted_inverse[:n] * weighted_inverse[n : 2 * n], axis=1)
    )
    off_variance = raw_variance[:n] + var_off_mean - 2.0 * cov_off_mean[:n]
    def_variance = (
        raw_variance[n : 2 * n]
        + var_def_mean
        - 2.0 * cov_def_mean[n : 2 * n]
    )
    centered_cross = (
        raw_cross
        - cov_def_mean[:n]
        - cov_off_mean[n : 2 * n]
        + cov_means
    )
    net_variance = off_variance + def_variance - 2.0 * centered_cross
    estimates = {
        "offense": 100.0 * solution.beta[:n],
        "defense": -100.0 * solution.beta[n : 2 * n],
    }
    estimates["net"] = estimates["offense"] + estimates["defense"]
    variances = {
        "offense": off_variance,
        "defense": def_variance,
        "net": net_variance,
    }
    output = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Poss_Off": off_counts,
            "Poss_Def": def_counts,
        }
    )
    for component in ("offense", "defense", "net"):
        estimate = estimates[component]
        standard_error = 100.0 * np.sqrt(np.clip(variances[component], 0.0, None))
        output[component] = estimate
        output[f"{component}_se"] = standard_error
        for level in (80, 95):
            critical = float(norm.ppf(0.5 + level / 200.0))
            output[f"{component}_ci{level}_low"] = estimate - critical * standard_error
            output[f"{component}_ci{level}_high"] = estimate + critical * standard_error
        output[f"{component}_probability_above_zero"] = norm.cdf(
            estimate / np.where(standard_error > 0, standard_error, np.nan)
        )
    output["uncertainty_method"] = "homoskedastic_analytic_ridge_sampling_covariance"
    output["uncertainty_status"] = "analytic_fixed_window_only"
    quality = {
        "rows": rows,
        "parameters": int(len(raw_beta)),
        "effective_df": effective_df,
        "residual_variance": float(residual_variance),
        "maximum_component_identity_error": float(
            np.abs(output["offense"] + output["defense"] - output["net"]).max()
        ),
    }
    return output, quality


@lru_cache(maxsize=32)
def _load_train_bundle(
    matrix_dir: str,
) -> tuple[dict, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(matrix_dir)
    return (
        json.loads((root / "manifest.json").read_text()),
        load_npz(root / "train_xtx.npz").tocsr(),
        np.load(root / "train_xty_centered.npy"),
        np.load(root / "player_ids.npy"),
        np.load(root / "train_off_possessions.npy"),
        np.load(root / "train_def_possessions.npy"),
    )


@lru_cache(maxsize=32)
def _load_evaluation_bundle(
    matrix_dir: str,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(matrix_dir)
    return (
        load_npz(root / "evaluation_game_design.npz").tocsr(),
        np.load(root / "evaluation_game_ids.npy"),
        np.load(root / "evaluation_actual_margin.npy"),
        np.load(root / "evaluation_intercept_multiplier.npy"),
        np.load(root / "evaluation_unknown_player_slots.npy"),
    )


def stored_evaluation_predictions(
    matrix_dir: str | Path,
    beta: np.ndarray,
    intercept: float,
) -> pd.DataFrame:
    """Return exact next-season game targets and predictions."""
    root = Path(matrix_dir).resolve()
    design, game_ids, actual, intercept_multiplier, unknown_slots = (
        _load_evaluation_bundle(str(root))
    )
    predicted = np.asarray(design @ beta).ravel() + intercept * intercept_multiplier
    return pd.DataFrame(
        {
            "game_id": game_ids.astype(str),
            "actual_margin": actual,
            "predicted_margin": predicted,
            "unknown_player_slots": unknown_slots,
        }
    )


def score_stored_evaluation(
    matrix_dir: str | Path,
    beta: np.ndarray,
    intercept: float,
) -> dict:
    """Score a stored coefficient vector on next-season game margins."""
    predictions = stored_evaluation_predictions(matrix_dir, beta, intercept)
    actual = predictions["actual_margin"].to_numpy()
    predicted = predictions["predicted_margin"].to_numpy()
    error = actual - predicted
    return {
        "games": int(len(actual)),
        "margin_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
    }
