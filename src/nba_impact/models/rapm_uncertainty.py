"""Game-cluster uncertainty for the frozen zero-prior normal RAPM model.

The bootstrap is the publication method. The analytic CR0 ridge sandwich is a
diagnostic. Both use the same terminal-lineup, zero-prior point estimator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve
from scipy.stats import norm

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import RapmConfig, RapmDesign, _penalty, build_design


@dataclass(frozen=True)
class RapmUncertaintyConfig:
    draws: int = 1000
    seed: int = 20260812
    method: str = "whole_game_bootstrap_stratified_by_season"
    contract_version: str = "normal_rapm_uncertainty_contract_v1"


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(f"{path.suffix}.partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def _solve(lhs, rhs: np.ndarray) -> np.ndarray:
    try:
        result, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:  # scipy before rtol
        result, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        result = spsolve(lhs.tocsc(), rhs)
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Ridge solver returned non-finite coefficients.")
    return result


def _recenter(
    beta: np.ndarray,
    off_counts: np.ndarray,
    def_counts: np.ndarray,
    n_players: int,
    *,
    intercept: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Apply the existing weighted-identification constraint as a linear map."""
    if not off_counts.any() or not def_counts.any():
        raise ValueError("A RAPM fit needs observed offensive and defensive rows.")
    off_weights = off_counts / off_counts.sum()
    def_weights = def_counts / def_counts.sum()
    transformed = np.asarray(beta, dtype=np.float64).copy()
    off_mean = float(off_weights @ transformed[:n_players])
    def_mean = float(def_weights @ transformed[n_players : 2 * n_players])
    transformed[:n_players] -= off_mean
    transformed[n_players : 2 * n_players] -= def_mean
    transformed_intercept = intercept + 5.0 * (off_mean + def_mean)

    # J maps raw coefficients into centered coefficient space. The intercept
    # has no effect on player intervals and is deliberately omitted here.
    columns = len(beta)
    transform = np.eye(columns, dtype=np.float64)
    transform[:n_players, :n_players] -= np.outer(
        np.ones(n_players), off_weights
    )
    transform[n_players : 2 * n_players, n_players : 2 * n_players] -= np.outer(
        np.ones(n_players), def_weights
    )
    return transformed, transformed_intercept, transform


def fit_weighted_zero_prior(
    design: RapmDesign,
    config: RapmConfig,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the normal estimator with integer game-bootstrap row multiplicities.

    ``weights`` is one for the observed fit. In bootstrap fits it is the sampled
    game multiplicity for every possession. This exactly represents resampling
    whole games without materializing copied possession rows.
    """
    n_rows = design.X.shape[0]
    row_weights = (
        np.ones(n_rows, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if row_weights.shape != (n_rows,) or (row_weights < 0).any():
        raise ValueError("RAPM bootstrap weights must be nonnegative by row.")
    total_weight = float(row_weights.sum())
    if total_weight <= 0:
        raise ValueError("RAPM bootstrap sample has no possessions.")
    n_players = len(design.players)
    weighted_y_mean = float(row_weights @ design.y / total_weight)
    weighted_x = design.X.multiply(row_weights[:, None])
    lhs = (design.X.T @ weighted_x).tocsr() + diags(
        _penalty(config, n_players), format="csr"
    )
    rhs = np.asarray(design.X.T @ (row_weights * (design.y - weighted_y_mean))).ravel()
    raw_beta = _solve(lhs, rhs)
    off_counts = np.asarray(
        row_weights @ design.X[:, :n_players]
    ).ravel()
    def_counts = np.asarray(
        row_weights @ design.X[:, n_players : 2 * n_players]
    ).ravel()
    beta, intercept, transform = _recenter(
        raw_beta,
        off_counts,
        def_counts,
        n_players,
        intercept=weighted_y_mean,
    )
    return beta, intercept, lhs, transform, np.concatenate([off_counts, def_counts])


def game_cluster_sandwich(
    design: RapmDesign,
    config: RapmConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return centered CR0 covariance, centered beta, and fitted intercept.

    The score is de-meaned because the point estimator estimates the intercept
    as the mean outcome before fitting the penalized player coefficients.
    """
    beta, intercept, lhs, transform, _ = fit_weighted_zero_prior(design, config)
    n_players = len(design.players)
    # Undo the centering solely to compute residuals under the uncentered fit.
    # Refit once here avoids relying on an inverse of a rank-deficient transform.
    y_mean = float(design.y.mean())
    raw_lhs = (design.X.T @ design.X).tocsr() + diags(
        _penalty(config, n_players), format="csr"
    )
    raw_beta = _solve(raw_lhs, np.asarray(design.X.T @ (design.y - y_mean)).ravel())
    residual = design.y - y_mean - np.asarray(design.X @ raw_beta).ravel()
    x_bar = np.asarray(design.X.sum(axis=0)).ravel() / len(design.y)
    cluster_keys = np.array(
        [f"{season}:{game}" for season, game in zip(design.seasons, design.game_ids)],
        dtype=object,
    )
    _, inverse = np.unique(cluster_keys, return_inverse=True)
    meat = np.zeros((design.X.shape[1], design.X.shape[1]), dtype=np.float64)
    for cluster in range(int(inverse.max()) + 1):
        rows = inverse == cluster
        cluster_residual = residual[rows]
        score = np.asarray(design.X[rows].T @ cluster_residual).ravel()
        score -= x_bar * float(cluster_residual.sum())
        meat += np.outer(score, score)
    inverse_lhs = np.linalg.inv(raw_lhs.toarray())
    raw_covariance = inverse_lhs @ meat @ inverse_lhs
    centered_covariance = transform @ raw_covariance @ transform.T
    centered_covariance = (centered_covariance + centered_covariance.T) / 2.0
    return centered_covariance, beta, intercept


def _rating_draw(
    design: RapmDesign,
    beta: np.ndarray,
    exposure: np.ndarray,
    draw: int,
) -> pd.DataFrame:
    n_players = len(design.players)
    off = beta[:n_players] * 100.0
    defense = -beta[n_players : 2 * n_players] * 100.0
    off_exposure, def_exposure = exposure[:n_players], exposure[n_players:]
    off = np.where(off_exposure > 0, off, np.nan)
    defense = np.where(def_exposure > 0, defense, np.nan)
    net = np.where(np.isfinite(off) & np.isfinite(defense), off + defense, np.nan)
    return pd.DataFrame(
        {
            "draw": draw,
            "player_id": design.players,
            "offense_per_100": off,
            "defense_per_100": defense,
            "net_per_100": net,
            "off_possessions": off_exposure.astype(np.int64),
            "def_possessions": def_exposure.astype(np.int64),
        }
    )


def _draw_weights(design: RapmDesign, seed: int, draw: int) -> tuple[np.ndarray, dict[str, int]]:
    """Sample a season-stratified whole-game bootstrap deterministically."""
    row_weights = np.zeros(len(design.y), dtype=np.int32)
    counts: dict[str, int] = {}
    for season in sorted(int(value) for value in np.unique(design.seasons)):
        rows = np.flatnonzero(design.seasons == season)
        games = np.array(sorted(np.unique(design.game_ids[rows])), dtype=object)
        rng = np.random.default_rng(np.random.SeedSequence([seed, draw, season]))
        sampled = rng.choice(games, size=len(games), replace=True)
        selected, multiplicity = np.unique(sampled, return_counts=True)
        counts[str(season)] = int(len(sampled))
        for game, repetitions in zip(selected, multiplicity):
            row_weights[rows[design.game_ids[rows] == game]] = int(repetitions)
    return row_weights, counts


def _frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _valid_checkpoint(path: Path, metadata_path: Path, draw: int, identity: str) -> bool:
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        rows = pd.read_parquet(path, columns=["draw", "player_id", "net_per_100"])
    except Exception:
        return False
    return (
        metadata.get("draw") == draw
        and metadata.get("identity") == identity
        and rows["draw"].nunique() == 1
        and int(rows["draw"].iloc[0]) == draw
        and rows["player_id"].is_unique
    )


def run_rapm_uncertainty(
    frame: pd.DataFrame,
    config: RapmConfig,
    uncertainty: RapmUncertaintyConfig,
    *,
    artifact_root: str | Path,
    names: pd.DataFrame | None = None,
    source_hashes: dict[str, str] | None = None,
) -> dict:
    """Fit and checkpoint a reproducible game-bootstrap uncertainty run."""
    if 2027 in config.seasons:
        raise ValueError("Season 2027 is reserved and cannot enter an uncertainty run.")
    if uncertainty.draws < 1:
        raise ValueError("At least one bootstrap draw is required.")
    design = build_design(frame, include_home=config.include_home)
    identity_payload = {
        "rapm_config": asdict(config),
        "uncertainty_config": asdict(uncertainty),
        "source_code": sha256_file(Path(__file__)),
        "frame_hash": _frame_hash(frame),
        "source_hashes": source_hashes or {},
    }
    identity = _json_hash(identity_payload)[:12]
    run_id = f"normal_rapm_uncertainty_v1_{identity}"
    output = Path(artifact_root) / "models" / "rapm_uncertainty" / run_id
    draw_root = output / "bootstrap_draws"
    draw_root.mkdir(parents=True, exist_ok=True)

    covariance, beta, intercept = game_cluster_sandwich(design, config)
    baseline = _rating_draw(
        design,
        beta,
        np.concatenate([design.off_possessions, design.def_possessions]),
        draw=-1,
    ).drop(columns="draw")
    n_players = len(design.players)
    for component, sign, left, right in (
        ("offense", 1.0, np.arange(n_players), np.arange(n_players)),
        ("defense", -1.0, np.arange(n_players, 2 * n_players), np.arange(n_players, 2 * n_players)),
    ):
        scale = sign * 100.0
        baseline[f"{component}_analytic_se"] = np.sqrt(
            np.clip(np.diag(covariance)[left], 0.0, None)
        ) * abs(scale)
    net_variance = (
        np.diag(covariance)[:n_players]
        + np.diag(covariance)[n_players : 2 * n_players]
        - 2.0 * np.diag(covariance[:n_players, n_players : 2 * n_players])
    ) * 10000.0
    baseline["net_analytic_se"] = np.sqrt(np.clip(net_variance, 0.0, None))
    baseline = baseline.rename(
        columns={
            "offense_per_100": "offense_estimate",
            "defense_per_100": "defense_estimate",
            "net_per_100": "net_estimate",
        }
    )

    for draw in range(uncertainty.draws):
        draw_path = draw_root / f"draw_{draw:04d}.parquet"
        metadata_path = draw_root / f"draw_{draw:04d}.json"
        if _valid_checkpoint(draw_path, metadata_path, draw, identity):
            continue
        weights, game_counts = _draw_weights(design, uncertainty.seed, draw)
        beta_draw, _, _, _, exposure = fit_weighted_zero_prior(design, config, weights)
        draw_frame = _rating_draw(design, beta_draw, exposure, draw)
        if not np.allclose(
            draw_frame["net_per_100"].dropna(),
            (draw_frame["offense_per_100"] + draw_frame["defense_per_100"]).dropna(),
            atol=1e-10,
        ):
            raise AssertionError("RAPM bootstrap component identity failed.")
        _write_parquet_atomic(draw_frame, draw_path)
        write_json_atomic(
            {
                "draw": draw,
                "identity": identity,
                "seed": uncertainty.seed,
                "game_counts_by_season": game_counts,
                "sampled_game_weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                "solver": "cg_then_direct_fallback",
            },
            metadata_path,
        )

    draws = pd.concat(
        [pd.read_parquet(draw_root / f"draw_{draw:04d}.parquet") for draw in range(uncertainty.draws)],
        ignore_index=True,
    )
    summary = baseline.copy()
    for component in ("offense", "defense", "net"):
        column = f"{component}_per_100"
        grouped = draws.groupby("player_id")[column]
        summary[f"{component}_draw_coverage"] = grouped.count().reindex(summary["player_id"]).to_numpy()
        summary[f"{component}_bootstrap_se"] = grouped.std(ddof=1).reindex(summary["player_id"]).to_numpy()
        for level, low, high in ((80, 0.10, 0.90), (95, 0.025, 0.975)):
            quantiles = grouped.quantile([low, high]).unstack().reindex(summary["player_id"])
            summary[f"{component}_ci{level}_low"] = quantiles[low].to_numpy()
            summary[f"{component}_ci{level}_high"] = quantiles[high].to_numpy()
        summary[f"{component}_probability_above_zero"] = grouped.apply(
            lambda values: float((values.dropna() > 0).mean()) if values.notna().any() else np.nan
        ).reindex(summary["player_id"]).to_numpy()
        estimate = summary[f"{component}_estimate"].to_numpy(dtype=float)
        analytic_se = summary[f"{component}_analytic_se"].to_numpy(dtype=float)
        for level in (80, 95):
            low, high = analytic_normal_interval(estimate, analytic_se, level / 100.0)
            summary[f"{component}_analytic_ci{level}_low"] = low
            summary[f"{component}_analytic_ci{level}_high"] = high
        bootstrap_width = summary[f"{component}_ci95_high"] - summary[f"{component}_ci95_low"]
        analytic_width = (
            summary[f"{component}_analytic_ci95_high"]
            - summary[f"{component}_analytic_ci95_low"]
        )
        summary[f"{component}_analytic_to_bootstrap_width_ratio"] = analytic_width / bootstrap_width
    summary["uncertainty_method"] = uncertainty.method
    summary["uncertainty_status"] = np.where(
        summary[[f"{component}_draw_coverage" for component in ("offense", "defense", "net")]].min(axis=1)
        >= uncertainty.draws,
        "bootstrap_complete",
        "bootstrap_component_missing_draws",
    )
    if names is not None and {"PLAYER_ID", "PLAYER_NAME"}.issubset(names.columns):
        lookup = names[["PLAYER_ID", "PLAYER_NAME"]].rename(
            columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}
        )
        summary = summary.merge(lookup, on="player_id", how="left", validate="one_to_one")
    summary = summary.sort_values("net_estimate", ascending=False, kind="stable").reset_index(drop=True)
    _write_parquet_atomic(summary, output / "ratings_uncertainty.parquet")
    _write_parquet_atomic(draws, output / "bootstrap_draws.parquet")
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_normal_rapm_game_bootstrap",
        "estimand_id": "trailing_observed_lineup_rapm_v1",
        "estimand": "retrospective terminal-lineup adjusted points per 100 possessions",
        "status": "research_uncertainty_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(config), "uncertainty": asdict(uncertainty)},
        "source_hashes": identity_payload,
        "quality": {
            "players": int(len(summary)),
            "draws_requested": uncertainty.draws,
            "draws_complete": uncertainty.draws,
            "games": int(len(np.unique(np.array([f"{s}:{g}" for s, g in zip(design.seasons, design.game_ids)])))),
            "rows": int(len(frame)),
            "component_identity_max_error": float(
                np.nanmax(np.abs(summary["net_estimate"] - summary["offense_estimate"] - summary["defense_estimate"]))
            ),
            "analytic_method": "CR0 game-cluster ridge sandwich diagnostic",
            "bootstrap_method": uncertainty.method,
            "analytic_bootstrap_agreement": {
                component: {
                    "high_exposure_players": int(
                        summary[["off_possessions", "def_possessions"]].min(axis=1).ge(2000).sum()
                    ),
                    "median_95_interval_width_ratio": float(
                        summary.loc[
                            summary[["off_possessions", "def_possessions"]].min(axis=1).ge(2000),
                            f"{component}_analytic_to_bootstrap_width_ratio",
                        ].median()
                    ),
                }
                for component in ("offense", "defense", "net")
            },
        },
        "metrics": {
            "players_with_complete_joint_draw_coverage": int(
                summary["uncertainty_status"].eq("bootstrap_complete").sum()
            ),
            "median_joint_draw_coverage": float(summary["net_draw_coverage"].median()),
            "median_net_bootstrap_standard_error": float(
                summary["net_bootstrap_se"].median()
            ),
        },
        "ratings_path": str((output / "ratings_uncertainty.parquet").resolve()),
        "draws_path": str((output / "bootstrap_draws.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Bootstrap is the publication uncertainty method; analytic CR0 is diagnostic only.",
            "Intervals quantify resampled-game variation, not ridge bias, lineup attribution error, or latent ability.",
            "Rows with missing component draws do not receive a complete joint interval.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run


def analytic_normal_interval(estimate: np.ndarray, standard_error: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic normal interval used only for analytic/bootstrap agreement QA."""
    z = float(norm.ppf((1.0 + level) / 2.0))
    return estimate - z * standard_error, estimate + z * standard_error
