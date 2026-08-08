"""Clean-room zero-prior RAPM baseline.

This is intentionally smaller than the legacy research engine. It fits one
transparent ridge model and labels its present data scope as unclassified; it does
not inherit date-based playoff rules, priors, garbage-time logic, or prior results.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.data.manifest import write_json_atomic
from nba_impact.data.normalize import normalize_legacy_possessions
from nba_impact.data.quality import audit_possession_frame, quarantine_invalid_games


@dataclass(frozen=True)
class RapmConfig:
    seasons: tuple[int, ...]
    lambda_off: float = 3000.0
    lambda_def: float = 3000.0
    lambda_home: float = 300.0
    include_home: bool = True
    game_types: tuple[str, ...] = ("regular",)
    data_scope: str = "game_id_prefix_classified"


@dataclass
class RapmDesign:
    X: csr_matrix
    y: np.ndarray
    players: np.ndarray
    game_ids: np.ndarray
    seasons: np.ndarray
    home_offense: np.ndarray
    off_possessions: np.ndarray
    def_possessions: np.ndarray


def load_legacy_possessions(
    cache_dir: str | Path,
    seasons: tuple[int, ...],
    *,
    game_types: tuple[str, ...] = ("regular",),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    quarantine_counts: dict[str, dict[str, int]] = {}
    for season in seasons:
        path = Path(cache_dir) / f"matchups_{season}.parquet"
        frame = pd.read_parquet(path)
        report = audit_possession_frame(frame, path=str(path), expected_season=season)
        if any(issue.code == "empty_partition" for issue in report.issues):
            raise ValueError(f"Possession quality gate failed for {season}: empty_partition=1")
        valid, rejected, counts = quarantine_invalid_games(frame)
        if not rejected.empty:
            quarantine_counts[str(season)] = counts
        clean_report = audit_possession_frame(valid, path=str(path), expected_season=season)
        if not clean_report.passed:
            failures = "; ".join(f"{issue.code}={issue.count}" for issue in clean_report.issues)
            raise ValueError(f"Possession quality gate failed after quarantine for {season}: {failures}")
        normalized = normalize_legacy_possessions(valid)
        frames.append(normalized.loc[normalized["game_type"].isin(game_types)])
    if not frames:
        raise ValueError("At least one season is required.")
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError(f"No possessions remain for game types {game_types}.")
    combined.attrs["quarantine_counts"] = quarantine_counts
    return combined


def build_design(frame: pd.DataFrame, include_home: bool = True) -> RapmDesign:
    away_players = frame.loc[:, AWAY_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home_players = frame.loc[:, HOME_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home_offense = frame["home_poss"].to_numpy(dtype=bool, copy=False)
    player_values = np.unique(np.concatenate([away_players.ravel(), home_players.ravel()]))
    players = np.asarray(sorted(int(value) for value in player_values), dtype=np.int64)
    n_rows = len(frame)
    n_players = len(players)
    n_columns = 2 * n_players + int(include_home)

    offense_players = np.where(home_offense[:, None], home_players, away_players)
    defense_players = np.where(home_offense[:, None], away_players, home_players)
    offense_indices = np.searchsorted(players, offense_players)
    defense_indices = np.searchsorted(players, defense_players)
    row_base = np.repeat(np.arange(n_rows, dtype=np.int64), 5)
    row_parts = [row_base, row_base]
    column_parts = [offense_indices.ravel(), n_players + defense_indices.ravel()]
    value_parts = [np.ones(n_rows * 5, dtype=np.float64), np.ones(n_rows * 5, dtype=np.float64)]
    if include_home:
        row_parts.append(np.arange(n_rows, dtype=np.int64))
        column_parts.append(np.full(n_rows, 2 * n_players, dtype=np.int64))
        value_parts.append(np.where(home_offense, 1.0, -1.0))

    off_counts = np.bincount(offense_indices.ravel(), minlength=n_players)
    def_counts = np.bincount(defense_indices.ravel(), minlength=n_players)

    matrix = csr_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(n_rows, n_columns),
        dtype=np.float64,
    )
    return RapmDesign(
        X=matrix,
        y=pd.to_numeric(frame["pts"], errors="raise").to_numpy(dtype=np.float64),
        players=players,
        game_ids=frame["gameid"].astype(str).to_numpy(),
        seasons=pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int32),
        home_offense=home_offense,
        off_possessions=off_counts,
        def_possessions=def_counts,
    )


def _penalty(config: RapmConfig, n_players: int) -> np.ndarray:
    values = np.concatenate(
        [
            np.full(n_players, config.lambda_off, dtype=np.float64),
            np.full(n_players, config.lambda_def, dtype=np.float64),
        ]
    )
    if config.include_home:
        values = np.append(values, config.lambda_home)
    return values


def fit_coefficients(
    design: RapmDesign,
    config: RapmConfig,
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    X = design.X if row_mask is None else design.X[row_mask]
    y = design.y if row_mask is None else design.y[row_mask]
    intercept = float(y.mean())
    lhs = (X.T @ X).tocsr() + diags(_penalty(config, len(design.players)), format="csr")
    rhs = X.T @ (y - intercept)
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta)

    # Offense and points-allowed defense are not separately level-identified:
    # every row contains five of each. Anchor each block to the possession-
    # weighted average player and adjust the intercept so predictions are exact.
    n_players = len(design.players)
    off_counts = np.asarray(X[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    off_mean = float(np.average(beta[:n_players], weights=off_counts))
    def_mean = float(np.average(beta[n_players : 2 * n_players], weights=def_counts))
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)
    return beta, intercept


def _game_margin_metrics(
    design: RapmDesign,
    beta: np.ndarray,
    intercept: float,
    test_mask: np.ndarray,
    train_mask: np.ndarray,
) -> dict:
    prediction = np.asarray(design.X[test_mask] @ beta).ravel() + intercept
    sign = np.where(design.home_offense[test_mask], 1.0, -1.0)
    test_rows = np.flatnonzero(test_mask)
    n_players = len(design.players)
    train_player_counts = np.asarray(design.X[train_mask, : 2 * n_players].sum(axis=0)).ravel()
    known_players = (train_player_counts[:n_players] + train_player_counts[n_players:]) > 0
    test_player_columns = design.X[test_mask, : 2 * n_players].indices % n_players
    # Every possession has ten player entries; aggregate unknown slots by row.
    unknown_entries = (~known_players[test_player_columns]).astype(np.int8)
    unknown_by_row = np.add.reduceat(unknown_entries, design.X[test_mask, : 2 * n_players].indptr[:-1])
    game_frame = pd.DataFrame(
        {
            "row": test_rows,
            "game_id": design.game_ids[test_mask],
            "actual": design.y[test_mask] * sign,
            "predicted": prediction * sign,
            "unknown_slots": unknown_by_row,
        }
    )
    games = game_frame.groupby("game_id", as_index=False).agg(
        actual_margin=("actual", "sum"),
        predicted_margin=("predicted", "sum"),
        unknown_player_slots=("unknown_slots", "sum"),
    )
    error = games["actual_margin"] - games["predicted_margin"]
    correlation = float(games[["actual_margin", "predicted_margin"]].corr().iloc[0, 1])
    predicted_variance = float(np.var(games["predicted_margin"], ddof=0))
    calibration_slope = (
        float(np.cov(games["actual_margin"], games["predicted_margin"], ddof=0)[0, 1] / predicted_variance)
        if predicted_variance > 0
        else float("nan")
    )
    calibration_intercept = float(
        games["actual_margin"].mean() - calibration_slope * games["predicted_margin"].mean()
    )
    return {
        "games": int(len(games)),
        "margin_rmse": float(math.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": correlation,
        "actual_margin_sd": float(games["actual_margin"].std(ddof=0)),
        "predicted_margin_sd": float(games["predicted_margin"].std(ddof=0)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "known_player_rate": float(known_players.mean()),
        "games_with_unknown_players": int((games["unknown_player_slots"] > 0).sum()),
    }


def ratings_table(
    design: RapmDesign,
    beta: np.ndarray,
    *,
    names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    n_players = len(design.players)
    offense = beta[:n_players] * 100.0
    defense = -beta[n_players : 2 * n_players] * 100.0
    table = pd.DataFrame(
        {
            "player_id": design.players,
            "offense_per_100": offense,
            "defense_per_100": defense,
            "net_per_100": offense + defense,
            "off_possessions": design.off_possessions,
            "def_possessions": design.def_possessions,
            "uncertainty_status": "not_estimated_v0",
        }
    )
    if names is not None and {"PLAYER_ID", "PLAYER_NAME"}.issubset(names.columns):
        lookup = names[["PLAYER_ID", "PLAYER_NAME"]].rename(
            columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}
        )
        lookup["player_id"] = pd.to_numeric(lookup["player_id"], errors="coerce").astype("Int64")
        table = table.merge(lookup.dropna(subset=["player_id"]), on="player_id", how="left")
    return table.sort_values("net_per_100", ascending=False).reset_index(drop=True)


def lineup_conditioned_retrodiction(design: RapmDesign, config: RapmConfig) -> dict:
    latest = int(max(config.seasons))
    train_mask = design.seasons < latest
    test_mask = design.seasons == latest
    if not train_mask.any() or not test_mask.any():
        return {"status": "not_available", "reason": "requires at least two seasons"}
    beta, intercept = fit_coefficients(design, config, train_mask)
    return {
        "status": "complete",
        "evaluation": "lineup_conditioned_retrodiction",
        "train_seasons": [season for season in config.seasons if season < latest],
        "test_season": latest,
        **_game_margin_metrics(design, beta, intercept, test_mask, train_mask),
        "warning": "Uses observed test-season lineups; this is retrodiction, not a deployable forecast.",
    }


def run_regularization_comparison(
    frame: pd.DataFrame,
    config: RapmConfig,
    lambda_pairs: tuple[tuple[float, float], ...],
    *,
    artifact_root: str | Path,
) -> dict:
    """Compare preregistered penalties on one chronological diagnostic fold.

    This intentionally does not select or promote a model. The latest season is
    the holdout and all earlier supplied seasons are training data; nested
    multi-season selection belongs in the next research layer.
    """
    design = build_design(frame, include_home=config.include_home)
    latest = int(max(config.seasons))
    train_mask = design.seasons < latest
    test_mask = design.seasons == latest
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Comparison requires at least one training season and one holdout season.")

    rows: list[dict] = []
    for lambda_off, lambda_def in lambda_pairs:
        candidate = RapmConfig(
            seasons=config.seasons,
            lambda_off=float(lambda_off),
            lambda_def=float(lambda_def),
            lambda_home=config.lambda_home,
            include_home=config.include_home,
            game_types=config.game_types,
            data_scope=config.data_scope,
        )
        beta, intercept = fit_coefficients(design, candidate, train_mask)
        rows.append(
            {
                "lambda_off": float(lambda_off),
                "lambda_def": float(lambda_def),
                **_game_margin_metrics(design, beta, intercept, test_mask, train_mask),
            }
        )

    run_id = f"rapm_compare_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm_comparisons" / run_id
    output.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(rows).sort_values(["margin_rmse", "lambda_off", "lambda_def"])
    results.to_parquet(output / "results.parquet", index=False)
    metrics = {
        "evaluation": "single_fold_lineup_conditioned_retrodiction",
        "train_seasons": [season for season in config.seasons if season < latest],
        "test_season": latest,
        "candidate_count": len(rows),
        "results": results.to_dict(orient="records"),
        "warning": "Diagnostic only: one outer fold cannot select or promote a production model.",
    }
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm_regularization_comparison",
        "estimand": "lineup_adjusted_descriptive_points_per_100",
        "status": "research_diagnostic_unverified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(config), "lambda_pairs": [list(pair) for pair in lambda_pairs]},
        "metrics": metrics,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run


def run_rapm(
    frame: pd.DataFrame,
    config: RapmConfig,
    *,
    artifact_root: str | Path,
    names: pd.DataFrame | None = None,
) -> dict:
    design = build_design(frame, include_home=config.include_home)
    beta, intercept = fit_coefficients(design, config)
    ratings = ratings_table(design, beta, names=names)
    metrics = {
        "rows": int(len(frame)),
        "games": int(frame["gameid"].nunique()),
        "players": int(len(design.players)),
        "intercept_points_per_possession": intercept,
        "in_sample_rmse": float(
            np.sqrt(np.mean((design.y - (np.asarray(design.X @ beta).ravel() + intercept)) ** 2))
        ),
        "quarantine_counts": frame.attrs.get("quarantine_counts", {}),
        "retrodiction": lineup_conditioned_retrodiction(design, config),
    }
    run_id = f"rapm_v0_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    write_json_atomic(asdict(config), output / "config.json")
    write_json_atomic(metrics, output / "metrics.json")
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm",
        "estimand": "lineup_adjusted_descriptive_points_per_100",
        "status": "research_baseline_unverified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "metrics": metrics,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
