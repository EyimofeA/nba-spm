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
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.normalize import normalize_legacy_possessions
from nba_impact.data.possessions import (
    AWAY_LINEUP_COLUMNS as CURRENT_AWAY_LINEUP_COLUMNS,
    HOME_LINEUP_COLUMNS as CURRENT_HOME_LINEUP_COLUMNS,
)
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


def load_current_possessions(
    possessions_path: str | Path,
    segments_path: str | Path,
    *,
    lineup_policy: str = "start",
    game_types: tuple[str, ...] = ("regular",),
) -> pd.DataFrame:
    """Adapt canonical current possessions to the transparent RAPM design contract.

    A possession can cross substitutions, so the lineup choice is explicit and
    researchable. ``start`` and ``terminal`` are sensitivity variants; neither
    is silently treated as ground truth.
    """
    if lineup_policy not in {"start", "terminal"}:
        raise ValueError("lineup_policy must be 'start' or 'terminal'")
    possessions = pd.read_parquet(possessions_path)
    possessions = possessions.loc[possessions["season_type"].isin(game_types)].copy()
    segments = pd.read_parquet(segments_path)
    segments = segments.loc[segments["possession_id"].isin(possessions["possession_id"])].copy()
    segments = segments.sort_values(["possession_id", "segment_number"], kind="stable")
    selected = (
        segments.groupby("possession_id", as_index=False, sort=False).head(1)
        if lineup_policy == "start"
        else segments.groupby("possession_id", as_index=False, sort=False).tail(1)
    )
    lineup_columns = [*CURRENT_HOME_LINEUP_COLUMNS, *CURRENT_AWAY_LINEUP_COLUMNS]
    selected = selected[["possession_id", *lineup_columns]]
    frame = possessions.merge(selected, on="possession_id", validate="one_to_one")
    rename = {
        **{f"away_player_{index}": f"a{index}" for index in range(1, 6)},
        **{f"home_player_{index}": f"h{index}" for index in range(1, 6)},
    }
    frame = frame.rename(columns=rename)
    output = pd.DataFrame(
        {
            "home_poss": frame["offense_is_home"].astype(int),
            "pts": frame["points"].astype(float),
            **{column: frame[column].astype("int64") for column in (*AWAY_PLAYER_COLUMNS, *HOME_PLAYER_COLUMNS)},
            "season": frame["season_end"].astype(int),
            "date": frame["game_date"],
            "period": frame["period"].astype(int),
            "num": frame["possession_number"].astype(int),
            "gameid": frame["game_id"].astype(str),
        }
    )
    output.attrs["lineup_policy"] = lineup_policy
    output.attrs["source_paths"] = [str(Path(possessions_path).resolve()), str(Path(segments_path).resolve())]
    return output


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
    games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
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
        "known_player_rate": float(games.attrs["known_player_rate"]),
        "games_with_unknown_players": int((games["unknown_player_slots"] > 0).sum()),
    }


def _game_margin_frame(
    design: RapmDesign,
    beta: np.ndarray,
    intercept: float,
    test_mask: np.ndarray,
    train_mask: np.ndarray,
) -> pd.DataFrame:
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
    games.attrs["known_player_rate"] = float(known_players.mean())
    return games


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


def _paired_season_game_bootstrap(
    error_table: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int,
) -> float:
    candidate_rows = error_table.loc[error_table["candidate"] == candidate]
    baseline_rows = error_table.loc[error_table["candidate"] == baseline]
    paired = candidate_rows.merge(
        baseline_rows,
        on=["test_season", "game_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    paired["loss_delta"] = paired["squared_error_candidate"] - paired["squared_error_baseline"]
    rng = np.random.default_rng(seed)
    seasons = sorted(paired["test_season"].unique())
    draws = np.empty(repetitions, dtype=np.float64)
    season_values = {
        season: paired.loc[paired["test_season"] == season, "loss_delta"].to_numpy()
        for season in seasons
    }
    for repetition in range(repetitions):
        sampled = [rng.choice(values, size=len(values), replace=True) for values in season_values.values()]
        draws[repetition] = np.concatenate(sampled).mean()
    return float(np.mean(draws < 0.0))


def run_walk_forward_comparison(
    frame: pd.DataFrame,
    config: RapmConfig,
    lambda_pairs: tuple[tuple[float, float], ...],
    test_seasons: tuple[int, ...],
    *,
    train_window: int,
    artifact_root: str | Path,
    bootstrap_repetitions: int = 2000,
    seed: int = 7,
) -> dict:
    """Evaluate fixed candidates across independent chronological outer folds."""
    if train_window < 1:
        raise ValueError("train_window must be positive")
    if not lambda_pairs:
        raise ValueError("provide at least one candidate; the first is the baseline")
    design = build_design(frame, include_home=config.include_home)
    available = set(int(value) for value in np.unique(design.seasons))
    fold_rows: list[dict] = []
    game_error_rows: list[dict] = []
    candidate_names = [f"off{off:g}_def{defense:g}" for off, defense in lambda_pairs]

    for test_season in test_seasons:
        train_seasons = tuple(range(test_season - train_window, test_season))
        missing = sorted(set((*train_seasons, test_season)) - available)
        if missing:
            raise ValueError(f"Fold ending {test_season} is missing seasons {missing}")
        train_mask = np.isin(design.seasons, train_seasons)
        test_mask = design.seasons == test_season
        for candidate_name, (lambda_off, lambda_def) in zip(candidate_names, lambda_pairs):
            candidate = RapmConfig(
                seasons=train_seasons,
                lambda_off=float(lambda_off),
                lambda_def=float(lambda_def),
                lambda_home=config.lambda_home,
                include_home=config.include_home,
                game_types=config.game_types,
                data_scope=config.data_scope,
            )
            beta, intercept = fit_coefficients(design, candidate, train_mask)
            metrics = _game_margin_metrics(design, beta, intercept, test_mask, train_mask)
            fold_rows.append(
                {
                    "candidate": candidate_name,
                    "lambda_off": float(lambda_off),
                    "lambda_def": float(lambda_def),
                    "train_start": train_seasons[0],
                    "train_end": train_seasons[-1],
                    "test_season": test_season,
                    **metrics,
                }
            )
            games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
            games["squared_error"] = (games["actual_margin"] - games["predicted_margin"]) ** 2
            for row in games[["game_id", "squared_error"]].itertuples(index=False):
                game_error_rows.append(
                    {
                        "candidate": candidate_name,
                        "test_season": test_season,
                        "game_id": row.game_id,
                        "squared_error": float(row.squared_error),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    errors = pd.DataFrame(game_error_rows)
    baseline = candidate_names[0]
    baseline_folds = folds.loc[folds["candidate"] == baseline, ["test_season", "margin_rmse"]].rename(
        columns={"margin_rmse": "baseline_rmse"}
    )
    summary_rows: list[dict] = []
    for index, candidate_name in enumerate(candidate_names):
        candidate_folds = folds.loc[folds["candidate"] == candidate_name].merge(
            baseline_folds, on="test_season", validate="one_to_one"
        )
        mean_rmse = float(candidate_folds["margin_rmse"].mean())
        baseline_mean = float(candidate_folds["baseline_rmse"].mean())
        fold_wins = int((candidate_folds["margin_rmse"] < candidate_folds["baseline_rmse"]).sum())
        probability = (
            0.5
            if candidate_name == baseline
            else _paired_season_game_bootstrap(
                errors,
                candidate_name,
                baseline,
                repetitions=bootstrap_repetitions,
                seed=seed + index,
            )
        )
        relative_improvement = (baseline_mean - mean_rmse) / baseline_mean if baseline_mean else 0.0
        if candidate_name == baseline:
            evidence_status = "baseline"
        elif len(test_seasons) < 3:
            evidence_status = "insufficient_folds"
        elif probability >= 0.95 and fold_wins >= math.ceil(0.7 * len(test_seasons)) and relative_improvement >= 0.01:
            evidence_status = "candidate_requires_untouched_confirmation"
        elif probability >= 0.90 and fold_wins >= math.ceil(0.5 * len(test_seasons)):
            evidence_status = "promising_research_challenger"
        else:
            evidence_status = "improvement_not_demonstrated"
        summary_rows.append(
            {
                "candidate": candidate_name,
                "folds": len(test_seasons),
                "mean_margin_rmse": mean_rmse,
                "mean_margin_correlation": float(candidate_folds["margin_correlation"].mean()),
                "fold_wins_vs_baseline": fold_wins,
                "relative_rmse_improvement": relative_improvement,
                "bootstrap_probability_better": probability,
                "evidence_status": evidence_status,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_margin_rmse")
    run_id = f"rapm_walk_forward_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm_walk_forward" / run_id
    output.mkdir(parents=True, exist_ok=False)
    folds.to_parquet(output / "fold_results.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    metrics = {
        "evaluation": "multi_fold_lineup_conditioned_retrodiction",
        "train_window": train_window,
        "test_seasons": list(test_seasons),
        "baseline": baseline,
        "bootstrap_repetitions": bootstrap_repetitions,
        "summary": summary.to_dict(orient="records"),
        "warning": "Observed future lineups are used. Promotion still requires an untouched confirmation season.",
    }
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm_walk_forward_comparison",
        "estimand": "lineup_adjusted_descriptive_points_per_100",
        "status": "research_evidence_unverified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "lambda_pairs": [list(pair) for pair in lambda_pairs],
            "test_seasons": list(test_seasons),
            "train_window": train_window,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
        },
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
    config_payload = {**asdict(config), "source_code_sha256": sha256_file(Path(__file__))}
    run_id = f"rapm_v0_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    write_json_atomic(config_payload, output / "config.json")
    write_json_atomic(metrics, output / "metrics.json")
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm",
        "estimand": "lineup_adjusted_descriptive_points_per_100",
        "status": "research_baseline_unverified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "metrics": metrics,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
