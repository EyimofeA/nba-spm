"""Lineup-adjusted shot residual research baseline.

The source contains the five players on each side for every shot, but not a
primary defender.  This model therefore estimates five-on-five *lineup*
associations after a player-neutral expected-shot model.  It must not be used
as a shooter-versus-defender or individual contest metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.expected_shot_quality import fit_and_predict_expected_shots


MODEL_VERSION = "lineup_shot_residual_v1"
OFFENSE_COLUMNS = tuple(f"offense_player_{index}" for index in range(1, 6))
DEFENSE_COLUMNS = tuple(f"defense_player_{index}" for index in range(1, 6))


def _require_panel(frame: pd.DataFrame) -> None:
    required = {
        "game_id", "season_end", "shooter_id", "shot_zone", "shot_value", "shot_made",
        *OFFENSE_COLUMNS, *DEFENSE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Shot panel lacks required lineup-residual columns: {missing}.")
    if frame.duplicated("shot_id").any():
        raise ValueError("Shot panel has duplicate shot IDs.")
    lineup = frame.loc[:, [*OFFENSE_COLUMNS, *DEFENSE_COLUMNS]]
    if lineup.isna().any().any():
        raise ValueError("Lineup shot residual model requires a complete ten-player lineup.")
    if (lineup.nunique(axis=1) != 10).any():
        raise ValueError("Lineup shot residual model requires ten distinct players per shot.")


def _split_games(game_ids: pd.Series) -> np.ndarray:
    """Use a deterministic, whole-game holdout without external randomness."""
    unique_games = pd.Index(game_ids.astype(str).unique())
    if len(unique_games) < 10:
        raise ValueError("Lineup residual holdout needs at least ten games.")
    test_games = {
        game_id
        for game_id in unique_games
        if int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) % 5 == 0
    }
    if not test_games:
        test_games = {str(unique_games[-1])}
    if len(test_games) == len(unique_games):
        test_games.remove(str(unique_games[0]))
    return game_ids.astype(str).isin(test_games).to_numpy()


def _build_design(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    players = np.sort(
        np.unique(frame.loc[:, [*OFFENSE_COLUMNS, *DEFENSE_COLUMNS]].to_numpy(dtype=np.int64).ravel())
    )
    player_index = {int(player): index for index, player in enumerate(players)}
    n_rows = len(frame)
    n_players = len(players)
    row_index = np.repeat(np.arange(n_rows, dtype=np.int64), 5)
    offense = frame.loc[:, OFFENSE_COLUMNS].to_numpy(dtype=np.int64)
    defense = frame.loc[:, DEFENSE_COLUMNS].to_numpy(dtype=np.int64)
    off_index = np.asarray([player_index[int(player)] for player in offense.ravel()], dtype=np.int64)
    def_index = np.asarray([player_index[int(player)] for player in defense.ravel()], dtype=np.int64)
    matrix = sparse.coo_matrix(
        (
            np.concatenate([np.ones(n_rows * 5), -np.ones(n_rows * 5)]),
            (
                np.concatenate([row_index, row_index]),
                np.concatenate([off_index, n_players + def_index]),
            ),
        ),
        shape=(n_rows, 2 * n_players),
        dtype=np.float64,
    ).tocsr()
    off_exposure = np.bincount(off_index, minlength=n_players)
    def_exposure = np.bincount(def_index, minlength=n_players)
    return matrix, players, off_exposure, def_exposure


def _fit_lineup_residual(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    *,
    train_mask: np.ndarray,
    ridge_penalty: float,
) -> tuple[np.ndarray, float]:
    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive.")
    train_x = matrix[train_mask]
    train_y = target[train_mask]
    intercept = float(train_y.mean())
    normal = (train_x.T @ train_x).tocsc()
    normal += sparse.eye(normal.shape[0], format="csc") * ridge_penalty
    beta = np.asarray(spsolve(normal, train_x.T @ (train_y - intercept))).ravel()
    if not np.isfinite(beta).all():
        raise ValueError("Lineup shot residual fit produced non-finite coefficients.")

    n_players = matrix.shape[1] // 2
    off_counts = np.asarray(matrix[:, :n_players].sum(axis=0)).ravel()
    def_counts = -np.asarray(matrix[:, n_players:].sum(axis=0)).ravel()
    off_mean = float(np.average(beta[:n_players], weights=off_counts))
    def_mean = float(np.average(beta[n_players:], weights=def_counts))
    beta[:n_players] -= off_mean
    beta[n_players:] -= def_mean
    # Rows contain five +1 offense and five -1 defense entries.
    intercept += 5.0 * off_mean - 5.0 * def_mean
    return beta, intercept


def _residual_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return {
        "shots": int(len(actual)),
        "baseline_rmse": float(np.sqrt(np.mean(actual**2))),
        "model_rmse": float(np.sqrt(np.mean(error**2))),
        "baseline_mae": float(np.mean(np.abs(actual))),
        "model_mae": float(np.mean(np.abs(error))),
        "residual_correlation": correlation,
    }


def _shooter_quality(frame: pd.DataFrame) -> pd.DataFrame:
    direct = frame[["shooter_id", "shot_class", "expected_points", "actual_points"]].copy()
    direct["attempts"] = 1
    grouped = direct.groupby(["shooter_id", "shot_class"], as_index=False).agg(
        shooter_attempts=("attempts", "sum"),
        shooter_expected_points=("expected_points", "sum"),
        shooter_actual_points=("actual_points", "sum"),
    )
    grouped["shooter_expected_points_per_shot"] = (
        grouped["shooter_expected_points"] / grouped["shooter_attempts"]
    )
    league = grouped.groupby("shot_class")["shooter_expected_points_per_shot"].transform("mean")
    grouped["shooter_quality_above_league_per_100_shots"] = (
        grouped["shooter_expected_points_per_shot"] - league
    ) * 100.0
    grouped["shooter_shotmaking_above_quality_per_100_shots"] = (
        (grouped["shooter_actual_points"] - grouped["shooter_expected_points"])
        / grouped["shooter_attempts"]
        * 100.0
    )
    return grouped.rename(columns={"shooter_id": "player_id"})


def _fit_class(
    frame: pd.DataFrame,
    *,
    ridge_penalty: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    matrix, players, off_exposure, def_exposure = _build_design(frame)
    target = frame["actual_points"].to_numpy(dtype=float) - frame["expected_points"].to_numpy(dtype=float)
    test_mask = _split_games(frame["game_id"])
    beta, intercept = _fit_lineup_residual(
        matrix, target, train_mask=~test_mask, ridge_penalty=ridge_penalty
    )
    holdout_metrics = _residual_metrics(target[test_mask], np.asarray(matrix[test_mask] @ beta).ravel() + intercept)
    full_beta, _ = _fit_lineup_residual(
        matrix, target, train_mask=np.ones(len(frame), dtype=bool), ridge_penalty=ridge_penalty
    )
    n_players = len(players)
    output = pd.DataFrame(
        {
            "player_id": players.astype(int),
            "lineup_offense_shotmaking_per_100_shots": full_beta[:n_players] * 100.0,
            "lineup_defense_contest_per_100_shots": full_beta[n_players:] * 100.0,
            "offense_lineup_shots": off_exposure.astype(int),
            "defense_lineup_shots": def_exposure.astype(int),
        }
    )
    output["lineup_net_residual_per_100_shots"] = (
        output["lineup_offense_shotmaking_per_100_shots"]
        + output["lineup_defense_contest_per_100_shots"]
    )
    return output, holdout_metrics


def build_lineup_shot_residual(
    panel_path: str | Path,
    *,
    artifact_root: str | Path,
    train_season_end: int = 2024,
    calibration_season_end: int = 2025,
    rating_season_end: int = 2026,
    ridge_penalty: float = 1000.0,
    c: float = 0.2,
    max_iter: int = 300,
) -> dict:
    """Build a descriptive five-on-five residual model, split rim/non-rim.

    The expected-shot model is fit on 2024, calibrated on 2025, and scores
    2026.  The line-up residual model then reserves deterministic whole 2026
    games for a within-season diagnostic.  That is not a future-season
    confirmation and cannot promote the model.
    """
    panel_path = Path(panel_path)
    panel = pd.read_parquet(panel_path)
    _require_panel(panel)
    train = panel.loc[panel["season_end"].eq(train_season_end)].copy()
    calibration = panel.loc[panel["season_end"].eq(calibration_season_end)].copy()
    target = panel.loc[panel["season_end"].eq(rating_season_end)].copy()
    if train.empty or calibration.empty or target.empty:
        raise ValueError("Shot residual model needs nonempty training, calibration, and rating seasons.")
    _, expected_make = fit_and_predict_expected_shots(
        train, calibration, target, c=c, max_iter=max_iter
    )
    target["expected_points"] = target["shot_value"].to_numpy(dtype=float) * expected_make
    target["actual_points"] = target["shot_value"].to_numpy(dtype=float) * target["shot_made"].to_numpy(dtype=float)
    target["shot_class"] = np.where(target["shot_zone"].eq("rim"), "rim", "non_rim")

    rows: list[pd.DataFrame] = []
    metrics: list[dict] = []
    for shot_class, class_frame in [
        ("all", target),
        ("rim", target.loc[target["shot_class"].eq("rim")]),
        ("non_rim", target.loc[target["shot_class"].eq("non_rim")]),
    ]:
        if len(class_frame) < 1000:
            raise ValueError(f"Shot class {shot_class} has too few shots for a residual fit.")
        ratings, evaluation = _fit_class(class_frame.reset_index(drop=True), ridge_penalty=ridge_penalty)
        ratings["shot_class"] = shot_class
        rows.append(ratings)
        metrics.append({"shot_class": shot_class, **evaluation})

    direct = _shooter_quality(target)
    ratings = pd.concat(rows, ignore_index=True).merge(
        direct,
        on=["player_id", "shot_class"],
        how="left",
        validate="one_to_one",
    )
    ratings["season_end"] = int(rating_season_end)
    ratings = ratings.sort_values(["shot_class", "lineup_net_residual_per_100_shots"], ascending=[True, False])

    config = {
        "model_version": MODEL_VERSION,
        "train_season_end": int(train_season_end),
        "calibration_season_end": int(calibration_season_end),
        "rating_season_end": int(rating_season_end),
        "ridge_penalty": float(ridge_penalty),
        "expected_shot_c": float(c),
        "expected_shot_max_iter": int(max_iter),
        "panel_sha256": sha256_file(panel_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "models" / "lineup_shot_residual" / f"{MODEL_VERSION}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(output / "holdout_metrics.parquet", index=False)
    run = {
        "run_id": output.name,
        "model_family": "five_on_five_lineup_adjusted_shot_residual_ridge",
        "status": "research_baseline",
        "estimand": (
            "Lineup association with field-goal points above a player-neutral expected-shot baseline, "
            "reported separately for rim and non-rim attempts."
        ),
        "evidence_status": "within_season_game_holdout_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "evaluation": metrics,
        "rating_data": {
            "shots": int(len(target)),
            "games": int(target["game_id"].nunique()),
            "rim_shots": int(target["shot_class"].eq("rim").sum()),
            "non_rim_shots": int(target["shot_class"].eq("non_rim").sum()),
        },
        "artifact_path": str(output.resolve()),
        "forbidden_interpretation": (
            "Primary defender value, shooter-versus-defender matchup result, causal shot contest, "
            "public defensive leaderboard, RAPM/SPM/AIO input, or future-season forecast."
        ),
        "next_gate": "Obtain a permitted shot-level defender assignment and test a frozen future-season comparison.",
    }
    write_json_atomic(run, output / "run.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the lineup-adjusted shot residual research baseline.")
    parser.add_argument("--panel", type=Path, default=Path("data/lake/silver/shot_defense_events.parquet"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--ridge-penalty", type=float, default=1000.0)
    args = parser.parse_args()
    print(json.dumps(
        build_lineup_shot_residual(
            args.panel,
            artifact_root=args.artifact_root,
            ridge_penalty=args.ridge_penalty,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
