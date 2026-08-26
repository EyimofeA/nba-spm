#!/usr/bin/env python3
"""Compare alternative priors inside one fixed single-season RAPM update."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit as _fit_box,
    _select_alpha,
)
from nba_impact.models.five_year_target_spm import _rating_table
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficient_center_path,
    load_legacy_possessions,
)
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.paths import LEGACY_POSSESSION_CACHE


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "aio_prior_bakeoff_v1"
RATING_SEASONS = (2021, 2022, 2023)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _team_context(schedule_root: Path, seasons: range) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        with gzip.open(schedule_root / f"leaguegamelog_{season}.json.gz", "rt") as handle:
            result = json.load(handle)["resultSets"][0]
        games = pd.DataFrame(result["rowSet"], columns=result["headers"])
        if games.groupby("GAME_ID").size().ne(2).any():
            raise ValueError(f"Schedule season {season} contains a non-paired game.")
        games["estimated_possessions"] = (
            games["FGA"] + 0.44 * games["FTA"] - games["OREB"] + games["TOV"]
        )
        opponent = games[["GAME_ID", "TEAM_ID", "PTS", "estimated_possessions"]].rename(
            columns={
                "TEAM_ID": "opponent_team_id",
                "PTS": "opponent_points",
                "estimated_possessions": "opponent_estimated_possessions",
            }
        )
        paired = games.merge(opponent, on="GAME_ID", how="inner")
        paired = paired.loc[paired["TEAM_ID"].ne(paired["opponent_team_id"])].copy()
        paired["game_possessions"] = (
            paired["estimated_possessions"] + paired["opponent_estimated_possessions"]
        ) / 2.0
        context = paired.groupby("TEAM_ID", as_index=False).agg(
            points=("PTS", "sum"),
            opponent_points=("opponent_points", "sum"),
            team_possessions=("game_possessions", "sum"),
        )
        context["team_offensive_rating"] = 100.0 * context["points"] / context["team_possessions"]
        context["team_defensive_rating"] = (
            100.0 * context["opponent_points"] / context["team_possessions"]
        )
        context["Season"] = int(season)
        rows.append(context)
    return pd.concat(rows, ignore_index=True)


def _annual_onoff(player_sheet_root: Path, team_context: pd.DataFrame, seasons: range) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        source = pd.read_parquet(player_sheet_root / f"{season}.parquet")
        columns = [
            "PLAYER_ID", "TEAM_ID", "OffPoss", "DefPoss", "OnOffRtg", "OnDefRtg"
        ]
        if missing := sorted(set(columns) - set(source.columns)):
            raise ValueError(f"Player sheet {season} is missing {missing}.")
        frame = source[columns].drop_duplicates().copy()
        if frame["PLAYER_ID"].duplicated().any():
            raise ValueError(
                f"Player sheet {season} contains more than one distinct row per player."
            )
        frame["Season"] = int(season)
        frame = frame.merge(
            team_context.loc[team_context["Season"].eq(season)],
            on=["Season", "TEAM_ID"],
            how="left",
            validate="many_to_one",
        )
        off_bench_possessions = frame["team_possessions"] - frame["OffPoss"]
        def_bench_possessions = frame["team_possessions"] - frame["DefPoss"]
        bench_offense = (
            frame["team_offensive_rating"] * frame["team_possessions"]
            - frame["OnOffRtg"] * frame["OffPoss"]
        ) / off_bench_possessions
        bench_defense = (
            frame["team_defensive_rating"] * frame["team_possessions"]
            - frame["OnDefRtg"] * frame["DefPoss"]
        ) / def_bench_possessions
        valid_offense = frame["OffPoss"].gt(0) & off_bench_possessions.gt(0)
        valid_defense = frame["DefPoss"].gt(0) & def_bench_possessions.gt(0)
        frame["raw_onoff_offense"] = (frame["OnOffRtg"] - bench_offense).where(valid_offense)
        frame["raw_onoff_defense"] = (bench_defense - frame["OnDefRtg"]).where(valid_defense)
        rows.append(
            frame[[
                "PLAYER_ID", "Season", "OffPoss", "DefPoss",
                "raw_onoff_offense", "raw_onoff_defense",
            ]]
        )
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual on-off context contains duplicate player-season keys.")
    return output


def _weighted_average(group: pd.DataFrame, value: str, weight: str) -> float:
    valid = group[value].notna() & group[weight].notna() & group[weight].gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(group.loc[valid, value], weights=group.loc[valid, weight]))


def _rolling_onoff(annual: pd.DataFrame, window_ends: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    for end in window_ends:
        window = annual.loc[annual["Season"].between(end - 4, end)]
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "Window_End": int(end),
                    "raw_onoff_offense_5y": _weighted_average(
                        group, "raw_onoff_offense", "OffPoss"
                    ),
                    "raw_onoff_defense_5y": _weighted_average(
                        group, "raw_onoff_defense", "DefPoss"
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Rolling on-off context contains duplicate keys.")
    return output


def _prior_frame(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    output = frame.rename(
        columns={
            "offense": "prior_offense_per_100",
            "defense": "prior_defense_per_100",
            "net": "prior_net_per_100",
        }
    ).copy()
    output["candidate"] = candidate
    return output[[
        "PLAYER_ID", "Window_End", "prior_offense_per_100",
        "prior_defense_per_100", "prior_net_per_100", "candidate",
    ]]


def _existing_spm_priors(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    source = source.loc[
        source["Window_End"].isin(RATING_SEASONS)
        & source["variant"].isin(["baseline", "selected_combined"])
    ].copy()
    source["candidate"] = source["variant"].map(
        {"baseline": "five_year_spm", "selected_combined": "selected_five_year_spm"}
    )
    return source[[
        "PLAYER_ID", "Window_End", "prior_offense_per_100",
        "prior_defense_per_100", "prior_net_per_100", "candidate",
    ]]


def _box_and_pipm_like_priors(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    onoff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).merge(
        onoff,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    box_rows: list[pd.DataFrame] = []
    pipm_rows: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Prior fold {season} lacks chronological history.")
        box = test[["PLAYER_ID", "Window_End"]].copy()
        pipm_like = test[["PLAYER_ID", "Window_End"]].copy()
        for side in ("offense", "defense"):
            target = f"target_{side}"
            alpha = _select_alpha(
                train.rename(columns={"Window_End": "Season"}),
                BOX_PIPM_STYLE_FEATURES,
                target,
                ALPHA_GRID,
            )
            box_model = _fit_box(
                train, BOX_PIPM_STYLE_FEATURES, target, alpha
            )
            train[f"box_{side}"] = box_model.predict(train.loc[:, BOX_PIPM_STYLE_FEATURES])
            box[side] = box_model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])

            context_field = f"raw_onoff_{side}_5y"
            residual_model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=100.0)),
                ]
            )
            residual_model.fit(
                train[[context_field]],
                train[target] - train[f"box_{side}"],
                ridge__sample_weight=train["sample_weight"],
            )
            pipm_like[side] = box[side] + residual_model.predict(test[[context_field]])
        for frame in (box, pipm_like):
            frame["net"] = frame["offense"] + frame["defense"]
        box_rows.append(box)
        pipm_rows.append(pipm_like)
    return (
        _prior_frame(pd.concat(box_rows, ignore_index=True), "box_pipm_style_prior"),
        _prior_frame(pd.concat(pipm_rows, ignore_index=True), "pipm_like_raw_onoff_prior"),
    )


def _paired_bootstrap(
    games: pd.DataFrame,
    *,
    baseline: str,
    draws: int = 10_000,
    seed: int = 20260826,
) -> pd.DataFrame:
    wide = games.pivot(
        index=["test_season", "game_id"], columns="candidate", values="squared_error"
    )
    if wide.isna().any().any():
        raise ValueError("Every candidate must score exactly the same games.")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for candidate in wide.columns:
        if candidate == baseline:
            continue
        season_delta = []
        season_arrays = []
        for _, frame in wide.groupby(level="test_season"):
            delta = (frame[candidate] - frame[baseline]).to_numpy(dtype=float)
            season_arrays.append(delta)
            season_delta.append(float(delta.mean()))
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = np.mean(
                [values[rng.integers(0, len(values), len(values))].mean() for values in season_arrays]
            )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "season_wins": int(sum(value < 0 for value in season_delta)),
                "mean_mse_delta": float(np.mean(season_delta)),
                "bootstrap_95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_95_high": float(np.quantile(samples, 0.975)),
                "probability_mse_delta_below_zero": float(np.mean(samples < 0)),
                "bootstrap_draws": int(draws),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mse_delta", kind="stable")


def _stored_game_frame(
    matrix_root: Path,
    design,
    beta: np.ndarray,
    intercept: float,
    rating_season: int,
) -> pd.DataFrame:
    """Map a one-season coefficient fit to the canonical stored game design."""
    matrix_dir = matrix_root / f"5y_end_{rating_season}"
    matrix_players = np.load(matrix_dir / "player_ids.npy")
    mapped = np.zeros(2 * len(matrix_players) + 1, dtype=float)
    source_index = {int(player_id): index for index, player_id in enumerate(design.players)}
    for target_index, player_id in enumerate(matrix_players):
        source = source_index.get(int(player_id))
        if source is None:
            continue
        mapped[target_index] = beta[source]
        mapped[len(matrix_players) + target_index] = beta[len(design.players) + source]
    mapped[-1] = beta[-1]
    return stored_evaluation_predictions(matrix_dir, mapped, intercept)


def _game_metrics(games: pd.DataFrame) -> dict:
    error = games["actual_margin"] - games["predicted_margin"]
    predicted_variance = float(np.var(games["predicted_margin"], ddof=0))
    slope = (
        float(
            np.cov(games["actual_margin"], games["predicted_margin"], ddof=0)[0, 1]
            / predicted_variance
        )
        if predicted_variance > 0
        else float("nan")
    )
    return {
        "games": int(len(games)),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": float(games["actual_margin"].corr(games["predicted_margin"])),
        "actual_margin_sd": float(games["actual_margin"].std(ddof=0)),
        "predicted_margin_sd": float(games["predicted_margin"].std(ddof=0)),
        "calibration_slope": slope,
        "calibration_intercept": float(
            games["actual_margin"].mean() - slope * games["predicted_margin"].mean()
        ),
        "games_with_unknown_players": int((games["unknown_player_slots"] > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-cache",
        type=Path,
        default=Path(
            os.environ.get("NBA_IMPACT_LEGACY_POSSESSION_CACHE", LEGACY_POSSESSION_CACHE)
        ),
    )
    parser.add_argument(
        "--evaluation-matrix-root",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument(
        "--player-sheet-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=ROOT / "data/lake/bronze/official_game_schedule_1997_2026",
    )
    parser.add_argument(
        "--spm-priors",
        type=Path,
        default=ROOT / "artifacts/models/five_year_spm_feature_research/five_year_spm_feature_research_v1_93c148510e/spm_predictions.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/research/spm_target_horizon_full/spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    if 2027 in RATING_SEASONS or 2027 in tuple(season + 1 for season in RATING_SEASONS):
        raise ValueError("Season 2027 is reserved and must not be loaded.")

    team = _team_context(args.schedule_root, range(2014, max(RATING_SEASONS) + 1))
    annual_onoff = _annual_onoff(
        args.player_sheet_root, team, range(2014, max(RATING_SEASONS) + 1)
    )
    rolling_onoff = _rolling_onoff(
        annual_onoff, tuple(range(2018, max(RATING_SEASONS) + 1))
    )
    features = pd.read_parquet(args.features)
    targets = pd.read_parquet(args.targets)
    box, pipm_like = _box_and_pipm_like_priors(features, targets, rolling_onoff)
    priors = pd.concat([_existing_spm_priors(args.spm_priors), box, pipm_like], ignore_index=True)
    if priors.duplicated(["PLAYER_ID", "Window_End", "candidate"]).any():
        raise ValueError("Prior panel contains duplicate candidate keys.")

    frame = load_legacy_possessions(
        args.legacy_cache,
        tuple(range(min(RATING_SEASONS), max(RATING_SEASONS) + 2)),
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    names_path = args.legacy_cache.parent / "all_names.csv"
    names = pd.read_csv(names_path)[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID")
    config = RapmConfig(
        seasons=tuple(range(min(RATING_SEASONS), max(RATING_SEASONS) + 2)),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        include_home=True,
        game_types=("regular",),
        data_scope="aio_prior_bakeoff_single_season_terminal",
    )

    metrics: list[dict] = []
    game_rows: list[pd.DataFrame] = []
    rating_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    candidates = (
        "zero_prior_rapm",
        "five_year_spm",
        "selected_five_year_spm",
        "box_pipm_style_prior",
        "pipm_like_raw_onoff_prior",
    )
    for rating_season in RATING_SEASONS:
        train_mask = design.seasons == rating_season
        test_season = rating_season + 1
        test_mask = design.seasons == test_season
        for candidate in candidates:
            if candidate == "zero_prior_rapm":
                center = np.zeros(design.X.shape[1], dtype=float)
                scale = 0.0
                coverage = {
                    "players_in_design": len(design.players),
                    "players_with_prior": 0,
                    "train_off_possession_coverage": 0.0,
                    "train_def_possession_coverage": 0.0,
                    "test_lineup_slot_coverage": 0.0,
                }
            else:
                source = priors.loc[priors["candidate"].eq(candidate)]
                center, coverage = build_prior_center(
                    design,
                    source,
                    prior_window_end=rating_season,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
                scale = 1.0
            beta, intercept = fit_coefficient_center_path(
                design,
                config,
                center,
                center_scales=(scale,),
                row_mask=train_mask,
            )[scale]
            games = _stored_game_frame(
                args.evaluation_matrix_root, design, beta, intercept, rating_season
            )
            metric = _game_metrics(games)
            metrics.append(
                {
                    "candidate": candidate,
                    "rating_season": rating_season,
                    "test_season": test_season,
                    **metric,
                }
            )
            games["candidate"] = candidate
            games["rating_season"] = rating_season
            games["test_season"] = test_season
            games["squared_error"] = (games["actual_margin"] - games["predicted_margin"]) ** 2
            game_rows.append(games)
            ratings = _rating_table(design, beta, train_mask, names)
            ratings["candidate"] = candidate
            ratings["rating_season"] = rating_season
            rating_rows.append(ratings)
            coverage_rows.append({"candidate": candidate, "rating_season": rating_season, **coverage})
        print(f"AIO prior fold {rating_season}->{test_season}: complete", flush=True)

    metric_frame = pd.DataFrame(metrics)
    game_frame = pd.concat(game_rows, ignore_index=True)
    rating_frame = pd.concat(rating_rows, ignore_index=True)
    coverage_frame = pd.DataFrame(coverage_rows)
    game_counts = game_frame.groupby(["test_season", "candidate"])["game_id"].nunique().unstack()
    if not game_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates did not score identical game counts.")
    actual_hashes = game_frame.assign(
        actual_key=lambda x: x["game_id"].astype(str) + ":" + x["actual_margin"].astype(str)
    ).groupby(["test_season", "candidate"])["actual_key"].apply(
        lambda x: hashlib.sha256("|".join(sorted(x)).encode()).hexdigest()
    ).unstack()
    if not actual_hashes.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates did not score identical game outcomes.")
    identity_error = float((rating_frame["offense"] + rating_frame["defense"] - rating_frame["net"]).abs().max())
    if identity_error > 1e-10:
        raise ValueError("Posterior offense plus defense must equal net.")
    bootstrap = _paired_bootstrap(
        game_frame,
        baseline="selected_five_year_spm",
    )
    summary = metric_frame.groupby("candidate", as_index=False).agg(
        folds=("test_season", "nunique"),
        mean_margin_rmse=("margin_rmse", "mean"),
        mean_margin_correlation=("margin_correlation", "mean"),
        minimum_games=("games", "min"),
    ).sort_values("mean_margin_rmse", kind="stable")

    source_paths = {
        "preregistered_contract": ROOT / "research/experiments/aio_prior_bakeoff_v1.yml",
        "spm_priors": args.spm_priors,
        "features": args.features,
        "targets": args.targets,
    }
    config_payload = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "lambda_off": 3000.0,
        "lambda_def": 3000.0,
        "lambda_home": 300.0,
        "center_scale": 1.0,
        "training_lineup_policy": "legacy_terminal",
        "evaluation_lineup_policy": "canonical_transition_terminal",
        "box_alpha_grid": list(ALPHA_GRID),
        "pipm_like_residual_alpha": 100.0,
        "source_hashes": {
            **{key: sha256_file(path) for key, path in source_paths.items()},
            "legacy_possessions": {
                str(season): sha256_file(args.legacy_cache / f"matchups_{season}.parquet")
                for season in range(min(RATING_SEASONS), max(RATING_SEASONS) + 2)
            },
            "player_sheets": {
                str(season): sha256_file(args.player_sheet_root / f"{season}.parquet")
                for season in range(2014, max(RATING_SEASONS) + 1)
            },
            "schedules": {
                str(season): sha256_file(args.schedule_root / f"leaguegamelog_{season}.json.gz")
                for season in range(2014, max(RATING_SEASONS) + 1)
            },
            "runner": sha256_file(Path(__file__)),
            "evaluation_matrices": {
                str(season): {
                    name: sha256_file(
                        args.evaluation_matrix_root / f"5y_end_{season}" / name
                    )
                    for name in (
                        "player_ids.npy",
                        "evaluation_game_design.npz",
                        "evaluation_game_ids.npy",
                        "evaluation_actual_margin.npy",
                        "evaluation_intercept_multiplier.npy",
                        "evaluation_unknown_player_slots.npy",
                    )
                }
                for season in RATING_SEASONS
            },
        },
    }
    identity = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:10]
    output = args.artifact_root / "research" / "aio_prior_bakeoff" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    priors.to_parquet(output / "priors.parquet", index=False)
    rating_frame.to_parquet(output / "posterior_ratings.parquet", index=False)
    metric_frame.to_parquet(output / "game_metrics.parquet", index=False)
    game_frame.to_parquet(output / "game_predictions.parquet", index=False)
    coverage_frame.to_parquet(output / "prior_coverage.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    bootstrap.to_parquet(output / "paired_bootstrap.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_development_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "quality": {
            "identical_game_rows": True,
            "component_identity_max_error": identity_error,
            "duplicate_prior_keys": 0,
            "season_2027_rows": 0,
        },
        "paths": {
            "priors": "priors.parquet",
            "posterior_ratings": "posterior_ratings.parquet",
            "game_metrics": "game_metrics.parquet",
            "game_predictions": "game_predictions.parquet",
            "prior_coverage": "prior_coverage.parquet",
            "summary": "summary.parquet",
            "paired_bootstrap": "paired_bootstrap.parquet",
        },
        "caveats": [
            "All scored seasons are reused development data; this cannot promote a model.",
            "BoxPIPM-style is the reproducible box-only portion, not full historical PIPM.",
            "The PIPM-like arm uses derived raw on-off, not a fully luck-adjusted on-off estimate.",
            "The PIPM-like prior and RAPM likelihood reuse same-season lineup outcomes.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
