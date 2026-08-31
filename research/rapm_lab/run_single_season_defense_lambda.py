"""Compare 3000 and 4500 defense penalties in one-season RAPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    load_unified_terminal_possessions,
    ratings_table,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/single_season_defense_lambda_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/single_season_defense_lambda"


def _paired_bootstrap(games: pd.DataFrame, draws: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    season_groups = [frame.reset_index(drop=True) for _, frame in games.groupby("season")]
    improvements = np.empty(draws, dtype=float)
    for draw in range(draws):
        reference_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for frame in season_groups:
            take = rng.integers(0, len(frame), len(frame))
            actual = frame["actual_margin"].to_numpy()[take]
            reference_errors.append((actual - frame["reference"].to_numpy()[take]) ** 2)
            candidate_errors.append((actual - frame["defense_4500"].to_numpy()[take]) ** 2)
        improvements[draw] = np.mean(np.concatenate(reference_errors)) - np.mean(
            np.concatenate(candidate_errors)
        )
    return {
        "draws": draws,
        "seed": seed,
        "reference_minus_candidate_mse": float(improvements.mean()),
        "lower_95": float(np.quantile(improvements, 0.025)),
        "upper_95": float(np.quantile(improvements, 0.975)),
        "probability_candidate_better": float(np.mean(improvements > 0)),
    }


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(int(season) for season in contract["seasons"])
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv",
        ROOT / "data/lake/silver/player_games.parquet",
    )
    evaluation_rows: list[dict] = []
    game_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    latest = max(seasons)
    for candidate in contract["candidates"]:
        label = str(candidate["label"])
        config = RapmConfig(
            seasons=seasons,
            lambda_off=float(candidate["lambda_off"]),
            lambda_def=float(candidate["lambda_def"]),
            lambda_home=float(candidate["lambda_home"]),
            data_scope="single_season_defense_lambda_research",
        )
        for train_season in seasons:
            train = design.seasons == train_season
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            if train_season == latest:
                ratings = ratings_table(design, beta, names=names)
                n_players = len(design.players)
                ratings["off_possessions"] = np.asarray(
                    design.X[train, :n_players].sum(axis=0)
                ).ravel()
                ratings["def_possessions"] = np.asarray(
                    design.X[train, n_players : 2 * n_players].sum(axis=0)
                ).ravel()
                ratings = ratings.loc[
                    ratings[["off_possessions", "def_possessions"]].min(axis=1).gt(0)
                ]
                ratings.insert(0, "variant", label)
                ratings.insert(1, "season", train_season)
                rating_frames.append(ratings)
            test_season = train_season + 1
            if test_season not in contract["evaluation_seasons"]:
                continue
            test = design.seasons == test_season
            prediction = intercept + np.asarray(design.X[test] @ beta).ravel()
            metrics, games = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            evaluation_rows.append(
                {
                    "variant": label,
                    "train_season": train_season,
                    "evaluation_season": test_season,
                    **metrics,
                }
            )
            games = games.rename(columns={"predicted_margin": label})
            games["season"] = test_season
            game_frames.append(games[["season", "gameid", "actual_margin", label]])
        print(f"completed {label}", flush=True)

    evaluation = pd.DataFrame(evaluation_rows)
    summaries = evaluation.groupby("variant", as_index=False).agg(
        seasons=("evaluation_season", "nunique"),
        mean_margin_rmse=("margin_rmse", "mean"),
        mean_margin_mae=("margin_mae", "mean"),
        mean_margin_correlation=("margin_correlation", "mean"),
    )
    by_variant = {
        label: pd.concat([frame for frame in game_frames if label in frame], ignore_index=True)
        for label in ("reference", "defense_4500")
    }
    paired = by_variant["reference"].merge(
        by_variant["defense_4500"],
        on=["season", "gameid", "actual_margin"],
        validate="one_to_one",
    )
    bootstrap = _paired_bootstrap(
        paired,
        draws=int(contract["bootstrap_draws"]),
        seed=int(contract["bootstrap_seed"]),
    )
    identity = hashlib.sha256(
        json.dumps(
            {"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"single_season_defense_lambda_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(output / "evaluation.parquet", index=False)
    summaries.to_parquet(output / "summary.parquet", index=False)
    paired.to_parquet(output / "game_predictions.parquet", index=False)
    pd.concat(rating_frames, ignore_index=True).to_parquet(
        output / "ratings_2026.parquet", index=False
    )
    manifest = {
        "run_id": output.name,
        "status": "research_comparison_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": contract,
        "summary": summaries.to_dict("records"),
        "paired_bootstrap": bootstrap,
        "quality": {
            "possessions": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "same_game_set": len(paired) == len(by_variant["reference"]) == len(by_variant["defense_4500"]),
        },
        "paths": {
            "evaluation": "evaluation.parquet",
            "summary": "summary.parquet",
            "game_predictions": "game_predictions.parquet",
            "ratings_2026": "ratings_2026.parquet"
        },
        "forbidden_interpretation": "This reused historical evidence does not promote a production penalty.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
