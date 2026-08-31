"""Compare 1-10 year RAPM windows and fixed age-conditional variants."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import (
    build_age_design,
    fit_age_adjusted_rapm,
    predict_age_adjusted_rapm,
)
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_unified_terminal_possessions,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rapm_horizon_age_sweep_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rapm_horizon_age_sweep"
AGE_DIR = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"


def _load_ages(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        frame = pd.read_parquet(AGE_DIR / f"{season}.parquet", columns=["PLAYER_ID", "AGE"])
        frame["Season"] = season
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    conflicts = panel.groupby(["Season", "PLAYER_ID"])["AGE"].nunique(dropna=True)
    if conflicts.gt(1).any():
        raise ValueError("The player sheet has conflicting ages within a season.")
    return panel.drop_duplicates(["Season", "PLAYER_ID"], keep="first")


def _paired_bootstrap(games: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for horizon, frame in games.groupby("horizon", sort=True):
        season_groups = [part.reset_index(drop=True) for _, part in frame.groupby("test_season")]
        differences = np.empty(draws, dtype=float)
        for draw in range(draws):
            normal_errors: list[np.ndarray] = []
            age_errors: list[np.ndarray] = []
            for part in season_groups:
                take = rng.integers(0, len(part), len(part))
                actual = part["actual_margin"].to_numpy()[take]
                normal_errors.append((actual - part["normal"].to_numpy()[take]) ** 2)
                age_errors.append((actual - part["age_conditional"].to_numpy()[take]) ** 2)
            differences[draw] = np.mean(np.concatenate(normal_errors)) - np.mean(
                np.concatenate(age_errors)
            )
        rows.append(
            {
                "horizon": horizon,
                "draws": draws,
                "normal_minus_age_mse": float(differences.mean()),
                "lower_95": float(np.quantile(differences, 0.025)),
                "upper_95": float(np.quantile(differences, 0.975)),
                "probability_age_better": float(np.mean(differences > 0)),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(int(value) for value in contract["seasons"])
    tests = set(int(value) for value in contract["common_evaluation_seasons"])
    started = time.perf_counter()
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    original = build_design(frame, include_home=True)
    season_mean = frame.groupby("season")["pts"].mean()
    centered_y = frame["pts"].to_numpy(dtype=float) - frame["season"].map(season_mean).to_numpy(dtype=float)
    design = replace(original, y=centered_y)
    ages = _load_ages(seasons)
    age_design = build_age_design(frame, ages)
    penalties = contract["penalties"]
    config = RapmConfig(
        seasons=seasons,
        lambda_off=float(penalties["lambda_off"]),
        lambda_def=float(penalties["lambda_def"]),
        lambda_home=float(penalties["lambda_home"]),
        data_scope="rapm_horizon_age_sweep",
    )
    metric_rows: list[dict] = []
    game_rows: list[pd.DataFrame] = []
    age_horizons = set(int(value) for value in contract["age_adjusted_horizons"])
    for horizon in contract["rolling_horizons"]:
        for test_season in sorted(tests):
            end = test_season - 1
            start = end - int(horizon) + 1
            train = (design.seasons >= start) & (design.seasons <= end)
            test = design.seasons == test_season
            if start < min(seasons) or not train.any() or not test.any():
                raise ValueError(f"Missing {horizon}y fold ending {end}.")
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            scoring_mean = float(original.y[train].mean())
            prediction = scoring_mean + intercept + np.asarray(design.X[test] @ beta).ravel()
            metrics, games = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            metric_rows.append(
                {
                    "model": "normal",
                    "horizon": f"{horizon}y",
                    "window_start": start,
                    "window_end": end,
                    "test_season": test_season,
                    **metrics,
                }
            )
            games = games.rename(columns={"predicted_margin": "normal"})
            if int(horizon) in age_horizons:
                age_fit = fit_age_adjusted_rapm(
                    design,
                    age_design,
                    config,
                    age_penalty=float(contract["age_penalty"]),
                    row_mask=train,
                )
                age_prediction = scoring_mean + predict_age_adjusted_rapm(
                    age_fit,
                    design,
                    age_design,
                    row_mask=test,
                    include_age=True,
                )
                age_metrics, age_games = game_margin_metrics(
                    frame.loc[test].reset_index(drop=True), age_prediction
                )
                metric_rows.append(
                    {
                        "model": "age_conditional",
                        "horizon": f"{horizon}y",
                        "window_start": start,
                        "window_end": end,
                        "test_season": test_season,
                        **age_metrics,
                    }
                )
                games = games.merge(
                    age_games.rename(columns={"predicted_margin": "age_conditional"})[
                        ["gameid", "actual_margin", "age_conditional"]
                    ],
                    on=["gameid", "actual_margin"],
                    validate="one_to_one",
                )
                games["horizon"] = f"{horizon}y"
                games["test_season"] = test_season
                game_rows.append(games)
            print(f"completed {horizon}y -> {test_season}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["model", "horizon"], as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            equal_season_mse=("margin_rmse", lambda values: float(np.mean(np.square(values)))),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_predicted_margin_sd=("predicted_margin_sd", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values(["model", "equal_season_mse", "horizon"], kind="stable")
    )
    paired_games = pd.concat(game_rows, ignore_index=True)
    bootstrap = _paired_bootstrap(
        paired_games,
        draws=int(contract["bootstrap_draws"]),
        seed=int(contract["bootstrap_seed"]),
    )
    identity = hashlib.sha256(
        json.dumps(
            {"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"rapm_horizon_age_sweep_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    paired_games.to_parquet(output / "age_game_predictions.parquet", index=False)
    bootstrap.to_parquet(output / "age_bootstrap.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "summary": summary.to_dict("records"),
        "age_bootstrap": bootstrap.to_dict("records"),
        "quality": {
            "possessions": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "age_slot_coverage": age_design.known_slots / age_design.total_slots,
            "identical_age_game_sets": True,
        },
        "paths": {
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "age_game_predictions": "age_game_predictions.parquet",
            "age_bootstrap": "age_bootstrap.parquet",
        },
        "forbidden_interpretation": "Independent confirmation, a neutral-age player ranking, or proof that one horizon is universally optimal.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
