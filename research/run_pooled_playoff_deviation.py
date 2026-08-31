"""Test heavily shrunk player playoff deviations on future postseasons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "rapm/data/possession_cache"
CONTRACT = ROOT / "research/experiments/pooled_playoff_deviation_v1.yml"
OUTPUT_ROOT = ROOT / "artifacts/research/pooled_playoff_deviation"
SEASONS = tuple(range(2019, 2024))
DEVIATION_PENALTIES = (3000.0, 10000.0, 30000.0, 100000.0, 300000.0)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260829


def _load() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = CACHE / f"matchups_{season}.parquet"
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["gameid"].astype(str).str[:3].isin(["002", "004"])].copy()
        frame["season_type"] = np.where(
            frame["gameid"].astype(str).str.startswith("004"), "playoffs", "regular"
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if int(combined["season"].max()) >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    if combined.duplicated(["gameid", "period", "num"]).any():
        raise ValueError("Duplicate possession keys in playoff pilot input.")
    return combined


def _paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    paired = baseline.merge(
        candidate,
        on="gameid",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("Baseline and challenger must score identical games.")
    actual = paired["actual_margin_baseline"].to_numpy(float)
    if not np.array_equal(actual, paired["actual_margin_candidate"].to_numpy(float)):
        raise ValueError("Actual margins differ across paired predictions.")
    base_error = actual - paired["predicted_margin_baseline"].to_numpy(float)
    candidate_error = actual - paired["predicted_margin_candidate"].to_numpy(float)
    per_game_delta = candidate_error**2 - base_error**2
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(paired), size=(draws, len(paired)))
    draws_delta = per_game_delta[sampled].mean(axis=1)
    return {
        "games": int(len(paired)),
        "mse_delta": float(per_game_delta.mean()),
        "mse_delta_95_low": float(np.quantile(draws_delta, 0.025)),
        "mse_delta_95_high": float(np.quantile(draws_delta, 0.975)),
        "probability_candidate_better": float(np.mean(draws_delta < 0)),
        "draws": int(draws),
        "seed": int(seed),
    }


def _fit_fold(frame: pd.DataFrame, target: int, penalty: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    regular_start = max(SEASONS[0], target - 4)
    eligible = frame.loc[
        frame["season"].between(regular_start, target)
        | (frame["season_type"].eq("playoffs") & frame["season"].lt(target))
    ].copy()
    target_playoffs = frame.loc[
        frame["season"].eq(target) & frame["season_type"].eq("playoffs")
    ].copy()
    combined = pd.concat([eligible, target_playoffs], ignore_index=True)
    design = build_design(combined)
    regular_mask = (
        combined["season"].between(regular_start, target)
        & combined["season_type"].eq("regular")
    ).to_numpy()
    prior_playoff_mask = (
        combined["season"].lt(target) & combined["season_type"].eq("playoffs")
    ).to_numpy()
    test_mask = (
        combined["season"].eq(target) & combined["season_type"].eq("playoffs")
    ).to_numpy()
    base_config = RapmConfig(
        seasons=tuple(range(regular_start, target + 1)),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        data_scope="pooled_playoff_deviation_regular_base",
    )
    base_beta, base_intercept = fit_coefficients(design, base_config, row_mask=regular_mask)
    base_prediction = base_intercept + np.asarray(design.X @ base_beta).ravel()
    residual_design = replace(design, y=design.y - base_prediction)
    deviation_config = RapmConfig(
        seasons=tuple(range(SEASONS[0], target)),
        lambda_off=penalty,
        lambda_def=penalty,
        lambda_home=1e12,
        data_scope="pooled_playoff_deviation_residual",
    )
    deviation_beta, deviation_intercept = fit_coefficients(
        residual_design, deviation_config, row_mask=prior_playoff_mask
    )
    candidate_prediction = (
        base_prediction + deviation_intercept + np.asarray(design.X @ deviation_beta).ravel()
    )
    test = combined.loc[test_mask].reset_index(drop=True)
    baseline_metrics, baseline_games = game_margin_metrics(test, base_prediction[test_mask])
    candidate_metrics, candidate_games = game_margin_metrics(test, candidate_prediction[test_mask])
    n_players = len(design.players)
    exposure = np.asarray(design.X[prior_playoff_mask, : 2 * n_players].sum(axis=0)).ravel()
    ratings = pd.DataFrame(
        {
            "target_season": target,
            "player_id": design.players,
            "playoff_offense_deviation_per_100": 100 * deviation_beta[:n_players],
            "playoff_defense_deviation_per_100": -100 * deviation_beta[n_players : 2 * n_players],
            "prior_playoff_offensive_possessions": exposure[:n_players],
            "prior_playoff_defensive_possessions": exposure[n_players:],
        }
    )
    ratings["playoff_net_deviation_per_100"] = (
        ratings["playoff_offense_deviation_per_100"]
        + ratings["playoff_defense_deviation_per_100"]
    )
    row = {
        "target_season": target,
        "regular_start": regular_start,
        "prior_playoff_seasons": "|".join(str(x) for x in range(SEASONS[0], target)),
        "deviation_penalty": penalty,
        "prior_playoff_games": int(combined.loc[prior_playoff_mask, "gameid"].nunique()),
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
    }
    games = baseline_games.merge(
        candidate_games,
        on=["gameid", "actual_margin"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    return row, games, ratings


def run() -> dict:
    frame = _load()
    selection_rows = []
    for penalty in DEVIATION_PENALTIES:
        row, _, _ = _fit_fold(frame, 2022, penalty)
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows)
    selected_penalty = float(
        selection.sort_values(["candidate_margin_rmse", "deviation_penalty"], kind="stable")
        .iloc[0]["deviation_penalty"]
    )
    diagnostic_row, diagnostic_games, ratings = _fit_fold(frame, 2023, selected_penalty)
    diagnostic = pd.DataFrame([diagnostic_row])
    base_games = diagnostic_games.rename(
        columns={"predicted_margin_baseline": "predicted_margin"}
    )[["gameid", "actual_margin", "predicted_margin"]]
    candidate_games = diagnostic_games.rename(
        columns={"predicted_margin_candidate": "predicted_margin"}
    )[["gameid", "actual_margin", "predicted_margin"]]
    bootstrap = _paired_bootstrap(base_games, candidate_games)
    source_hashes = {
        str(season): sha256_file(CACHE / f"matchups_{season}.parquet") for season in SEASONS
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(CONTRACT),
                "sources": source_hashes,
                "selected_penalty": selected_penalty,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"pooled_playoff_deviation_v1_{identity}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(output / "selection.parquet", index=False)
    diagnostic.to_parquet(output / "diagnostic.parquet", index=False)
    diagnostic_games.to_parquet(output / "diagnostic_games.parquet", index=False)
    ratings.to_parquet(output / "playoff_deviations.parquet", index=False)
    decision = (
        "research_challenger"
        if bootstrap["mse_delta_95_high"] < 0
        else "research_null"
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": decision,
        "scope": "historical_2019_2023_pilot",
        "selected_deviation_penalty": selected_penalty,
        "selection": selection.to_dict("records"),
        "diagnostic": diagnostic_row,
        "paired_bootstrap": bootstrap,
        "source_hashes": source_hashes,
        "contract_hash": sha256_file(CONTRACT),
        "untouched_confirmation_season": 2027,
        "forbidden_interpretation": "A standalone playoff rating, causal clutch skill, or current production evidence.",
        "artifacts": {
            "selection": "selection.parquet",
            "diagnostic": "diagnostic.parquet",
            "diagnostic_games": "diagnostic_games.parquet",
            "playoff_deviations": "playoff_deviations.parquet",
        },
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
