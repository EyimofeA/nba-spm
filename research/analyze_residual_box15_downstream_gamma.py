#!/usr/bin/env python3
"""Select residual offense and defense scales on earlier downstream games."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from run_aio_prior_canonical_followup import _center, _remap_annual, _solve

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "residual_box15_downstream_gamma_v1"
CONTRACT = ROOT / "research/experiments/residual_box15_downstream_gamma_v1.yml"
SOURCE_RUN = ROOT / "artifacts/research/residual_box15_spm/residual_box15_spm_v1_427c5f2c25"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
RATING_SEASONS = tuple(range(2022, 2027))
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Downstream gamma contract ID changed.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _candidate_surface(priors: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[tuple[float, float], np.ndarray]], dict[int, np.ndarray]]:
    annual, _ = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    game_rows = []
    rating_beta: dict[int, dict[tuple[float, float], np.ndarray]] = {}
    rating_players: dict[int, np.ndarray] = {}
    for season in RATING_SEASONS:
        matrix_dir = MATRIX_ROOT / f"5y_end_{season}"
        players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[season], players)
        control = priors.loc[
            priors["candidate"].eq("box15") & priors["Window_End"].eq(season)
        ]
        residual = priors.loc[
            priors["candidate"].eq("box15_residual") & priors["Window_End"].eq(season)
        ]
        control_center, _ = _center(control, bundle)
        residual_center, _ = _center(residual, bundle)
        n = len(players)
        delta = residual_center - control_center
        offense_center = control_center.copy()
        offense_center[:n] += delta[:n]
        defense_center = control_center.copy()
        defense_center[n : 2 * n] += delta[n : 2 * n]
        beta_base, intercept_base = _solve(bundle, control_center, scale=1.0)
        beta_off, intercept_off = _solve(bundle, offense_center, scale=1.0)
        beta_def, intercept_def = _solve(bundle, defense_center, scale=1.0)
        rating_players[season] = players
        rating_beta[season] = {}
        if season in EVALUATED_RATING_SEASONS:
            game_base = stored_evaluation_predictions(matrix_dir, beta_base, intercept_base)
            game_off = stored_evaluation_predictions(matrix_dir, beta_off, intercept_off)
            game_def = stored_evaluation_predictions(matrix_dir, beta_def, intercept_def)
        for gamma_offense, gamma_defense in itertools.product(GAMMAS, GAMMAS):
            rating_beta[season][(gamma_offense, gamma_defense)] = (
                beta_base
                + gamma_offense * (beta_off - beta_base)
                + gamma_defense * (beta_def - beta_base)
            )
            if season in EVALUATED_RATING_SEASONS:
                row = game_base.copy()
                row["predicted_margin"] = (
                    game_base["predicted_margin"]
                    + gamma_offense * (game_off["predicted_margin"] - game_base["predicted_margin"])
                    + gamma_defense * (game_def["predicted_margin"] - game_base["predicted_margin"])
                )
                row["rating_season"] = season
                row["test_season"] = season + 1
                row["gamma_offense"] = gamma_offense
                row["gamma_defense"] = gamma_defense
                row["squared_error"] = (row["actual_margin"] - row["predicted_margin"]) ** 2
                game_rows.append(row)
    return pd.concat(game_rows, ignore_index=True), rating_beta, rating_players


def _select(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in RATING_SEASONS:
        history = surface.loc[surface["rating_season"].lt(season)]
        if history.empty:
            selected_offense = selected_defense = 0.0
            history_folds = 0
            selected_mse = None
        else:
            scores = (
                history.groupby(["gamma_offense", "gamma_defense", "rating_season"], as_index=False)["squared_error"].mean()
                .groupby(["gamma_offense", "gamma_defense"], as_index=False)
                .agg(equal_season_mse=("squared_error", "mean"), history_folds=("rating_season", "nunique"))
            )
            winner = scores.sort_values(
                ["equal_season_mse", "gamma_offense", "gamma_defense"], kind="stable"
            ).iloc[0]
            selected_offense = float(winner["gamma_offense"])
            selected_defense = float(winner["gamma_defense"])
            history_folds = int(winner["history_folds"])
            selected_mse = float(winner["equal_season_mse"])
        rows.append(
            {
                "rating_season": season,
                "gamma_offense": selected_offense,
                "gamma_defense": selected_defense,
                "history_folds": history_folds,
                "history_equal_season_mse": selected_mse,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    contract = _load_contract()
    priors = pd.read_parquet(SOURCE_RUN / "priors.parquet")
    if priors["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered the downstream gamma experiment.")
    surface, rating_beta, rating_players = _candidate_surface(priors)
    selection = _select(surface)
    game_rows = []
    rating_rows = []
    for row in selection.itertuples(index=False):
        season = int(row.rating_season)
        gamma = (float(row.gamma_offense), float(row.gamma_defense))
        players = rating_players[season]
        n = len(players)
        for candidate, selected_gamma in (
            ("box15_aio", (0.0, 0.0)),
            ("box15_residual_tuned_aio", gamma),
        ):
            beta = rating_beta[season][selected_gamma]
            ratings = pd.DataFrame(
                {
                    "PLAYER_ID": players,
                    "rating_season": season,
                    "candidate": candidate,
                    "offense": 100.0 * beta[:n],
                    "defense": -100.0 * beta[n : 2 * n],
                }
            )
            ratings["net"] = ratings["offense"] + ratings["defense"]
            rating_rows.append(ratings)
            if season in EVALUATED_RATING_SEASONS:
                games = surface.loc[
                    surface["rating_season"].eq(season)
                    & surface["gamma_offense"].eq(selected_gamma[0])
                    & surface["gamma_defense"].eq(selected_gamma[1])
                ].copy()
                games["candidate"] = candidate
                game_rows.append(games)
    games = pd.concat(game_rows, ignore_index=True)
    ratings = pd.concat(rating_rows, ignore_index=True)
    base.MODEL_ORDER = ("box15_aio", "box15_residual_tuned_aio")
    base.PRIMARY_PAIRS = {frozenset(base.MODEL_ORDER)}
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(games, draws=5000, seed=20260829)
    pair = paired.iloc[0]
    challenger_delta = -float(pair["mean_mse_delta"])
    challenger_low = -float(pair["bootstrap_95_high"])
    challenger_high = -float(pair["bootstrap_95_low"])
    config = {
        "experiment_id": EXPERIMENT_ID,
        "gamma_grid": list(GAMMAS),
        "source_run_sha256": sha256_file(SOURCE_RUN / "run.json"),
        "contract_sha256": sha256_file(CONTRACT),
        "runner_sha256": sha256_file(Path(__file__)),
        "bootstrap": {"draws": 5000, "seed": 20260829, "unit": "whole game within test season"},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/residual_box15_downstream_gamma" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "gamma_surface.parquet": surface,
        "selection.parquet": selection,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_challenger" if challenger_high < 0 else "research_null",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
            "season_2027_loaded": False,
        },
        "decision": {
            "challenger": "box15_residual_tuned_aio",
            "reference": "box15_aio",
            "mean_mse_delta": challenger_delta,
            "bootstrap_95_low": challenger_low,
            "bootstrap_95_high": challenger_high,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(selection.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(run["decision"], indent=2))


if __name__ == "__main__":
    main()
