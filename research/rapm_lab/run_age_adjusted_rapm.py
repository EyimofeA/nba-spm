"""Fit and validate same-age long-window RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import (
    age_curve,
    build_age_design,
    fit_age_adjusted_rapm,
    predict_age_adjusted_rapm,
)
from nba_impact.models.aging_resolution import load_age_panel
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    load_unified_terminal_possessions,
    ratings_table,
)
from research.rapm_lab.run_rubberband_progress_rapm import _paired_rmse_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "research/experiments/age_adjusted_rapm_v1.json"
CACHE = REPO_ROOT / "rapm/data/possession_cache"
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
PLAYER_GAMES = REPO_ROOT / "data/lake/silver/player_games.parquet"
NAMES = REPO_ROOT / "rapm/data/all_names.csv"
AGE_DIR = REPO_ROOT / "data/raw/playersheets/year_totals"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/age_adjusted_rapm"


def _config(contract: dict) -> RapmConfig:
    penalty = contract["player_penalties"]
    return RapmConfig(
        seasons=tuple(int(value) for value in contract["seasons"]),
        lambda_off=float(penalty["lambda_off"]),
        lambda_def=float(penalty["lambda_def"]),
        lambda_home=float(penalty["lambda_home"]),
        data_scope="same_age_long_window_rapm_research",
    )


def _evaluate(frame: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray) -> tuple[dict, pd.DataFrame]:
    metrics, games = game_margin_metrics(frame.reset_index(drop=True), prediction)
    metrics["possession_rmse"] = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    return metrics, games


def run(
    output_root: Path = OUTPUT_ROOT,
    contract_path: Path = CONTRACT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Age-adjusted RAPM requires a frozen contract.")
    if contract["untouched_confirmation_season"] in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    seasons = tuple(int(value) for value in contract["seasons"])
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(REPO_ROOT / "src/nba_impact/models/age_adjusted_rapm.py"),
        "legacy": {
            str(season): sha256_file(CACHE / f"matchups_{season}.parquet")
            for season in seasons
            if season < 2024
        },
        "possessions": sha256_file(POSSESSIONS),
        "segments": sha256_file(SEGMENTS),
        "ages": {
            str(season): sha256_file(AGE_DIR / f"{season}.csv") for season in seasons
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"{contract['experiment_id']}_{identity}"
    output = output_root / run_id
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    ages = load_age_panel(AGE_DIR, seasons)
    player_design = build_design(frame, include_home=True)
    age_design = build_age_design(
        frame,
        ages,
        minimum_age=int(contract["minimum_age"]),
        maximum_age=int(contract["maximum_age"]),
        reference_age=int(contract["reference_age"]),
    )
    config = _config(contract)

    selection_test_season = int(contract["selection_test_season"])
    selection_train = player_design.seasons < selection_test_season
    selection_test = player_design.seasons == selection_test_season
    selection_rows = []
    for age_penalty in contract["age_penalties"]:
        fit = fit_age_adjusted_rapm(
            player_design,
            age_design,
            config,
            age_penalty=float(age_penalty),
            row_mask=selection_train,
        )
        prediction = predict_age_adjusted_rapm(
            fit,
            player_design,
            age_design,
            row_mask=selection_test,
            include_age=True,
        )
        metrics, _ = _evaluate(
            frame.loc[selection_test], player_design.y[selection_test], prediction
        )
        selection_rows.append({"age_penalty": float(age_penalty), **metrics})
    selection = pd.DataFrame(selection_rows).sort_values(
        ["margin_rmse", "age_penalty"], kind="stable"
    )
    selected_penalty = float(selection.iloc[0]["age_penalty"])
    selection.to_parquet(output / "selection.parquet", index=False)

    diagnostic_season = int(contract["reused_diagnostic_season"])
    train = player_design.seasons < diagnostic_season
    test = player_design.seasons == diagnostic_season
    normal_beta, normal_intercept = fit_coefficients(player_design, config, row_mask=train)
    normal_prediction = normal_intercept + np.asarray(
        player_design.X[test] @ normal_beta
    ).ravel()
    age_fit = fit_age_adjusted_rapm(
        player_design,
        age_design,
        config,
        age_penalty=selected_penalty,
        row_mask=train,
    )
    neutral_prediction = predict_age_adjusted_rapm(
        age_fit,
        player_design,
        age_design,
        row_mask=test,
        include_age=False,
    )
    conditional_prediction = predict_age_adjusted_rapm(
        age_fit,
        player_design,
        age_design,
        row_mask=test,
        include_age=True,
    )
    predictions = {
        "normal": normal_prediction,
        "same_age_27": neutral_prediction,
        "age_conditional": conditional_prediction,
    }
    evaluation_rows = []
    game_predictions = {}
    for variant, prediction in predictions.items():
        metrics, games = _evaluate(frame.loc[test], player_design.y[test], prediction)
        evaluation_rows.append({"variant": variant, **metrics})
        game_predictions[variant] = games
        games.assign(variant=variant).to_parquet(
            output / f"game_predictions_{variant}.parquet", index=False
        )
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_parquet(output / "evaluation.parquet", index=False)
    bootstrap = {
        variant: _paired_rmse_bootstrap(
            game_predictions["normal"],
            game_predictions[variant],
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]) + offset,
        )
        for offset, variant in enumerate(("same_age_27", "age_conditional"))
    }

    full_normal_beta, _ = fit_coefficients(player_design, config)
    full_age_fit = fit_age_adjusted_rapm(
        player_design,
        age_design,
        config,
        age_penalty=selected_penalty,
    )
    curve = age_curve(full_age_fit)
    curve.to_parquet(output / "age_curve.parquet", index=False)
    names = load_current_player_names(NAMES, PLAYER_GAMES)
    normal_ratings = ratings_table(player_design, full_normal_beta, names=names).rename(
        columns={
            "offense_per_100": "normal_offense",
            "defense_per_100": "normal_defense",
            "net_per_100": "normal_net",
        }
    )
    age_ratings = ratings_table(
        player_design, full_age_fit.player_coefficients, names=names
    ).rename(
        columns={
            "offense_per_100": "age27_offense",
            "defense_per_100": "age27_defense",
            "net_per_100": "age27_net",
        }
    )
    ratings = normal_ratings.drop(columns="uncertainty_status").merge(
        age_ratings[["player_id", "age27_offense", "age27_defense", "age27_net"]],
        on="player_id",
        validate="one_to_one",
    )
    ratings["age27_net_change"] = ratings["age27_net"] - ratings["normal_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)

    qualified = ratings.loc[
        ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(10_000)
    ]
    manifest = {
        "run_id": run_id,
        "status": "research_diagnostic",
        "model_family": "same_age_long_window_rapm",
        "estimand_id": contract["estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload, "selected_age_penalty": selected_penalty},
        "quality": {
            "possessions": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "age_slot_coverage": float(age_design.known_slots / age_design.total_slots),
            "known_age_slots": age_design.known_slots,
            "total_age_slots": age_design.total_slots,
            "qualified_players": int(len(qualified)),
            "maximum_rating_identity_error": float(
                np.abs(
                    ratings["age27_offense"]
                    + ratings["age27_defense"]
                    - ratings["age27_net"]
                ).max()
            ),
        },
        "selection": selection.to_dict("records"),
        "evaluation": evaluation.to_dict("records"),
        "bootstrap_vs_normal": bootstrap,
        "rating_effect": {
            "net_correlation_with_normal": float(
                ratings[["normal_net", "age27_net"]].corr().iloc[0, 1]
            ),
            "mean_absolute_net_change": float(ratings["age27_net_change"].abs().mean()),
        },
        "paths": {
            "selection": "selection.parquet",
            "evaluation": "evaluation.parquet",
            "age_curve": "age_curve.parquet",
            "ratings": "ratings.parquet",
        },
        "forbidden_interpretation": (
            "This long-window age-27 counterfactual is not single-season impact, "
            "a career ranking, biological causality, or a forecast."
        ),
    }
    write_json_atomic(manifest, run_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.contract), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
