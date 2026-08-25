"""Tune a five-year, time-decayed RAPM with actual-age lineup controls."""

from __future__ import annotations

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
    current_age_player_coefficients,
    fit_age_adjusted_rapm,
    predict_age_adjusted_rapm,
    season_decay_weights,
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


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/time_decay_actual_age_5y_rapm_v1.json"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/time_decay_actual_age_5y_rapm"
CACHE = ROOT / "rapm/data/possession_cache"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
AGE_DIR = ROOT / "data/raw/playersheets/year_totals"


def _metrics(
    frame: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    metrics, games = game_margin_metrics(frame.reset_index(drop=True), prediction)
    metrics["possession_rmse"] = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    return metrics, games


def _candidate_key(half_life: float | None, age_penalty: float, player_penalty: float) -> str:
    half = "none" if half_life is None else f"{half_life:g}"
    return f"half={half}|age={age_penalty:g}|player={player_penalty:g}"


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Time-decay tuning requires a frozen contract.")
    all_seasons = tuple(
        sorted(
            set(contract["selection_train_seasons"])
            | {contract["selection_test_season"]}
            | set(contract["diagnostic_train_seasons"])
            | {contract["diagnostic_test_season"]}
            | set(contract["final_fit_seasons"])
        )
    )
    if contract["untouched_confirmation_season"] in all_seasons:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract": sha256_file(CONTRACT),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "src/nba_impact/models/age_adjusted_rapm.py"),
        "possessions": sha256_file(POSSESSIONS),
        "segments": sha256_file(SEGMENTS),
        "legacy": {
            str(season): sha256_file(CACHE / f"matchups_{season}.parquet")
            for season in all_seasons
            if season < 2024
        },
        "ages": {
            str(season): sha256_file(AGE_DIR / f"{season}.csv") for season in all_seasons
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"time_decay_actual_age_5y_rapm_v1_{identity}"
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        all_seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    ages = load_age_panel(AGE_DIR, all_seasons)
    design = build_design(frame, include_home=True)
    age_design = build_age_design(
        frame,
        ages,
        minimum_age=int(contract["minimum_age"]),
        maximum_age=int(contract["maximum_age"]),
        reference_age=int(contract["reference_age"]),
    )
    selection_train = np.isin(design.seasons, contract["selection_train_seasons"])
    selection_test = design.seasons == int(contract["selection_test_season"])
    baseline_config = RapmConfig(
        seasons=tuple(contract["selection_train_seasons"]),
        lambda_off=float(contract["base_player_penalty"]),
        lambda_def=float(contract["base_player_penalty"]),
        lambda_home=float(contract["lambda_home"]),
        data_scope="five_year_unweighted_no_age_baseline",
    )
    baseline_beta, baseline_intercept = fit_coefficients(
        design, baseline_config, row_mask=selection_train
    )
    baseline_prediction = baseline_intercept + np.asarray(
        design.X[selection_test] @ baseline_beta
    ).ravel()
    baseline_metrics, baseline_games = _metrics(
        frame.loc[selection_test], design.y[selection_test], baseline_prediction
    )

    cache: dict[str, dict] = {}

    def score_candidate(
        half_life: float | None,
        age_penalty: float,
        player_penalty: float,
    ) -> dict:
        key = _candidate_key(half_life, age_penalty, player_penalty)
        if key in cache:
            return cache[key]
        config = RapmConfig(
            seasons=tuple(contract["selection_train_seasons"]),
            lambda_off=player_penalty,
            lambda_def=player_penalty,
            lambda_home=float(contract["lambda_home"]),
            data_scope="five_year_time_decay_actual_age_selection",
        )
        weights = season_decay_weights(
            design.seasons,
            window_end=max(contract["selection_train_seasons"]),
            half_life_years=half_life,
        )
        fit = fit_age_adjusted_rapm(
            design,
            age_design,
            config,
            age_penalty=age_penalty,
            row_mask=selection_train,
            row_weights=weights,
        )
        prediction = predict_age_adjusted_rapm(
            fit, design, age_design, row_mask=selection_test, include_age=True
        )
        metrics, _ = _metrics(
            frame.loc[selection_test], design.y[selection_test], prediction
        )
        result = {
            "key": key,
            "half_life_years": half_life,
            "age_penalty": age_penalty,
            "player_penalty": player_penalty,
            **metrics,
        }
        cache[key] = result
        return result

    base_age = float(contract["base_age_penalty"])
    base_player = float(contract["base_player_penalty"])
    half_life_stage = [
        score_candidate(value, base_age, base_player)
        for value in contract["half_lives_years"]
    ]
    selected_half_life = min(half_life_stage, key=lambda row: row["margin_rmse"])[
        "half_life_years"
    ]
    age_stage = [
        score_candidate(selected_half_life, float(value), base_player)
        for value in contract["age_penalties"]
    ]
    selected_age = min(age_stage, key=lambda row: row["margin_rmse"])["age_penalty"]
    penalty_stage = [
        score_candidate(selected_half_life, selected_age, float(value))
        for value in contract["player_penalties"]
    ]
    selected = min(penalty_stage, key=lambda row: row["margin_rmse"])
    selection = pd.DataFrame(cache.values()).sort_values(
        ["margin_rmse", "key"], kind="stable"
    )
    selection.to_parquet(output / "selection.parquet", index=False)

    diagnostic_train = np.isin(design.seasons, contract["diagnostic_train_seasons"])
    diagnostic_test = design.seasons == int(contract["diagnostic_test_season"])
    diagnostic_config = RapmConfig(
        seasons=tuple(contract["diagnostic_train_seasons"]),
        lambda_off=float(selected["player_penalty"]),
        lambda_def=float(selected["player_penalty"]),
        lambda_home=float(contract["lambda_home"]),
        data_scope="five_year_time_decay_actual_age_diagnostic",
    )
    diagnostic_baseline_config = RapmConfig(
        seasons=tuple(contract["diagnostic_train_seasons"]),
        lambda_off=base_player,
        lambda_def=base_player,
        lambda_home=float(contract["lambda_home"]),
        data_scope="five_year_unweighted_no_age_diagnostic",
    )
    diagnostic_baseline_beta, diagnostic_baseline_intercept = fit_coefficients(
        design, diagnostic_baseline_config, row_mask=diagnostic_train
    )
    diagnostic_baseline_prediction = diagnostic_baseline_intercept + np.asarray(
        design.X[diagnostic_test] @ diagnostic_baseline_beta
    ).ravel()
    diagnostic_baseline_metrics, diagnostic_baseline_games = _metrics(
        frame.loc[diagnostic_test],
        design.y[diagnostic_test],
        diagnostic_baseline_prediction,
    )
    diagnostic_weights = season_decay_weights(
        design.seasons,
        window_end=max(contract["diagnostic_train_seasons"]),
        half_life_years=selected["half_life_years"],
    )
    diagnostic_fit = fit_age_adjusted_rapm(
        design,
        age_design,
        diagnostic_config,
        age_penalty=float(selected["age_penalty"]),
        row_mask=diagnostic_train,
        row_weights=diagnostic_weights,
    )
    diagnostic_prediction = predict_age_adjusted_rapm(
        diagnostic_fit,
        design,
        age_design,
        row_mask=diagnostic_test,
        include_age=True,
    )
    diagnostic_metrics, diagnostic_games = _metrics(
        frame.loc[diagnostic_test], design.y[diagnostic_test], diagnostic_prediction
    )
    pd.DataFrame(
        [
            {"variant": "baseline", **diagnostic_baseline_metrics},
            {"variant": "time_decay_actual_age", **diagnostic_metrics},
        ]
    ).to_parquet(output / "diagnostic.parquet", index=False)
    diagnostic_games.assign(variant="time_decay_actual_age").to_parquet(
        output / "diagnostic_game_predictions.parquet", index=False
    )
    bootstrap = _paired_rmse_bootstrap(
        diagnostic_baseline_games,
        diagnostic_games,
        draws=int(contract["paired_game_bootstrap_draws"]),
        seed=int(contract["paired_game_bootstrap_seed"]),
    )

    final_mask = np.isin(design.seasons, contract["final_fit_seasons"])
    final_config = RapmConfig(
        seasons=tuple(contract["final_fit_seasons"]),
        lambda_off=float(selected["player_penalty"]),
        lambda_def=float(selected["player_penalty"]),
        lambda_home=float(contract["lambda_home"]),
        data_scope="five_year_time_decay_actual_age_final_candidate",
    )
    final_weights = season_decay_weights(
        design.seasons,
        window_end=max(contract["final_fit_seasons"]),
        half_life_years=selected["half_life_years"],
    )
    final_fit = fit_age_adjusted_rapm(
        design,
        age_design,
        final_config,
        age_penalty=float(selected["age_penalty"]),
        row_mask=final_mask,
        row_weights=final_weights,
    )
    current_coefficients, current_age_coverage = current_age_player_coefficients(
        final_fit,
        design,
        ages,
        season=max(contract["final_fit_seasons"]),
    )
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet"
    )
    base_ratings = ratings_table(
        design, final_fit.player_coefficients, names=names
    ).rename(
        columns={
            "offense_per_100": "base_offense",
            "defense_per_100": "base_defense",
            "net_per_100": "base_net",
        }
    )
    current_ratings = ratings_table(design, current_coefficients, names=names).rename(
        columns={
            "offense_per_100": "offense",
            "defense_per_100": "defense",
            "net_per_100": "net",
        }
    )
    ratings = current_ratings.drop(columns="uncertainty_status").merge(
        base_ratings[["player_id", "base_offense", "base_defense", "base_net"]],
        on="player_id",
        validate="one_to_one",
    )
    ratings["age_offense_adjustment"] = ratings["offense"] - ratings["base_offense"]
    ratings["age_defense_adjustment"] = ratings["defense"] - ratings["base_defense"]
    ratings["age_net_adjustment"] = ratings["net"] - ratings["base_net"]
    n_players = len(design.players)
    final_off_exposure = np.asarray(
        design.X[final_mask, :n_players].sum(axis=0)
    ).ravel()
    final_def_exposure = np.asarray(
        design.X[final_mask, n_players : 2 * n_players].sum(axis=0)
    ).ravel()
    exposure = pd.DataFrame(
        {
            "player_id": design.players,
            "off_possessions": final_off_exposure,
            "def_possessions": final_def_exposure,
        }
    )
    ratings = ratings.drop(columns=["off_possessions", "def_possessions"]).merge(
        exposure, on="player_id", validate="one_to_one"
    )
    current_player_ids = set(
        pd.to_numeric(
            ages.loc[
                pd.to_numeric(ages["Season"]).eq(max(contract["final_fit_seasons"])),
                "PLAYER_ID",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    ratings = ratings.loc[ratings["player_id"].isin(current_player_ids)].copy()
    ratings["window_start"] = min(contract["final_fit_seasons"])
    ratings["window_end"] = max(contract["final_fit_seasons"])
    ratings.to_parquet(output / "ratings.parquet", index=False)
    age_curve(final_fit).to_parquet(output / "age_curve.parquet", index=False)

    selection_delta = float(selected["margin_rmse"] - baseline_metrics["margin_rmse"])
    diagnostic_delta = float(
        diagnostic_metrics["margin_rmse"] - diagnostic_baseline_metrics["margin_rmse"]
    )
    promoted = (
        selection_delta < 0
        and diagnostic_delta < 0
        and bootstrap["upper_95"] < 0
        and diagnostic_metrics["margin_correlation"]
        >= diagnostic_baseline_metrics["margin_correlation"] - 0.01
    )
    manifest = {
        "run_id": output.name,
        "status": "production_candidate" if promoted else "research_challenger",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "estimand_id": contract["estimand_id"],
        "contract": {**contract, **identity_payload},
        "selected": selected,
        "selection_baseline": baseline_metrics,
        "selection_rmse_delta": selection_delta,
        "diagnostic": {
            "baseline": diagnostic_baseline_metrics,
            "candidate": diagnostic_metrics,
            "rmse_delta": diagnostic_delta,
            "paired_bootstrap": bootstrap,
        },
        "promotion_gate_passed": promoted,
        "quality": {
            "possessions": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "lineup_age_slot_coverage": float(age_design.known_slots / age_design.total_slots),
            "current_age_player_coverage_all_fitted_players": current_age_coverage,
            "published_current_players": int(len(ratings)),
            "published_current_age_coverage": 1.0,
            "maximum_rating_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
        },
        "paths": {
            "selection": "selection.parquet",
            "diagnostic": "diagnostic.parquet",
            "ratings": "ratings.parquet",
            "age_curve": "age_curve.parquet",
        },
        "forbidden_interpretation": (
            "Season 2026 is a reused diagnostic. This is not an untouched confirmation, "
            "a same-age leaderboard, or a forecast independent of observed lineups."
        ),
    }
    write_json_atomic(manifest, run_path)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
