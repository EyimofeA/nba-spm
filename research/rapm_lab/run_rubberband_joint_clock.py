"""Fit actual-clock rubber-band columns jointly inside possession RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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
    ratings_table,
)
from nba_impact.models.rubberband_score_state import (
    clock_margin_curve,
    fit_clock_margin_rapm,
    predict_clock_margin_rapm,
)
from research.rapm_lab.run_rubberband_progress_rapm import (
    _paired_rmse_bootstrap,
    load_clocked_terminal_possessions,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rubberband_joint_clock_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rubberband_joint_clock"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
NAMES = ROOT / "rapm/data/all_names.csv"


def _config(contract: dict) -> RapmConfig:
    penalty = contract["player_penalties"]
    return RapmConfig(
        seasons=tuple(int(value) for value in contract["seasons"]),
        lambda_off=float(penalty["lambda_off"]),
        lambda_def=float(penalty["lambda_def"]),
        lambda_home=float(penalty["lambda_home"]),
        data_scope="joint_actual_clock_score_context_research",
    )


def _metrics(frame: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray):
    metrics, games = game_margin_metrics(frame, prediction)
    return (
        {
            "possessions": int(len(frame)),
            "possession_rmse": float(math.sqrt(np.mean((actual - prediction) ** 2))),
            **metrics,
        },
        games,
    )


def _baseline(design, config, train_mask, test_mask):
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    return np.asarray(design.X[test_mask] @ beta).ravel() + intercept


def _candidate(
    source,
    design,
    config,
    spec,
    context_penalty,
    train_mask,
    test_mask,
):
    fit = fit_clock_margin_rapm(
        design,
        source,
        config,
        time_buckets=int(spec["time_buckets"]),
        margin_clip=float(spec["margin_clip"]),
        context_penalty=float(context_penalty),
        row_mask=train_mask,
    )
    neutral = predict_clock_margin_rapm(
        fit, design, source, row_mask=test_mask, include_context=False
    )
    conditional = predict_clock_margin_rapm(
        fit, design, source, row_mask=test_mask, include_context=True
    )
    return fit, neutral, conditional


def run_experiment(
    contract_path: Path = CONTRACT,
    output_root: Path = OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Joint clock experiment requires a frozen contract")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched")
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(
            ROOT / "src/nba_impact/models/rubberband_score_state.py"
        ),
        "possessions": sha256_file(POSSESSIONS),
        "segments": sha256_file(SEGMENTS),
        "player_games": sha256_file(PLAYER_GAMES),
        "names": sha256_file(NAMES),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = output_root / f"{contract['experiment_id']}_{identity}"
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    seasons = tuple(int(value) for value in contract["seasons"])
    source = load_clocked_terminal_possessions(POSSESSIONS, SEGMENTS, seasons)
    design = build_design(source, include_home=True)
    config = _config(contract)
    spec = contract["clock_context"]

    selection_train = source["season"].isin(
        contract["selection_fit_seasons"]
    ).to_numpy()
    selection_test = source["season"].eq(contract["selection_season"]).to_numpy()
    selection_frame = source.loc[selection_test].reset_index(drop=True)
    baseline_prediction = _baseline(
        design, config, selection_train, selection_test
    )
    selection_baseline_metrics, _ = _metrics(
        selection_frame, design.y[selection_test], baseline_prediction
    )
    selection_rows: list[dict] = []
    for penalty in spec["context_penalties"]:
        _, neutral, conditional = _candidate(
            source,
            design,
            config,
            spec,
            penalty,
            selection_train,
            selection_test,
        )
        neutral_metrics, _ = _metrics(
            selection_frame, design.y[selection_test], neutral
        )
        conditional_metrics, _ = _metrics(
            selection_frame, design.y[selection_test], conditional
        )
        selection_rows.append(
            {
                "context_penalty": float(penalty),
                **{f"neutral_{key}": value for key, value in neutral_metrics.items()},
                **{
                    f"conditional_{key}": value
                    for key, value in conditional_metrics.items()
                },
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["neutral_margin_rmse", "neutral_margin_correlation"],
        ascending=[True, False],
        kind="stable",
    )
    selected_penalty = float(selection.iloc[0]["context_penalty"])
    selection.to_parquet(output / "selection.parquet", index=False)

    diagnostic_train = source["season"].isin(
        contract["diagnostic_fit_seasons"]
    ).to_numpy()
    diagnostic_test = source["season"].eq(contract["diagnostic_season"]).to_numpy()
    diagnostic_frame = source.loc[diagnostic_test].reset_index(drop=True)
    baseline_prediction = _baseline(
        design, config, diagnostic_train, diagnostic_test
    )
    diagnostic_fit, neutral, conditional = _candidate(
        source,
        design,
        config,
        spec,
        selected_penalty,
        diagnostic_train,
        diagnostic_test,
    )
    baseline_metrics, baseline_games = _metrics(
        diagnostic_frame, design.y[diagnostic_test], baseline_prediction
    )
    neutral_metrics, neutral_games = _metrics(
        diagnostic_frame, design.y[diagnostic_test], neutral
    )
    conditional_metrics, conditional_games = _metrics(
        diagnostic_frame, design.y[diagnostic_test], conditional
    )
    bootstrap = {
        "neutral_player_only": _paired_rmse_bootstrap(
            baseline_games,
            neutral_games,
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]),
        ),
        "conditional_score_path": _paired_rmse_bootstrap(
            baseline_games,
            conditional_games,
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]) + 1,
        ),
    }
    evaluation = pd.DataFrame(
        [
            {"variant": "normal", "prediction_mode": "neutral_player_only", **baseline_metrics},
            {"variant": "joint_clock_margin", "prediction_mode": "neutral_player_only", **neutral_metrics},
            {"variant": "joint_clock_margin", "prediction_mode": "conditional_score_path", **conditional_metrics},
        ]
    )
    evaluation.to_parquet(output / "evaluation.parquet", index=False)

    full_fit = fit_clock_margin_rapm(
        design,
        source,
        config,
        time_buckets=int(spec["time_buckets"]),
        margin_clip=float(spec["margin_clip"]),
        context_penalty=selected_penalty,
    )
    normal_beta, _ = fit_coefficients(design, config)
    names = load_current_player_names(NAMES, PLAYER_GAMES)
    normal = ratings_table(design, normal_beta, names=names).rename(
        columns={
            "offense_per_100": "normal_offense",
            "defense_per_100": "normal_defense",
            "net_per_100": "normal_net",
        }
    )
    candidate = ratings_table(
        design, full_fit.player_coefficients, names=names
    ).rename(
        columns={
            "offense_per_100": "joint_offense",
            "defense_per_100": "joint_defense",
            "net_per_100": "joint_net",
        }
    )
    ratings = normal.drop(columns="uncertainty_status").merge(
        candidate[["player_id", "joint_offense", "joint_defense", "joint_net"]],
        on="player_id",
        validate="one_to_one",
    )
    ratings["joint_net_change"] = ratings["joint_net"] - ratings["normal_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)
    curve = clock_margin_curve(full_fit)
    curve.to_parquet(output / "curve.parquet", index=False)

    rmse_delta = float(neutral_metrics["margin_rmse"] - baseline_metrics["margin_rmse"])
    correlation_delta = float(
        neutral_metrics["margin_correlation"]
        - baseline_metrics["margin_correlation"]
    )
    gate = contract["promotion_gate"]
    passes = bool(
        rmse_delta < float(gate["maximum_rmse_delta"])
        and correlation_delta >= -float(gate["maximum_correlation_decline"])
        and bootstrap["neutral_player_only"]["upper_95"]
        < float(gate["paired_rmse_interval_upper"])
    )
    qualified = ratings.loc[
        ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(3000)
    ]
    run = {
        "run_id": output.name,
        "status": "research_challenger" if passes else "research_null",
        "model_family": "joint_actual_clock_margin_rapm",
        "estimand_id": contract["estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload},
        "selected_context_penalty": selected_penalty,
        "selection_baseline": selection_baseline_metrics,
        "selection": selection.to_dict("records"),
        "diagnostic": {
            "baseline": baseline_metrics,
            "neutral_player_only": neutral_metrics,
            "conditional_score_path": conditional_metrics,
            "rmse_delta_candidate_minus_baseline": rmse_delta,
            "correlation_delta_candidate_minus_baseline": correlation_delta,
            "paired_game_bootstrap": bootstrap,
            "passes_promotion_gate": passes,
        },
        "rating_effect": {
            "qualified_players": int(len(qualified)),
            "net_correlation_with_normal": float(
                ratings[["normal_net", "joint_net"]].corr().iloc[0, 1]
            ),
            "mean_absolute_net_change": float(
                ratings["joint_net_change"].abs().mean()
            ),
        },
        "quality": {
            "possessions": int(len(source)),
            "games": int(source["gameid"].nunique()),
            "season_2027_loaded": False,
            "maximum_rating_identity_error": float(
                np.abs(
                    ratings["joint_offense"]
                    + ratings["joint_defense"]
                    - ratings["joint_net"]
                ).max()
            ),
        },
        "decision": "promote_for_confirmation" if passes else "reject_and_keep_normal_rapm",
        "artifacts": {
            "selection": "selection.parquet",
            "evaluation": "evaluation.parquet",
            "ratings": "ratings.parquet",
            "curve": "curve.parquet",
        },
        "forbidden_interpretation": (
            "Score margin is endogenous. The conditional score path is not a pregame "
            "forecast, and this reused diagnostic cannot promote a public rating."
        ),
    }
    write_json_atomic(run, run_path)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.contract, args.output), indent=2))


if __name__ == "__main__":
    main()
