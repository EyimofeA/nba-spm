"""Reproduce JE's exact score-state indicators inside the current RAPM design."""

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
    load_unified_terminal_possessions,
    ratings_table,
)
from nba_impact.models.rubberband_score_state import (
    annotate_offense_margin_before,
    fit_score_state_rapm,
    predict_score_state_rapm,
    score_state_curve,
)
from research.rapm_lab.run_rubberband_progress_rapm import _paired_rmse_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rubberband_je_replication_v1.json"
)
DEFAULT_CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = (
    REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
)
DEFAULT_PLAYER_GAMES = REPO_ROOT / "data" / "lake" / "silver" / "player_games.parquet"
DEFAULT_NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "rubberband_je"


def _config(contract: dict) -> RapmConfig:
    penalties = contract["player_penalties"]
    return RapmConfig(
        seasons=tuple(int(value) for value in contract["seasons"]),
        lambda_off=float(penalties["lambda_off"]),
        lambda_def=float(penalties["lambda_def"]),
        lambda_home=float(penalties["lambda_home"]),
        data_scope="je_score_state_rapm_research",
    )


def _row_set_hash(frame: pd.DataFrame) -> str:
    keys = frame[["gameid", "period", "num"]].astype(str)
    values = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_experiment(
    contract_path: Path = DEFAULT_CONTRACT,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("JE replication requires a frozen research contract.")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "rubberband_score_state.py"
        ),
        "legacy_possessions": {
            str(season): sha256_file(DEFAULT_CACHE / f"matchups_{season}.parquet")
            for season in contract["seasons"]
            if int(season) < 2024
        },
        "possessions": sha256_file(DEFAULT_POSSESSIONS),
        "segments": sha256_file(DEFAULT_SEGMENTS),
        "player_games": sha256_file(DEFAULT_PLAYER_GAMES),
        "names": sha256_file(DEFAULT_NAMES),
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
    source = load_unified_terminal_possessions(
        DEFAULT_CACHE,
        DEFAULT_POSSESSIONS,
        DEFAULT_SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    source = annotate_offense_margin_before(source)
    design = build_design(source, include_home=True)
    margins = source["offense_margin_before"].to_numpy(dtype=float)
    train_mask = source["season"].isin(contract["fit_seasons"]).to_numpy()
    test_mask = source["season"].eq(contract["reused_diagnostic_season"]).to_numpy()
    state = contract["score_state"]
    config = _config(contract)

    normal_beta, normal_intercept = fit_coefficients(
        design, config, row_mask=train_mask
    )
    score_fit = fit_score_state_rapm(
        design,
        margins,
        config,
        minimum=int(state["minimum"]),
        maximum=int(state["maximum"]),
        state_penalty=float(state["penalty"]),
        row_mask=train_mask,
    )
    normal_prediction = (
        np.asarray(design.X[test_mask] @ normal_beta).ravel() + normal_intercept
    )
    neutral_prediction = predict_score_state_rapm(
        score_fit,
        design,
        margins,
        row_mask=test_mask,
        include_score_state=False,
    )
    conditional_prediction = predict_score_state_rapm(
        score_fit,
        design,
        margins,
        row_mask=test_mask,
        include_score_state=True,
    )
    test_frame = source.loc[test_mask].reset_index(drop=True)
    predictions = {
        "normal_neutral": normal_prediction,
        "je_neutral": neutral_prediction,
        "je_conditional": conditional_prediction,
    }
    evaluation_rows: list[dict] = []
    evaluation_games: dict[str, pd.DataFrame] = {}
    for name, prediction in predictions.items():
        metrics, games = game_margin_metrics(test_frame, prediction)
        evaluation_rows.append(
            {
                "variant": "normal" if name == "normal_neutral" else "je_categorical",
                "prediction_mode": (
                    "conditional_score_path"
                    if name == "je_conditional"
                    else "neutral_player_only"
                ),
                "possession_rmse": float(
                    math.sqrt(np.mean((design.y[test_mask] - prediction) ** 2))
                ),
                **metrics,
            }
        )
        evaluation_games[name] = games
        games.assign(variant=name).to_parquet(
            output / f"game_predictions_{name}.parquet", index=False
        )
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_parquet(output / "evaluation.parquet", index=False)

    bootstrap = {
        "neutral_player_only": _paired_rmse_bootstrap(
            evaluation_games["normal_neutral"],
            evaluation_games["je_neutral"],
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]),
        ),
        "conditional_score_path": _paired_rmse_bootstrap(
            evaluation_games["normal_neutral"],
            evaluation_games["je_conditional"],
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]) + 1,
        ),
    }

    curve = score_state_curve(score_fit)
    curve = curve.loc[
        curve["margin"].between(
            int(state["display_minimum"]), int(state["display_maximum"])
        )
    ].reset_index(drop=True)
    curve.to_parquet(output / "score_state_curve.parquet", index=False)

    full_normal_beta, _ = fit_coefficients(design, config)
    full_score_fit = fit_score_state_rapm(
        design,
        margins,
        config,
        minimum=int(state["minimum"]),
        maximum=int(state["maximum"]),
        state_penalty=float(state["penalty"]),
    )
    names = load_current_player_names(DEFAULT_NAMES, DEFAULT_PLAYER_GAMES)
    normal_ratings = ratings_table(design, full_normal_beta, names=names).rename(
        columns={
            "offense_per_100": "normal_offense",
            "defense_per_100": "normal_defense",
            "net_per_100": "normal_net",
        }
    )
    je_ratings = ratings_table(
        design, full_score_fit.player_coefficients, names=names
    ).rename(
        columns={
            "offense_per_100": "je_offense",
            "defense_per_100": "je_defense",
            "net_per_100": "je_net",
        }
    )
    ratings = normal_ratings.drop(columns="uncertainty_status").merge(
        je_ratings[
            ["player_id", "je_offense", "je_defense", "je_net"]
        ],
        on="player_id",
        validate="one_to_one",
    )
    ratings["je_net_change"] = ratings["je_net"] - ratings["normal_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)

    curve_lookup = dict(
        zip(curve["margin"], curve["effect_points_per_100_vs_tie"])
    )
    qualified = ratings.loc[
        ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(5000)
    ]
    run = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "model_family": "joint_categorical_score_state_rapm",
        "estimand_id": contract["estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload},
        "quality": {
            "row_set_hash": _row_set_hash(source),
            "possessions": int(len(source)),
            "train_possessions": int(train_mask.sum()),
            "diagnostic_possessions": int(test_mask.sum()),
            "games": int(source["gameid"].nunique()),
            "train_state_counts_sum_to_rows": bool(
                score_fit.state_counts.sum() == train_mask.sum()
            ),
            "tie_effect_points_per_100": float(curve_lookup[0]),
            "qualified_rating_players": int(len(qualified)),
            "maximum_je_rating_identity_error": float(
                np.abs(
                    ratings["je_offense"] + ratings["je_defense"] - ratings["je_net"]
                ).max()
            ),
        },
        "score_state_effects_points_per_100_vs_tie": {
            str(margin): float(curve_lookup[margin])
            for margin in (-20, -10, 10, 20)
        },
        "evaluation": evaluation.to_dict("records"),
        "bootstrap_vs_normal": bootstrap,
        "rating_effect": {
            "net_correlation_with_normal": float(
                ratings[["normal_net", "je_net"]].corr().iloc[0, 1]
            ),
            "mean_absolute_net_change": float(ratings["je_net_change"].abs().mean()),
        },
        "artifacts": {
            "score_state_curve": "score_state_curve.parquet",
            "evaluation": "evaluation.parquet",
            "ratings": "ratings.parquet",
        },
        "decision_rule": (
            "The JE curve is a descriptive score-state control. Promote adjusted player "
            "ratings only if neutral held-out game-margin RMSE improves with a paired "
            "95% interval below zero."
        ),
        "forbidden_interpretation": (
            "The curve is not a causal effort response. Score margin is endogenous, and "
            "the conditional path uses observed future game state."
        ),
    }
    write_json_atomic(run, run_path)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.contract, args.output), indent=2))


if __name__ == "__main__":
    main()
