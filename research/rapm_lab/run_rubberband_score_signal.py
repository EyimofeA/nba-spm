"""Compare categorical, linear, and spline score-state controls in RAPM."""

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
from nba_impact.models.rubberband_score_state import annotate_offense_margin_before
from nba_impact.models.score_context_rapm import (
    clipped_linear_score_design,
    fit_context_rapm,
    predict_context_rapm,
    signed_score_bucket_design,
    spline_score_design,
)
from research.rapm_lab.run_rubberband_progress_rapm import _paired_rmse_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rubberband_score_signal_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rubberband_score_signal"
CACHE = ROOT / "rapm/data/possession_cache"
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
        data_scope="joint_score_signal_rapm_research",
    )


def _shape_specs(contract: dict) -> list[dict]:
    shapes = contract["candidate_shapes"]
    result = [{"family": "signed_buckets", "shape": "signed_buckets"}]
    result.extend(
        {
            "family": "clipped_linear",
            "shape": f"linear_clip_{clip:g}",
            "clip": float(clip),
        }
        for clip in shapes["clipped_linear"]["clips"]
    )
    result.extend(
        {
            "family": "cubic_spline",
            "shape": f"spline_knots_{knots}",
            "clip": float(shapes["cubic_spline"]["clip"]),
            "degree": int(shapes["cubic_spline"]["degree"]),
            "n_knots": int(knots),
        }
        for knots in shapes["cubic_spline"]["n_knots"]
    )
    return result


def _context(margins: np.ndarray, train_mask: np.ndarray, spec: dict):
    if spec["family"] == "signed_buckets":
        return signed_score_bucket_design(margins)
    if spec["family"] == "clipped_linear":
        return clipped_linear_score_design(margins, clip=spec["clip"])
    return spline_score_design(
        margins,
        train_mask=train_mask,
        clip=spec["clip"],
        n_knots=spec["n_knots"],
        degree=spec["degree"],
    )


def _metrics(frame: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray):
    metrics, games = game_margin_metrics(frame, prediction)
    return (
        {
            "possession_rmse": float(math.sqrt(np.mean((actual - prediction) ** 2))),
            **metrics,
        },
        games,
    )


def _baseline(design, config, train_mask, test_mask):
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    return np.asarray(design.X[test_mask] @ beta).ravel() + intercept


def _clean_winner(row: pd.Series) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in row.items()
        if not pd.isna(value)
    }


def run_experiment(
    contract_path: Path = CONTRACT,
    output_root: Path = OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Score-signal experiment requires a frozen contract")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched")
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "src/nba_impact/models/score_context_rapm.py"),
        "legacy_possessions": {
            str(season): sha256_file(CACHE / f"matchups_{season}.parquet")
            for season in contract["seasons"]
            if int(season) < 2024
        },
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
    source = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    source = annotate_offense_margin_before(source)
    if source["season"].eq(int(contract["untouched_confirmation_season"])).any():
        raise ValueError("Season 2027 rows entered the score-signal experiment")
    margins = source["offense_margin_before"].to_numpy(dtype=float)
    design = build_design(source, include_home=True)
    config = _config(contract)

    selection_train = source["season"].isin(
        contract["selection_fit_seasons"]
    ).to_numpy()
    selection_test = source["season"].eq(contract["selection_season"]).to_numpy()
    selection_frame = source.loc[selection_test].reset_index(drop=True)
    baseline_prediction = _baseline(
        design, config, selection_train, selection_test
    )
    selection_baseline, _ = _metrics(
        selection_frame, design.y[selection_test], baseline_prediction
    )
    selection_rows: list[dict] = []
    for spec in _shape_specs(contract):
        context, labels = _context(margins, selection_train, spec)
        for penalty in contract["context_penalties"]:
            fit = fit_context_rapm(
                design,
                context,
                config,
                context_penalty=float(penalty),
                row_mask=selection_train,
            )
            neutral = predict_context_rapm(
                fit,
                design,
                context,
                row_mask=selection_test,
                include_context=False,
            )
            conditional = predict_context_rapm(
                fit,
                design,
                context,
                row_mask=selection_test,
                include_context=True,
            )
            neutral_metrics, _ = _metrics(
                selection_frame, design.y[selection_test], neutral
            )
            conditional_metrics, _ = _metrics(
                selection_frame, design.y[selection_test], conditional
            )
            selection_rows.append(
                {
                    **spec,
                    "context_columns": len(labels),
                    "context_penalty": float(penalty),
                    **{
                        f"neutral_{key}": value
                        for key, value in neutral_metrics.items()
                    },
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
    winner = _clean_winner(selection.iloc[0])
    selection.to_parquet(output / "selection.parquet", index=False)

    diagnostic_train = source["season"].isin(
        contract["diagnostic_fit_seasons"]
    ).to_numpy()
    diagnostic_test = source["season"].eq(contract["diagnostic_season"]).to_numpy()
    diagnostic_frame = source.loc[diagnostic_test].reset_index(drop=True)
    baseline_prediction = _baseline(
        design, config, diagnostic_train, diagnostic_test
    )
    winner_spec = next(
        spec for spec in _shape_specs(contract) if spec["shape"] == winner["shape"]
    )
    context, labels = _context(margins, diagnostic_train, winner_spec)
    fit = fit_context_rapm(
        design,
        context,
        config,
        context_penalty=float(winner["context_penalty"]),
        row_mask=diagnostic_train,
    )
    neutral = predict_context_rapm(
        fit,
        design,
        context,
        row_mask=diagnostic_test,
        include_context=False,
    )
    conditional = predict_context_rapm(
        fit,
        design,
        context,
        row_mask=diagnostic_test,
        include_context=True,
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
            {"variant": winner["shape"], "prediction_mode": "neutral_player_only", **neutral_metrics},
            {"variant": winner["shape"], "prediction_mode": "conditional_score_path", **conditional_metrics},
        ]
    )
    evaluation.to_parquet(output / "evaluation.parquet", index=False)

    full_mask = np.ones(len(source), dtype=bool)
    full_context, full_labels = _context(margins, full_mask, winner_spec)
    full_fit = fit_context_rapm(
        design,
        full_context,
        config,
        context_penalty=float(winner["context_penalty"]),
        row_mask=full_mask,
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
            "offense_per_100": "candidate_offense",
            "defense_per_100": "candidate_defense",
            "net_per_100": "candidate_net",
        }
    )
    ratings = normal.drop(columns="uncertainty_status").merge(
        candidate[
            ["player_id", "candidate_offense", "candidate_defense", "candidate_net"]
        ],
        on="player_id",
        validate="one_to_one",
    )
    ratings["candidate_net_change"] = ratings["candidate_net"] - ratings["normal_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)

    context_effect = np.asarray(
        full_context @ full_fit.context_coefficients
    ).ravel()
    curve = (
        pd.DataFrame(
            {
                "margin": margins,
                "effect_points_per_100": 100.0 * context_effect,
            }
        )
        .loc[lambda frame: frame["margin"].between(-30, 30)]
        .groupby("margin", as_index=False)["effect_points_per_100"]
        .mean()
    )
    tie = float(curve.loc[curve["margin"].eq(0), "effect_points_per_100"].iloc[0])
    curve["effect_points_per_100_vs_tie"] = curve["effect_points_per_100"] - tie
    curve.to_parquet(output / "curve.parquet", index=False)
    pd.DataFrame(
        {"context_column": full_labels, "coefficient": full_fit.context_coefficients}
    ).to_parquet(output / "context_coefficients.parquet", index=False)

    rmse_delta = float(neutral_metrics["margin_rmse"] - baseline_metrics["margin_rmse"])
    correlation_delta = float(
        neutral_metrics["margin_correlation"] - baseline_metrics["margin_correlation"]
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
        "model_family": "joint_score_signal_rapm",
        "estimand_id": contract["estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload},
        "selection_baseline": selection_baseline,
        "selection_winner": winner,
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
                ratings[["normal_net", "candidate_net"]].corr().iloc[0, 1]
            ),
            "mean_absolute_net_change": float(
                ratings["candidate_net_change"].abs().mean()
            ),
        },
        "quality": {
            "possessions": int(len(source)),
            "games": int(source["gameid"].nunique()),
            "selection_candidates": int(len(selection)),
            "season_2027_loaded": bool(
                source["season"]
                .eq(int(contract["untouched_confirmation_season"]))
                .any()
            ),
            "maximum_rating_identity_error": float(
                np.abs(
                    ratings["candidate_offense"]
                    + ratings["candidate_defense"]
                    - ratings["candidate_net"]
                ).max()
            ),
        },
        "decision": "promote_for_confirmation" if passes else "reject_and_keep_normal_rapm",
        "artifacts": {
            "selection": "selection.parquet",
            "evaluation": "evaluation.parquet",
            "ratings": "ratings.parquet",
            "curve": "curve.parquet",
            "context_coefficients": "context_coefficients.parquet",
        },
        "forbidden_interpretation": (
            "Score margin is endogenous. Conditional score-path evaluation is not a "
            "pregame forecast, and reused 2026 evidence cannot promote a public rating."
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
