"""Compare no, age, score, and joint age-score RAPM controls."""

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
from scipy.sparse import hstack

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import build_age_design
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
from nba_impact.models.rubberband_score_state import annotate_offense_margin_before
from nba_impact.models.score_context_rapm import (
    clipped_linear_score_design,
    fit_context_rapm,
    predict_context_rapm,
    signed_score_bucket_design,
    spline_age_design,
    spline_score_design,
)
from research.rapm_lab.run_rubberband_progress_rapm import _paired_rmse_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rubberband_score_signal_v1.json"
SCORE_OUTPUT = ROOT / "research/rapm_lab/outputs/rubberband_score_signal"
OUTPUT = ROOT / "research/rapm_lab/outputs/age_score_context"
CACHE = ROOT / "rapm/data/possession_cache"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
NAMES = ROOT / "rapm/data/all_names.csv"
AGE_DIR = ROOT / "data/raw/playersheets/year_totals"


def _latest_score_run(expected_contract_hash: str) -> tuple[dict, Path]:
    paths = sorted(
        SCORE_OUTPUT.glob("rubberband_score_signal_v1_*/run.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        run = json.loads(path.read_text())
        if run.get("config", {}).get("contract") == expected_contract_hash:
            return run, path.parent
    raise FileNotFoundError(
        "Run the score-signal comparison for the current frozen contract first"
    )


def _config(contract: dict) -> RapmConfig:
    penalty = contract["player_penalties"]
    return RapmConfig(
        seasons=tuple(int(value) for value in contract["seasons"]),
        lambda_off=float(penalty["lambda_off"]),
        lambda_def=float(penalty["lambda_def"]),
        lambda_home=float(penalty["lambda_home"]),
        data_scope="joint_age_score_context_rapm_research",
    )


def _score_context(margins: np.ndarray, train_mask: np.ndarray, winner: dict):
    family = winner["family"]
    if family == "signed_buckets":
        return signed_score_bucket_design(margins)
    if family == "clipped_linear":
        return clipped_linear_score_design(margins, clip=float(winner["clip"]))
    return spline_score_design(
        margins,
        train_mask=train_mask,
        clip=float(winner["clip"]),
        n_knots=int(winner["n_knots"]),
        degree=int(winner["degree"]),
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


def _plain_prediction(design, config, train_mask, test_mask):
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    return np.asarray(design.X[test_mask] @ beta).ravel() + intercept


def run_experiment(
    contract_path: Path = CONTRACT,
    output_root: Path = OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    contract_hash = sha256_file(contract_path)
    score_run, _ = _latest_score_run(contract_hash)
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Age-score comparison requires a frozen contract")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched")
    identity_payload = {
        "contract": contract_hash,
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "src/nba_impact/models/score_context_rapm.py"),
        "score_run": score_run["run_id"],
        "legacy_possessions": {
            str(season): sha256_file(CACHE / f"matchups_{season}.parquet")
            for season in contract["seasons"]
            if int(season) < 2024
        },
        "possessions": sha256_file(POSSESSIONS),
        "segments": sha256_file(SEGMENTS),
        "player_games": sha256_file(PLAYER_GAMES),
        "names": sha256_file(NAMES),
        "ages": {
            str(season): sha256_file(AGE_DIR / f"{season}.csv")
            for season in contract["seasons"]
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = output_root / f"age_score_context_v1_{identity}"
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
        raise ValueError("Season 2027 rows entered the age-score experiment")
    margins = source["offense_margin_before"].to_numpy(dtype=float)
    design = build_design(source, include_home=True)
    config = _config(contract)
    age_spec = contract["age_spline"]
    ages = load_age_panel(AGE_DIR, seasons)
    categorical_age = build_age_design(
        source,
        ages,
        minimum_age=int(age_spec["minimum_age"]),
        maximum_age=int(age_spec["maximum_age"]),
        reference_age=int(age_spec["reference_age"]),
    )
    score_winner = score_run["selection_winner"]

    selection_train = source["season"].isin(
        contract["selection_fit_seasons"]
    ).to_numpy()
    selection_test = source["season"].eq(contract["selection_season"]).to_numpy()
    selection_frame = source.loc[selection_test].reset_index(drop=True)
    normal_prediction = _plain_prediction(
        design, config, selection_train, selection_test
    )
    normal_selection, _ = _metrics(
        selection_frame, design.y[selection_test], normal_prediction
    )
    score_context, score_labels = _score_context(
        margins, selection_train, score_winner
    )
    score_fit = fit_context_rapm(
        design,
        score_context,
        config,
        context_penalty=float(score_winner["context_penalty"]),
        row_mask=selection_train,
    )
    score_prediction = predict_context_rapm(
        score_fit,
        design,
        score_context,
        row_mask=selection_test,
        include_context=False,
    )
    score_selection, _ = _metrics(
        selection_frame, design.y[selection_test], score_prediction
    )

    age_rows: list[dict] = []
    age_contexts: dict[int, tuple] = {}
    for knots in age_spec["n_knots"]:
        age_contexts[int(knots)] = spline_age_design(
            categorical_age,
            n_knots=int(knots),
            degree=int(age_spec["degree"]),
        )
        age_context, age_labels = age_contexts[int(knots)]
        for penalty in age_spec["penalties"]:
            fit = fit_context_rapm(
                design,
                age_context,
                config,
                context_penalty=float(penalty),
                row_mask=selection_train,
            )
            prediction = predict_context_rapm(
                fit,
                design,
                age_context,
                row_mask=selection_test,
                include_context=True,
            )
            metrics, _ = _metrics(
                selection_frame, design.y[selection_test], prediction
            )
            age_rows.append(
                {
                    "age_n_knots": int(knots),
                    "age_columns": len(age_labels),
                    "age_penalty": float(penalty),
                    **metrics,
                }
            )
    age_selection = pd.DataFrame(age_rows).sort_values(
        ["margin_rmse", "margin_correlation"],
        ascending=[True, False],
        kind="stable",
    )
    age_winner = age_selection.iloc[0].to_dict()
    age_context, age_labels = age_contexts[int(age_winner["age_n_knots"])]

    combined_context = hstack([age_context, score_context], format="csr")
    age_count = age_context.shape[1]
    score_count = score_context.shape[1]
    include_age_only = np.r_[
        np.ones(age_count, dtype=bool), np.zeros(score_count, dtype=bool)
    ]
    combined_rows: list[dict] = []
    for age_penalty in age_spec["penalties"]:
        for score_penalty in contract["context_penalties"]:
            fit = fit_context_rapm(
                design,
                combined_context,
                config,
                context_penalty=np.r_[
                    np.full(age_count, float(age_penalty)),
                    np.full(score_count, float(score_penalty)),
                ],
                row_mask=selection_train,
            )
            prediction = predict_context_rapm(
                fit,
                design,
                combined_context,
                row_mask=selection_test,
                include_context=True,
                include_columns=include_age_only,
            )
            metrics, _ = _metrics(
                selection_frame, design.y[selection_test], prediction
            )
            combined_rows.append(
                {
                    "age_n_knots": int(age_winner["age_n_knots"]),
                    "age_penalty": float(age_penalty),
                    "score_shape": score_winner["shape"],
                    "score_penalty": float(score_penalty),
                    **metrics,
                }
            )
    combined_selection = pd.DataFrame(combined_rows).sort_values(
        ["margin_rmse", "margin_correlation"],
        ascending=[True, False],
        kind="stable",
    )
    combined_winner = combined_selection.iloc[0].to_dict()
    age_selection.to_parquet(output / "age_selection.parquet", index=False)
    combined_selection.to_parquet(
        output / "combined_selection.parquet", index=False
    )

    diagnostic_train = source["season"].isin(
        contract["diagnostic_fit_seasons"]
    ).to_numpy()
    diagnostic_test = source["season"].eq(contract["diagnostic_season"]).to_numpy()
    diagnostic_frame = source.loc[diagnostic_test].reset_index(drop=True)
    normal_prediction = _plain_prediction(
        design, config, diagnostic_train, diagnostic_test
    )
    score_context, score_labels = _score_context(
        margins, diagnostic_train, score_winner
    )
    score_fit = fit_context_rapm(
        design,
        score_context,
        config,
        context_penalty=float(score_winner["context_penalty"]),
        row_mask=diagnostic_train,
    )
    score_prediction = predict_context_rapm(
        score_fit,
        design,
        score_context,
        row_mask=diagnostic_test,
        include_context=False,
    )
    age_context, age_labels = spline_age_design(
        categorical_age,
        n_knots=int(age_winner["age_n_knots"]),
        degree=int(age_spec["degree"]),
    )
    age_fit = fit_context_rapm(
        design,
        age_context,
        config,
        context_penalty=float(age_winner["age_penalty"]),
        row_mask=diagnostic_train,
    )
    age_prediction = predict_context_rapm(
        age_fit,
        design,
        age_context,
        row_mask=diagnostic_test,
        include_context=True,
    )
    combined_context = hstack([age_context, score_context], format="csr")
    combined_fit = fit_context_rapm(
        design,
        combined_context,
        config,
        context_penalty=np.r_[
            np.full(age_context.shape[1], float(combined_winner["age_penalty"])),
            np.full(score_context.shape[1], float(combined_winner["score_penalty"])),
        ],
        row_mask=diagnostic_train,
    )
    include_age_only = np.r_[
        np.ones(age_context.shape[1], dtype=bool),
        np.zeros(score_context.shape[1], dtype=bool),
    ]
    combined_prediction = predict_context_rapm(
        combined_fit,
        design,
        combined_context,
        row_mask=diagnostic_test,
        include_context=True,
        include_columns=include_age_only,
    )
    age_neutral_prediction = predict_context_rapm(
        age_fit,
        design,
        age_context,
        row_mask=diagnostic_test,
        include_context=False,
    )
    combined_neutral_prediction = predict_context_rapm(
        combined_fit,
        design,
        combined_context,
        row_mask=diagnostic_test,
        include_context=False,
    )
    prediction_modes = {
        "player_only": {
            "normal": normal_prediction,
            "age_only": age_neutral_prediction,
            "score_only": score_prediction,
            "age_plus_score": combined_neutral_prediction,
        },
        "pregame_context": {
            "normal": normal_prediction,
            "age_only": age_prediction,
            "score_only": score_prediction,
            "age_plus_score": combined_prediction,
        },
    }
    metrics: dict[str, dict[str, dict]] = {}
    game_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    evaluation_rows: list[dict] = []
    for mode, predictions in prediction_modes.items():
        metrics[mode] = {}
        game_predictions[mode] = {}
        for variant, prediction in predictions.items():
            variant_metrics, variant_games = _metrics(
                diagnostic_frame, design.y[diagnostic_test], prediction
            )
            metrics[mode][variant] = variant_metrics
            game_predictions[mode][variant] = variant_games
            evaluation_rows.append(
                {"prediction_mode": mode, "variant": variant, **variant_metrics}
            )
    evaluation = pd.DataFrame(
        evaluation_rows
    )
    evaluation.to_parquet(output / "evaluation.parquet", index=False)
    bootstrap: dict[str, dict[str, dict]] = {}
    for mode_offset, mode in enumerate(("player_only", "pregame_context")):
        mode_games = game_predictions[mode]
        bootstrap[mode] = {
            variant: _paired_rmse_bootstrap(
                mode_games["normal"],
                mode_games[variant],
                draws=int(contract["paired_game_bootstrap_draws"]),
                seed=(
                    int(contract["paired_game_bootstrap_seed"])
                    + 10
                    + 10 * mode_offset
                    + variant_offset
                ),
            )
            for variant_offset, variant in enumerate(
                ("age_only", "score_only", "age_plus_score")
            )
        }
        bootstrap[mode]["age_plus_score_vs_age_only"] = _paired_rmse_bootstrap(
            mode_games["age_only"],
            mode_games["age_plus_score"],
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=(
                int(contract["paired_game_bootstrap_seed"])
                + 19
                + 10 * mode_offset
            ),
        )
    gate = contract["promotion_gate"]
    gate_passes = {
        mode: {
            variant: bool(
                metrics[mode][variant]["margin_rmse"]
                - metrics[mode]["normal"]["margin_rmse"]
                < float(gate["maximum_rmse_delta"])
                and metrics[mode][variant]["margin_correlation"]
                - metrics[mode]["normal"]["margin_correlation"]
                >= -float(gate["maximum_correlation_decline"])
                and bootstrap[mode][variant]["upper_95"]
                < float(gate["paired_rmse_interval_upper"])
            )
            for variant in ("age_only", "score_only", "age_plus_score")
        }
        for mode in ("player_only", "pregame_context")
    }
    any_gate_pass = any(
        passed for mode in gate_passes.values() for passed in mode.values()
    )

    full_mask = np.ones(len(source), dtype=bool)
    full_score, _ = _score_context(margins, full_mask, score_winner)
    full_age, _ = spline_age_design(
        categorical_age,
        n_knots=int(age_winner["age_n_knots"]),
        degree=int(age_spec["degree"]),
    )
    full_combined = hstack([full_age, full_score], format="csr")
    normal_beta, _ = fit_coefficients(design, config)
    full_age_fit = fit_context_rapm(
        design,
        full_age,
        config,
        context_penalty=float(age_winner["age_penalty"]),
        row_mask=full_mask,
    )
    full_score_fit = fit_context_rapm(
        design,
        full_score,
        config,
        context_penalty=float(score_winner["context_penalty"]),
        row_mask=full_mask,
    )
    full_combined_fit = fit_context_rapm(
        design,
        full_combined,
        config,
        context_penalty=np.r_[
            np.full(full_age.shape[1], float(combined_winner["age_penalty"])),
            np.full(full_score.shape[1], float(combined_winner["score_penalty"])),
        ],
        row_mask=full_mask,
    )
    names = load_current_player_names(NAMES, PLAYER_GAMES)
    rating_specs = {
        "normal": normal_beta,
        "age": full_age_fit.player_coefficients,
        "score": full_score_fit.player_coefficients,
        "combined": full_combined_fit.player_coefficients,
    }
    ratings = None
    for prefix, coefficients in rating_specs.items():
        table = ratings_table(design, coefficients, names=names).rename(
            columns={
                "offense_per_100": f"{prefix}_offense",
                "defense_per_100": f"{prefix}_defense",
                "net_per_100": f"{prefix}_net",
            }
        )
        keep = [
            "player_id",
            "player_name",
            "off_possessions",
            "def_possessions",
            f"{prefix}_offense",
            f"{prefix}_defense",
            f"{prefix}_net",
        ]
        if ratings is None:
            ratings = table[keep]
        else:
            ratings = ratings.merge(
                table[["player_id", f"{prefix}_offense", f"{prefix}_defense", f"{prefix}_net"]],
                on="player_id",
                validate="one_to_one",
            )
    assert ratings is not None
    ratings["combined_net_change"] = ratings["combined_net"] - ratings["normal_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)

    run = {
        "run_id": output.name,
        "status": "research_challenger" if any_gate_pass else "research_null",
        "model_family": "blocked_age_score_context_comparison",
        "estimand_id": contract["age_score_estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload},
        "selection": {
            "normal": normal_selection,
            "score_only": score_selection,
            "age_winner": age_winner,
            "combined_winner": combined_winner,
        },
        "diagnostic": {
            "metrics": metrics,
            "paired_game_bootstrap": bootstrap,
            "passes_reused_diagnostic_gate": gate_passes,
        },
        "quality": {
            "possessions": int(len(source)),
            "games": int(source["gameid"].nunique()),
            "age_slot_coverage": float(
                categorical_age.known_slots / categorical_age.total_slots
            ),
            "score_shape": score_winner["shape"],
            "score_columns": len(score_labels),
            "age_columns": len(age_labels),
            "season_2027_loaded": bool(
                source["season"]
                .eq(int(contract["untouched_confirmation_season"]))
                .any()
            ),
        },
        "decision": (
            "Retain as a research challenger pending untouched confirmation."
            if any_gate_pass
            else "Do not promote. No player-only or pregame-context variant clears "
            "the paired reused-diagnostic gate."
        ),
        "artifacts": {
            "age_selection": "age_selection.parquet",
            "combined_selection": "combined_selection.parquet",
            "evaluation": "evaluation.parquet",
            "ratings": "ratings.parquet",
        },
        "forbidden_interpretation": (
            "These are nuisance-control diagnostics on reused 2026 games, not causal "
            "age effects, causal effort, forecasts, or public rating promotions."
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
