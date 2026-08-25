"""Select a five-point score-state control jointly with scalar RAPM penalties."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "research/experiments/rubberband_5pt_lambda_v1.json"
DEFAULT_OUTPUT = ROOT / "research/rapm_lab/outputs/rubberband_5pt_lambda"
CACHE = ROOT / "rapm/data/possession_cache"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
NAMES = ROOT / "rapm/data/all_names.csv"


def _config(seasons: list[int], candidate: dict, home: float) -> RapmConfig:
    return RapmConfig(
        seasons=tuple(seasons),
        lambda_off=float(candidate["lambda_off"]),
        lambda_def=float(candidate["lambda_def"]),
        lambda_home=float(home),
        data_scope="five_point_score_bucket_lambda_research",
    )


def _predict_plain(design, config, train_mask, test_mask):
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    return np.asarray(design.X[test_mask] @ beta).ravel() + intercept, beta


def _predict_bucket(design, margins, config, spec, train_mask, test_mask):
    fit = fit_score_state_rapm(
        design,
        margins,
        config,
        minimum=int(spec["minimum"]),
        maximum=int(spec["maximum"]),
        bucket_width=int(spec["width"]),
        state_penalty=float(spec["penalty"]),
        row_mask=train_mask,
    )
    prediction = predict_score_state_rapm(
        fit,
        design,
        margins,
        row_mask=test_mask,
        include_score_state=False,
    )
    return prediction, fit


def _evaluate(frame, y, prediction, label, candidate, mode):
    metrics, games = game_margin_metrics(frame, prediction)
    return (
        {
            "label": label,
            "lambda_name": candidate["name"],
            "score_control": mode,
            "lambda_off": float(candidate["lambda_off"]),
            "lambda_def": float(candidate["lambda_def"]),
            "possession_rmse": float(math.sqrt(np.mean((y - prediction) ** 2))),
            **metrics,
        },
        games,
    )


def run_experiment(contract_path: Path = DEFAULT_CONTRACT, output_root: Path = DEFAULT_OUTPUT) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("The experiment contract must be frozen before fitting.")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract_sha256": sha256_file(contract_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(ROOT / "src/nba_impact/models/rubberband_score_state.py"),
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:10]
    output = output_root / f"{contract['experiment_id']}_{identity}"
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    seasons = [int(value) for value in contract["seasons"]]
    source = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    source = annotate_offense_margin_before(source)
    design = build_design(source, include_home=True)
    margins = source["offense_margin_before"].to_numpy(dtype=float)
    bucket = contract["score_bucket"]
    home = float(contract["lambda_home"])

    selection_train = source["season"].isin(contract["selection_fit_seasons"]).to_numpy()
    selection_test = source["season"].eq(contract["selection_season"]).to_numpy()
    selection_frame = source.loc[selection_test].reset_index(drop=True)
    selection_rows = []
    for candidate in contract["lambda_candidates"]:
        config = _config(seasons, candidate, home)
        plain_prediction, _ = _predict_plain(design, config, selection_train, selection_test)
        row, _ = _evaluate(
            selection_frame,
            design.y[selection_test],
            plain_prediction,
            f"plain__{candidate['name']}",
            candidate,
            "none",
        )
        selection_rows.append(row)
        bucket_prediction, _ = _predict_bucket(
            design, margins, config, bucket, selection_train, selection_test
        )
        row, _ = _evaluate(
            selection_frame,
            design.y[selection_test],
            bucket_prediction,
            f"bucket5__{candidate['name']}",
            candidate,
            "five_point_buckets",
        )
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows).sort_values(
        ["margin_rmse", "margin_correlation"], ascending=[True, False], kind="stable"
    )
    winner = selection.iloc[0].to_dict()
    selection.to_parquet(output / "selection.parquet", index=False)

    diagnostic_train = source["season"].isin(contract["diagnostic_fit_seasons"]).to_numpy()
    diagnostic_test = source["season"].eq(contract["diagnostic_season"]).to_numpy()
    diagnostic_frame = source.loc[diagnostic_test].reset_index(drop=True)
    baseline_candidate = next(c for c in contract["lambda_candidates"] if c["name"] == "baseline")
    winner_candidate = next(c for c in contract["lambda_candidates"] if c["name"] == winner["lambda_name"])
    baseline_config = _config(seasons, baseline_candidate, home)
    winner_config = _config(seasons, winner_candidate, home)
    baseline_prediction, _ = _predict_plain(
        design, baseline_config, diagnostic_train, diagnostic_test
    )
    if winner["score_control"] == "five_point_buckets":
        winner_prediction, diagnostic_bucket_fit = _predict_bucket(
            design, margins, winner_config, bucket, diagnostic_train, diagnostic_test
        )
    else:
        winner_prediction, _ = _predict_plain(
            design, winner_config, diagnostic_train, diagnostic_test
        )
        diagnostic_bucket_fit = None
    baseline_row, baseline_games = _evaluate(
        diagnostic_frame,
        design.y[diagnostic_test],
        baseline_prediction,
        "baseline",
        baseline_candidate,
        "none",
    )
    winner_row, winner_games = _evaluate(
        diagnostic_frame,
        design.y[diagnostic_test],
        winner_prediction,
        "selected_candidate",
        winner_candidate,
        winner["score_control"],
    )
    diagnostic = pd.DataFrame([baseline_row, winner_row])
    diagnostic.to_parquet(output / "diagnostic.parquet", index=False)
    bootstrap = _paired_rmse_bootstrap(
        baseline_games,
        winner_games,
        draws=int(contract["paired_game_bootstrap_draws"]),
        seed=int(contract["paired_game_bootstrap_seed"]),
    )

    full_mask = np.ones(len(source), dtype=bool)
    baseline_beta, _ = fit_coefficients(design, baseline_config, row_mask=full_mask)
    if winner["score_control"] == "five_point_buckets":
        full_bucket_fit = fit_score_state_rapm(
            design,
            margins,
            winner_config,
            minimum=int(bucket["minimum"]),
            maximum=int(bucket["maximum"]),
            bucket_width=int(bucket["width"]),
            state_penalty=float(bucket["penalty"]),
        )
        winner_beta = full_bucket_fit.player_coefficients
        curve = score_state_curve(full_bucket_fit)
    else:
        winner_beta, _ = fit_coefficients(design, winner_config, row_mask=full_mask)
        curve = pd.DataFrame()
    if not curve.empty:
        curve.to_parquet(output / "score_bucket_curve.parquet", index=False)
    names = load_current_player_names(NAMES, PLAYER_GAMES)
    baseline_ratings = ratings_table(design, baseline_beta, names=names).rename(
        columns={"offense_per_100": "baseline_offense", "defense_per_100": "baseline_defense", "net_per_100": "baseline_net"}
    )
    winner_ratings = ratings_table(design, winner_beta, names=names).rename(
        columns={"offense_per_100": "candidate_offense", "defense_per_100": "candidate_defense", "net_per_100": "candidate_net"}
    )
    ratings = baseline_ratings.merge(
        winner_ratings[["player_id", "candidate_offense", "candidate_defense", "candidate_net"]],
        on="player_id",
        validate="one_to_one",
    )
    ratings["candidate_net_change"] = ratings["candidate_net"] - ratings["baseline_net"]
    ratings.to_parquet(output / "ratings.parquet", index=False)

    rmse_delta = float(winner_row["margin_rmse"] - baseline_row["margin_rmse"])
    correlation_delta = float(winner_row["margin_correlation"] - baseline_row["margin_correlation"])
    gate = contract["promotion_gate"]
    passes = bool(
        rmse_delta < float(gate["maximum_rmse_delta"])
        and correlation_delta >= -float(gate["maximum_correlation_decline"])
        and float(bootstrap["upper_95"]) < float(gate["paired_rmse_interval_upper"])
    )
    run = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_family": "five_point_score_bucket_and_lambda_selection",
        "config": {**contract, **identity_payload},
        "selection_winner": winner,
        "diagnostic": {
            "baseline": baseline_row,
            "candidate": winner_row,
            "rmse_delta_candidate_minus_baseline": rmse_delta,
            "correlation_delta_candidate_minus_baseline": correlation_delta,
            "paired_game_bootstrap": bootstrap,
            "passes_promotion_gate": passes,
        },
        "quality": {
            "possessions": int(len(source)),
            "games": int(source["gameid"].nunique()),
            "seasons": seasons,
            "maximum_baseline_identity_error": float(np.abs(ratings["baseline_offense"] + ratings["baseline_defense"] - ratings["baseline_net"]).max()),
            "maximum_candidate_identity_error": float(np.abs(ratings["candidate_offense"] + ratings["candidate_defense"] - ratings["candidate_net"]).max()),
        },
        "decision": "propagate" if passes else "reject_and_keep_baseline",
        "artifacts": {
            "selection": "selection.parquet",
            "diagnostic": "diagnostic.parquet",
            "ratings": "ratings.parquet",
            "score_bucket_curve": "score_bucket_curve.parquet" if not curve.empty else None,
        },
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
