#!/usr/bin/env python3
"""Build a ten-minute-safe combined validation and interpretability report."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.combined_validation_interpretability import (
    align_game_predictions,
    align_prior_predictions,
    build_aio_component_ledger,
    build_factor_skill_panel,
    game_metrics,
    paired_game_bootstrap,
    score_game_predictions,
    score_prior_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "combined_validation_interpretability_v1"
CONTRACT = ROOT / "research/experiments/combined_validation_interpretability_v1.yml"
FULL_RUN = ROOT / (
    "artifacts/research/full_spm_history_ablation/"
    "full_spm_history_ablation_v1_2eb5eb428c"
)
STRICT_RUN = ROOT / (
    "artifacts/research/impact_validation_v2/impact_validation_v2_gate_a_090cb2d323"
)
BOX_RUN = ROOT / (
    "artifacts/research/final_box_feature_ladder/"
    "final_box_feature_ladder_v1_8bb26f12e7"
)
INTERPRET_RUN = ROOT / (
    "artifacts/research/final_box_interpretability/"
    "final_box_interpretability_v1_652799efb6"
)
FACTOR_RUN = ROOT / (
    "artifacts/research/historical_factor_residual_tournament/"
    "historical_factor_residual_tournament_v2_c06bdebcd5"
)
FEATURES = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_8be676bd0f/five_year_features.parquet"
)
TARGETS = ROOT / (
    "artifacts/models/five_year_target_spm/"
    "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_report_only",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must equal {value!r}.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    if contract["runtime_contract"]["refits_allowed"]:
        raise ValueError("The combined report must not refit models.")
    return json.loads(json.dumps(contract, default=str))


def _write(output: Path, name: str, frame: pd.DataFrame, files: dict) -> None:
    frame.to_parquet(output / name, index=False)
    files[name] = {
        "path": name,
        "rows": int(len(frame)),
        "sha256": sha256_file(output / name),
    }


def main() -> None:
    started = time.monotonic()
    contract = _load_contract()
    draws = int(contract["runtime_contract"]["bootstrap_draws"])
    seed = int(contract["runtime_contract"]["bootstrap_seed"])
    groups = {
        name: tuple(features)
        for name, features in contract["interpretability"]["exact_groups"].items()
    }

    priors = pd.read_parquet(FULL_RUN / "priors.parquet")
    targets = pd.read_parquet(TARGETS)
    aligned_priors, prior_audit = align_prior_predictions(
        priors, targets, candidates=("full_spm", "box_pipm")
    )
    prior_folds, prior_summary = score_prior_predictions(aligned_priors)

    games = pd.read_parquet(FULL_RUN / "game_predictions.parquet")
    game_candidates = ("box_pipm_aio", "full_spm_aio", "zero_prior_rapm")
    aligned_games, game_audit = align_game_predictions(
        games,
        candidates=game_candidates,
        key_columns=("rating_season", "test_season", "game_id"),
    )
    game_folds, game_summary = score_game_predictions(
        aligned_games, fold_columns=("rating_season", "test_season")
    )
    game_bootstrap = paired_game_bootstrap(
        aligned_games,
        candidate="full_spm_aio",
        reference="box_pipm_aio",
        season_column="test_season",
        draws=draws,
        seed=seed,
    )

    strict_games = pd.read_parquet(STRICT_RUN / "game_predictions.parquet")
    aligned_strict, strict_audit = align_game_predictions(
        strict_games,
        candidates=("box15_aio", "zero_prior_rapm"),
        key_columns=("season", "fold", "game_id"),
    )
    strict_summary = pd.DataFrame(
        [
            {"candidate": candidate, **game_metrics(frame)}
            for candidate, frame in aligned_strict.groupby("candidate", sort=True)
        ]
    ).sort_values("mse")
    strict_source = pd.read_parquet(STRICT_RUN / "pooled_metrics.parquet").sort_values(
        "candidate"
    )
    strict_check = strict_summary.sort_values("candidate")
    for metric in ("mse", "rmse", "mae", "correlation", "calibration_slope"):
        difference = abs(
            strict_check[metric].to_numpy() - strict_source[metric].to_numpy()
        )
        if difference.max() >= 1e-9:
            raise AssertionError(f"Strict source parity failed for {metric}.")
    strict_bootstrap = pd.read_parquet(STRICT_RUN / "paired_bootstrap.parquet")

    feature_panel = pd.read_parquet(
        FEATURES, columns=["PLAYER_ID", "Window_End", *BOX_PIPM_STYLE_FEATURES]
    )
    box_priors = pd.read_parquet(BOX_RUN / "priors.parquet")
    leaderboard = pd.read_parquet(INTERPRET_RUN / "active_2026_leaderboard.parquet")
    models = {
        side: joblib.load(BOX_RUN / "models" / f"2026_box_15_{side}.joblib")
        for side in ("offense", "defense")
    }
    component_ledger, player_summary, interpretation_quality = build_aio_component_ledger(
        feature_panel=feature_panel,
        raw_priors=box_priors,
        active_leaderboard=leaderboard,
        models=models,
        feature_names=BOX_PIPM_STYLE_FEATURES,
        groups=groups,
    )
    factor_predictions = pd.read_parquet(FACTOR_RUN / "factor_predictions.parquet")
    factor_skills = build_factor_skill_panel(
        factor_predictions,
        leaderboard.loc[leaderboard["candidate"].eq("box_15_aio")],
    )
    factor_metrics = pd.read_parquet(FACTOR_RUN / "factor_metrics.parquet")
    factor_quality = (
        factor_metrics.loc[
            factor_metrics["rating_season"].le(2025)
            & factor_metrics["factor"].isin(
                ("shooting_ts", "turnover_avoidance", "opponent_oreb_prevention")
            )
        ]
        .groupby(["factor", "component", "candidate"], as_index=False)
        .agg(
            folds=("rating_season", "nunique"),
            mean_r2=("r2", "mean"),
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
        )
    )
    factor_downstream = pd.read_parquet(FACTOR_RUN / "summary.parquet").loc[
        lambda frame: frame["candidate"].isin(
            (
                "box15_aio",
                "box15_ts_aio",
                "box15_turnover_aio",
                "box15_oreb_aio",
                "box15_all_factors_aio",
            )
        )
    ]
    factor_manifest = json.loads((FACTOR_RUN / "run.json").read_text())
    factor_gate = pd.DataFrame(factor_manifest["decisions"])
    dependence = pd.read_parquet(INTERPRET_RUN / "group_permutation_summary.parquet")

    trajectories = pd.read_parquet(BOX_RUN / "ratings.parquet").loc[
        lambda frame: frame["candidate"].eq("box_15_aio")
        & frame["Poss_Off"].gt(0)
        & frame["Poss_Def"].gt(0)
    ]
    active_identity = leaderboard.loc[
        leaderboard["candidate"].eq("box_15_aio"),
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"],
    ]
    trajectories = trajectories.merge(
        active_identity, on="PLAYER_ID", how="inner", validate="many_to_one"
    ).sort_values(["PLAYER_NAME", "rating_season"])

    row_set_audit = pd.DataFrame(
        [
            {
                "panel": "prior_target_fit",
                "candidate_count": prior_audit["candidate_count"],
                "evaluation_units": prior_audit["common_player_windows"],
                "unit": "player_window",
                "rows_per_candidate": prior_audit["rows_per_candidate"],
                "folds": prior_audit["seasons"],
                "identical_row_set": prior_audit["identical_row_set"],
                "outcome_parity": "passed",
                "season_2027_rows": prior_audit["season_2027_rows"],
            },
            {
                "panel": "next_season_game_prediction",
                "candidate_count": game_audit["candidate_count"],
                "evaluation_units": game_audit["common_games"],
                "unit": "whole_game",
                "rows_per_candidate": game_audit["rows_per_candidate"],
                "folds": int(aligned_games["test_season"].nunique()),
                "identical_row_set": game_audit["identical_row_set"],
                "outcome_parity": "passed",
                "season_2027_rows": game_audit["season_2027_rows"],
            },
            {
                "panel": "strict_same_season_reconstruction",
                "candidate_count": strict_audit["candidate_count"],
                "evaluation_units": strict_audit["common_games"],
                "unit": "whole_game",
                "rows_per_candidate": strict_audit["rows_per_candidate"],
                "folds": int(aligned_strict["fold"].nunique()),
                "identical_row_set": strict_audit["identical_row_set"],
                "outcome_parity": "passed",
                "season_2027_rows": strict_audit["season_2027_rows"],
            },
        ]
    )
    source_paths = {
        "contract": CONTRACT,
        "full_run_manifest": FULL_RUN / "run.json",
        "full_priors": FULL_RUN / "priors.parquet",
        "full_game_predictions": FULL_RUN / "game_predictions.parquet",
        "strict_run_manifest": STRICT_RUN / "run.json",
        "strict_game_predictions": STRICT_RUN / "game_predictions.parquet",
        "strict_pooled_metrics": STRICT_RUN / "pooled_metrics.parquet",
        "strict_paired_bootstrap": STRICT_RUN / "paired_bootstrap.parquet",
        "box_run_manifest": BOX_RUN / "run.json",
        "box_priors": BOX_RUN / "priors.parquet",
        "box_ratings": BOX_RUN / "ratings.parquet",
        "box_offense_model": BOX_RUN / "models/2026_box_15_offense.joblib",
        "box_defense_model": BOX_RUN / "models/2026_box_15_defense.joblib",
        "interpret_run_manifest": INTERPRET_RUN / "run.json",
        "active_leaderboard": INTERPRET_RUN / "active_2026_leaderboard.parquet",
        "global_model_dependence": INTERPRET_RUN / "group_permutation_summary.parquet",
        "factor_run_manifest": FACTOR_RUN / "run.json",
        "factor_predictions": FACTOR_RUN / "factor_predictions.parquet",
        "factor_metrics": FACTOR_RUN / "factor_metrics.parquet",
        "factor_summary": FACTOR_RUN / "summary.parquet",
        "features": FEATURES,
        "targets": TARGETS,
        "implementation": ROOT
        / "src/nba_impact/models/combined_validation_interpretability.py",
        "box_pipm_style_implementation": ROOT
        / "src/nba_impact/models/box_pipm_style.py",
        "validation_math_implementation": ROOT
        / "src/nba_impact/models/impact_validation_suite.py",
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "contract": contract,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/research/combined_validation_interpretability" / (
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    files: dict = {}
    outputs = {
        "prior_target_fit_folds.parquet": prior_folds,
        "prior_target_fit_summary.parquet": prior_summary,
        "next_season_game_folds.parquet": game_folds,
        "next_season_game_summary.parquet": game_summary,
        "next_season_paired_bootstrap.parquet": game_bootstrap,
        "strict_same_season_summary.parquet": strict_summary,
        "strict_same_season_paired_bootstrap.parquet": strict_bootstrap,
        "player_aio_component_ledger_2026.parquet": component_ledger,
        "player_aio_summary_2026.parquet": player_summary,
        "player_factor_skills_2026.parquet": factor_skills,
        "factor_skill_quality.parquet": factor_quality,
        "factor_downstream_quality.parquet": factor_downstream,
        "factor_downstream_gate.parquet": factor_gate,
        "global_model_dependence.parquet": dependence,
        "player_aio_trajectories_2021_2026.parquet": trajectories,
        "row_set_audit.parquet": row_set_audit,
    }
    for name, frame in outputs.items():
        _write(output, name, frame, files)
    elapsed = time.monotonic() - started
    if elapsed > 600:
        raise RuntimeError(f"Combined report exceeded ten minutes: {elapsed:.1f}s.")
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "report_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "model_refits": 0,
        "season_2027_loaded": False,
        "config": config,
        "quality": {
            "row_sets": row_set_audit.to_dict(orient="records"),
            "interpretability": interpretation_quality,
            "panels_combined_into_one_score": False,
        },
        "files": files,
        "interpretation_boundaries": [
            "The three validation panels estimate different quantities and are not one composite score.",
            "Next-season games use observed future lineups and are not a deployable pregame forecast.",
            "The strict same-season panel covers only its score-conserved eligible subset.",
            "Exact Box15 contribution groups are model accounting, not causal effects.",
            "Factor skill scores use factor-target units and are not additive to AIO points per 100.",
            "Season 2027 was not loaded and remains untouched confirmation.",
        ],
    }
    write_json_atomic(manifest, output / "run.json")
    print(output)
    print(f"runtime_seconds={elapsed:.3f}")
    print("\nNext-season game prediction")
    print(game_summary.to_string(index=False))
    print("\nFull SPM AIO minus Box15 AIO")
    print(game_bootstrap.to_string(index=False))
    print("\nStrict same-season reconstruction")
    print(strict_summary.to_string(index=False))


if __name__ == "__main__":
    main()
