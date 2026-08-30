#!/usr/bin/env python3
"""Run frozen Impact Validation V2 Gate A on the 2021 regular season."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.blocked_game_box_features import (
    audit_annual_source_parity,
    audit_legacy_game_target_parity,
    build_player_game_box15_ledger,
    load_gabriel_game_finals,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.impact_validation_v2 import (
    paired_whole_game_mse_bootstrap,
    pooled_game_metrics,
    run_gate_a,
)
from nba_impact.models.rapm import build_design, load_legacy_possessions


ROOT = Path(__file__).resolve().parents[1]
RATING_SEASON = 2021
EXPERIMENT_ID = "impact_validation_v2_gate_a"


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "validation_contract_v2",
        "contract_id": "impact_validation_v2",
        "status": "frozen_design_before_next_model_run",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must equal {value!r}.")
    primary = contract["retrospective_impact"]["primary_test"]
    if int(primary["folds"]) != 5 or primary["split_unit"] != "whole game":
        raise ValueError("Gate A requires five whole-game folds.")
    if int(contract["common_rules"]["uncertainty_draws"]) != 5_000:
        raise ValueError("Gate A requires 5,000 uncertainty draws.")
    return contract


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _decision(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame, contract: dict
) -> dict[str, object]:
    gate = contract["common_rules"]["promotion_gate"]
    indexed = metrics.set_index("candidate")
    candidate = indexed.loc["box15_aio"]
    reference = indexed.loc["zero_prior_rapm"]
    boot = bootstrap.iloc[0]
    tests = {
        "minimum_rmse_improvement": bool(
            boot["rmse_improvement_reference_minus_candidate"]
            >= float(gate["minimum_rmse_improvement_points_per_game"])
        ),
        "paired_mse_interval_favors_candidate": bool(
            boot["bootstrap_95_high"]
            < float(gate["paired_mse_interval_upper_bound"])
        ),
        "correlation_decline_within_limit": bool(
            candidate["correlation"] - reference["correlation"]
            >= -float(gate["maximum_correlation_decline"])
        ),
        "calibration_slope_distance_within_limit": bool(
            abs(candidate["calibration_slope"] - 1.0)
            - abs(reference["calibration_slope"] - 1.0)
            <= float(gate["maximum_calibration_slope_distance_increase"])
        ),
    }
    return {
        "classification": (
            "passes_gate_a_for_further_research"
            if all(tests.values())
            else "fails_gate_a"
        ),
        "tests": tests,
        "segment_gate": "not_scored_in_single_season_engine_validation",
        "production_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/impact_validation_v2.yml",
    )
    parser.add_argument(
        "--possession-cache", type=Path, default=ROOT / "rapm/data/possession_cache"
    )
    parser.add_argument(
        "--espn-player-box",
        type=Path,
        default=(
            ROOT
            / "data/lake/bronze/llimllib_nba_data/espn/player_box.parquet"
        ),
    )
    parser.add_argument(
        "--gabriel-pbp-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_merged_playbyplay/old_data",
    )
    parser.add_argument(
        "--player-sheet",
        type=Path,
        default=(
            ROOT
            / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals/2021.parquet"
        ),
    )
    historical_root = (
        ROOT
        / "artifacts/research/historical_box15_extension/historical_box15_extension_v1_08ff4c34ff"
    )
    parser.add_argument(
        "--annual-features",
        type=Path,
        default=historical_root / "annual_box15_features.parquet",
    )
    parser.add_argument(
        "--five-year-features",
        type=Path,
        default=historical_root / "five_year_box15_features.parquet",
    )
    parser.add_argument(
        "--five-year-targets",
        type=Path,
        default=historical_root / "five_year_targets.parquet",
    )
    parser.add_argument("--draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    contract = _load_contract(args.contract)
    if args.draws != int(contract["common_rules"]["uncertainty_draws"]):
        raise ValueError("Draw count must match the frozen validation contract.")
    sources = {
        "contract": args.contract,
        "possessions": args.possession_cache / f"matchups_{RATING_SEASON}.parquet",
        "espn_player_box": args.espn_player_box,
        "gabriel_player_sheet": args.player_sheet,
        "annual_features": args.annual_features,
        "five_year_features": args.five_year_features,
        "five_year_targets": args.five_year_targets,
        "runner": Path(__file__),
        "box_ledger_builder": ROOT
        / "src/nba_impact/data/blocked_game_box_features.py",
        "validation_engine": ROOT
        / "src/nba_impact/models/impact_validation_v2.py",
    }
    gabriel_paths = sorted(args.gabriel_pbp_root.glob("*_2021_rs.parquet"))
    if len(gabriel_paths) < 29:
        raise FileNotFoundError("Gate A requires the complete 2021 Gabriel PBP set.")
    sources.update(
        {
            f"gabriel_pbp_{path.stem.split('_')[0]}": path
            for path in gabriel_paths
        }
    )
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError(f"Gate A sources are missing: {missing}.")

    parity, parity_quality = audit_annual_source_parity(
        espn_player_box_path=args.espn_player_box,
        gabriel_pbp_root=args.gabriel_pbp_root,
        player_sheet_path=args.player_sheet,
        season=RATING_SEASON,
    )
    if not parity_quality["passed"]:
        failed = parity.loc[~parity["passed"], "feature_numerator"].tolist()
        raise ValueError(f"Annual Box15 source parity failed for {failed}.")

    possessions = load_legacy_possessions(
        args.possession_cache, (RATING_SEASON,), game_types=("regular",)
    )
    game_finals = load_gabriel_game_finals(
        args.gabriel_pbp_root,
        season=RATING_SEASON,
        game_ids=possessions["gameid"].astype(str).unique(),
    )
    game_target_parity, game_target_quality = audit_legacy_game_target_parity(
        possessions, game_finals
    )
    eligible_games = set(
        game_target_parity.loc[game_target_parity["strict_eligible"], "game_id"]
    )
    possessions = possessions.loc[
        possessions["gameid"].astype(str).isin(eligible_games)
    ].copy()
    if len(eligible_games) < 400:
        raise ValueError("Strict score-conserved regulation universe is too small.")
    ledger, ledger_quality = build_player_game_box15_ledger(
        possessions,
        espn_player_box_path=args.espn_player_box,
        gabriel_pbp_root=args.gabriel_pbp_root,
        season=RATING_SEASON,
    )
    if ledger_quality["player_game_source_join_coverage"] < 0.99:
        raise ValueError("Player-game source coverage fell below 99 percent.")
    design = build_design(possessions, include_home=True)
    results = run_gate_a(
        design=design,
        possessions=possessions,
        player_game_ledger=ledger,
        historical_annual=pd.read_parquet(args.annual_features),
        historical_five_year=pd.read_parquet(args.five_year_features),
        historical_targets=pd.read_parquet(args.five_year_targets),
        rating_season=RATING_SEASON,
        folds=5,
    )
    game_predictions = results["game_predictions"]
    pooled_metrics = pooled_game_metrics(game_predictions)
    bootstrap, bootstrap_draws = paired_whole_game_mse_bootstrap(
        game_predictions, draws=args.draws, seed=args.seed
    )
    decision = _decision(pooled_metrics, bootstrap, contract)

    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity_input = {
        "experiment_id": EXPERIMENT_ID,
        "rating_season": RATING_SEASON,
        "folds": 5,
        "draws": args.draws,
        "seed": args.seed,
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(
        json.dumps(identity_input, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research"
        / "impact_validation_v2"
        / f"impact_validation_v2_gate_a_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    frames = {
        "annual_source_parity.parquet": parity,
        "game_target_parity.parquet": game_target_parity,
        "player_game_box15_ledger.parquet": ledger,
        "fold_assignments.parquet": results["fold_assignments"],
        "game_predictions.parquet": game_predictions,
        "fold_metrics.parquet": results["fold_metrics"],
        "pooled_metrics.parquet": pooled_metrics,
        "fold_priors.parquet": results["fold_priors"],
        "fold_features.parquet": results["fold_features"],
        "prior_coverage.parquet": results["prior_coverage"],
        "model_selection.parquet": results["model_selection"],
        "paired_bootstrap.parquet": bootstrap,
        "bootstrap_draws.parquet": bootstrap_draws,
    }
    for name, frame in frames.items():
        frame.to_parquet(output / name, index=False)
    artifact_hashes = {
        name: sha256_file(output / name) for name in sorted(frames)
    }
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete_research_result",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "retrospective same-season player impact conditional on observed lineups",
        "rating_season": RATING_SEASON,
        "comparison": ["zero_prior_rapm", "box15_aio"],
        "rapm_penalties": {"offense": 3000, "defense": 3000, "home": 300},
        "folds": 5,
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "whole_game"},
        "quality": {
            "annual_source_parity": parity_quality,
            "game_target_parity": game_target_quality,
            "player_game_ledger": ledger_quality,
            "games_scored_per_candidate": int(
                game_predictions.groupby("candidate")["game_id"].nunique().min()
            ),
            "duplicate_game_candidate_rows": int(
                game_predictions.duplicated(["candidate", "game_id"]).sum()
            ),
            "season_2022_or_later_rows": int(
                game_predictions["season"].ge(2022).sum()
            ),
        },
        "decision": decision,
        "metrics": pooled_metrics.to_dict(orient="records"),
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes,
        "paths": {name.removesuffix(".parquet"): name for name in frames},
        "source_paths": {name: _relative(path) for name, path in sources.items()},
        "forbidden_interpretation": [
            "This strict regulation-game engine validation is not production promotion evidence.",
            "Observed held-game lineups make this retrospective reconstruction, not a pregame forecast.",
            "The strict game universe excludes overtime and every cache game whose side-specific points do not match the Gabriel final score.",
            "The frozen shared-field Box15 pooling uses offensive exposure for all 15 fields.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
