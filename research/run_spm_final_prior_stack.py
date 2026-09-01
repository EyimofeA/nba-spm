#!/usr/bin/env python3
"""Test the final combination of retained SPM prior challengers."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic

try:
    from run_aio_prior_complementarity import (
        canonical_frame_hash,
        _load_contract as load_source_contract,
        _load_features,
        _load_matrix,
        _metric_row,
        _paired_bootstrap,
        _precision_mode_games,
        _prior_target_rows,
        _reference_panels,
        _select_precision_modes,
        _selected_predictions,
        _selected_uniform_parameters,
        _uniform_basis,
        _uniform_grid,
        assert_component_identity,
        assert_identical_games,
        coefficient_center,
        select_from_history,
    )
except ModuleNotFoundError:
    from research.run_aio_prior_complementarity import (
        canonical_frame_hash,
        _load_contract as load_source_contract,
        _load_features,
        _load_matrix,
        _metric_row,
        _paired_bootstrap,
        _precision_mode_games,
        _prior_target_rows,
        _reference_panels,
        _select_precision_modes,
        _selected_predictions,
        _selected_uniform_parameters,
        _uniform_basis,
        _uniform_grid,
        assert_component_identity,
        assert_identical_games,
        coefficient_center,
        select_from_history,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_final_prior_stack_v1"
CONTRACT = ROOT / "research/experiments/spm_final_prior_stack_v1.yml"
SOURCE_RUN = ROOT / "artifacts/research/aio_prior_complementarity/aio_prior_complementarity_v1_4d83e381af"
CONSENSUS_RUN = ROOT / "artifacts/research/spm_consensus_complementarity/spm_consensus_complementarity_v1_8f49b7448f"
OUTPUT_ROOT = ROOT / "artifacts/research/spm_final_prior_stack"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_contract(path: Path = CONTRACT) -> dict:
    contract = yaml.safe_load(path.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment ID changed.")
    if tuple(contract["seasons"]["rating"]) != tuple(range(2016, 2026)):
        raise ValueError("Rating folds changed.")
    if int(contract["evaluation"]["bootstrap_draws"]) != 5000:
        raise ValueError("The frozen comparison requires 5,000 draws.")
    return contract


def combine_prior_deltas(
    box: pd.DataFrame,
    consensus: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    consensus_offense_weight: float,
    consensus_defense_weight: float,
    defense_residual_weight: float,
) -> pd.DataFrame:
    """Add selected consensus and defense-residual increments to Box15."""
    keys = ["PLAYER_ID"]
    merged = (
        box[keys + ["prior_offense", "prior_defense"]]
        .merge(
            consensus[keys + ["prior_offense", "prior_defense"]],
            on=keys,
            suffixes=("_box", "_consensus"),
            validate="one_to_one",
        )
        .merge(
            residual[keys + ["prior_defense"]].rename(columns={"prior_defense": "prior_defense_residual"}),
            on=keys,
            validate="one_to_one",
        )
    )
    output = pd.DataFrame(
        {
            "PLAYER_ID": merged["PLAYER_ID"],
            "prior_offense": merged["prior_offense_box"]
            + consensus_offense_weight
            * (merged["prior_offense_consensus"] - merged["prior_offense_box"]),
            "prior_defense": merged["prior_defense_box"]
            + consensus_defense_weight
            * (merged["prior_defense_consensus"] - merged["prior_defense_box"])
            + defense_residual_weight
            * (merged["prior_defense_residual"] - merged["prior_defense_box"]),
        }
    )
    output["prior_net"] = output["prior_offense"] + output["prior_defense"]
    return output


def _source_priors() -> dict[int, dict[str, pd.DataFrame]]:
    source = pd.read_parquet(SOURCE_RUN / "priors.parquet")
    consensus = pd.read_parquet(CONSENSUS_RUN / "matched_priors.parquet")
    rows: dict[int, dict[str, pd.DataFrame]] = {}
    for season in range(2016, 2026):
        candidates = {
            "box15": source.loc[
                source["design"].eq("target_excluded")
                & source["candidate"].eq("box15")
                & source["rating_season"].eq(season)
            ].copy(),
            "rich": source.loc[
                source["design"].eq("target_excluded")
                & source["candidate"].eq("rich")
                & source["rating_season"].eq(season)
            ].copy(),
            "residual": source.loc[
                source["design"].eq("target_excluded")
                & source["candidate"].eq("def_residual_outcome")
                & source["rating_season"].eq(season)
            ].copy(),
            "consensus": consensus.loc[
                consensus["design"].eq("target_excluded")
                & consensus["candidate"].eq("box15_consensus")
                & consensus["rating_season"].eq(season)
            ].copy(),
        }
        if any(frame.empty for frame in candidates.values()):
            raise ValueError(f"Source priors are incomplete for {season}.")
        common = set.intersection(*(set(frame["PLAYER_ID"]) for frame in candidates.values()))
        rows[season] = {
            name: frame.loc[frame["PLAYER_ID"].isin(common)].sort_values("PLAYER_ID").reset_index(drop=True)
            for name, frame in candidates.items()
        }
    return rows


def _combination_grid(contract: dict, sources: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    source_contract = load_source_contract()
    control = source_contract["rapm"]["aio_control"]
    values = contract["combination"]
    factor_cache = {}
    rows = []
    for season, candidates in sources.items():
        matrix = _load_matrix(season)
        components = {}
        for name in ("box15", "consensus", "residual"):
            center, _ = coefficient_center(candidates[name], matrix)
            components[name] = _uniform_basis(
                matrix,
                center,
                float(control["lambda_off"]),
                float(control["lambda_def"]),
                float(control["lambda_home"]),
                factor_cache,
            )
        base, box_off, box_def = components["box15"]
        _, consensus_off, consensus_def = components["consensus"]
        _, _, residual_def = components["residual"]
        for offense_weight in map(float, values["consensus_offense_weight"]):
            for defense_weight in map(float, values["consensus_defense_weight"]):
                for residual_weight in map(float, values["defense_residual_weight"]):
                    predicted = (
                        base
                        + box_off
                        + box_def
                        + offense_weight * (consensus_off - box_off)
                        + defense_weight * (consensus_def - box_def)
                        + residual_weight * (residual_def - box_def)
                    )
                    rows.append(
                        {
                            "rating_season": season,
                            "outcome_season": season + 1,
                            "consensus_offense_weight": offense_weight,
                            "consensus_defense_weight": defense_weight,
                            "defense_residual_weight": residual_weight,
                            **_metric_row(matrix.actual_margin, predicted),
                        }
                    )
    return pd.DataFrame(rows)


def _selected_stack_priors(
    contract: dict,
    sources: dict[int, dict[str, pd.DataFrame]],
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "consensus_offense_weight",
        "consensus_defense_weight",
        "defense_residual_weight",
    ]
    combination = contract["combination"]
    selection_outcomes = tuple(map(int, contract["seasons"]["design_selection_outcomes"]))
    priors = []
    selections = []
    for season, candidates in sources.items():
        chosen = select_from_history(
            grid,
            columns,
            {name: float(value) for name, value in combination["defaults"].items()},
            current_outcome_season=season + 1,
            selection_outcomes=selection_outcomes,
            minimum_folds=int(combination["minimum_prior_folds"]),
        )
        prior = combine_prior_deltas(
            candidates["box15"],
            candidates["consensus"],
            candidates["residual"],
            **chosen,
        )
        prior["design"] = "target_excluded"
        prior["candidate"] = "final_stack"
        prior["rating_season"] = season
        prior["input_season"] = season
        priors.append(prior)
        selections.append({"rating_season": season, "outcome_season": season + 1, **chosen})
    return pd.concat(priors, ignore_index=True), pd.DataFrame(selections)


def _support_priors(sources: dict[int, dict[str, pd.DataFrame]], stack: pd.DataFrame) -> pd.DataFrame:
    rows = [stack]
    for season, candidates in sources.items():
        for name in ("box15", "rich"):
            frame = candidates[name].copy()
            frame["design"] = "target_excluded"
            frame["candidate"] = name
            frame["rating_season"] = season
            frame["input_season"] = season
            rows.append(frame)
    output = pd.concat(rows, ignore_index=True, sort=False)
    if output.duplicated(["design", "candidate", "rating_season", "PLAYER_ID"]).any():
        raise ValueError("Final-stack prior keys are not unique.")
    assert_component_identity(
        output.rename(columns={"prior_offense": "offense", "prior_defense": "defense", "prior_net": "net"})
    )
    return output


def _fold_metrics(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, stage, season), frame in games.groupby(["candidate", "stage", "outcome_season"], sort=True):
        rows.append(
            {
                "candidate": candidate,
                "stage": stage,
                "rating_season": season - 1,
                "outcome_season": season,
                "games": len(frame),
                **_metric_row(frame["actual_margin"].to_numpy(), frame["predicted_margin"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _decision(contract: dict, folds: pd.DataFrame, paired: pd.DataFrame) -> dict:
    reference = "current_control__box15"
    candidate = "target_excluded__final_stack"
    later = folds.loc[
        folds["stage"].eq("aio")
        & folds["outcome_season"].isin(contract["seasons"]["later_diagnostic_outcomes"])
        & folds["candidate"].isin((reference, candidate))
    ]
    wide = later.pivot(index="outcome_season", columns="candidate", values=["mse", "correlation"])
    reference_mse = float(wide["mse"][reference].mean())
    candidate_mse = float(wide["mse"][candidate].mean())
    rmse_gain = math.sqrt(reference_mse) - math.sqrt(candidate_mse)
    correlation_loss = float(wide["correlation"][reference].mean() - wide["correlation"][candidate].mean())
    wins = int((wide["mse"][candidate] < wide["mse"][reference]).sum())
    interval = paired.loc[paired["candidate"].eq(candidate)].iloc[0]
    passes = bool(
        rmse_gain >= float(contract["evaluation"]["practical_rmse_gate"])
        and float(interval["bootstrap_95_high"]) < 0
        and correlation_loss <= float(contract["evaluation"]["maximum_correlation_loss"])
        and wins >= int(contract["evaluation"]["minimum_later_fold_wins"])
    )
    return {
        "classification": "hybrid prior selected" if passes else "Box15 retained",
        "passes_numeric_gate": passes,
        "later_rmse_gain_points_per_game": rmse_gain,
        "later_mean_correlation_loss": correlation_loss,
        "later_fold_wins": wins,
        "stop_broad_retrospective_spm_feature_search": not passes,
    }


def main() -> None:
    contract = _load_contract()
    source_contract = load_source_contract()
    sources = _source_priors()
    grid = _combination_grid(contract, sources)
    stack, selections = _selected_stack_priors(contract, sources, grid)
    priors = _support_priors(sources, stack)
    references = _reference_panels(source_contract)
    _, rich_features, _, _, _ = _load_features(source_contract)
    targets = _prior_target_rows(priors, references, rich_features)
    matrices = {season: _load_matrix(season) for season in range(2016, 2026)}
    uniform_grid = _uniform_grid(source_contract, priors, matrices, {})
    uniform_selection = _selected_uniform_parameters(source_contract, uniform_grid)
    precision_games, multipliers = _precision_mode_games(
        source_contract,
        priors,
        targets,
        matrices,
        uniform_selection,
    )
    precision_selection = _select_precision_modes(source_contract, precision_games)
    games, ratings, _ = _selected_predictions(
        source_contract,
        priors,
        matrices,
        precision_games,
        precision_selection,
        uniform_selection,
        targets,
    )
    reference_games = pd.read_parquet(SOURCE_RUN / "game_predictions.parquet")
    reference_games = reference_games.loc[
        reference_games["candidate"].eq("current_control__box15")
        & reference_games["stage"].eq("aio")
    ].copy()
    games = pd.concat([games, reference_games], ignore_index=True, sort=False)
    assert_identical_games(games.loc[games["stage"].eq("aio")])
    folds = _fold_metrics(games)
    paired = _paired_bootstrap(
        games,
        reference="current_control__box15",
        stage="aio",
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )
    decision = _decision(contract, folds, paired)

    source_paths = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "source_manifest": SOURCE_RUN / "run.json",
        "consensus_manifest": CONSENSUS_RUN / "run.json",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    if output.exists():
        print(output)
        print(json.dumps(decision, indent=2))
        return
    frames = {
        "combination_grid.parquet": grid,
        "combination_selections.parquet": selections,
        "priors.parquet": priors,
        "uniform_penalty_grid.parquet": uniform_grid,
        "uniform_parameter_selections.parquet": uniform_selection,
        "precision_mode_selections.parquet": precision_selection,
        "precision_multipliers.parquet": multipliers,
        "game_predictions.parquet": games,
        "ratings.parquet": ratings,
        "fold_metrics.parquet": folds,
        "paired_bootstrap.parquet": paired,
    }
    output.mkdir(parents=True)
    files = {}
    for name, frame in frames.items():
        _atomic_parquet(frame, output / name)
        files[name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
            "row_set_sha256": canonical_frame_hash(frame, frame.columns[: min(4, len(frame.columns))]),
        }
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": source_hashes[name]}
            for name, path in source_paths.items()
        },
        "quality": {
            "identical_games": True,
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
        },
        "decision": decision,
        "files": files,
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
