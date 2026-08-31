#!/usr/bin/env python3
"""Isolate the defensive features behind the mechanism-family result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES

import run_full_spm_history_ablation as base
import run_mechanism_feature_challenger as family


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "defense_mechanism_screen_v1"
CONTRACT = ROOT / "research/experiments/defense_mechanism_screen_v1.yml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-run", type=Path, required=True)
    args = parser.parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")

    box = tuple(BOX_PIPM_STYLE_FEATURES)
    candidates = {
        name: {"offense": box, "defense": (*box, *tuple(features))}
        for name, features in contract["candidates"].items()
    }
    candidate_names = tuple(candidates)
    model_order = (
        *candidate_names,
        "zero_prior_rapm",
        *(f"{candidate}_aio" for candidate in candidate_names),
    )
    primary_pairs = {
        frozenset(("box_15", candidate)) for candidate in candidate_names[1:]
    } | {
        frozenset(("box_15_aio", f"{candidate}_aio"))
        for candidate in candidate_names[1:]
    }

    panel, _ = base._load_panel(
        family.FEATURE_RUN / "five_year_features.parquet",
        family.TARGETS,
        family.FEATURE_RUN / "run.json",
        (),
    )
    mechanism_path = args.mechanism_run / "five_year_features.parquet"
    panel = panel.merge(
        pd.read_parquet(mechanism_path),
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    selected = {feature for sides in candidates.values() for feature in sides["defense"]}
    if panel[list(selected)].isna().any().any():
        raise ValueError("The defensive screen contains missing selected inputs.")

    priors, target_metrics, selections, models = family._fit_priors(
        panel, candidates
    )
    base.PRIOR_CANDIDATES = candidate_names
    base.MODEL_ORDER = model_order
    base.PRIMARY_PAIRS = primary_pairs
    annual, reconstruction = base._annual_bundles(
        family.POSSESSION_CACHE, family.MATRIX_ROOT
    )
    ratings, games, coverage = base._score_models(
        priors, annual, family.MATRIX_ROOT
    )
    folds, summary = base._game_metrics_frames(games)
    draws = int(contract["evaluation"]["uncertainty"]["draws"])
    intervals, paired = base.paired_game_bootstrap(
        games,
        draws=draws,
        seed=int(contract["evaluation"]["uncertainty"]["seed"]),
    )

    sources = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "family_runner": ROOT / "research/run_mechanism_feature_challenger.py",
        "mechanism_manifest": args.mechanism_run / "run.json",
        "mechanism_features": mechanism_path,
        "base_features": family.FEATURE_RUN / "five_year_features.parquet",
        "targets": family.TARGETS,
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_features": {
            name: list(sides["defense"]) for name, sides in candidates.items()
        },
        "rating_seasons": list(base.RATING_SEASONS),
        "evaluated_rating_seasons": list(base.EVALUATED_RATING_SEASONS),
        "bootstrap_draws": draws,
        "sources": {
            name: {
                "path": family._relative(path),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / (
        "artifacts/research/defense_mechanism_screen/"
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "model_selection.parquet": selections,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    model_root = output / "models"
    model_root.mkdir()
    for (season, candidate, side), model in models.items():
        joblib.dump(model, model_root / f"{season}_{candidate}_{side}.joblib")

    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "missing_selected_values": 0,
            "identical_games_within_fold": True,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(paired.loc[paired["primary_comparison"]].to_string(index=False))


if __name__ == "__main__":
    main()
