#!/usr/bin/env python3
"""Run one chronological FullSPM versus BoxSPM future-game pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "walk_forward_spm_pilot_v1"
RATING_SEASON = 2024
TEST_SEASON = 2025
MODEL_ORDER = (
    "full_spm",
    "box_pipm",
    "zero_prior_rapm",
    "full_spm_aio",
    "box_pipm_aio",
)
CONTRACT = ROOT / "research/experiments/walk_forward_spm_pilot_v1.yml"
FEATURE_RUN = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_4ffd1e34df"
)
TARGETS = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"


def main() -> None:
    contract = json.loads(
        json.dumps(yaml.safe_load(CONTRACT.read_text()), default=str)
    )
    cutoff = contract["information_cutoff"]
    if (
        contract["experiment_id"] != EXPERIMENT_ID
        or cutoff["rating_season"] != RATING_SEASON
        or cutoff["test_season"] != TEST_SEASON
    ):
        raise ValueError("Pilot contract changed.")

    panel, selected = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    panel = panel.loc[panel["Window_End"].le(RATING_SEASON)].copy()
    if panel["Window_End"].max() != RATING_SEASON:
        raise ValueError("Pilot panel does not end at the rating cutoff.")
    base.RATING_SEASONS = (RATING_SEASON,)
    base.EVALUATED_RATING_SEASONS = (RATING_SEASON,)
    base.PRIOR_CANDIDATES = ("full_spm", "box_pipm")
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {
        frozenset(("full_spm", "box_pipm")),
        frozenset(("full_spm_aio", "box_pipm_aio")),
    }

    priors, target_metrics, selections, _ = base._fit_priors(panel, selected, ())
    annual = {}
    for season in range(2020, RATING_SEASON):
        frame = base.load_legacy_possessions(
            POSSESSION_CACHE, (season,), game_types=("regular",)
        )
        annual[season] = base._annual_from_frame(frame, season)
    annual[RATING_SEASON], quality = base._recover_annual(
        MATRIX_ROOT / f"5y_end_{RATING_SEASON}", RATING_SEASON, annual
    )
    reconstruction = pd.DataFrame([quality])
    ratings, games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    folds, summary = base._game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = base.paired_game_bootstrap(
        games,
        draws=int(contract["evaluation"]["uncertainty"]["draws"]),
        seed=int(contract["evaluation"]["uncertainty"]["seed"]),
    )

    sources = {
        "contract": CONTRACT,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "matrix": MATRIX_ROOT / f"5y_end_{RATING_SEASON}/manifest.json",
        "runner": Path(__file__),
        **{
            f"possessions_{season}": POSSESSION_CACHE / f"matchups_{season}.parquet"
            for season in range(2020, RATING_SEASON)
        },
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_season": RATING_SEASON,
        "test_season": TEST_SEASON,
        "feature_counts": {side: len(selected[side]) for side in ("offense", "defense")},
        "models": list(MODEL_ORDER),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        ROOT
        / "artifacts/research/walk_forward_spm_pilot"
        / f"{EXPERIMENT_ID}_{identity}"
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
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_one_fold_pilot_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games": True,
            "scored_games": int(games["game_id"].nunique()),
            "maximum_loaded_feature_window": int(panel["Window_End"].max()),
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
        },
        "files": {},
        "forbidden_interpretation": (
            "One reused chronological fold cannot select or promote either model."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")

    requested = summary.loc[
        summary["candidate"].isin(
            ("full_spm", "box_pipm", "full_spm_aio", "box_pipm_aio")
        )
    ]
    pairs = bootstrap_pairs.loc[bootstrap_pairs["primary_comparison"]]
    print(output)
    print(requested.to_string(index=False))
    print("\nPaired MSE comparisons")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
