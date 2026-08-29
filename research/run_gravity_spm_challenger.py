#!/usr/bin/env python3
"""Test dynamic shooting gravity in Box15 and the full SPM/AIO."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES

import run_final_box_feature_ladder as ladder
import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "gravity_spm_challenger_v1"
CONTRACT = ROOT / "research/experiments/gravity_spm_challenger_v1.yml"
FEATURE_RUN = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f"
TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
GRAVITY_RUN = ROOT / "artifacts/research/laser_breaker/laser_breaker_v1_75d8ef37c1"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
CANDIDATES = (
    "box_15",
    "box_15_plus_gravity",
    "full_spm",
    "full_spm_plus_gravity",
)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("The gravity SPM contract changed.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    panel, selected = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    gravity = pd.read_parquet(GRAVITY_RUN / "gravity_panel.parquet")[
        ["PLAYER_ID", "Window_End", "court_signal_gravity"]
    ]
    panel = panel.merge(
        gravity,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    coverage = (
        panel.assign(observed=panel["court_signal_gravity"].notna())
        .groupby("Window_End", as_index=False)
        .agg(players=("PLAYER_ID", "size"), observed=("observed", "sum"))
    )
    coverage["coverage"] = coverage["observed"] / coverage["players"]
    panel["court_signal_gravity"] = panel["court_signal_gravity"].fillna(0.0)
    candidates = {
        "box_15": {
            "offense": BOX_PIPM_STYLE_FEATURES,
            "defense": BOX_PIPM_STYLE_FEATURES,
        },
        "box_15_plus_gravity": {
            "offense": (*BOX_PIPM_STYLE_FEATURES, "court_signal_gravity"),
            "defense": BOX_PIPM_STYLE_FEATURES,
        },
        "full_spm": selected,
        "full_spm_plus_gravity": {
            "offense": (*selected["offense"], "court_signal_gravity"),
            "defense": selected["defense"],
        },
    }
    priors, target_metrics, alphas, coefficients, _ = ladder._fit_priors(
        panel, candidates
    )
    base.PRIOR_CANDIDATES = CANDIDATES
    base.MODEL_ORDER = (
        *CANDIDATES,
        "zero_prior_rapm",
        *(f"{candidate}_aio" for candidate in CANDIDATES),
    )
    base.PRIMARY_PAIRS = {
        frozenset(("box_15", "box_15_plus_gravity")),
        frozenset(("box_15_aio", "box_15_plus_gravity_aio")),
        frozenset(("full_spm", "full_spm_plus_gravity")),
        frozenset(("full_spm_aio", "full_spm_plus_gravity_aio")),
    }
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, prior_coverage = base._score_models(priors, annual, MATRIX_ROOT)
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games, draws=5000, seed=20260829
    )
    sources = {
        "contract": CONTRACT,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "gravity_run": GRAVITY_RUN / "run.json",
        "gravity_panel": GRAVITY_RUN / "gravity_panel.parquet",
        "runner": Path(__file__),
        **{
            f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json"
            for season in base.RATING_SEASONS
        },
        **{
            f"possessions_{season}": POSSESSION_CACHE
            / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_order": list(CANDIDATES),
        "model_order": list(base.MODEL_ORDER),
        "bootstrap_draws": 5000,
        "bootstrap_seed": 20260829,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/gravity_spm_challenger" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "feature_coverage.parquet": coverage,
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "alpha_selection.parquet": alphas,
        "coefficients.parquet": coefficients,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
        "prior_coverage.parquet": prior_coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in files.items():
        frame.to_parquet(output / name, index=False)
    primary = paired.loc[paired["primary_comparison"]].copy()
    run = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "public_model_changed": False,
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "season_2027_loaded": False,
        },
        "primary_comparisons": primary.to_dict("records"),
        "files": {
            name: {"rows": len(frame), "sha256": sha256_file(output / name)}
            for name, frame in files.items()
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
