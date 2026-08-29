#!/usr/bin/env python3
"""Test one shooting-threat addition against the frozen Box15 AIO prior."""

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
EXPERIMENT_ID = "box15_shooting_threat_v1"
CONTRACT = ROOT / "research/experiments/box15_shooting_threat_v1.yml"
FEATURE_RUN = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f"
TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
SHOT_RUN = ROOT / "artifacts/research/shot_model_suite/shot_model_suite_v1_2494cca535"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
CANDIDATES = ("box_15", "box_15_plus_shooting_threat")


def _contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("The experiment contract changed.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def main() -> None:
    contract = _contract()
    panel, _ = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    threat = pd.read_parquet(SHOT_RUN / "five_year_shooting_threat.parquet")[
        ["PLAYER_ID", "Window_End", "shooting_threat_p100"]
    ]
    panel = panel.merge(
        threat,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    coverage = (
        panel.assign(observed=panel["shooting_threat_p100"].notna())
        .groupby("Window_End", as_index=False)
        .agg(players=("PLAYER_ID", "size"), observed=("observed", "sum"))
    )
    coverage["coverage"] = coverage["observed"] / coverage["players"]
    panel["shooting_threat_p100"] = panel["shooting_threat_p100"].fillna(0.0)
    candidates = {
        "box_15": {
            "offense": BOX_PIPM_STYLE_FEATURES,
            "defense": BOX_PIPM_STYLE_FEATURES,
        },
        "box_15_plus_shooting_threat": {
            "offense": (*BOX_PIPM_STYLE_FEATURES, "shooting_threat_p100"),
            "defense": BOX_PIPM_STYLE_FEATURES,
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
        frozenset(CANDIDATES),
        frozenset(f"{candidate}_aio" for candidate in CANDIDATES),
    }
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, prior_coverage = base._score_models(priors, annual, MATRIX_ROOT)
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games, draws=5000, seed=20260829
    )
    box = summary.set_index("candidate")
    challenger = "box_15_plus_shooting_threat_aio"
    control = "box_15_aio"
    selected = challenger if (
        box.loc[challenger, "mean_margin_rmse"] < box.loc[control, "mean_margin_rmse"]
        and box.loc[challenger, "mean_margin_correlation"]
        >= box.loc[control, "mean_margin_correlation"] - 0.01
    ) else control
    sources = {
        "contract": CONTRACT,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "shot_run": SHOT_RUN / "run.json",
        "shooting_threat": SHOT_RUN / "five_year_shooting_threat.parquet",
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "candidates": list(CANDIDATES),
        "model_order": list(base.MODEL_ORDER),
        "bootstrap_draws": 5000,
        "bootstrap_seed": 20260829,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/box15_shooting_threat" / f"{EXPERIMENT_ID}_{identity}"
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
    run = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_under_reused_gate": selected,
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
        "files": {
            name: {"rows": len(frame), "sha256": sha256_file(output / name)}
            for name, frame in files.items()
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print(paired.loc[paired["primary_comparison"]].to_string(index=False))


if __name__ == "__main__":
    main()
