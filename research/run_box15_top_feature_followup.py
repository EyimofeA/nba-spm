#!/usr/bin/env python3
"""Test the strongest audited non-Box15 fields in the final AIO."""

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
EXPERIMENT_ID = "box15_top_feature_followup_v1"
CONTRACT = ROOT / "research/experiments/box15_top_feature_followup_v1.yml"
OUTPUT_ROOT = ROOT / "artifacts/research/box15_top_feature_followup"


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment ID changed.")
    if contract.get("status") != "frozen_reused_diagnostic":
        raise ValueError("The follow-up must remain a reused diagnostic.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _candidate_features(contract: dict, selected: dict[str, tuple[str, ...]]):
    box = tuple(BOX_PIPM_STYLE_FEATURES)
    offense = tuple(contract["features"]["offense"])
    defense = tuple(contract["features"]["defense"])
    for side, fields in (("offense", offense), ("defense", defense)):
        if missing := sorted(set(fields) - set(selected[side])):
            raise ValueError(f"Selected {side} fields are missing: {missing}")
    return {
        "box_15": {"offense": box, "defense": box},
        "box_plus_top_offense": {
            "offense": tuple(dict.fromkeys((*box, *offense))),
            "defense": box,
        },
        "box_plus_top_defense": {
            "offense": box,
            "defense": tuple(dict.fromkeys((*box, *defense))),
        },
        "box_plus_top_both": {
            "offense": tuple(dict.fromkeys((*box, *offense))),
            "defense": tuple(dict.fromkeys((*box, *defense))),
        },
    }


def main() -> None:
    contract = _load_contract()
    panel, selected = base._load_panel(
        ladder.FEATURE_RUN / "five_year_features.parquet",
        ladder.TARGETS,
        ladder.FEATURE_RUN / "run.json",
        (),
    )
    candidates = _candidate_features(contract, selected)
    expected = tuple(contract["candidates"])
    if tuple(candidates) != expected:
        raise ValueError("Candidate order differs from the frozen contract.")

    priors, target_metrics, alphas, _, _ = ladder._fit_priors(panel, candidates)
    annual, reconstruction = base._annual_bundles(
        ladder.POSSESSION_CACHE, ladder.MATRIX_ROOT
    )
    base.PRIOR_CANDIDATES = expected
    base.MODEL_ORDER = ladder._model_order(expected)
    base.PRIMARY_PAIRS = {
        frozenset(("box_15_aio", f"{candidate}_aio"))
        for candidate in expected
        if candidate != "box_15"
    }
    ratings, games, coverage = base._score_models(
        priors, annual, ladder.MATRIX_ROOT
    )
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games, draws=ladder.BOOTSTRAP_DRAWS, seed=ladder.BOOTSTRAP_SEED
    )

    aio = intervals.loc[intervals["candidate"].str.endswith("_aio")].merge(
        summary[["candidate", "mean_margin_correlation"]],
        on="candidate",
        validate="one_to_one",
    )
    box_mse = float(
        aio.loc[aio["candidate"].eq("box_15_aio"), "equal_season_mse"].iloc[0]
    )
    box_corr = float(
        aio.loc[
            aio["candidate"].eq("box_15_aio"), "mean_margin_correlation"
        ].iloc[0]
    )
    comparisons = paired.loc[
        paired["primary_comparison"]
        & paired["candidate"].eq("box_15_aio")
    ].copy()
    challenger_interval = {
        row.reference: (-float(row.bootstrap_95_high), -float(row.bootstrap_95_low))
        for row in comparisons.itertuples(index=False)
    }
    aio["point_mse_below_box"] = aio["equal_season_mse"].lt(box_mse)
    aio["passes_correlation_guard"] = aio["mean_margin_correlation"].ge(
        box_corr - 0.01
    )
    aio["paired_interval_below_zero_vs_box"] = [
        candidate != "box_15_aio"
        and challenger_interval[candidate][1] < 0
        for candidate in aio["candidate"]
    ]
    aio["passes_replacement_gate"] = (
        aio["point_mse_below_box"]
        & aio["passes_correlation_guard"]
        & aio["paired_interval_below_zero_vs_box"]
    )
    eligible = aio.loc[aio["passes_replacement_gate"]].sort_values(
        "equal_season_mse", kind="stable"
    )
    selected_aio = (
        "box_15_aio" if eligible.empty else str(eligible.iloc[0]["candidate"])
    )
    aio["selected"] = aio["candidate"].eq(selected_aio)

    sources = {
        "contract": CONTRACT,
        "features": ladder.FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": ladder.FEATURE_RUN / "run.json",
        "targets": ladder.TARGETS,
        "runner": Path(__file__),
    }
    config = {
        "contract_sha256": sha256_file(CONTRACT),
        "candidate_order": list(expected),
        "model_order": list(base.MODEL_ORDER),
        "alpha_grid": list(ladder.ALPHA_GRID),
        "bootstrap_draws": ladder.BOOTSTRAP_DRAWS,
        "bootstrap_seed": ladder.BOOTSTRAP_SEED,
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
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "alpha_selection.parquet": alphas,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
        "selection.parquet": aio,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_aio": selected_aio,
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "season_2027_loaded": False,
        },
        "files": {},
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(aio.sort_values("equal_season_mse").to_string(index=False))


if __name__ == "__main__":
    main()
