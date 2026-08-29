#!/usr/bin/env python3
"""Validate the expanded Box15 history on identical current-era games."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit,
    _select_alpha,
)
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _prior_frame

import run_final_box_feature_ladder as ladder
import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "historical_box15_validation_v1"
CONTRACT = ROOT / "research/experiments/historical_box15_validation_v1.yml"
HISTORICAL_RUN = (
    ROOT
    / "artifacts/research/historical_box15_extension"
    / "historical_box15_extension_v1_08ff4c34ff"
)
OUTPUT_ROOT = ROOT / "artifacts/research/historical_box15_validation"
CANDIDATES = ("box15_original_history", "box15_expanded_history")


def _panel(features: Path, targets: Path) -> pd.DataFrame:
    feature = pd.read_parquet(features)
    target = pd.read_parquet(targets)
    panel = feature.merge(
        target,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def _fit_priors(original: pd.DataFrame, expanded: pd.DataFrame):
    panels = {
        "box15_original_history": original,
        "box15_expanded_history": expanded,
    }
    prior_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    alpha_rows: list[dict] = []
    for season in base.RATING_SEASONS:
        for candidate, panel in panels.items():
            train = panel.loc[panel["Window_End"].lt(season)].copy()
            test = panel.loc[panel["Window_End"].eq(season)].copy()
            if train["Window_End"].nunique() < 3 or test.empty:
                raise ValueError(f"{candidate} lacks history for {season}.")
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                target = f"target_{side}"
                alpha = _select_alpha(
                    train.rename(columns={"Window_End": "Season"}),
                    BOX_PIPM_STYLE_FEATURES,
                    target,
                    ladder.ALPHA_GRID,
                )
                model = _fit(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
                prediction = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
                prior[side] = prediction
                alpha_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "selected_alpha": alpha,
                        "training_window_min": int(train["Window_End"].min()),
                        "training_window_max": int(train["Window_End"].max()),
                        "training_windows": int(train["Window_End"].nunique()),
                    }
                )
                metric_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "component": side,
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prediction,
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
            prior["net"] = prior["offense"] + prior["defense"]
            metric_rows.append(
                {
                    "rating_season": season,
                    "candidate": candidate,
                    "component": "net",
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["net"].to_numpy(dtype=float),
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prior_rows.append(_prior_frame(prior, candidate))
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(alpha_rows),
    )


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    original, _ = base._load_panel(
        ladder.FEATURE_RUN / "five_year_features.parquet",
        ladder.TARGETS,
        ladder.FEATURE_RUN / "run.json",
        (),
    )
    expanded = _panel(
        HISTORICAL_RUN / "five_year_box15_features.parquet",
        HISTORICAL_RUN / "five_year_targets.parquet",
    )
    priors, target_metrics, alphas = _fit_priors(original, expanded)

    annual, reconstruction = base._annual_bundles(
        ladder.POSSESSION_CACHE, ladder.MATRIX_ROOT
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
    ratings, games, coverage = base._score_models(
        priors, annual, ladder.MATRIX_ROOT
    )
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games, draws=ladder.BOOTSTRAP_DRAWS, seed=ladder.BOOTSTRAP_SEED
    )

    original_aio = "box15_original_history_aio"
    expanded_aio = "box15_expanded_history_aio"
    interval_lookup = intervals.set_index("candidate")
    summary_lookup = summary.set_index("candidate")
    rmse_increase = float(
        interval_lookup.loc[expanded_aio, "equal_season_rmse"]
        - interval_lookup.loc[original_aio, "equal_season_rmse"]
    )
    correlation_decline = float(
        summary_lookup.loc[original_aio, "mean_margin_correlation"]
        - summary_lookup.loc[expanded_aio, "mean_margin_correlation"]
    )
    gate = contract["retention_gate"]
    retention = pd.DataFrame(
        [
            {
                "comparison": "expanded_minus_original",
                "equal_season_rmse_increase": rmse_increase,
                "mean_margin_correlation_decline": correlation_decline,
                "maximum_allowed_rmse_increase": float(
                    gate["maximum_equal_season_rmse_increase"]
                ),
                "maximum_allowed_correlation_decline": float(
                    gate["maximum_mean_margin_correlation_decline"]
                ),
                "passes_retention_gate": (
                    rmse_increase
                    <= float(gate["maximum_equal_season_rmse_increase"])
                    and correlation_decline
                    <= float(gate["maximum_mean_margin_correlation_decline"])
                ),
            }
        ]
    )

    source_paths = {
        "contract": CONTRACT,
        "historical_run": HISTORICAL_RUN / "run.json",
        "historical_features": HISTORICAL_RUN / "five_year_box15_features.parquet",
        "historical_targets": HISTORICAL_RUN / "five_year_targets.parquet",
        "original_features": ladder.FEATURE_RUN / "five_year_features.parquet",
        "original_targets": ladder.TARGETS,
        "runner": Path(__file__),
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
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
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
        "retention_gate.parquet": retention,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "quality": {
            "expanded_training_window_ends": [
                int(expanded["Window_End"].min()),
                int(expanded["Window_End"].max()),
            ],
            "retention_gate_passed": bool(retention.iloc[0]["passes_retention_gate"]),
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "season_2027_loaded": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashes[name],
            }
            for name, path in source_paths.items()
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
    print(retention.to_string(index=False))
    print(
        intervals.loc[intervals["candidate"].str.endswith("_aio")]
        .sort_values("equal_season_mse")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
