#!/usr/bin/env python3
"""Fit the corrected full and correlation-pruned five-year SPM contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "compact_spm_comparison_v1"
FEATURE_RUN = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e"
)
TARGETS = ROOT / (
    "artifacts/models/five_year_target_spm/"
    "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
COMPACT_CONFIG = ROOT / "configs/models/compact_spm_correlated_v1.json"
CONTRACT = ROOT / "research/experiments/compact_spm_comparison_v1.yml"
MATRIX_ROOT = ROOT / (
    "research/rapm_lab/outputs/rolling_5y_2014_2026/"
    "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
MODEL_ORDER = (
    "full_spm",
    "compact_spm",
    "box_pipm",
    "zero_prior_rapm",
    "full_spm_aio",
    "compact_spm_aio",
    "box_pipm_aio",
)
PRIMARY_PAIRS = {
    frozenset(("full_spm", "compact_spm")),
    frozenset(("full_spm", "box_pipm")),
    frozenset(("compact_spm", "box_pipm")),
    frozenset(("full_spm_aio", "compact_spm_aio")),
    frozenset(("full_spm_aio", "box_pipm_aio")),
    frozenset(("compact_spm_aio", "box_pipm_aio")),
}


def _rename(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    return frame.assign(
        candidate=frame["candidate"].replace(
            {source: target, f"{source}_aio": f"{target}_aio"}
        )
    )


def _assert_identical_games(games: pd.DataFrame) -> None:
    for season, frame in games.groupby("test_season"):
        counts = frame.groupby("candidate")["game_id"].nunique()
        if set(counts.index) != set(MODEL_ORDER) or counts.nunique() != 1:
            raise ValueError(f"Candidates do not score identical {season} games.")
        outcomes = frame.groupby("candidate").apply(
            lambda group: hashlib.sha256(
                "|".join(
                    sorted(
                        group["game_id"].astype(str)
                        + ":"
                        + group["actual_margin"].astype(str)
                    )
                ).encode()
            ).hexdigest(),
            include_groups=False,
        )
        if outcomes.nunique() != 1:
            raise ValueError(f"Candidates do not share {season} outcomes.")


def main() -> None:
    experiment = json.loads(
        json.dumps(yaml.safe_load(CONTRACT.read_text()), default=str)
    )
    if experiment.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")

    panel, full_features = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    compact = json.loads(COMPACT_CONFIG.read_text())
    compact_features = {
        side: tuple(
            feature
            for feature in full_features[side]
            if feature not in set(compact["dropped_features"][side])
        )
        for side in ("offense", "defense")
    }

    full_priors, full_targets, full_selection, full_models = base._fit_priors(
        panel, full_features, ()
    )
    compact_priors, compact_targets, compact_selection, compact_models = (
        base._fit_priors(panel, compact_features, ())
    )

    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    full_ratings, full_games, full_coverage = base._score_models(
        full_priors, annual, MATRIX_ROOT
    )
    compact_ratings, compact_games, compact_coverage = base._score_models(
        compact_priors, annual, MATRIX_ROOT
    )

    full_keep = {"full_spm", "box_pipm"}
    full_score_keep = {
        "full_spm",
        "box_pipm",
        "zero_prior_rapm",
        "full_spm_aio",
        "box_pipm_aio",
    }
    compact_prior = _rename(
        compact_priors.loc[compact_priors["candidate"].eq("full_spm")],
        "full_spm",
        "compact_spm",
    )
    priors = pd.concat(
        [full_priors.loc[full_priors["candidate"].isin(full_keep)], compact_prior],
        ignore_index=True,
    )
    target_metrics = pd.concat(
        [
            full_targets.loc[full_targets["candidate"].isin(full_keep)],
            _rename(
                compact_targets.loc[compact_targets["candidate"].eq("full_spm")],
                "full_spm",
                "compact_spm",
            ),
        ],
        ignore_index=True,
    )
    selections = pd.concat(
        [
            full_selection.loc[full_selection["candidate"].isin(full_keep)],
            _rename(
                compact_selection.loc[
                    compact_selection["candidate"].eq("full_spm")
                ],
                "full_spm",
                "compact_spm",
            ),
        ],
        ignore_index=True,
    )
    ratings = pd.concat(
        [
            full_ratings.loc[full_ratings["candidate"].isin(full_score_keep)],
            _rename(
                compact_ratings.loc[
                    compact_ratings["candidate"].isin(
                        ("full_spm", "full_spm_aio")
                    )
                ],
                "full_spm",
                "compact_spm",
            ),
        ],
        ignore_index=True,
    )
    games = pd.concat(
        [
            full_games.loc[full_games["candidate"].isin(full_score_keep)],
            _rename(
                compact_games.loc[
                    compact_games["candidate"].isin(
                        ("full_spm", "full_spm_aio")
                    )
                ],
                "full_spm",
                "compact_spm",
            ),
        ],
        ignore_index=True,
    )
    coverage = pd.concat(
        [
            full_coverage.loc[full_coverage["candidate"].isin(full_keep)],
            _rename(
                compact_coverage.loc[
                    compact_coverage["candidate"].eq("full_spm")
                ],
                "full_spm",
                "compact_spm",
            ),
        ],
        ignore_index=True,
    )

    _assert_identical_games(games)
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = PRIMARY_PAIRS
    folds, summary = base._game_metrics_frames(games)
    intervals, pairs = base.paired_game_bootstrap(
        games,
        draws=int(experiment["evaluation"]["uncertainty"]["draws"]),
        seed=int(experiment["evaluation"]["uncertainty"]["seed"]),
    )

    source_paths = {
        "contract": CONTRACT,
        "compact_config": COMPACT_CONFIG,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "runner": Path(__file__),
        "base_runner": ROOT / "research/run_full_spm_history_ablation.py",
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(base.RATING_SEASONS),
        "evaluated_rating_seasons": list(base.EVALUATED_RATING_SEASONS),
        "feature_counts": {
            "full_offense": len(full_features["offense"]),
            "full_defense": len(full_features["defense"]),
            "compact_offense": len(compact_features["offense"]),
            "compact_defense": len(compact_features["defense"]),
            "box": len(base.BOX_PIPM_STYLE_FEATURES),
        },
        "model_order": list(MODEL_ORDER),
        "rapm_penalties": {"offense": 3000, "defense": 3000, "home": 300},
        "center_scale": 1,
        "bootstrap": experiment["evaluation"]["uncertainty"],
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / (
        "artifacts/research/compact_spm_comparison/"
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
        "paired_bootstrap.parquet": pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)

    model_root = output / "models"
    model_root.mkdir()
    for (season, candidate, side), model in full_models.items():
        if candidate in full_keep:
            joblib.dump(model, model_root / f"{season}_{candidate}_{side}.joblib")
    for (season, candidate, side), model in compact_models.items():
        if candidate == "full_spm":
            joblib.dump(
                model, model_root / f"{season}_compact_spm_{side}.joblib"
            )

    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season game margin from prior-season player ratings",
        "contract": experiment,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "panel_missing_selected_values": int(
                panel[
                    list(
                        set(compact_features["offense"])
                        | set(compact_features["defense"])
                    )
                ]
                .isna()
                .sum()
                .sum()
            ),
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
        },
        "files": {},
        "forbidden_interpretation": (
            "The test seasons contain reused evidence and cannot promote a public model."
        ),
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
    print("\nPrimary comparisons")
    print(
        pairs.loc[pairs["primary_comparison"]]
        .sort_values("mean_mse_delta")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
