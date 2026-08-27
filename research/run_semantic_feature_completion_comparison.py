#!/usr/bin/env python3
"""Compare fully finite SPM panels with current and BoxPIPM baselines."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "semantic_feature_completion_comparison_v1"
FEATURE_RUN = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_8be676bd0f"
)
TARGETS = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
OLD_RUN = (
    ROOT
    / "artifacts/research/full_spm_history_ablation"
    / "full_spm_history_ablation_v1_2eb5eb428c"
)
MATRIX_ROOT = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
MATCHUP_FEATURES = (
    "matchup_opponent_adjusted_points_saved_p100_eb",
    "matchup_fga_suppressed_vs_scorer_p100_eb",
    "matchup_shotmaking_points_saved_vs_scorer_p100_eb",
    "matchup_three_pa_suppressed_vs_scorer_p100_eb",
    "matchup_turnovers_forced_vs_scorer_p100_eb",
    "matchup_assists_suppressed_vs_scorer_p100_eb",
    "matchup_shooting_fouls_prevented_vs_scorer_p100_eb",
    "matchup_blocks_p100",
)
MODEL_ORDER = (
    "complete_spm",
    "no_matchup_spm",
    "current_missing_spm",
    "box_pipm",
    "zero_prior_rapm",
    "complete_spm_aio",
    "no_matchup_spm_aio",
    "current_missing_spm_aio",
    "box_pipm_aio",
)
PRIMARY_PAIRS = {
    frozenset(("complete_spm", "no_matchup_spm")),
    frozenset(("complete_spm", "current_missing_spm")),
    frozenset(("complete_spm", "box_pipm")),
    frozenset(("complete_spm_aio", "no_matchup_spm_aio")),
    frozenset(("complete_spm_aio", "current_missing_spm_aio")),
    frozenset(("complete_spm_aio", "box_pipm_aio")),
}


def _rename_new(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        candidate=frame["candidate"].replace(
            {
                "full_spm": "complete_spm",
                "full_spm_aio": "complete_spm_aio",
                "history_complete_spm": "no_matchup_spm",
                "history_complete_spm_aio": "no_matchup_spm_aio",
            }
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
    panel, selected = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        MATCHUP_FEATURES,
    )
    priors, target_metrics, selections, _models = base._fit_priors(
        panel, selected, MATCHUP_FEATURES
    )
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, new_games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    priors = _rename_new(priors)
    target_metrics = _rename_new(target_metrics)
    selections = _rename_new(selections)
    ratings = _rename_new(ratings)
    coverage = _rename_new(coverage)
    new_games = _rename_new(new_games)

    old_games = pd.read_parquet(OLD_RUN / "game_predictions.parquet")
    old_games = old_games.loc[
        old_games["candidate"].isin(("full_spm", "full_spm_aio"))
    ].copy()
    old_games["candidate"] = old_games["candidate"].replace(
        {
            "full_spm": "current_missing_spm",
            "full_spm_aio": "current_missing_spm_aio",
        }
    )
    games = pd.concat([new_games, old_games], ignore_index=True)
    _assert_identical_games(games)

    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = PRIMARY_PAIRS
    folds, summary = base._game_metrics_frames(games)
    intervals, pairs = base.paired_game_bootstrap(
        games, draws=5000, seed=20260827
    )

    sources = {
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "old_game_predictions": OLD_RUN / "game_predictions.parquet",
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(base.RATING_SEASONS),
        "evaluated_rating_seasons": list(base.EVALUATED_RATING_SEASONS),
        "season_2027": "forbidden",
        "matchup_features_removed": list(MATCHUP_FEATURES),
        "model_order": list(MODEL_ORDER),
        "bootstrap": {
            "draws": 5000,
            "seed": 20260827,
            "unit": "whole game within test season",
            "aggregation": "equal-season mean MSE",
        },
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        ROOT
        / "artifacts/research/semantic_feature_completion_comparison"
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
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season game margin from prior-season player ratings",
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "panel_missing_values": int(
                panel[list(selected["offense"] + selected["defense"])]
                .isna()
                .sum()
                .sum()
            ),
            "identical_games_within_fold": True,
            "season_2027_loaded": False,
        },
        "files": {},
        "forbidden_interpretation": (
            "All five test seasons are reused evidence. This run cannot promote a public model."
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
