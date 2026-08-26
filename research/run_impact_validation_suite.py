#!/usr/bin/env python3
"""Run the frozen multi-test validation suite for statistical RAPM priors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.impact_validation_suite import (
    DEFAULT_TEST_WEIGHTS,
    build_adjacent_annual_metrics,
    composite_ranking,
    evaluate_midseason_adaptation,
    game_metrics,
    paired_game_mse_intervals,
)
from nba_impact.models.rapm import RapmConfig, load_legacy_possessions


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "impact_validation_suite_v1"
COMPOSITE_CANDIDATES = (
    "five_year_spm",
    "selected_five_year_spm",
    "box_pipm_style_prior",
)
ALL_GAME_CANDIDATES = (*COMPOSITE_CANDIDATES, "zero_prior_rapm")


def _load_priors(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    source = pd.concat(frames, ignore_index=True)
    required = {
        "PLAYER_ID",
        "Window_End",
        "candidate",
        "prior_offense_per_100",
        "prior_defense_per_100",
        "prior_net_per_100",
    }
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"Prior artifacts are missing {missing}.")
    source = source.loc[source["candidate"].isin(COMPOSITE_CANDIDATES)].copy()
    source = source.drop_duplicates(
        ["PLAYER_ID", "Window_End", "candidate"], keep="last"
    )
    source = source.rename(
        columns={
            "Window_End": "Season",
            "prior_offense_per_100": "offense",
            "prior_defense_per_100": "defense",
            "prior_net_per_100": "net",
        }
    )
    return source[["PLAYER_ID", "Season", "candidate", "offense", "defense", "net"]]


def _load_targets(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    if "variant" in source:
        source = source.loc[source["variant"].eq("sqrt_possessions")].copy()
    required = {
        "PLAYER_ID",
        "Season",
        "Poss_Off",
        "Poss_Def",
        "target_offense",
        "target_defense",
        "target_net",
    }
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"Annual target artifact is missing {missing}.")
    source["sample_weight"] = (
        source[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1).pow(0.5)
    )
    return source[
        [
            "PLAYER_ID",
            "Season",
            "sample_weight",
            "target_offense",
            "target_defense",
            "target_net",
        ]
    ].drop_duplicates(["PLAYER_ID", "Season"])


def _load_ages(root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        path = root / f"{season}.parquet"
        frame = pd.read_parquet(path, columns=["PLAYER_ID", "AGE"])
        frame = frame.dropna(subset=["PLAYER_ID", "AGE"]).drop_duplicates("PLAYER_ID")
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
        frame["AGE"] = pd.to_numeric(frame["AGE"], errors="raise")
        frame["Season"] = season
        rows.append(frame[["PLAYER_ID", "Season", "AGE"]])
    return pd.concat(rows, ignore_index=True)


def _load_next_season_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["candidate"].isin(ALL_GAME_CANDIDATES)].copy()
    frame = frame.rename(columns={"test_season": "season"})
    required = {
        "game_id",
        "actual_margin",
        "predicted_margin",
        "candidate",
        "season",
        "rating_season",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Next-season predictions are missing {missing}.")
    if frame.duplicated(["candidate", "season", "game_id"]).any():
        raise ValueError("Next-season predictions contain duplicate candidate games.")
    actual_counts = frame.groupby(["season", "game_id"])["actual_margin"].nunique()
    if actual_counts.gt(1).any():
        raise ValueError("Candidates disagree on a game target.")
    coverage = frame.groupby(["candidate", "season"])["game_id"].nunique().unstack()
    if coverage.nunique(axis=0).gt(1).any():
        raise ValueError("Candidates must score identical games in every season.")
    frame["test_id"] = "next_season_game_margin"
    return frame


def _game_fold_metrics(frame: pd.DataFrame, test_id: str) -> pd.DataFrame:
    rows = []
    for (candidate, season), fold in frame.groupby(["candidate", "season"], sort=True):
        rows.append(
            {
                "test_id": test_id,
                "candidate": candidate,
                "season": int(season),
                **game_metrics(fold),
            }
        )
    return pd.DataFrame(rows)


def _composite_input(
    next_game: pd.DataFrame,
    midseason: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for source, test_id in (
        (next_game, "next_season_game_margin"),
        (midseason, "midseason_adaptation"),
    ):
        for row in source.loc[source["candidate"].isin(COMPOSITE_CANDIDATES)].itertuples():
            rows.append(
                {
                    "test_id": test_id,
                    "candidate": row.candidate,
                    "fold": int(row.season),
                    "value": float(row.mse),
                    "higher_is_better": False,
                }
            )
    variants = {
        "same_season_rapm_fit": "raw",
        "forward_annual_impact": "aging_adjusted",
        "reverse_annual_impact": "aging_adjusted",
    }
    for test_id, variant in variants.items():
        selected = annual.loc[
            annual["test_id"].eq(test_id)
            & annual["variant"].eq(variant)
            & annual["component"].eq("net")
            & annual["candidate"].isin(COMPOSITE_CANDIDATES)
        ]
        for row in selected.itertuples():
            rows.append(
                {
                    "test_id": test_id,
                    "candidate": row.candidate,
                    "fold": int(row.season),
                    "value": float(row.correlation),
                    "higher_is_better": True,
                }
            )
    scores = pd.DataFrame(rows).dropna(subset=["value"])
    for test_id, fold in scores.groupby("test_id"):
        expected = set(COMPOSITE_CANDIDATES)
        for season, season_frame in fold.groupby("fold"):
            if set(season_frame["candidate"]) != expected:
                raise ValueError(f"{test_id} fold {season} lacks a complete candidate set.")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--possession-cache", type=Path, required=True)
    parser.add_argument(
        "--prior-artifacts",
        type=Path,
        nargs="+",
        default=[
            ROOT / "artifacts/research/aio_prior_bakeoff/aio_prior_bakeoff_v1_0a3591a402/priors.parquet",
            ROOT / "artifacts/research/aio_prior_canonical_followup/aio_prior_canonical_followup_v1_8c61405875/priors.parquet",
        ],
    )
    parser.add_argument(
        "--next-game-artifacts",
        type=Path,
        nargs="+",
        default=[
            ROOT / "artifacts/research/aio_prior_bakeoff/aio_prior_bakeoff_v1_0a3591a402/game_predictions.parquet",
            ROOT / "artifacts/research/aio_prior_canonical_followup/aio_prior_canonical_followup_v1_8c61405875/game_predictions.parquet",
        ],
    )
    parser.add_argument(
        "--annual-targets",
        type=Path,
        default=ROOT / "artifacts/research/spm_weight_ablation/spm_weight_ablation_v1_9a4136a6d7/oof_predictions.parquet",
    )
    parser.add_argument(
        "--age-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
    )
    parser.add_argument("--midseason-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if 2027 in args.midseason_seasons:
        raise ValueError("Season 2027 is reserved and cannot enter this suite.")

    priors = _load_priors(args.prior_artifacts)
    targets = _load_targets(args.annual_targets)
    age_seasons = tuple(sorted(targets["Season"].unique().astype(int)))
    ages = _load_ages(args.age_root, age_seasons)
    annual_metrics, annual_matches = build_adjacent_annual_metrics(
        priors,
        targets,
        candidates=COMPOSITE_CANDIDATES,
        ages=ages,
    )

    mid_metric_frames: list[pd.DataFrame] = []
    mid_prediction_frames: list[pd.DataFrame] = []
    mid_coverage_frames: list[pd.DataFrame] = []
    for season in args.midseason_seasons:
        possessions = load_legacy_possessions(args.possession_cache, (season,))
        metrics, predictions, coverage = evaluate_midseason_adaptation(
            possessions,
            priors,
            season=season,
            candidates=ALL_GAME_CANDIDATES,
            config=RapmConfig(seasons=(season,)),
        )
        mid_metric_frames.append(metrics)
        mid_prediction_frames.append(predictions)
        mid_coverage_frames.append(coverage)
    mid_metrics = pd.concat(mid_metric_frames, ignore_index=True)
    mid_predictions = pd.concat(mid_prediction_frames, ignore_index=True)
    mid_coverage = pd.concat(mid_coverage_frames, ignore_index=True)

    next_predictions = _load_next_season_predictions(args.next_game_artifacts)
    next_metrics = _game_fold_metrics(next_predictions, "next_season_game_margin")
    score_input = _composite_input(next_metrics, mid_metrics, annual_metrics)
    composite, ranked_folds = composite_ranking(score_input)
    per_test = ranked_folds.groupby(["test_id", "candidate"], as_index=False).agg(
        folds=("fold", "nunique"),
        mean_value=("value", "mean"),
        mean_percentile_score=("percentile_score", "mean"),
    )
    per_test["priority"] = per_test["test_id"].map(
        {test_id: index + 1 for index, test_id in enumerate(DEFAULT_TEST_WEIGHTS)}
    )
    per_test["weight"] = per_test["test_id"].map(DEFAULT_TEST_WEIGHTS)
    per_test = per_test.sort_values(["priority", "mean_percentile_score"], ascending=[True, False])

    sources = {
        "contract": ROOT / "research/experiments/impact_validation_suite_v1.yml",
        "builder": ROOT / "src/nba_impact/models/impact_validation_suite.py",
        "runner": Path(__file__),
        "annual_targets": args.annual_targets,
        **{f"prior_{index}": path for index, path in enumerate(args.prior_artifacts)},
        **{f"next_game_{index}": path for index, path in enumerate(args.next_game_artifacts)},
        **{
            f"possession_{season}": args.possession_cache / f"matchups_{season}.parquet"
            for season in args.midseason_seasons
        },
        **{f"ages_{season}": args.age_root / f"{season}.parquet" for season in age_seasons},
    }
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity_payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_hashes": source_hashes,
        "midseason_seasons": args.midseason_seasons,
        "weights": DEFAULT_TEST_WEIGHTS,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = args.artifact_root / "research" / "impact_validation_suite" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)

    tables = {
        "composite_ranking": composite,
        "per_test_summary": per_test,
        "ranked_fold_scores": ranked_folds,
        "annual_metrics": annual_metrics,
        "annual_matches": annual_matches,
        "midseason_metrics": mid_metrics,
        "midseason_game_predictions": mid_predictions,
        "midseason_coverage": mid_coverage,
        "midseason_paired_intervals": paired_game_mse_intervals(mid_predictions),
        "next_season_game_metrics": next_metrics,
        "next_season_paired_intervals": paired_game_mse_intervals(next_predictions),
    }
    for name, table in tables.items():
        table.to_parquet(output / f"{name}.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "multi-test validation of frozen statistical priors",
        "config": {
            "candidates": list(COMPOSITE_CANDIDATES),
            "game_baseline": "zero_prior_rapm",
            "midseason_seasons": args.midseason_seasons,
            "test_weights": DEFAULT_TEST_WEIGHTS,
            "composite_method": "weighted_mean_within_fold_percentile_rank",
            "fixed_rapm": {"lambda_off": 3000.0, "lambda_def": 3000.0, "lambda_home": 300.0},
            "source_hashes": source_hashes,
        },
        "quality": {
            "next_season_games": int(next_predictions[["season", "game_id"]].drop_duplicates().shape[0]),
            "midseason_games": int(mid_predictions[["season", "game_id"]].drop_duplicates().shape[0]),
            "annual_matched_rows": int(len(annual_matches)),
            "season_2027_rows": 0,
        },
        "results": {
            "composite_ranking": composite.to_dict(orient="records"),
            "ordered_test_summary": per_test.to_dict(orient="records"),
        },
        "paths": {name: f"{name}.parquet" for name in tables},
        "caveats": [
            "All scored seasons are reused historical evidence, not independent confirmation.",
            "The composite averages within-fold ranks, never raw MSE, correlation, or R-squared.",
            "Observed future lineups isolate rating quality but do not create a deployable roster forecast.",
            "Reverse annual impact is a low-weight diagnostic with no production interpretation.",
            "The BoxPIPM-style candidate is a box-only reproduction, not the complete historical PIPM model.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(json.dumps({"run_id": run["run_id"], "ranking": run["results"]["composite_ranking"]}, indent=2))


if __name__ == "__main__":
    main()
