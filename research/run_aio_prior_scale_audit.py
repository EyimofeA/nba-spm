#!/usr/bin/env python3
"""Chronologically tune AIO prior-center strength on saved evaluation games."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from run_aio_prior_canonical_followup import _center, _remap_annual, _solve
from run_full_spm_history_ablation import _annual_bundles


ROOT = Path(__file__).resolve().parents[1]
SCALES = (0.25, 0.5, 0.75, 1.0)
CANDIDATES = ("full_spm", "box_pipm")


def _bootstrap(games: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    names = tuple(sorted(games["candidate"].unique()))
    by_season = []
    for _, frame in games.groupby("test_season", sort=True):
        wide = frame.pivot(index="game_id", columns="candidate", values="squared_error")
        if tuple(sorted(wide.columns)) != names or wide.isna().any().any():
            raise ValueError("Scale candidates must score identical games.")
        by_season.append(wide.loc[:, names].to_numpy(dtype=float))
    point = np.mean([values.mean(axis=0) for values in by_season], axis=0)
    rng = np.random.default_rng(seed)
    sampled = np.empty((draws, len(names)))
    for draw in range(draws):
        sampled[draw] = np.mean(
            [
                values[rng.integers(0, len(values), len(values))].mean(axis=0)
                for values in by_season
            ],
            axis=0,
        )
    rows = []
    for left, right in itertools.combinations(range(len(names)), 2):
        delta = sampled[:, left] - sampled[:, right]
        low, high = np.quantile(delta, [0.025, 0.975])
        rows.append(
            {
                "candidate": names[left],
                "reference": names[right],
                "mean_mse_delta": float(point[left] - point[right]),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(delta < 0)),
                "draws": draws,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/full_spm_history_ablation"
            / "full_spm_history_ablation_v1_2eb5eb428c"
        ),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=(
            ROOT
            / "research/rapm_lab/outputs/rolling_5y_2014_2026"
            / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
        ),
    )
    parser.add_argument(
        "--possession-cache",
        type=Path,
        default=ROOT / "rapm/data/possession_cache",
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/research")
    args = parser.parse_args()

    priors_path = args.source_run / "priors.parquet"
    manifest_path = args.source_run / "run.json"
    if not priors_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("The source refit needs priors.parquet and run.json.")
    priors = pd.read_parquet(priors_path)
    annual, _ = _annual_bundles(args.possession_cache, args.matrix_root)

    grid_rows: list[pd.DataFrame] = []
    for rating_season in range(2021, 2026):
        matrix_dir = args.matrix_root / f"5y_end_{rating_season}"
        players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[rating_season], players)
        for candidate in CANDIDATES:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(rating_season)
            ]
            center, _ = _center(prior, bundle)
            for scale in SCALES:
                beta, intercept = _solve(bundle, center, scale=scale)
                frame = stored_evaluation_predictions(matrix_dir, beta, intercept)
                frame["prior"] = candidate
                frame["scale"] = scale
                frame["rating_season"] = rating_season
                frame["test_season"] = rating_season + 1
                frame["squared_error"] = (
                    frame["actual_margin"] - frame["predicted_margin"]
                ) ** 2
                grid_rows.append(frame)
    grid = pd.concat(grid_rows, ignore_index=True)

    selection_rows = []
    scored_rows = []
    for rating_season in range(2022, 2026):
        for prior in CANDIDATES:
            history = grid.loc[
                grid["prior"].eq(prior) & grid["rating_season"].lt(rating_season)
            ]
            # Each outcome season receives equal weight. Otherwise a season
            # with more stored games would have more influence on selection.
            scale_mse = (
                history.groupby(["scale", "test_season"])["squared_error"]
                .mean()
                .groupby("scale")
                .mean()
            )
            selected_scale = float(scale_mse.idxmin())
            selection_rows.append(
                {
                    "rating_season": rating_season,
                    "prior": prior,
                    "selected_scale": selected_scale,
                    "selection_rating_seasons": ",".join(
                        map(str, sorted(history["rating_season"].unique()))
                    ),
                    "selection_games": int(history["game_id"].nunique()),
                }
            )
            for label, scale in (("nested", selected_scale), ("unit", 1.0)):
                scored = grid.loc[
                    grid["prior"].eq(prior)
                    & grid["rating_season"].eq(rating_season)
                    & grid["scale"].eq(scale)
                ].copy()
                scored["candidate"] = f"{prior}_{label}"
                scored_rows.append(scored)
    games = pd.concat(scored_rows, ignore_index=True)
    folds = (
        games.groupby(["candidate", "test_season"], as_index=False)
        .agg(games=("game_id", "size"), mse=("squared_error", "mean"))
    )
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            games=("games", "sum"),
            equal_season_mse=("mse", "mean"),
        )
        .sort_values("equal_season_mse")
    )
    summary["rmse"] = np.sqrt(summary["equal_season_mse"])
    intervals = _bootstrap(games, draws=args.draws, seed=args.seed)
    unknown = (
        games.loc[games["candidate"].eq("full_spm_nested")]
        .groupby("test_season", as_index=False)
        .agg(
            games=("game_id", "nunique"),
            games_with_unknown_players=("unknown_player_slots", lambda x: int((x > 0).sum())),
            unknown_player_slots=("unknown_player_slots", "sum"),
        )
    )
    unknown["possession_rows"] = unknown["test_season"].map(
        {
            season + 1: json.loads(
                (args.matrix_root / f"5y_end_{season}/manifest.json").read_text()
            )["evaluation"]["possession_rows"]
            for season in range(2022, 2026)
        }
    )
    unknown["unknown_slot_fraction"] = (
        unknown["unknown_player_slots"] / (10.0 * unknown["possession_rows"])
    )

    source_hashes = {
        "source_manifest": sha256_file(manifest_path),
        "source_priors": sha256_file(priors_path),
        "runner": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"aio_prior_scale_audit_v1_{identity}"
    output = args.artifact_root / "aio_prior_scale_audit" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(selection_rows).to_parquet(output / "selections.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    intervals.to_parquet(output / "paired_intervals.parquet", index=False)
    unknown.to_parquet(output / "unknown_player_exposure.parquet", index=False)
    games.to_parquet(output / "game_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": json.loads(manifest_path.read_text())["run_id"],
        "scale_grid": list(SCALES),
        "selection_rule": "For each rating season, select the scale on all strictly earlier stored next-season games.",
        "scored_test_seasons": [2023, 2024, 2025, 2026],
        "season_2027_loaded": False,
        "source_hashes": source_hashes,
        "forbidden_interpretation": "Reused folds cannot promote a public model.",
        "paths": {
            "selections": "selections.parquet",
            "summary": "summary.parquet",
            "paired_intervals": "paired_intervals.parquet",
            "unknown_player_exposure": "unknown_player_exposure.parquet",
            "game_predictions": "game_predictions.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print(pd.DataFrame(selection_rows).to_string(index=False))
    print(intervals.to_string(index=False))
    print(unknown.to_string(index=False))


if __name__ == "__main__":
    main()
