"""Build 1/3/5/6/full RAPM target panels and compare next-season fit."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, load_current_player_names, load_unified_terminal_possessions


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rapm_target_horizon_bakeoff_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rapm_target_horizon_bakeoff"
REFERENCE_5Y = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/rolling_ratings.parquet"


def _ratings(design, beta, mask, names, *, horizon: str, start: int, end: int) -> pd.DataFrame:
    n = len(design.players)
    off_counts = np.asarray(design.X[mask, :n].sum(axis=0)).ravel()
    def_counts = np.asarray(design.X[mask, n : 2 * n].sum(axis=0)).ravel()
    offense = 100.0 * beta[:n]
    defense = -100.0 * beta[n : 2 * n]
    frame = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": offense,
            "defense": defense,
            "net": offense + defense,
            "Poss_Off": off_counts,
            "Poss_Def": def_counts,
            "horizon": horizon,
            "window_start": start,
            "window_end": end,
        }
    )
    frame = frame.loc[frame[["Poss_Off", "Poss_Def"]].min(axis=1).gt(0)]
    return frame.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(contract["seasons"])
    started = time.perf_counter()
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    original_design = build_design(frame, include_home=True)
    season_mean = frame.groupby("season")["pts"].mean()
    adjusted_y = frame["pts"].to_numpy(dtype=float) - frame["season"].map(season_mean).to_numpy(dtype=float)
    design = replace(original_design, y=adjusted_y)
    penalties = contract["penalties"]
    config = RapmConfig(
        seasons=seasons,
        lambda_off=penalties["lambda_off"],
        lambda_def=penalties["lambda_def"],
        lambda_home=penalties["lambda_home"],
        data_scope="rapm_target_horizon_bakeoff",
    )
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv",
        ROOT / "data/lake/silver/player_games.parquet",
    )
    rating_frames = []
    evaluation_rows = []
    common_tests = set(contract["common_evaluation_seasons"])
    for horizon in contract["rolling_horizons"]:
        for end in range(min(seasons) + horizon - 1, max(seasons) + 1):
            start = end - horizon + 1
            train = (design.seasons >= start) & (design.seasons <= end)
            beta, _ = fit_coefficients(design, config, row_mask=train)
            rating_frames.append(
                _ratings(design, beta, train, names, horizon=f"{horizon}y", start=start, end=end)
            )
            test_season = end + 1
            if test_season in common_tests:
                test = design.seasons == test_season
                prediction = float(original_design.y[train].mean()) + np.asarray(design.X[test] @ beta).ravel()
                metrics, _ = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
                evaluation_rows.append(
                    {"horizon": f"{horizon}y", "window_start": start, "window_end": end, "test_season": test_season, **metrics}
                )
            print(f"{horizon}y {start}-{end}", flush=True)
    full_start, full_end = contract["full_span"]
    full = (design.seasons >= full_start) & (design.seasons <= full_end)
    full_beta, _ = fit_coefficients(design, config, row_mask=full)
    rating_frames.append(
        _ratings(design, full_beta, full, names, horizon="full", start=full_start, end=full_end)
    )
    ratings = pd.concat(rating_frames, ignore_index=True)
    evaluation = pd.DataFrame(evaluation_rows)
    summary = evaluation.groupby("horizon", as_index=False).agg(
        seasons=("test_season", "nunique"),
        mean_margin_rmse=("margin_rmse", "mean"),
        mean_margin_mae=("margin_mae", "mean"),
        mean_margin_correlation=("margin_correlation", "mean"),
    ).sort_values(["mean_margin_rmse", "horizon"], kind="stable")
    reference = pd.read_parquet(REFERENCE_5Y)
    five = ratings.loc[ratings["horizon"].eq("5y")]
    parity = five.merge(
        reference[["PLAYER_ID", "window_end", "offense", "defense", "net"]],
        on=["PLAYER_ID", "window_end"],
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    maximum_5y_error = max(
        float((parity[component] - parity[f"{component}_reference"]).abs().max())
        for component in ("offense", "defense", "net")
    )
    if maximum_5y_error > 1e-6:
        raise AssertionError("Five-year horizon does not reproduce the validated rolling run.")
    identity = hashlib.sha256(json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"rapm_target_horizon_bakeoff_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    evaluation.to_parquet(output / "evaluation.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "summary": summary.to_dict("records"),
        "quality": {
            "possessions": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "rating_rows": int(len(ratings)),
            "maximum_5y_reference_error": maximum_5y_error,
        },
        "paths": {"ratings": "ratings.parquet", "evaluation": "evaluation.parquet", "summary": "summary.parquet"},
        "forbidden_interpretation": "Untouched confirmation, an SPM feature-model comparison, or proof that a horizon is universally optimal.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
