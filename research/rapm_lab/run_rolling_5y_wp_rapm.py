"""Build leakage-safe progress-WP states and rolling five-year WP-RAPM."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    load_unified_terminal_possessions,
    ratings_table,
)
from nba_impact.models.win_probability_rapm import build_conserved_wp_target


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rolling_5y_wp_rapm_2014_2026_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rolling_5y_wp_rapm"


def _states(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.sort_values(["gameid", "period", "num"], kind="stable").reset_index(drop=True).copy()
    home = result["home_poss"].astype(bool).to_numpy()
    points = result["pts"].to_numpy(dtype=float)
    result["_home_points"] = np.where(home, points, 0.0)
    result["_away_points"] = np.where(home, 0.0, points)
    grouped = result.groupby("gameid", sort=False)
    result["home_score_before"] = grouped["_home_points"].cumsum() - result["_home_points"]
    result["away_score_before"] = grouped["_away_points"].cumsum() - result["_away_points"]
    result["home_score_diff_before"] = result["home_score_before"] - result["away_score_before"]
    result["possession_index_before"] = grouped.cumcount().astype(int)
    finals = grouped[["_home_points", "_away_points"]].sum()
    home_win = finals["_home_points"].gt(finals["_away_points"])
    result["home_win"] = result["gameid"].map(home_win).astype(int)
    if result["home_win"].isna().any():
        raise ValueError("Every possession must resolve to a final game result.")
    progress = np.minimum(result["possession_index_before"].to_numpy(dtype=float) / 200.0, 1.25)
    remaining = np.maximum(200.0 - result["possession_index_before"].to_numpy(dtype=float), 1.0)
    score = result["home_score_diff_before"].to_numpy(dtype=float)
    features = pd.DataFrame(
        {
            "home_score_diff": score,
            "possession_progress": progress,
            "is_overtime": result["period"].gt(4).to_numpy(dtype=float),
            "score_pressure": score / np.sqrt(remaining),
            "score_late_interaction": score * progress,
            "home_possession": np.where(home, 1.0, -1.0),
        }
    )
    result["possession_id"] = (
        result["gameid"].astype(str)
        + "-"
        + result["period"].astype(str)
        + "-"
        + result["num"].astype(str)
    )
    if result["possession_id"].duplicated().any():
        result["possession_id"] += "-" + grouped.cumcount().astype(str)
    return result.drop(columns=["_home_points", "_away_points"]), features


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    start_source, end_source = (int(v) for v in contract["surface_source_seasons"])
    if end_source >= int(contract["untouched_confirmation_season"]):
        raise ValueError("Season 2027 must remain untouched.")
    started = time.perf_counter()
    source = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        tuple(range(start_source, end_source + 1)),
        transition_season=2024,
        game_types=("regular",),
    )
    states, features = _states(source)
    first_rating = int(contract["rating_start_season"])
    last_rating = int(contract["rating_end_season"])
    stride = int(contract["surface_training_stride"])
    probabilities = np.full(len(states), np.nan, dtype=float)
    surface_rows = []
    for season in range(first_rating, last_rating + 1):
        train = states["season"].lt(season).to_numpy()
        sampled = train & states["possession_index_before"].mod(stride).eq(0).to_numpy()
        test = states["season"].eq(season).to_numpy()
        model = LogisticRegression(
            C=float(contract["surface_logistic_c"]),
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(features.loc[sampled], states.loc[sampled, "home_win"])
        probability = model.predict_proba(features.loc[test])[:, 1]
        probabilities[test] = probability
        outcome = states.loc[test, "home_win"].to_numpy(dtype=int)
        surface_rows.append(
            {
                "season": season,
                "train_seasons": f"{start_source}-{season - 1}",
                "train_rows": int(sampled.sum()),
                "test_rows": int(test.sum()),
                "games": int(states.loc[test, "gameid"].nunique()),
                "brier": float(brier_score_loss(outcome, probability)),
                "log_loss": float(log_loss(outcome, probability)),
                "auc": float(roc_auc_score(outcome, probability)),
            }
        )
    rating_rows = states["season"].between(first_rating, last_rating).to_numpy()
    rating_frame = states.loc[rating_rows].copy().reset_index(drop=True)
    rating_frame["probability_context"] = probabilities[rating_rows]
    if rating_frame["probability_context"].isna().any():
        raise AssertionError("Every rating possession needs an earlier-season WP surface.")
    target, conservation = build_conserved_wp_target(rating_frame)
    target["pts"] = target["offense_wp_change"]
    design = build_design(target, include_home=True)
    window = int(contract["rating_window"])
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv",
        ROOT / "data/lake/silver/player_games.parquet",
    )
    ratings_rows = []
    window_rows = []
    for end in range(first_rating + window - 1, last_rating + 1):
        begin = end - window + 1
        mask = (design.seasons >= begin) & (design.seasons <= end)
        config = RapmConfig(
            seasons=tuple(range(begin, end + 1)),
            lambda_off=float(contract["lambda_off"]),
            lambda_def=float(contract["lambda_def"]),
            lambda_home=float(contract["lambda_home"]),
            data_scope="rolling_five_year_progress_wp_rapm",
        )
        beta, intercept = fit_coefficients(design, config, row_mask=mask)
        table = ratings_table(design, beta, names=names)
        player_count = len(design.players)
        table["off_possessions"] = np.asarray(
            design.X[mask, :player_count].sum(axis=0)
        ).ravel()
        table["def_possessions"] = np.asarray(
            design.X[mask, player_count : 2 * player_count].sum(axis=0)
        ).ravel()
        table = table.loc[
            table[["off_possessions", "def_possessions"]].min(axis=1).gt(0)
        ].copy()
        for component in ("offense", "defense", "net"):
            # ratings_table already multiplies coefficient units by 100.
            table[f"{component}_wp_percentage_points_per_100"] = table[f"{component}_per_100"]
        table["window_start"] = begin
        table["window_end"] = end
        ratings_rows.append(table)
        prediction = intercept + np.asarray(design.X[mask] @ beta).ravel()
        window_rows.append(
            {
                "window_start": begin,
                "window_end": end,
                "possessions": int(mask.sum()),
                "players": int(len(table)),
                "target_rmse": float(math.sqrt(np.mean((design.y[mask] - prediction) ** 2))),
            }
        )
    ratings = pd.concat(ratings_rows, ignore_index=True)
    surface = pd.DataFrame(surface_rows)
    windows = pd.DataFrame(window_rows)
    identity = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(CONTRACT),
                "runner": sha256_file(Path(__file__)),
                "legacy": {str(s): sha256_file(ROOT / f"rapm/data/possession_cache/matchups_{s}.parquet") for s in range(start_source, 2024)},
                "current": sha256_file(ROOT / "data/lake/silver/possessions.parquet"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"rolling_5y_wp_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    surface.to_parquet(output / "surface_metrics.parquet", index=False)
    windows.to_parquet(output / "windows.parquet", index=False)
    conservation.to_parquet(output / "game_conservation.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "research_progress_surface",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "quality": {
            "source_possessions": int(len(states)),
            "rating_possessions": int(len(rating_frame)),
            "rating_games": int(rating_frame["gameid"].nunique()),
            "maximum_conservation_error": float(conservation["conservation_error"].abs().max()),
            "surface_brier_range": [float(surface["brier"].min()), float(surface["brier"].max())],
            "surface_auc_range": [float(surface["auc"].min()), float(surface["auc"].max())],
        },
        "windows": windows.to_dict("records"),
        "artifacts": {"ratings": "ratings.parquet", "surface_metrics": "surface_metrics.parquet", "windows": "windows.parquet"},
        "forbidden_interpretation": "Exact-clock WP credit, ordinary points impact, or a forecast of portable player strength.",
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
