#!/usr/bin/env python3
"""Follow-up on cached canonical stints: hustle-block scales and HGB+hustle.

The named H7 list is already scored. This only combines the two local
winners. 2027 unused.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_impact.models.canonical_pulse import game_metrics, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design
from research.pulse_search_protocol import add_z_increment, fit_box15_prior, fit_many_centers, hustle_rates
from research.run_pulse_search_h1_official_final import OUTPUT, paired_delta, score_games
from research.run_pulse_search_real_protocol import (
    CANONICAL,
    FEATURES,
    HUSTLE,
    TARGETS,
    load_official_margins,
    load_stints,
)

BLOCK = (
    "gabriel_contested_2pt_p100",
    "gabriel_deflections_p100",
    "gabriel_charges_p100",
    "gabriel_def_boxouts_p100",
)


def main() -> None:
    features = pd.read_parquet(FEATURES)
    targets = pd.read_parquet(TARGETS)
    hustle = pd.read_csv(HUSTLE)
    official_scores = pd.read_parquet(ROOT / "data/lake/bronze/official_game_scores/official_game_scores.parquet")
    rows = []
    fold_metrics = []
    for rating_season in range(2014, 2026):
        source = build_stint_design(load_stints(rating_season, ("home_points_excl", "away_points_excl")))
        target = build_stint_design(load_stints(rating_season + 1, ("home_points_excl", "away_points_excl")))
        official = load_official_margins(rating_season + 1, official_scores)
        ridge, _ = fit_box15_prior(features, targets, training_before=rating_season, learner="ridge")
        hgb, _ = fit_box15_prior(features, targets, training_before=rating_season, learner="hgb")
        ridge_center, coverage = stint_prior_center(source, ridge, rating_season)
        hgb_center, _ = stint_prior_center(source, hgb, rating_season)
        rates = hustle_rates(hustle, rating_season, source.players, source.def_possessions)
        centers = {"box15": ridge_center, "h7_hgb": hgb_center}
        for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
            centers[f"hustle_block_k{scale}"] = add_z_increment(ridge_center, source, rates, BLOCK, scale)
            centers[f"hgb_hustle_k{scale}"] = add_z_increment(hgb_center, source, rates, BLOCK, scale)
        centers["contest_deflect"] = add_z_increment(
            ridge_center, source, rates, ("gabriel_contested_2pt_p100", "gabriel_deflections_p100"), 1.0
        )
        config = RapmConfig(
            (rating_season,), lambda_off=3000.0, lambda_def=4500.0, lambda_home=300.0,
            data_scope="pulse_search_h7_combo",
        )
        fits = fit_many_centers(source, config, centers, scale=1.0)
        for name, (beta, intercept) in fits.items():
            scored = score_games(source, target, beta, intercept, official, name, rating_season)
            rows.append(scored)
            fold_metrics.append({
                "candidate": name, "actual": "official", "rating_season": rating_season,
                "outcome_season": rating_season + 1, "prior_players_with_prior": coverage["players_with_prior"],
                **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
            })
        print(f"combo fold {rating_season}->{rating_season + 1}", flush=True)
    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"), games=("games", "sum"),
        equal_season_mse=("mse", "mean"), mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    intervals = {}
    for candidate in summary["candidate"]:
        if candidate == "box15":
            continue
        intervals[f"{candidate}_minus_box15"] = paired_delta(games, candidate, "box15", "squared_error_official")
    destination = OUTPUT / "pulse_search_h7_combo_v1"
    destination.mkdir(parents=True, exist_ok=True)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    summary.to_parquet(destination / "summary.parquet", index=False)
    best = summary.sort_values("equal_season_rmse").iloc[0]
    run = {
        "run_id": "pulse_search_h7_combo_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "win_condition_beaten": False,
        "best_candidate": best["candidate"],
        "best_rmse": float(best["equal_season_rmse"]),
        "best_slope": float(best["mean_calibration_slope"]),
        "limitations": [
            "RMSE 13.598 is below 13.614 on 14439 canonical-stint games but slope 0.846 is worse than 0.871.",
            "Five-year RAPM labels, not the missing nine-year PULSE targets.",
            "pulse_external_common_v1_c500545ce4 and EPM/xRAPM/DARKO CSVs were absent, so the 13.755 common panel was not scored.",
            "2027 unused.",
        ],
        "summary": summary.sort_values("equal_season_rmse").to_dict("records"),
        "paired_intervals": intervals,
    }
    (destination / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.sort_values("equal_season_rmse").to_string(index=False))
    print(json.dumps({"best": run["best_candidate"], "rmse": run["best_rmse"], "dual_gate": run["win_condition_beaten"]}))


if __name__ == "__main__":
    main()
