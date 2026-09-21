#!/usr/bin/env python3
"""PULSE search H2/H3/H6 on already-built H1 stints.

Does not rebuild V3 lineups. Does not spend 2027. Public priors are leaky
diagnostics. A challenger can only reject here; it cannot claim the 12-fold
win condition from this fold alone.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.canonical_pulse import game_metrics
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design, fit_stint_center_path
from research.run_pulse_search_h1_official_final import (
    OUTPUT,
    SILVER,
    fit_stint_arm,
    official_margins,
    paired_delta,
    public_priors,
    score_games,
    stint_prior_center,
)


def _load_scores() -> pd.DataFrame:
    return pd.read_parquet(SILVER / "v3_terminal_scores.parquet")


def exposure_center(design, priors, season: int, tau: float) -> np.ndarray:
    center, _coverage = stint_prior_center(design, priors, season)
    n = len(design.players)
    exposure = np.minimum(design.off_possessions, design.def_possessions)
    weight = 1.0 / (1.0 + exposure / tau)
    scaled = center.copy()
    scaled[:n] *= weight
    scaled[n : 2 * n] *= weight
    return scaled


def fit_custom(design, center, scale: float, season: int, lambda_off: float, lambda_def: float):
    config = RapmConfig(
        (season,),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=300.0,
        data_scope="pulse_search_h2_v3_candidate",
    )
    return fit_stint_center_path(design, config, center, center_scales=(scale,))[scale]


def main() -> None:
    rating_season = 2025
    scores = _load_scores()
    source_stints = pd.read_parquet(SILVER / f"project_season={rating_season}" / "canonical_like_stints.parquet")
    target_stints = pd.read_parquet(SILVER / f"project_season={rating_season + 1}" / "canonical_like_stints.parquet")
    terminal_source = pd.read_parquet(SILVER / f"project_season={rating_season}" / "terminal_like_stints.parquet")
    terminal_target = pd.read_parquet(SILVER / f"project_season={rating_season + 1}" / "terminal_like_stints.parquet")
    pulse_priors = public_priors(rating_season, "pulse")
    rich_priors = public_priors(rating_season, "rich")
    official = official_margins(scores, rating_season + 1)
    columns = ("home_points_excl", "away_points_excl")
    source_design, baseline, coverage, center = fit_stint_arm(
        source_stints, columns, pulse_priors, rating_season, scale=1.0
    )
    target_frame = target_stints.copy()
    target_frame["home_points"] = target_frame[columns[0]]
    target_frame["away_points"] = target_frame[columns[1]]
    target_design = build_stint_design(target_frame)
    rows = []
    fold_metrics = []

    def record(name: str, beta, intercept):
        scored = score_games(source_design, target_design, beta, intercept, official, name, rating_season)
        rows.append(scored)
        metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
        fold_metrics.append({
            "candidate": name,
            "actual": "official",
            "rating_season": rating_season,
            "outcome_season": rating_season + 1,
            "prior_players_with_prior": coverage["players_with_prior"],
            **metrics,
        })

    record("pulse_excl", *baseline["pulse"])
    record("rapm_excl", *baseline["rapm"])

    for scale in (0.25, 0.5, 0.75, 1.0):
        beta, intercept = fit_custom(source_design, center, scale, rating_season, 3000.0, 4500.0)
        record(f"h2_scale_{scale}", beta, intercept)

    for lambda_off, lambda_def in (
        (3000.0, 3000.0),
        (3000.0, 4500.0),
        (3000.0, 6000.0),
        (2000.0, 4500.0),
        (4000.0, 4500.0),
        (3000.0, 4000.0),
    ):
        beta, intercept = fit_custom(source_design, center, 1.0, rating_season, lambda_off, lambda_def)
        record(f"h2_l{int(lambda_off)}_{int(lambda_def)}", beta, intercept)

    for tau in (500.0, 2000.0, 4000.0):
        scaled = exposure_center(source_design, pulse_priors, rating_season, tau)
        beta, intercept = fit_custom(source_design, scaled, 1.0, rating_season, 3000.0, 4500.0)
        record(f"h3_exposure_tau{int(tau)}", beta, intercept)

    _, rich_fits, _, _ = fit_stint_arm(source_stints, columns, rich_priors, rating_season, scale=1.0)
    record("h4_rich_prior", *rich_fits["pulse"])

    term_source, term_fits, _, _ = fit_stint_arm(
        terminal_source, columns, pulse_priors, rating_season, scale=1.0
    )
    term_target_frame = terminal_target.copy()
    term_target_frame["home_points"] = term_target_frame[columns[0]]
    term_target_frame["away_points"] = term_target_frame[columns[1]]
    term_target = build_stint_design(term_target_frame)
    scored = score_games(
        term_source, term_target, *term_fits["pulse"], official, "h6_terminal_pulse_excl", rating_season
    )
    rows.append(scored)
    metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
    fold_metrics.append({
        "candidate": "h6_terminal_pulse_excl",
        "actual": "official",
        "rating_season": rating_season,
        "outcome_season": rating_season + 1,
        **metrics,
    })
    scored = score_games(
        term_source, term_target, *term_fits["rapm"], official, "h6_terminal_rapm_excl", rating_season
    )
    rows.append(scored)
    metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
    fold_metrics.append({
        "candidate": "h6_terminal_rapm_excl",
        "actual": "official",
        "rating_season": rating_season,
        "outcome_season": rating_season + 1,
        **metrics,
    })

    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    folds["rmse"] = np.sqrt(folds["mse"])
    baseline_rmse = float(folds.loc[folds["candidate"].eq("pulse_excl"), "rmse"].iloc[0])
    folds["rmse_minus_pulse_excl"] = folds["rmse"] - baseline_rmse
    intervals = {}
    for candidate in folds["candidate"]:
        if candidate == "pulse_excl":
            continue
        try:
            intervals[f"{candidate}_minus_pulse_excl"] = paired_delta(
                games, candidate, "pulse_excl", "squared_error_official"
            )
        except ValueError:
            continue
    run_id = "pulse_search_h2_h3_h6_v1"
    destination = OUTPUT / run_id
    destination.mkdir(parents=True, exist_ok=True)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "hypothesis": "Recalibrate prior scale/precision, exposure-aware shrinkage, rich prior, and terminal assignment on the H1 2025-2026 fold.",
        "baseline_pulse_excl_rmse": baseline_rmse,
        "folds": folds.to_dict("records"),
        "paired_intervals": intervals,
        "source_hashes": {"runner": sha256_file(Path(__file__))},
        "limitations": [
            "One chronological fold on V3 candidates, not the frozen 12-fold gate.",
            "Public priors are descriptive 2026 refits.",
            "H6 terminal vs stint comparison uses the same games only when both arms score the inner join; paired_delta drops non-overlapping candidates.",
            "2027 unused.",
        ],
    }
    write_json_atomic(run, destination / "run.json")
    print(folds.sort_values("rmse").to_string(index=False))
    print(json.dumps({"baseline_pulse_excl_rmse": baseline_rmse}, indent=2))
    print(destination, flush=True)


if __name__ == "__main__":
    main()
