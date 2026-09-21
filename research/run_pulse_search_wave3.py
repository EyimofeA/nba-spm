#!/usr/bin/env python3
"""Wave 3: nine-year PULSE labels, hustle+HGB recalibration, official-final panel.

Research-only. Does not promote. Does not spend 2027.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_impact.models.canonical_pulse import game_metrics, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design
from research.pulse_search_protocol import (
    HUSTLE_BLOCK,
    PULSE_RMSE_GATE,
    PULSE_SLOPE_GATE,
    add_z_increment,
    fit_box15_prior,
    fit_precision_jobs,
    fold_internal_affine,
    hustle_rates,
    keep_calibrated_variant,
    scale_side_center,
)
from research.pulse_search_wave3_external import load_external_ratings, score_common_panel, summarize_folds
from research.pulse_search_wave3_stints import (
    NINE_YEAR_ROOT,
    fit_nine_year_targets,
    prepare_historical_stints,
    utc_now,
)
from research.run_pulse_search_h1_official_final import OUTPUT, paired_delta, score_games
from research.run_pulse_search_real_protocol import (
    FEATURES,
    HUSTLE,
    OFFICIAL,
    load_official_margins,
    load_stints,
)


WAVE3_ROOT = OUTPUT / "pulse_search_wave3_v1"


def calibration_jobs(hustle_center: np.ndarray, n_players: int) -> list[dict]:
    jobs = []
    for scale in (0.5, 0.75, 1.0, 1.25):
        jobs.append({"name": f"hgb_hustle_scale{scale}", "center": hustle_center, "prior_scale": scale})
    for off_scale, def_scale in ((1.0, 0.7), (1.0, 1.3), (0.7, 1.0), (1.3, 1.0)):
        jobs.append(
            {
                "name": f"hgb_hustle_side_o{off_scale}_d{def_scale}",
                "center": scale_side_center(
                    hustle_center, n_players, offense_scale=off_scale, defense_scale=def_scale
                ),
                "prior_scale": 1.0,
            }
        )
    for off_mult, def_mult in ((1.0, 1.5), (1.5, 1.0), (0.75, 1.25)):
        jobs.append(
            {
                "name": f"hgb_hustle_prec_o{off_mult}_d{def_mult}",
                "center": hustle_center,
                "prior_scale": 1.0,
                "offense_penalty_mult": off_mult,
                "defense_penalty_mult": def_mult,
            }
        )
    return jobs


def run_twelve_folds(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    features = pd.read_parquet(FEATURES)
    hustle = pd.read_csv(HUSTLE)
    official_scores = pd.read_parquet(OFFICIAL)
    rows = []
    fold_metrics = []
    prior_quality = []
    season_fits: dict[int, dict] = {}
    for rating_season in range(2014, 2026):
        source = build_stint_design(load_stints(rating_season, ("home_points_excl", "away_points_excl")))
        target = build_stint_design(load_stints(rating_season + 1, ("home_points_excl", "away_points_excl")))
        official = load_official_margins(rating_season + 1, official_scores)
        ridge, quality = fit_box15_prior(features, targets, training_before=rating_season, learner="ridge")
        hgb, _ = fit_box15_prior(features, targets, training_before=rating_season, learner="hgb")
        quality["rating_season"] = rating_season
        prior_quality.append(quality)
        ridge_center, coverage = stint_prior_center(source, ridge, rating_season)
        hgb_center, _ = stint_prior_center(source, hgb, rating_season)
        rates = hustle_rates(hustle, rating_season, source.players, source.def_possessions)
        hustle_center = add_z_increment(hgb_center, source, rates, HUSTLE_BLOCK, 1.0)
        n = len(source.players)
        jobs = [
            {"name": "rapm", "center": np.zeros(source.X.shape[1]), "prior_scale": 1.0},
            {"name": "box15", "center": ridge_center, "prior_scale": 1.0},
            {"name": "hgb", "center": hgb_center, "prior_scale": 1.0},
            {"name": "hgb_hustle_k1.0", "center": hustle_center, "prior_scale": 1.0},
            *calibration_jobs(hustle_center, n),
        ]
        config = RapmConfig(
            (rating_season,),
            lambda_off=3000.0,
            lambda_def=4500.0,
            lambda_home=300.0,
            data_scope="pulse_search_wave3_nine_year",
        )
        fits = fit_precision_jobs(source, config, jobs)
        for name, (beta, intercept) in fits.items():
            scored = score_games(source, target, beta, intercept, official, name, rating_season)
            rows.append(scored)
            fold_metrics.append(
                {
                    "candidate": name,
                    "actual": "official",
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    "prior_players_with_prior": coverage["players_with_prior"],
                    **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
                }
            )
        season_fits[rating_season] = fits
        print(f"wave3 fold {rating_season}->{rating_season + 1}", flush=True)
    games = pd.concat(rows, ignore_index=True)
    affine = fold_internal_affine(games, "hgb_hustle_k1.0")
    games = pd.concat([games, affine], ignore_index=True)
    for season, group in affine.groupby("outcome_season"):
        fold_metrics.append(
            {
                "candidate": "hgb_hustle_k1.0_affine",
                "actual": "official",
                "rating_season": int(season) - 1,
                "outcome_season": int(season),
                **game_metrics(group.assign(actual_margin=group["official_margin"])),
            }
        )
    folds = pd.DataFrame(fold_metrics)
    return games, folds, pd.DataFrame(prior_quality), season_fits


def kept_calibrated(summary: pd.DataFrame, box15_rmse: float) -> list[str]:
    kept = []
    for row in summary.itertuples(index=False):
        if row.candidate in {"box15", "rapm"}:
            continue
        if keep_calibrated_variant(row.equal_season_rmse, row.mean_calibration_slope, box15_rmse):
            kept.append(row.candidate)
    return kept


def box15_intervals(games: pd.DataFrame, summary: pd.DataFrame) -> dict:
    intervals = {}
    for candidate in summary["candidate"]:
        if candidate == "box15":
            continue
        try:
            intervals[f"{candidate}_minus_box15"] = paired_delta(
                games, candidate, "box15", "squared_error_official"
            )
        except Exception as exc:
            intervals[f"{candidate}_minus_box15"] = {"status": "unscored", "error": str(exc)}
    return intervals


def write_wave3_run(
    stint_report: dict,
    target_report: dict,
    games: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    prior_quality: pd.DataFrame,
    common: dict,
    epm: pd.DataFrame,
    xrapm: pd.DataFrame,
    darko: pd.DataFrame,
    kept: list[str],
    intervals: dict,
) -> dict:
    WAVE3_ROOT.mkdir(parents=True, exist_ok=True)
    games.to_parquet(WAVE3_ROOT / "game_predictions.parquet", index=False)
    folds.to_parquet(WAVE3_ROOT / "fold_metrics.parquet", index=False)
    summary.to_parquet(WAVE3_ROOT / "summary.parquet", index=False)
    prior_quality.to_parquet(WAVE3_ROOT / "prior_quality.parquet", index=False)
    if common.get("status") == "scored":
        common["games_frame"].to_parquet(WAVE3_ROOT / "common_game_predictions.parquet", index=False)
        common["folds_frame"].to_parquet(WAVE3_ROOT / "common_fold_metrics.parquet", index=False)
        common["summary_frame"].to_parquet(WAVE3_ROOT / "common_summary.parquet", index=False)
        common_public = {key: value for key, value in common.items() if not key.endswith("_frame")}
    else:
        common_public = common
    box15 = summary.loc[summary["candidate"].eq("box15")].iloc[0]
    best = summary.iloc[0]
    dual = bool(
        keep_calibrated_variant(float(best.equal_season_rmse), float(best.mean_calibration_slope), PULSE_RMSE_GATE)
        and common_public.get("status") == "scored"
    )
    run = {
        "run_id": WAVE3_ROOT.name,
        "created_at": utc_now(),
        "status": "research_pilot_not_promotion",
        "win_condition_beaten": False,
        "dual_gate_note": "Promotion remains false even if a local RMSE/slope pair looks better.",
        "stints": stint_report,
        "nine_year_targets": target_report,
        "box15_nine_year_rmse": float(box15.equal_season_rmse),
        "box15_nine_year_slope": float(box15.mean_calibration_slope),
        "kept_calibrated_variants": kept,
        "best_candidate": best.candidate,
        "best_rmse": float(best.equal_season_rmse),
        "best_slope": float(best.mean_calibration_slope),
        "beat_13614_without_worse_slope": keep_calibrated_variant(
            float(best.equal_season_rmse), float(best.mean_calibration_slope), PULSE_RMSE_GATE
        ),
        "paired_intervals": intervals,
        "summary": summary.to_dict("records"),
        "common_panel": common_public,
        "external_coverage": {
            "epm_rows": int(len(epm)),
            "xrapm_rows": int(len(xrapm)),
            "darko_rows": int(len(darko)),
            "lebron_rows": 0,
        },
        "limitations": [
            "Nine-year RAPM is rebuilt on reconstructed canonical stints; 2014 stint hash does not match frozen pulse_canonical_v1_cd3c14750a.",
            "EPM is end-of-season expected EPM from dunksandthrees.com/epm, not the locked actual EPM_All_Seasons.csv.",
            "DARKO/LEBRON source CSVs are used only if a public file parsed.",
            "2027 unused. No promotion.",
        ],
        "pulse_rmse_gate": PULSE_RMSE_GATE,
        "pulse_slope_gate": PULSE_SLOPE_GATE,
        "would_have_beaten_dual_if_common_matched": dual,
    }
    (WAVE3_ROOT / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "best": run["best_candidate"],
                "rmse": run["best_rmse"],
                "slope": run["best_slope"],
                "kept": kept,
                "common": common_public.get("status"),
                "dual": run["win_condition_beaten"],
            }
        ),
        flush=True,
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("stints", "targets", "all"))
    args = parser.parse_args()
    WAVE3_ROOT.mkdir(parents=True, exist_ok=True)
    print("wave3 historical stints", flush=True)
    stint_report = prepare_historical_stints(1997, 2013)
    if args.stage == "stints":
        from nba_impact.data.manifest import write_json_atomic

        write_json_atomic({"created_at": utc_now(), "stints": stint_report}, WAVE3_ROOT / "stints.json")
        print(json.dumps(stint_report, default=str)[:4000], flush=True)
        return
    print("wave3 nine-year RAPM", flush=True)
    target_report = fit_nine_year_targets(2005, 2025)
    if args.stage == "targets":
        print(json.dumps(target_report, default=str)[:4000], flush=True)
        return
    targets = pd.read_parquet(NINE_YEAR_ROOT / "targets.parquet")
    games, folds, prior_quality, season_fits = run_twelve_folds(targets)
    summary = summarize_folds(folds)
    box15 = summary.loc[summary["candidate"].eq("box15")].iloc[0]
    kept = kept_calibrated(summary, float(box15.equal_season_rmse))
    intervals = box15_intervals(games, summary)
    print(summary.to_string(index=False), flush=True)
    epm, xrapm, darko, external = load_external_ratings()
    common = score_common_panel(season_fits, external)
    write_wave3_run(
        stint_report,
        target_report,
        games,
        folds,
        summary,
        prior_quality,
        common,
        epm,
        xrapm,
        darko,
        kept,
        intervals,
    )


if __name__ == "__main__":
    main()
