#!/usr/bin/env python3
"""Non-leaky chronological PULSE search: contested-2PT then H7.

Uses canonical stints when present. Otherwise builds the largest V3
chronological set with past-only Box15 priors. Scores official-final margins
when LeagueGameLog or box sums exist; never spends 2027.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.canonical_pulse import game_metrics, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design
from research.pulse_search_protocol import (
    add_z_increment,
    fit_box15_prior,
    fit_constrained,
    fit_many_centers,
    game_split_mask,
    hustle_rates,
    segments_to_stints_mode,
)
from research.build_canonical_lineup_stints import build_season
from research.run_pulse_search_fetch_support import build_offline_official_scores
from research.run_pulse_search_h1_official_final import (
    BOX_LOGS,
    OUTPUT,
    SILVER,
    V3_NESTED,
    _normalize_box,
    build_player_games,
    build_score_reference,
    build_sources,
    official_margins,
    paired_delta,
    score_games,
)


FEATURES = ROOT / (
    "artifacts/research/historical_box15_extension/"
    "historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet"
)
TARGETS = ROOT / (
    "artifacts/research/historical_box15_extension/"
    "historical_box15_extension_v1_08ff4c34ff/five_year_targets.parquet"
)
HUSTLE = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/hustle.csv"
CANONICAL = ROOT / "data/lake/silver/canonical_lineup_stints"
OFFICIAL = ROOT / "data/lake/bronze/official_game_scores/official_game_scores.parquet"
FETCHED_BOX = ROOT / "data/lake/bronze/nba_player_game_logs"
CANONICAL_PULSE_RMSE = 13.614
COMMON_PANEL_RMSE = 13.755


def espn_player_games(espn: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    from research.run_pulse_search_h1_official_final import minutes_value

    frame = espn.copy()
    frame["game_id"] = frame["game_id"].map(canonical_game_id)
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype("int64")
    frame["starter"] = pd.to_numeric(frame["starter"], errors="coerce").fillna(0).astype(bool)
    frame["minutes_seconds"] = frame["minutes_played"].map(minutes_value)
    frame["player_name"] = frame["name"].astype(str)
    games = scores[["game_id", "project_season", "game_date", "home_team_id", "away_team_id"]].copy()
    games["game_id"] = games["game_id"].map(canonical_game_id)
    merged = frame.merge(games, on="game_id", how="inner")
    home_flag = pd.to_numeric(merged["home"], errors="coerce").fillna(0).eq(1)
    merged["team_id"] = np.where(home_flag, merged["home_team_id"], merged["away_team_id"])
    merged["season_end"] = merged["project_season"].astype(int)
    merged["season_label"] = merged["season_end"].map(lambda year: f"{year - 1}-{str(year)[-2:]}")
    merged["season_type"] = "regular"
    return merged[[
        "game_id", "season_end", "season_label", "season_type", "game_date",
        "team_id", "player_id", "player_name", "starter", "minutes_seconds",
    ]].drop_duplicates(["game_id", "player_id"])


def combined_box() -> pd.DataFrame:
    parts = []
    if BOX_LOGS.exists():
        parts.append(_normalize_box(pd.read_parquet(BOX_LOGS)))
    if FETCHED_BOX.exists():
        for path in sorted(FETCHED_BOX.glob("season=*/regular.parquet")):
            parts.append(_normalize_box(pd.read_parquet(path)))
    if not parts:
        return pd.DataFrame()
    box = pd.concat(parts, ignore_index=True)
    box = box.drop_duplicates(["game_id", "player_id"], keep="last")
    return box


def load_official_margins(season: int, fallback: pd.DataFrame) -> pd.DataFrame:
    if OFFICIAL.exists():
        scores = pd.read_parquet(OFFICIAL)
        games = scores.loc[scores["project_season"].eq(season) & scores["season_type"].eq("regular")].copy()
        if not games.empty:
            games["game_id"] = games["game_id"].map(canonical_game_id)
            games["official_margin"] = games["home_score"].astype(float) - games["away_score"].astype(float)
            games["v3_margin"] = games["official_margin"]
            games["box_margin"] = games["official_margin"]
            return games[["game_id", "official_margin", "v3_margin", "box_margin", "home_score", "away_score"]]
    return official_margins(fallback, season)


def load_stints(season: int, point_columns: tuple[str, str]) -> pd.DataFrame:
    canonical = CANONICAL / f"season={season}" / "regular.parquet"
    if canonical.exists():
        frame = pd.read_parquet(canonical)
        frame["home_points_excl"] = frame["home_points"]
        frame["away_points_excl"] = frame["away_points"]
        return frame
    path = SILVER / f"project_season={season}" / "canonical_like_stints.parquet"
    frame = pd.read_parquet(path)
    frame["home_points"] = frame[point_columns[0]]
    frame["away_points"] = frame[point_columns[1]]
    return frame


def season_has_stints(season: int) -> bool:
    return (CANONICAL / f"season={season}" / "regular.parquet").exists() or (
        SILVER / f"project_season={season}" / "canonical_like_stints.parquet"
    ).exists()


def parse_seasons(raw: str) -> tuple[int, ...]:
    seasons = (
        tuple(int(item) for item in raw.split(",") if item.strip())
        if raw
        else tuple(range(2014, 2027))
    )
    if len(seasons) < 2:
        raise SystemExit(f"Need at least two seasons, found {seasons}")
    return seasons


def rebuild_canonical_stints(seasons: tuple[int, ...], official_table: pd.DataFrame) -> list[dict]:
    canonical_quality = []
    for season in seasons:
        destination = CANONICAL / f"season={season}" / "regular.parquet"
        if destination.exists():
            canonical_quality.append({"season": season, "status": "existing"})
            continue
        try:
            quality = build_season(season, official_table, CANONICAL)
            canonical_quality.append(quality)
            print(
                f"canonical {season}: stints={quality['stints']} "
                f"native={quality['native_score_reconciliation']:.3f} "
                f"tech={quality['model_plus_technical_reconciliation']:.3f}",
                flush=True,
            )
        except Exception as exc:
            canonical_quality.append({"season": season, "status": "failed", "error": str(exc)})
            print(f"canonical {season} failed: {exc}", flush=True)
    return canonical_quality


def _write_espn_player_games(v3_scores: pd.DataFrame, players: Path) -> None:
    espn_path = ROOT / "data/lake/bronze/llimllib_nba_data/espn/player_box.parquet"
    espn = pd.read_parquet(espn_path)
    espn_games = espn_player_games(espn, v3_scores)
    if players.exists():
        espn_games = pd.concat([pd.read_parquet(players), espn_games], ignore_index=True)
        espn_games = espn_games.drop_duplicates(["game_id", "player_id"], keep="last")
    players.parent.mkdir(parents=True, exist_ok=True)
    espn_games.to_parquet(players, index=False)


def maybe_build_v3_fallback(
    seasons: tuple[int, ...], score_metrics: dict, box: pd.DataFrame
) -> tuple[list, dict]:
    missing = [season for season in seasons if not season_has_stints(season)]
    if not missing:
        return [], score_metrics
    v3_seasons = tuple(
        season
        for season in missing
        if (V3_NESTED / f"project_season={season}" / "regular.parquet").exists()
    )
    if not v3_seasons:
        return [], score_metrics
    v3_scores, v3_metrics = build_score_reference(v3_seasons, box)
    score_metrics = {**score_metrics, "v3_fallback": v3_metrics}
    players = SILVER / "player_games.parquet"
    espn_path = ROOT / "data/lake/bronze/llimllib_nba_data/espn/player_box.parquet"
    if espn_path.exists():
        _write_espn_player_games(v3_scores, players)
    elif not players.exists():
        box_path = SILVER / "combined_box.parquet"
        if not box.empty:
            box.to_parquet(box_path, index=False)
        build_player_games(v3_scores, box_path if not box.empty else BOX_LOGS, players, v3_seasons)
    summaries = build_sources(v3_seasons, v3_scores, players)
    print(json.dumps({"built": summaries}, indent=2), flush=True)
    return summaries, score_metrics


def h7_centers(ridge_center, source_design, rates, contest_center, features, targets, rating_season, prior_quality):
    extra_centers = {
        "rapm": np.zeros(source_design.X.shape[1]),
        "box15": ridge_center,
        "contest2": contest_center,
        "h7_deflections": add_z_increment(ridge_center, source_design, rates, ("gabriel_deflections_p100",), 1.0),
        "h7_charges": add_z_increment(ridge_center, source_design, rates, ("gabriel_charges_p100",), 1.0),
        "h7_boxouts": add_z_increment(ridge_center, source_design, rates, ("gabriel_def_boxouts_p100",), 1.0),
        "h7_contest3": add_z_increment(ridge_center, source_design, rates, ("gabriel_contested_3pt_p100",), 1.0),
        "h7_hustle_block": add_z_increment(
            ridge_center,
            source_design,
            rates,
            (
                "gabriel_contested_2pt_p100",
                "gabriel_deflections_p100",
                "gabriel_charges_p100",
                "gabriel_def_boxouts_p100",
            ),
            1.0,
        ),
    }
    for learner in ("elasticnet", "extratrees", "hgb"):
        learner_priors, learner_quality = fit_box15_prior(
            features, targets, training_before=rating_season, learner=learner
        )
        learner_quality["rating_season"] = rating_season
        prior_quality.append(learner_quality)
        extra_centers[f"h7_{learner}"] = stint_prior_center(source_design, learner_priors, rating_season)[0]
    return extra_centers


def score_fractional_arm(rating_season, ridge_priors, config, target_design, official, coverage):
    poss_path = SILVER / f"project_season={rating_season}" / "possessions_with_tech.parquet"
    seg_path = SILVER / f"project_season={rating_season}" / "segments.parquet"
    if not (poss_path.exists() and seg_path.exists()):
        return None
    fractional = segments_to_stints_mode(
        pd.read_parquet(poss_path), pd.read_parquet(seg_path), mode="fractional"
    )
    frac_design = build_stint_design(fractional.assign(
        home_points=lambda frame: frame["home_points_excl"],
        away_points=lambda frame: frame["away_points_excl"],
    ))
    frac_center, _ = stint_prior_center(frac_design, ridge_priors, rating_season)
    frac_fits = fit_many_centers(frac_design, config, {"frac": frac_center}, scale=1.0)
    scored = score_games(
        frac_design, target_design, *frac_fits["frac"], official, "h7_fractional", rating_season
    )
    return scored, {
        "candidate": "h7_fractional",
        "actual": "official",
        "rating_season": rating_season,
        "outcome_season": rating_season + 1,
        "prior_players_with_prior": coverage["players_with_prior"],
        **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
    }


def score_rating_fold(rating_season, features, targets, hustle, scores):
    source_design = build_stint_design(load_stints(rating_season, ("home_points_excl", "away_points_excl")))
    target_design = build_stint_design(load_stints(rating_season + 1, ("home_points_excl", "away_points_excl")))
    official = load_official_margins(rating_season + 1, scores)
    ridge_priors, quality = fit_box15_prior(features, targets, training_before=rating_season, learner="ridge")
    quality["rating_season"] = rating_season
    ridge_center, coverage = stint_prior_center(source_design, ridge_priors, rating_season)
    rates = hustle_rates(hustle, rating_season, source_design.players, source_design.def_possessions)
    contest_center = add_z_increment(
        ridge_center, source_design, rates, ("gabriel_contested_2pt_p100",), 1.0
    )
    prior_quality = [quality]
    extra_centers = h7_centers(
        ridge_center, source_design, rates, contest_center, features, targets, rating_season, prior_quality
    )
    config = RapmConfig(
        (rating_season,),
        lambda_off=3000.0,
        lambda_def=4500.0,
        lambda_home=300.0,
        data_scope="pulse_search_real_protocol",
    )
    fits = fit_many_centers(source_design, config, extra_centers, scale=1.0)
    first_mask = game_split_mask(source_design, second_half=False)
    second_mask = game_split_mask(source_design, second_half=True)
    if first_mask.sum() > 100 and second_mask.sum() > 100:
        fits["h7_half_likelihood"] = fit_many_centers(
            source_design, config, {"box15": ridge_center}, scale=1.0, row_mask=first_mask
        )["box15"]
        fits["h7_current_second_half"] = fit_many_centers(
            source_design, config, {"box15": ridge_center}, scale=1.0, row_mask=second_mask
        )["box15"]
    fits["h7_constrained"] = fit_constrained(source_design, config, ridge_center, scale=1.0)
    fits["h7_constrained_contest"] = fit_constrained(source_design, config, contest_center, scale=1.0)
    rows = []
    fold_metrics = []
    fractional = score_fractional_arm(
        rating_season, ridge_priors, config, target_design, official, coverage
    )
    if fractional is not None:
        rows.append(fractional[0])
        fold_metrics.append(fractional[1])
    stint_source = "canonical" if (CANONICAL / f"season={rating_season}/regular.parquet").exists() else "v3"
    for name, (beta, intercept) in fits.items():
        scored = score_games(source_design, target_design, beta, intercept, official, name, rating_season)
        rows.append(scored)
        fold_metrics.append({
            "candidate": name,
            "actual": "official",
            "rating_season": rating_season,
            "outcome_season": rating_season + 1,
            "prior_players_with_prior": coverage["players_with_prior"],
            "stint_source": stint_source,
            **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
        })
    print(
        f"fold {rating_season}->{rating_season + 1} games={fold_metrics[-1]['games']} "
        f"prior_rows={quality['training_rows']}",
        flush=True,
    )
    return rows, fold_metrics, prior_quality


def rating_seasons_with_stints(seasons: tuple[int, ...]) -> list[int]:
    return [
        season
        for season in seasons
        if season + 1 in seasons and season_has_stints(season) and season_has_stints(season + 1)
    ]


def summarize_protocol(folds: pd.DataFrame) -> pd.DataFrame:
    summary = folds.groupby(["candidate"], as_index=False).agg(
        folds=("outcome_season", "nunique"),
        games=("games", "sum"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    summary["beats_canonical_13614"] = summary["equal_season_rmse"].lt(CANONICAL_PULSE_RMSE)
    summary["beats_common_13755"] = summary["equal_season_rmse"].lt(COMMON_PANEL_RMSE)
    return summary


def paired_intervals_vs_baseline(games: pd.DataFrame, summary: pd.DataFrame) -> dict:
    intervals = {}
    baseline = "box15"
    for candidate in sorted(summary["candidate"].unique()):
        if candidate == baseline:
            continue
        if set(games.loc[games["candidate"].eq(candidate), "game_id"]) != set(
            games.loc[games["candidate"].eq(baseline), "game_id"]
        ):
            continue
        try:
            intervals[f"{candidate}_minus_{baseline}"] = paired_delta(
                games, candidate, baseline, "squared_error_official"
            )
        except Exception as exc:
            intervals[f"{candidate}_minus_{baseline}"] = {"error": str(exc)}
    return intervals


def write_real_protocol_run(
    seasons: tuple[int, ...],
    rating_seasons: list[int],
    games: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    prior_quality: list,
    score_metrics: dict,
    summaries: list,
    intervals: dict,
) -> dict:
    run_id = "pulse_search_real_protocol_v1"
    destination = OUTPUT / run_id
    destination.mkdir(parents=True, exist_ok=True)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    summary.to_parquet(destination / "summary.parquet", index=False)
    pd.DataFrame(prior_quality).to_parquet(destination / "prior_quality.parquet", index=False)
    best = summary.sort_values("equal_season_rmse").iloc[0]
    dual_gate = bool(
        best["beats_canonical_13614"] and best["beats_common_13755"] and int(best["folds"]) >= 8
    )
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "win_condition_beaten": dual_gate,
        "best_candidate": best["candidate"],
        "best_rmse": float(best["equal_season_rmse"]),
        "canonical_reference_rmse": CANONICAL_PULSE_RMSE,
        "common_panel_reference_rmse": COMMON_PANEL_RMSE,
        "seasons": list(seasons),
        "rating_seasons": rating_seasons,
        "score_metrics": score_metrics,
        "season_build": summaries,
        "prior_label": "five_year_rapm_window_end_lt_rating_season",
        "limitations": [
            "Canonical stints used when present; otherwise V3 historical candidates.",
            "Box15 labels are five-year RAPM, not the missing nine-year PULSE targets.",
            "Priors are past-only (training_before=rating_season). Public 2026 refits unused.",
            "Official-final uses LeagueGameLog when downloaded, else box-sum/V3 proxy.",
            "pulse_external_common_v1_c500545ce4 game list and EPM/xRAPM/DARKO CSVs were absent.",
            "Equal-season RMSE on this panel is not automatically the frozen 12-fold 13.614 number.",
            "2027 unused.",
        ],
        "summary": summary.sort_values("equal_season_rmse").to_dict("records"),
        "paired_intervals": intervals,
        "source_hashes": {
            "features": sha256_file(FEATURES),
            "targets": sha256_file(TARGETS),
            "runner": sha256_file(Path(__file__)),
            "protocol": sha256_file(ROOT / "research/pulse_search_protocol.py"),
        },
    }
    write_json_atomic(run, destination / "run.json")
    print(summary.sort_values("equal_season_rmse").to_string(index=False))
    print(json.dumps({"best": run["best_candidate"], "rmse": run["best_rmse"], "dual_gate": dual_gate}, indent=2))
    print(destination, flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons)
    SILVER.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    score_build = build_offline_official_scores(seasons)
    print(json.dumps({"official_scores": score_build}, indent=2), flush=True)
    official_table = pd.read_parquet(OFFICIAL)
    canonical_quality = rebuild_canonical_stints(seasons, official_table)
    scores = official_table.copy()
    score_metrics = {
        "games": int(len(scores)),
        "official_games": int(len(scores)),
        "score_source": "offline_espn_classic_v3",
        "canonical_quality": canonical_quality,
        **score_build,
    }
    print(json.dumps({"seasons": seasons, "score_metrics_games": score_metrics["games"]}, indent=2), flush=True)
    summaries, score_metrics = maybe_build_v3_fallback(seasons, score_metrics, combined_box())
    features = pd.read_parquet(FEATURES)
    targets = pd.read_parquet(TARGETS)
    hustle = pd.read_csv(HUSTLE)
    rating_seasons = rating_seasons_with_stints(seasons)
    rows = []
    fold_metrics = []
    prior_quality = []
    for rating_season in rating_seasons:
        fold_rows, fold_rows_metrics, quality = score_rating_fold(
            rating_season, features, targets, hustle, scores
        )
        rows.extend(fold_rows)
        fold_metrics.extend(fold_rows_metrics)
        prior_quality.extend(quality)
    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    summary = summarize_protocol(folds)
    write_real_protocol_run(
        seasons,
        rating_seasons,
        games,
        folds,
        summary,
        prior_quality,
        score_metrics,
        summaries,
        paired_intervals_vs_baseline(games, summary),
    )


if __name__ == "__main__":
    main()
