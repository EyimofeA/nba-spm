#!/usr/bin/env python3
"""Smallest decisive PULSE H1 pilot: official-final vs technical-FT-excluded targets.

Rebuilds 2025-2026 regular-season V3 possession/lineup candidates, fits the
PULSE update (3000/4500/300) with and without identified technical free-throw
points, and scores identical next-season games on V3-terminal finals and on
box-sum finals.

LeagueGameLog is blocked in this environment. V3 last scoreHome/scoreAway is
the score-conserving possession target. Box player-point sums are an
independent official-final proxy. Public PULSE priors are descriptive refits
through 2026; pulse_* arms are leaky diagnostics that can only reject.

This is a research candidate. It does not replace canonical stints or spend 2027.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.historical_v3_lineups import build_historical_v3_lineup_candidate
from nba_impact.data.historical_v3_possession_lineups import (
    build_historical_v3_possession_lineup_candidate,
)
from nba_impact.data.historical_v3_possessions import build_historical_v3_possession_candidates
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.canonical_pulse import game_metrics, stint_prior_center
from research.pulse_search_h1_scores import (
    BOX_LOGS,
    OUTPUT,
    SCORES,
    SILVER,
    V3_NESTED,
    V3_PARENT,
    _normalize_box,
    build_player_games,
    build_score_reference,
    minutes_value,
    v3_terminal_scores,
)
from research.pulse_search_h1_stints import (
    _attach_possession_meta,
    apply_technical_points,
    fit_stint_arm,
    official_margins,
    paired_delta,
    public_priors,
    score_games,
    segments_to_stints,
    segments_to_terminal,
    tag_segment_technicals,
    technical_free_throws,
)

def build_sources(seasons: tuple[int, ...], scores: pd.DataFrame, players: Path) -> list[dict]:
    possession_root = SILVER / "possession_candidates"
    quality_poss = SILVER / "possession_quality.parquet"
    if not all((possession_root / f"project_season={season}" / "regular.parquet").exists() for season in seasons):
        build_historical_v3_possession_candidates(
            V3_NESTED,
            SCORES,
            possession_root,
            quality_poss,
            SILVER / "manifests",
            seasons=seasons,
            season_types=("regular",),
        )
    summaries = []
    for season in seasons:
        season_root = SILVER / f"project_season={season}"
        season_root.mkdir(parents=True, exist_ok=True)
        stints_path = season_root / "lineup_stints.parquet"
        lineup_quality = season_root / "lineup_quality.parquet"
        if not stints_path.exists():
            report = build_historical_v3_lineup_candidate(
                V3_PARENT,
                players,
                SCORES,
                stints_path,
                lineup_quality,
                season_root / "lineup_report.json",
                SILVER / "manifests",
                project_season=season,
                season_type="regular",
            )
            print(
                f"lineups {season}: passed={report['passed_game_count']} "
                f"quarantined={report['quarantined_game_count']}",
                flush=True,
            )
        attached_poss = season_root / "possessions.parquet"
        segments = season_root / "segments.parquet"
        assigned = season_root / "assigned_actions.parquet"
        attach_quality = season_root / "attach_quality.parquet"
        if not attached_poss.exists():
            attach = build_historical_v3_possession_lineup_candidate(
                V3_PARENT,
                possession_root / f"project_season={season}" / "regular.parquet",
                quality_poss,
                stints_path,
                lineup_quality,
                attached_poss,
                segments,
                assigned,
                attach_quality,
                season_root / "attach_report.json",
                SILVER / "manifests",
                project_season=season,
                season_type="regular",
            )
            print(
                f"attach {season}: emitted={attach['emitted_game_count']} "
                f"rejected={attach['rejected_after_attachment_count']}",
                flush=True,
            )
        v3 = pd.read_parquet(V3_NESTED / f"project_season={season}" / "regular.parquet")
        tech = technical_free_throws(v3)
        assigned_frame = pd.read_parquet(assigned)
        possessions = apply_technical_points(
            pd.read_parquet(attached_poss),
            assigned_frame,
            tech,
        )
        tagged_segments = tag_segment_technicals(pd.read_parquet(segments), assigned_frame, tech)
        stints = segments_to_stints(possessions, tagged_segments)
        terminal = segments_to_terminal(possessions, tagged_segments)
        stints.to_parquet(season_root / "canonical_like_stints.parquet", index=False)
        terminal.to_parquet(season_root / "terminal_like_stints.parquet", index=False)
        possessions.to_parquet(season_root / "possessions_with_tech.parquet", index=False)
        summaries.append({
            "season": season,
            "possession_games": int(possessions["game_id"].nunique()),
            "stints": int(len(stints)),
            "terminal_rows": int(len(terminal)),
            "technical_points": int(possessions["technical_points"].sum()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2025,2026")
    args = parser.parse_args()
    seasons = tuple(int(item) for item in args.seasons.split(",") if item.strip())
    SILVER.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    box = _normalize_box(pd.read_parquet(BOX_LOGS)) if BOX_LOGS.exists() else pd.DataFrame()
    scores, score_metrics = build_score_reference(seasons, box)
    print(json.dumps({"score_metrics": score_metrics}, indent=2), flush=True)
    players = SILVER / "player_games.parquet"
    if not players.exists():
        build_player_games(scores, BOX_LOGS, players, seasons)
    summaries = build_sources(seasons, scores, players)
    print(json.dumps(summaries, indent=2), flush=True)

    rating_seasons = [season for season in seasons if season + 1 in seasons]
    rows = []
    fold_metrics = []
    for rating_season in rating_seasons:
        source_stints = pd.read_parquet(SILVER / f"project_season={rating_season}" / "canonical_like_stints.parquet")
        target_stints = pd.read_parquet(SILVER / f"project_season={rating_season + 1}" / "canonical_like_stints.parquet")
        priors = public_priors(rating_season)
        official = official_margins(scores, rating_season + 1)
        for target_name, columns in {
            "excl": ("home_points_excl", "away_points_excl"),
            "incl": ("home_points", "away_points"),
        }.items():
            source_design, fits, coverage, _center = fit_stint_arm(
                source_stints, columns, priors, rating_season, scale=1.0
            )
            target_frame = target_stints.copy()
            target_frame["home_points"] = target_frame[columns[0]]
            target_frame["away_points"] = target_frame[columns[1]]
            target_design = build_stint_design(target_frame)
            for model, (beta, intercept) in fits.items():
                scored = score_games(
                    source_design, target_design, beta, intercept, official,
                    f"{model}_{target_name}", rating_season,
                )
                rows.append(scored)
                for actual, label in (("official_margin", "official"), ("excl_margin", "excl"), ("v3_margin", "v3")):
                    metrics = game_metrics(scored.assign(actual_margin=scored[actual]))
                    fold_metrics.append({
                        "candidate": f"{model}_{target_name}",
                        "actual": label,
                        "rating_season": rating_season,
                        "outcome_season": rating_season + 1,
                        "prior_players_with_prior": coverage["players_with_prior"],
                        **metrics,
                    })
            for scale in (0.5, 0.75):
                _, scaled, _, _ = fit_stint_arm(source_stints, columns, priors, rating_season, scale=scale)
                scored = score_games(
                    source_design, target_design, *scaled["pulse"], official,
                    f"pulse_{target_name}_scale{scale}", rating_season,
                )
                rows.append(scored)
                metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
                fold_metrics.append({
                    "candidate": f"pulse_{target_name}_scale{scale}",
                    "actual": "official",
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    **metrics,
                })
        print(f"H1 fold {rating_season}->{rating_season + 1} complete", flush=True)

    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    summary = folds.groupby(["candidate", "actual"], as_index=False).agg(
        folds=("outcome_season", "nunique"),
        games=("games", "sum"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    core = games.loc[games["candidate"].isin(["pulse_incl", "pulse_excl", "rapm_incl", "rapm_excl"])]
    intervals = {
        "pulse_incl_minus_pulse_excl_official_mse": paired_delta(
            core, "pulse_incl", "pulse_excl", "squared_error_official"
        ),
        "rapm_incl_minus_rapm_excl_official_mse": paired_delta(
            core, "rapm_incl", "rapm_excl", "squared_error_official"
        ),
        "pulse_incl_minus_rapm_incl_official_mse": paired_delta(
            core, "pulse_incl", "rapm_incl", "squared_error_official"
        ),
        "pulse_excl_minus_rapm_excl_official_mse": paired_delta(
            core, "pulse_excl", "rapm_excl", "squared_error_official"
        ),
    }
    run_id = "pulse_search_h1_official_final_v1"
    destination = OUTPUT / run_id
    destination.mkdir(parents=True, exist_ok=True)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    summary.to_parquet(destination / "summary.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "hypothesis": "Score and fit the official-final target, including technical free throws.",
        "seasons": list(seasons),
        "score_metrics": score_metrics,
        "season_build": summaries,
        "source_hashes": {
            "scores": sha256_file(SCORES),
            "runner": sha256_file(Path(__file__)),
        },
        "limitations": [
            "V3 historical possession/lineup candidates, not canonical CDN/Gabriel stints.",
            "LeagueGameLog blocked; official proxy is box player-point sums, else V3 terminal scores.",
            "Public PULSE priors are descriptive refits through 2026; pulse_* arms are leaky diagnostics.",
            "Pilot folds only; not the frozen 12-fold gate.",
            "2027 unused.",
        ],
        "summary": summary.to_dict("records"),
        "paired_intervals": intervals,
    }
    write_json_atomic(run, destination / "run.json")
    print(summary.to_string(index=False))
    print(json.dumps(intervals, indent=2))
    print(destination, flush=True)


if __name__ == "__main__":
    main()
