#!/usr/bin/env python3
"""Historical stint rebuild and nine-year RAPM targets for PULSE wave 3."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.manifest import sha256_file, write_json_atomic
from research.build_canonical_lineup_stints import build_season, convert_lineups
from research.build_canonical_rapm_targets import fit_window
from research.run_pulse_search_fetch_support import (
    EVENT_ROOT,
    classic_event_scores,
    download_canonical_inputs,
)
from research.run_pulse_search_h1_official_final import OUTPUT
from research.run_pulse_search_real_protocol import CANONICAL, OFFICIAL


NINE_YEAR_ROOT = OUTPUT / "canonical_nine_year_rapm_v1"
SCORE_COLUMNS = [
    "project_season",
    "season_type",
    "game_id",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "score_source",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce_game_date(game: pd.DataFrame, season: int) -> pd.DataFrame:
    output = game.copy()
    fallback = f"{season}-01-01"
    if "game_date" not in output.columns:
        output["game_date"] = fallback
        return output
    parsed = pd.to_datetime(output["game_date"], errors="coerce")
    output["game_date"] = parsed.fillna(pd.Timestamp(fallback)).dt.strftime("%Y-%m-%d")
    return output


def lineup_native_scores(season: int) -> pd.DataFrame:
    rows = pd.read_csv(convert_lineups(season))
    home = rows.loc[rows["location_team"].eq("home")].copy()
    home["game_id"] = home["game_id"].map(canonical_game_id)
    aggregations = {
        "home_score": ("pts_team", "sum"),
        "away_score": ("pts_opp", "sum"),
        "home_tricode": ("team", "first"),
        "away_tricode": ("opp", "first"),
    }
    if "game_date" in home.columns:
        aggregations["game_date"] = ("game_date", "first")
    game = home.groupby("game_id", as_index=False).agg(**aggregations)
    game = coerce_game_date(game, season)
    events = pd.read_parquet(
        EVENT_ROOT / f"season={season}" / "regular.parquet",
        columns=["GAME_ID", "PLAYER1_TEAM_ID", "PLAYER1_TEAM_ABBREVIATION", "_season_type"],
    )
    events = events.loc[events["_season_type"].eq("rg")].copy()
    events["game_id"] = events["GAME_ID"].map(canonical_game_id)
    teams = (
        events.dropna(subset=["PLAYER1_TEAM_ID", "PLAYER1_TEAM_ABBREVIATION"])
        .assign(
            team_id=lambda frame: pd.to_numeric(frame["PLAYER1_TEAM_ID"], errors="raise").astype("int64"),
            tricode=lambda frame: frame["PLAYER1_TEAM_ABBREVIATION"].astype(str).str.strip(),
        )
        .drop_duplicates(["game_id", "tricode"])
    )
    lookup = {
        game_id: dict(zip(frame["tricode"], frame["team_id"]))
        for game_id, frame in teams.groupby("game_id")
    }
    game["home_team_id"] = [
        lookup.get(game_id, {}).get(str(tricode).strip())
        for game_id, tricode in zip(game["game_id"], game["home_tricode"])
    ]
    game["away_team_id"] = [
        lookup.get(game_id, {}).get(str(tricode).strip())
        for game_id, tricode in zip(game["game_id"], game["away_tricode"])
    ]
    game["project_season"] = season
    game["season_type"] = "regular"
    game["score_source"] = "lineup_native_point_sums"
    return game.dropna(subset=["home_team_id", "away_team_id", "home_score", "away_score"])


def historical_official_scores(season: int) -> pd.DataFrame:
    native = lineup_native_scores(season)
    try:
        classic = classic_event_scores(season)
    except Exception as exc:
        print(f"classic scores {season} failed: {exc}", flush=True)
        classic = pd.DataFrame()
    if classic.empty:
        return native
    classic = classic.copy()
    classic["game_id"] = classic["game_id"].map(canonical_game_id)
    native["game_id"] = native["game_id"].map(canonical_game_id)
    classic = classic.loc[:, [column for column in SCORE_COLUMNS if column in classic.columns]]
    filled = native.merge(classic, on="game_id", how="left", suffixes=("", "_classic"))
    for column in ("home_score", "away_score", "home_team_id", "away_team_id", "game_date"):
        classic_column = f"{column}_classic"
        if classic_column in filled.columns:
            filled[column] = filled[classic_column].combine_first(filled[column])
    filled["score_source"] = np.where(
        filled["home_score_classic"].notna() if "home_score_classic" in filled.columns else False,
        "classic_pbp_last_SCORE",
        filled["score_source"],
    )
    filled["project_season"] = season
    filled["season_type"] = "regular"
    return filled[SCORE_COLUMNS].dropna(subset=["home_team_id", "away_team_id", "home_score", "away_score"])


def merge_official_scores(new_rows: pd.DataFrame) -> dict:
    existing = pd.read_parquet(OFFICIAL) if OFFICIAL.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
    combined["game_id"] = combined["game_id"].map(canonical_game_id)
    combined = combined.drop_duplicates("game_id", keep="first")
    OFFICIAL.parent.mkdir(parents=True, exist_ok=True)
    for season, group in combined.groupby("project_season"):
        destination = OFFICIAL.parent / f"project_season={int(season)}" / "regular.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(destination, index=False)
    combined.to_parquet(OFFICIAL, index=False)
    return {
        "games": int(len(combined)),
        "seasons": sorted(int(year) for year in combined["project_season"].unique()),
        "sha256": sha256_file(OFFICIAL),
    }


def prepare_historical_stints(start: int = 1997, end: int = 2013) -> dict:
    download = download_canonical_inputs(start, end)
    scores_rows = []
    quality = []
    official_table = pd.read_parquet(OFFICIAL) if OFFICIAL.exists() else pd.DataFrame()
    for season in range(start, end + 1):
        destination = CANONICAL / f"season={season}" / "regular.parquet"
        if destination.exists():
            quality.append({"season": season, "status": "existing"})
            continue
        scores_rows.append(historical_official_scores(season))
        official_table = pd.concat([official_table, scores_rows[-1]], ignore_index=True)
        official_table = official_table.drop_duplicates("game_id", keep="first")
        try:
            quality.append(build_season(season, official_table, CANONICAL))
            print(
                f"canonical {season}: stints={quality[-1]['stints']} "
                f"native={quality[-1]['native_score_reconciliation']:.3f}",
                flush=True,
            )
        except Exception as exc:
            quality.append({"season": season, "status": "failed", "error": str(exc)})
            print(f"canonical {season} failed: {exc}", flush=True)
    if scores_rows:
        merge_official_scores(pd.concat(scores_rows, ignore_index=True))
    return {
        "download": {"failed": sum(1 for row in download["files"] if row.get("status") == "failed")},
        "stints": quality,
    }


def fit_nine_year_targets(start_end: int = 2005, last_end: int = 2025) -> dict:
    checkpoints = NINE_YEAR_ROOT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    missing_seasons = []
    rows = []
    for end in range(start_end, last_end + 1):
        path = checkpoints / f"target_9y_{end}.parquet"
        seasons = tuple(range(end - 8, end + 1))
        absent = [season for season in seasons if not (CANONICAL / f"season={season}" / "regular.parquet").exists()]
        if path.exists():
            rows.append(pd.read_parquet(path))
            continue
        if absent:
            missing_seasons.append({"window_end": end, "missing": absent})
            print(f"nine-year {end}: missing stints {absent}", flush=True)
            continue
        ratings = fit_window(seasons, 3000).rename(
            columns={"offense": "target_offense", "defense": "target_defense", "net": "target_net"}
        )
        ratings["Window_End"] = end
        ratings["horizon"] = 9
        ratings["target_variant"] = "normal"
        ratings.to_parquet(path, index=False)
        rows.append(ratings)
        print(f"Nine-year RAPM ending {end}: {len(ratings)} players", flush=True)
    if not rows:
        raise FileNotFoundError("No nine-year RAPM windows could be fit.")
    targets = pd.concat(rows, ignore_index=True)
    destination = NINE_YEAR_ROOT / "targets.parquet"
    targets.to_parquet(destination, index=False)
    run = {
        "run_id": NINE_YEAR_ROOT.name,
        "created_at": utc_now(),
        "target_horizon": 9,
        "target_variant": "normal",
        "windows": sorted(int(value) for value in targets["Window_End"].unique()),
        "target_rows": int(len(targets)),
        "missing_windows": missing_seasons,
        "skipped_manifest_passed": True,
        "skipped_annual_rapm": True,
        "sha256": sha256_file(destination),
    }
    write_json_atomic(run, NINE_YEAR_ROOT / "run.json")
    return run
