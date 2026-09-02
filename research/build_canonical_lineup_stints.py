#!/usr/bin/env python3
"""Build score-conserving, technical-FT-excluded lineup stints for RAPM."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
PLAYER_ID = re.compile(r"^\s*(\d+)\s+")


def elapsed_seconds(period: int, clock: object) -> float:
    text = str(clock).strip()
    iso = re.fullmatch(r"PT(\d+)M([\d.]+)S", text)
    if iso:
        remaining = 60 * int(iso.group(1)) + float(iso.group(2))
    else:
        minutes, seconds = text.split(":", maxsplit=1)
        remaining = 60 * int(minutes) + float(seconds)
    period_length = 720 if period <= 4 else 300
    start = 720 * min(period - 1, 4) + 300 * max(period - 5, 0)
    return start + period_length - remaining


def lineup_ids(value: object) -> tuple[int, ...]:
    ids = []
    for player in str(value).split(","):
        match = PLAYER_ID.match(player)
        if not match:
            raise ValueError(f"Could not parse lineup member: {player!r}")
        ids.append(int(match.group(1)))
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError(f"Lineup is not five unique players: {value!r}")
    return tuple(sorted(ids))


def convert_lineups(season: int) -> Path:
    source = (
        ROOT
        / "data/lake/bronze/canonical_historical_lineups"
        / f"season={season}"
        / "regular.rds"
    )
    destination = source.with_suffix(".csv.gz")
    if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            ["Rscript", str(ROOT / "research/convert_historical_lineups.R"), str(source), str(destination)],
            check=True,
        )
    return destination


def technical_free_throws(season: int) -> pd.DataFrame:
    classic = (
        ROOT
        / "data/lake/bronze/canonical_historical_events"
        / f"season={season}"
        / "regular.parquet"
    )
    if classic.exists():
        events = pd.read_parquet(
            classic,
            columns=[
                "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD", "PCTIMESTRING",
                "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION",
                "PLAYER1_TEAM_ID", "_season_type",
            ],
        )
        events = events.loc[events["_season_type"].eq("rg")].copy()
        free_throw = events["EVENTMSGTYPE"].eq(3)
        descriptions = events.loc[
            free_throw,
            ["HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION"],
        ]
        technical = pd.Series(False, index=descriptions.index)
        missed = pd.Series(False, index=descriptions.index)
        for column in descriptions:
            values = descriptions[column].astype("string")
            technical |= values.str.contains("technical", case=False, na=False)
            missed |= values.str.contains(r"\bMISS\b", case=False, na=False)
        made = pd.Series(False, index=events.index)
        made.loc[descriptions.index] = technical & ~missed
        output = events.loc[made, [
            "GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING", "PLAYER1_TEAM_ID"
        ]].rename(
            columns={
                "GAME_ID": "game_id", "EVENTNUM": "event_order", "PERIOD": "period",
                "PCTIMESTRING": "clock", "PLAYER1_TEAM_ID": "team_id",
            }
        )
    else:
        v3 = (
            ROOT
            / "data/lake/bronze/nba_data_archive_scoring"
            / "revision=dfa8fa43"
            / "nbastatsv3"
            / f"project_season={season}"
            / "regular.parquet"
        )
        events = pd.read_parquet(
            v3,
            columns=[
                "gameId", "actionId", "period", "clock", "teamId", "actionType",
                "subType", "description",
            ],
        )
        made = events["actionType"].astype(str).str.casefold().eq("free throw")
        made &= events["subType"].astype(str).str.contains("technical", case=False, na=False)
        made &= ~events["description"].astype(str).str.contains(
            r"^\s*MISS\b", case=False, na=False
        )
        output = events.loc[
            made, ["gameId", "actionId", "period", "clock", "teamId"]
        ].rename(
            columns={
                "gameId": "game_id", "actionId": "event_order", "teamId": "team_id"
            }
        )
    output["game_id"] = output["game_id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    ).str.zfill(10)
    output["team_id"] = pd.to_numeric(output["team_id"], errors="raise").astype("int64")
    output["elapsed_seconds"] = [
        elapsed_seconds(int(period), clock)
        for period, clock in zip(output["period"], output["clock"], strict=True)
    ]
    output["points"] = 1
    return output.sort_values(["game_id", "period", "elapsed_seconds", "event_order"])


def audit_existing_season(
    season: int,
    scores: pd.DataFrame,
    destination: Path,
) -> dict:
    output = pd.read_parquet(destination)
    official = scores.loc[
        scores["project_season"].eq(season) & scores["season_type"].eq("regular")
    ].copy()
    official["game_id"] = official["game_id"].astype(str).str.zfill(10)
    game = output.groupby("game_id", as_index=False).agg(
        home_points_model=("home_points", "sum"),
        away_points_model=("away_points", "sum"),
        home_technical_points=("home_technical_points_excluded", "sum"),
        away_technical_points=("away_technical_points_excluded", "sum"),
    ).merge(
        official[["game_id", "home_score", "away_score"]],
        on="game_id",
        how="outer",
        validate="one_to_one",
    )
    native = (
        (game["home_points_model"] + game["home_technical_points"]).eq(game["home_score"])
        & (game["away_points_model"] + game["away_technical_points"]).eq(game["away_score"])
    )
    tech = technical_free_throws(season)
    excluded = int(
        output["home_technical_points_excluded"].sum()
        + output["away_technical_points_excluded"].sum()
    )
    player_columns = [
        f"{side}_player_{number}"
        for side in ("home", "away")
        for number in range(1, 6)
    ]
    return {
        "season": season,
        "official_games": int(len(official)),
        "lineup_games": int(output["game_id"].nunique()),
        "stints": int(len(output)),
        "valid_lineup_fraction": float(output[player_columns].nunique(axis=1).eq(10).mean()),
        "native_score_reconciliation": float(native.mean()),
        "model_plus_technical_reconciliation": float(native.mean()),
        "technical_points_excluded": excluded,
        "unmatched_technical_free_throws": int(len(tech) - excluded),
        "relative_path": str(destination.relative_to(ROOT)),
        "sha256": sha256_file(destination),
    }


def build_season(season: int, scores: pd.DataFrame, output_root: Path) -> dict:
    source = convert_lineups(season)
    rows = pd.read_csv(source)
    rows = rows.loc[rows["location_team"].eq("home")].copy()
    rows["game_id"] = rows["game_id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    ).str.zfill(10)
    rows["home_lineup"] = rows["lineup_team"].map(lineup_ids)
    rows["away_lineup"] = rows["lineup_opp"].map(lineup_ids)
    if rows.duplicated(["game_id", "period", "stint"]).any():
        raise ValueError(f"Season {season} has duplicate home stint rows")

    official = scores.loc[
        scores["project_season"].eq(season) & scores["season_type"].eq("regular")
    ].copy()
    official["game_id"] = official["game_id"].astype(str).str.zfill(10)
    rows = rows.merge(
        official[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]],
        on="game_id", how="left", validate="many_to_one",
    )
    if rows[["home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError(f"Season {season} lineup games do not match official games")

    tech = technical_free_throws(season)
    rows["home_technical_points"] = 0
    rows["away_technical_points"] = 0
    unmatched = 0
    grouped_indices = {
        game_id: group.index.to_numpy()
        for game_id, group in rows.groupby("game_id", sort=False)
    }
    for event in tech.itertuples(index=False):
        indices = grouped_indices.get(str(event.game_id))
        if indices is None:
            unmatched += 1
            continue
        candidates = rows.loc[indices]
        period_candidates = candidates.loc[candidates["period"].eq(int(event.period))]
        contains = candidates["period"].eq(int(event.period)) & candidates[
            "secs_game_start"
        ].le(float(event.elapsed_seconds)) & candidates["secs_game_end"].ge(
            float(event.elapsed_seconds)
        )
        matching = candidates.loc[contains].copy()
        identity_rows = matching if not matching.empty else period_candidates
        if identity_rows.empty:
            unmatched += 1
            continue
        if int(event.team_id) == int(identity_rows["home_team_id"].iloc[0]):
            point_column, technical_column = "pts_team", "home_technical_points"
        elif int(event.team_id) == int(identity_rows["away_team_id"].iloc[0]):
            point_column, technical_column = "pts_opp", "away_technical_points"
        else:
            unmatched += 1
            continue
        # A substitution can share the technical-FT clock. Choose a matching
        # interval that still contains an unassigned point for this side.
        matching = matching.loc[
            matching[point_column].sub(matching[technical_column]).gt(0)
        ].copy()
        if matching.empty:
            # Some source stints place same-clock substitutions on the other
            # side of the event boundary. Fall back to the nearest scoring
            # stint in the same period; the technical point is removed from
            # the model, so this does not invent a player-impact outcome.
            matching = period_candidates.loc[
                period_candidates[point_column]
                .sub(period_candidates[technical_column])
                .gt(0)
            ].copy()
            if matching.empty:
                unmatched += 1
                continue
        matching["clock_distance"] = (
            (matching["secs_game_start"] + matching["secs_game_end"]) / 2
            - float(event.elapsed_seconds)
        ).abs()
        index = int(matching.sort_values(["clock_distance", "stint"]).index[0])
        rows.at[index, technical_column] += 1

    rows["home_points_model"] = rows["pts_team"] - rows["home_technical_points"]
    rows["away_points_model"] = rows["pts_opp"] - rows["away_technical_points"]
    if (rows[["home_points_model", "away_points_model"]] < 0).any().any():
        raise ValueError(f"Season {season} has negative points after technical FT exclusion")
    game = rows.groupby("game_id", as_index=False).agg(
        home_points=("pts_team", "sum"), away_points=("pts_opp", "sum"),
        home_technical_points=("home_technical_points", "sum"),
        away_technical_points=("away_technical_points", "sum"),
        home_points_model=("home_points_model", "sum"),
        away_points_model=("away_points_model", "sum"),
    ).merge(
        official[["game_id", "home_score", "away_score"]], on="game_id",
        how="outer", validate="one_to_one",
    )
    game["native_score_reconciled"] = game["home_points"].eq(game["home_score"]) & game[
        "away_points"
    ].eq(game["away_score"])
    game["model_plus_technical_reconciled"] = (
        game["home_points_model"] + game["home_technical_points"]
    ).eq(game["home_score"]) & (
        game["away_points_model"] + game["away_technical_points"]
    ).eq(game["away_score"])

    output = pd.DataFrame(
        {
            "season": season,
            "game_id": rows["game_id"],
            "period": rows["period"].astype(int),
            "stint": rows["stint"].astype(int),
            "start_seconds": rows["secs_game_start"],
            "end_seconds": rows["secs_game_end"],
            "home_team_id": rows["home_team_id"].astype(int),
            "away_team_id": rows["away_team_id"].astype(int),
            "home_possessions": rows["poss_team"].astype(int),
            "away_possessions": rows["poss_opp"].astype(int),
            "home_points": rows["home_points_model"].astype(int),
            "away_points": rows["away_points_model"].astype(int),
            "home_technical_points_excluded": rows["home_technical_points"].astype(int),
            "away_technical_points_excluded": rows["away_technical_points"].astype(int),
        }
    )
    for side in ("home", "away"):
        for number in range(1, 6):
            output[f"{side}_player_{number}"] = [
                lineup[number - 1] for lineup in rows[f"{side}_lineup"]
            ]
    destination = output_root / f"season={season}" / "regular.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    return {
        "season": season,
        "official_games": int(len(official)),
        "lineup_games": int(rows["game_id"].nunique()),
        "stints": int(len(output)),
        "valid_lineup_fraction": float(
            output[[f"{side}_player_{number}" for side in ("home", "away") for number in range(1, 6)]]
            .nunique(axis=1).eq(10).mean()
        ),
        "native_score_reconciliation": float(game["native_score_reconciled"].mean()),
        "model_plus_technical_reconciliation": float(
            game["model_plus_technical_reconciled"].mean()
        ),
        "technical_points_excluded": int(tech["points"].sum() - unmatched),
        "unmatched_technical_free_throws": int(unmatched),
        "relative_path": str(destination.relative_to(ROOT)),
        "sha256": sha256_file(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1997)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    scores = pd.read_parquet(
        ROOT / "data/lake/bronze/official_game_scores/official_game_scores.parquet"
    )
    output_root = ROOT / "data/lake/silver/canonical_lineup_stints"
    quality = []
    for season in range(args.start, args.end + 1):
        destination = output_root / f"season={season}" / "regular.parquet"
        quality.append(
            audit_existing_season(season, scores, destination)
            if destination.exists() and not args.rebuild
            else build_season(season, scores, output_root)
        )
    quality_frame = pd.DataFrame(quality)
    quality_path = output_root / "season_quality.parquet"
    quality_frame.to_parquet(quality_path, index=False)
    manifest = {
        "schema_version": "canonical_lineup_stints_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": [args.start, args.end],
        "passed": bool(
            quality_frame["native_score_reconciliation"].ge(.99).all()
            and quality_frame["model_plus_technical_reconciliation"].ge(.99).all()
            and quality_frame["valid_lineup_fraction"].ge(.95).all()
            and quality_frame["unmatched_technical_free_throws"].eq(0).all()
        ),
        "minimum_native_score_reconciliation": float(quality_frame["native_score_reconciliation"].min()),
        "minimum_model_plus_technical_reconciliation": float(quality_frame["model_plus_technical_reconciliation"].min()),
        "minimum_valid_lineup_fraction": float(quality_frame["valid_lineup_fraction"].min()),
        "unmatched_technical_free_throws": int(quality_frame["unmatched_technical_free_throws"].sum()),
        "technical_points_excluded": int(quality_frame["technical_points_excluded"].sum()),
        "quality_sha256": sha256_file(quality_path),
    }
    write_json_atomic(manifest, output_root / "manifest.json")
    print(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise ValueError("Canonical lineup stint contract failed")


if __name__ == "__main__":
    main()
