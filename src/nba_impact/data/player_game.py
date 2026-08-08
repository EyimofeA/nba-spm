"""Canonical player-game box scores and exact starter seeds."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


_ISO_MINUTES = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:([0-9.]+)S)?$")


def minutes_to_seconds(value: object) -> float:
    """Parse NBA ``MM:SS`` or ISO-8601 duration strings."""
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
    match = _ISO_MINUTES.fullmatch(text)
    if match:
        hours, minutes, seconds = match.groups()
        return float(hours or 0) * 3600.0 + float(minutes or 0) * 60.0 + float(seconds or 0)
    raise ValueError(f"Unsupported minutes value: {value!r}")


def _load_espn(path: Path) -> pd.DataFrame:
    columns = [
        "game_id",
        "player_id",
        "team",
        "home",
        "starter",
        "played",
        "dAvgPos",
        "oNetPts",
        "dNetPts",
        "tNetPts",
        "oUsg",
        "dUsg",
        "plusMinusPoints",
        "oWPA",
        "dWPA",
        "tWPA",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["game_id"] = frame["game_id"].map(canonical_game_id)
    frame = frame.rename(
        columns={
            "player_id": "player_id",
            "team": "espn_team_tricode",
            "home": "espn_home",
            "starter": "espn_starter",
            "played": "espn_played",
            "dAvgPos": "espn_defensive_average_position",
            "oNetPts": "espn_offensive_net_points",
            "dNetPts": "espn_defensive_net_points",
            "tNetPts": "espn_total_net_points",
            "oUsg": "espn_offensive_usage",
            "dUsg": "espn_defensive_usage",
            "plusMinusPoints": "espn_plus_minus",
            "oWPA": "espn_offensive_wpa",
            "dWPA": "espn_defensive_wpa",
            "tWPA": "espn_total_wpa",
        }
    )
    return frame


def build_player_games(
    box_path: str | Path,
    espn_path: str | Path,
    game_dim_path: str | Path,
    destination: str | Path,
    manifest_dir: str | Path,
) -> dict:
    """Build a validated player-game table; NBA box starters are authoritative."""
    box_source = Path(box_path)
    espn_source = Path(espn_path)
    game_source = Path(game_dim_path)
    games = pd.read_parquet(game_source)
    box = pd.read_parquet(
        box_source,
        columns=[
            "gameId",
            "teamId",
            "personId",
            "firstName",
            "familyName",
            "position",
            "comment",
            "minutes",
        ],
    )
    box["game_id"] = box["gameId"].map(canonical_game_id)
    box = box.loc[box["game_id"].isin(set(games["game_id"]))].copy()
    box = box.rename(
        columns={
            "teamId": "team_id",
            "personId": "player_id",
            "firstName": "first_name",
            "familyName": "family_name",
            "position": "starter_position",
        }
    )
    box["starter_position"] = box["starter_position"].fillna("").astype(str).str.strip()
    box["starter"] = box["starter_position"].ne("")
    box["minutes_seconds"] = box["minutes"].map(minutes_to_seconds)
    box["played"] = box["minutes_seconds"].gt(0)
    box["player_name"] = (
        box["first_name"].fillna("").str.strip() + " " + box["family_name"].fillna("").str.strip()
    ).str.strip()

    game_columns = [
        "game_id",
        "season_start",
        "season_end",
        "season_label",
        "season_type",
        "game_date",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
        "max_period",
    ]
    box = box.merge(games[game_columns], on="game_id", how="left", validate="many_to_one")
    box["team_side"] = np.select(
        [box["team_id"].eq(box["home_team_id"]), box["team_id"].eq(box["away_team_id"])],
        ["home", "away"],
        default="unknown",
    )
    box["team_tricode"] = np.where(
        box["team_side"].eq("home"), box["home_team_tricode"], box["away_team_tricode"]
    )

    espn = _load_espn(espn_source)
    espn_duplicate_rows = int(espn.duplicated(["game_id", "player_id"], keep=False).sum())
    if espn_duplicate_rows:
        espn = espn.drop_duplicates(["game_id", "player_id"], keep="first")
    box = box.merge(espn, on=["game_id", "player_id"], how="left", validate="one_to_one")
    box["espn_available"] = box["espn_starter"].notna()

    starter_team = box.groupby(["game_id", "team_id"], as_index=False).agg(
        starter_count=("starter", "sum"), player_rows=("player_id", "size")
    )
    game_team_count = box.groupby("game_id")["team_id"].nunique()
    boxed_games = set(box["game_id"])
    espn_games = set(espn["game_id"])
    expected_team_seconds = (
        2880.0 + (box["max_period"].clip(lower=4) - 4) * 300.0
    ) * 5.0
    minute_checks = box.assign(expected_team_seconds=expected_team_seconds).groupby(
        ["game_id", "team_id"], as_index=False
    ).agg(
        actual_seconds=("minutes_seconds", "sum"),
        expected_seconds=("expected_team_seconds", "first"),
    )
    minute_checks["absolute_error"] = (
        minute_checks["actual_seconds"] - minute_checks["expected_seconds"]
    ).abs()

    starter_overlap = box.loc[box["espn_available"]]
    issues = {
        "duplicate_player_games": int(box.duplicated(["game_id", "player_id"], keep=False).sum()),
        "duplicate_espn_player_games": espn_duplicate_rows,
        "missing_game_boxes": int(len(set(games["game_id"]) - boxed_games)),
        "invalid_team_identity_rows": int(box["team_side"].eq("unknown").sum()),
        "games_without_two_teams": int(game_team_count.ne(2).sum()),
        "team_games_without_five_starters": int(starter_team["starter_count"].ne(5).sum()),
        "team_games_minutes_off_by_over_five_seconds": int(minute_checks["absolute_error"].gt(5).sum()),
        "starter_disagreements_with_espn": int(
            starter_overlap["starter"].ne(starter_overlap["espn_starter"].astype(bool)).sum()
        ),
    }
    critical = {
        "duplicate_player_games",
        "missing_game_boxes",
        "invalid_team_identity_rows",
        "games_without_two_teams",
        "team_games_without_five_starters",
        "team_games_minutes_off_by_over_five_seconds",
    }
    passed = not any(issues[key] for key in critical)

    output_columns = [
        "game_id",
        "season_start",
        "season_end",
        "season_label",
        "season_type",
        "game_date",
        "team_id",
        "team_tricode",
        "team_side",
        "player_id",
        "first_name",
        "family_name",
        "player_name",
        "starter_position",
        "starter",
        "played",
        "minutes",
        "minutes_seconds",
        "comment",
        "espn_available",
        "espn_team_tricode",
        "espn_starter",
        "espn_played",
        "espn_defensive_average_position",
        "espn_offensive_net_points",
        "espn_defensive_net_points",
        "espn_total_net_points",
        "espn_offensive_usage",
        "espn_defensive_usage",
        "espn_plus_minus",
        "espn_offensive_wpa",
        "espn_defensive_wpa",
        "espn_total_wpa",
    ]
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    box.loc[:, output_columns].sort_values(["game_id", "team_side", "starter", "player_id"]).to_parquet(
        temporary, index=False
    )
    temporary.replace(output)

    source_files = [box_source, espn_source, game_source]
    source_records = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    ]
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_records]).encode("utf-8")
    ).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"player_games_{identity}",
        "dataset": "player_games",
        "grain": "one player in one NBA game",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "row_count": int(len(box)),
        "game_count": int(box["game_id"].nunique()),
        "player_count": int(box["player_id"].nunique()),
        "espn_game_count": int(len(boxed_games & espn_games)),
        "espn_missing_game_count": int(len(boxed_games - espn_games)),
        "issues": issues,
        "license_note": (
            "The llimllib/nba_data repository does not declare a license. ESPN-derived fields are "
            "research-only inputs and must not be redistributed until rights are clarified."
        ),
        "path": str(output.resolve()),
        "source_files": source_records,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
