"""Build the canonical game dimension from source-native event partitions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .manifest import sha256_file, write_json_atomic


def canonical_game_id(value: object) -> str:
    return f"{int(value):010d}"


def _partition(path: Path) -> tuple[int, str]:
    return int(path.parent.name.split("=", maxsplit=1)[1]), path.stem


def _events_table(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "nbastatsv3").rglob("*.parquet")):
        season_start, season_type = _partition(path)
        frame = pd.read_parquet(
            path,
            columns=[
                "gameId",
                "actionId",
                "period",
                "location",
                "teamId",
                "teamTricode",
                "scoreHome",
                "scoreAway",
            ],
        )
        frame["game_id"] = frame["gameId"].map(canonical_game_id)
        games = frame.groupby("game_id", as_index=False).agg(
            event_rows=("actionId", "size"),
            event_actions=("actionId", "nunique"),
            max_period=("period", "max"),
            home_score=("scoreHome", "max"),
            away_score=("scoreAway", "max"),
        )
        teams = (
            frame.loc[(frame["teamId"] > 0) & frame["location"].isin(["h", "v"])]
            .groupby(["game_id", "location"], as_index=False)
            .agg(
                team_id=("teamId", "first"),
                team_tricode=("teamTricode", "first"),
                distinct_team_ids=("teamId", "nunique"),
            )
        )
        home = teams.loc[teams["location"] == "h"].rename(
            columns={
                "team_id": "home_team_id",
                "team_tricode": "home_team_tricode",
                "distinct_team_ids": "home_team_id_count",
            }
        )
        away = teams.loc[teams["location"] == "v"].rename(
            columns={
                "team_id": "away_team_id",
                "team_tricode": "away_team_tricode",
                "distinct_team_ids": "away_team_id_count",
            }
        )
        games = games.merge(
            home[["game_id", "home_team_id", "home_team_tricode", "home_team_id_count"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        ).merge(
            away[["game_id", "away_team_id", "away_team_tricode", "away_team_id_count"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        games["source_season"] = season_start
        games["season_start"] = season_start
        games["season_end"] = season_start + 1
        games["season_label"] = f"{season_start}-{str(season_start + 1)[-2:]}"
        games["season_type"] = season_type
        frames.append(games)
    if not frames:
        raise ValueError(f"No NBA Stats V3 partitions found under {root}")
    return pd.concat(frames, ignore_index=True)


def _shot_games(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "shotdetail").rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["GAME_ID", "GAME_DATE", "HTM", "VTM"])
        frame["game_id"] = frame["GAME_ID"].map(canonical_game_id)
        frame["shot_game_date"] = pd.to_datetime(
            frame["GAME_DATE"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
        games = frame.groupby("game_id", as_index=False).agg(
            shot_game_date=("shot_game_date", "first"),
            shot_home_tricode=("HTM", "first"),
            shot_away_tricode=("VTM", "first"),
            shot_rows=("GAME_ID", "size"),
            shot_date_values=("shot_game_date", "nunique"),
        )
        frames.append(games)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "game_id",
                "shot_game_date",
                "shot_home_tricode",
                "shot_away_tricode",
                "shot_rows",
                "shot_date_values",
            ]
        )
    )


def _pbp_games(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "pbpstats").rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["GAMEID", "GAMEDATE"])
        frame["game_id"] = frame["GAMEID"].map(canonical_game_id)
        frame["pbp_game_date"] = pd.to_datetime(frame["GAMEDATE"], errors="coerce")
        games = frame.groupby("game_id", as_index=False).agg(
            pbp_game_date=("pbp_game_date", "first"),
            pbp_rows=("GAMEID", "size"),
            pbp_date_values=("pbp_game_date", "nunique"),
        )
        frames.append(games)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["game_id", "pbp_game_date", "pbp_rows", "pbp_date_values"])
    )


def _matchup_games(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "matchups").rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["game_id", "home_team_id", "away_team_id"])
        frame["game_id"] = frame["game_id"].map(canonical_game_id)
        games = frame.groupby("game_id", as_index=False).agg(
            matchup_rows=("home_team_id", "size"),
            matchup_home_team_id=("home_team_id", "first"),
            matchup_away_team_id=("away_team_id", "first"),
            matchup_home_team_values=("home_team_id", "nunique"),
            matchup_away_team_values=("away_team_id", "nunique"),
        )
        frames.append(games)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "game_id",
                "matchup_rows",
                "matchup_home_team_id",
                "matchup_away_team_id",
                "matchup_home_team_values",
                "matchup_away_team_values",
            ]
        )
    )


def build_game_dimension(root: str | Path, destination: str | Path, manifest_dir: str | Path) -> dict:
    source_root = Path(root)
    games = _events_table(source_root)
    shots = _shot_games(source_root)
    pbp = _pbp_games(source_root)
    matchups = _matchup_games(source_root)
    games = games.merge(shots, on="game_id", how="left", validate="one_to_one")
    games = games.merge(pbp, on="game_id", how="left", validate="one_to_one")
    games = games.merge(matchups, on="game_id", how="left", validate="one_to_one")

    games["shot_game_date"] = pd.to_datetime(games["shot_game_date"], errors="coerce")
    games["pbp_game_date"] = pd.to_datetime(games["pbp_game_date"], errors="coerce")
    games["game_date"] = games["shot_game_date"].combine_first(games["pbp_game_date"])
    games["game_date_source"] = "shotdetail"
    games.loc[games["shot_game_date"].isna() & games["pbp_game_date"].notna(), "game_date_source"] = "pbpstats"
    games.loc[games["game_date"].isna(), "game_date_source"] = pd.NA
    games["home_score"] = games["home_score"].astype("Int64")
    games["away_score"] = games["away_score"].astype("Int64")
    games["home_margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = games["home_margin"] > 0
    games["is_overtime"] = games["max_period"] > 4
    games["has_shotdetail"] = games["shot_rows"].notna()
    games["has_pbpstats"] = games["pbp_rows"].notna()
    games["has_matchups"] = games["matchup_rows"].notna()

    issues = {
        "duplicate_game_ids": int(games.duplicated("game_id", keep=False).sum()),
        "missing_game_dates": int(games["game_date"].isna().sum()),
        "missing_final_scores": int(games[["home_score", "away_score"]].isna().any(axis=1).sum()),
        "invalid_team_identity": int(
            (
                games[["home_team_id", "away_team_id"]].isna().any(axis=1)
                | games["home_team_id"].eq(games["away_team_id"])
                | games["home_team_id_count"].ne(1)
                | games["away_team_id_count"].ne(1)
            ).sum()
        ),
        "event_action_key_mismatch": int(games["event_rows"].ne(games["event_actions"]).sum()),
        "conflicting_source_dates": int(
            (
                games["shot_game_date"].notna()
                & games["pbp_game_date"].notna()
                & games["shot_game_date"].ne(games["pbp_game_date"])
            ).sum()
        ),
        "shot_team_code_mismatch": int(
            (
                games["has_shotdetail"]
                & (
                    games["home_team_tricode"].ne(games["shot_home_tricode"])
                    | games["away_team_tricode"].ne(games["shot_away_tricode"])
                )
            ).sum()
        ),
        "matchup_team_id_mismatch": int(
            (
                games["has_matchups"]
                & (
                    games["home_team_id"].ne(games["matchup_home_team_id"])
                    | games["away_team_id"].ne(games["matchup_away_team_id"])
                )
            ).sum()
        ),
    }
    critical_keys = {
        "duplicate_game_ids",
        "missing_game_dates",
        "missing_final_scores",
        "invalid_team_identity",
        "event_action_key_mismatch",
        "conflicting_source_dates",
        "shot_team_code_mismatch",
        "matchup_team_id_mismatch",
    }
    passed = not any(issues[key] for key in critical_keys)

    output_columns = [
        "game_id",
        "source_season",
        "season_start",
        "season_end",
        "season_label",
        "season_type",
        "game_date",
        "game_date_source",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
        "home_score",
        "away_score",
        "home_margin",
        "home_win",
        "max_period",
        "is_overtime",
        "event_rows",
        "has_shotdetail",
        "has_pbpstats",
        "has_matchups",
    ]
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    games.loc[:, output_columns].sort_values(["game_date", "game_id"]).to_parquet(temporary, index=False)
    temporary.replace(output)

    source_files = []
    for path in sorted(source_root.rglob("*.parquet")):
        source_files.append(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_files]).encode("utf-8")
    ).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"game_dim_{identity}",
        "dataset": "game_dim",
        "grain": "one NBA game",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "row_count": int(len(games)),
        "season_labels": sorted(games["season_label"].unique().tolist()),
        "season_type_counts": {
            str(key): int(value) for key, value in games["season_type"].value_counts().items()
        },
        "source_coverage": {
            "shotdetail_games": int(games["has_shotdetail"].sum()),
            "pbpstats_games": int(games["has_pbpstats"].sum()),
            "matchup_games": int(games["has_matchups"].sum()),
        },
        "scope_note": "Regular season and playoffs only; play-in games are not present in the source archive.",
        "issues": issues,
        "path": str(output.resolve()),
        "source_files": source_files,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
