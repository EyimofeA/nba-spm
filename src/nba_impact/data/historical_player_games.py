"""Build a strict, separate historical player-game table from local ESPN boxes.

This is deliberately not a replacement for ``player_games.parquet``. The
historical ESPN mirror starts in 2019 and has a known 2020 coverage gap. Each
emitted game is reconciled to an official game record and local V3 play-by-play
team identities and game length. Rejected and absent games stay in the quality
table.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic
from .official_boxscore import infer_official_starters, load_official_boxscore_rows


_OFFICIAL_COLUMNS = (
    "project_season", "season_type", "game_id", "game_date", "home_team_id", "away_team_id",
)
_ESPN_COLUMNS = ("season", "game_id", "player_id", "team", "home", "name", "starter", "minutes_played", "played")
_V3_COLUMNS = ("gameId", "period", "teamId", "teamTricode")
_TRICODE_ALIASES = {"BRK": "BKN", "NOR": "NOP", "PHO": "PHX", "SAN": "SAS", "CHA": "CHO"}
_CLOCK_MINUTES = re.compile(r"^(\d+):(\d{1,2})$")


def _canonical_game_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Game ID cannot be null.")
    return f"{int(str(value)):010d}"


def _season_label(season_end: int) -> str:
    return f"{season_end - 1}-{str(season_end)[-2:]}"


def _minutes_to_seconds(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return 0.0
    match = _CLOCK_MINUTES.fullmatch(text)
    if not match or int(match.group(2)) > 60:
        return float("nan")
    return float(int(match.group(1)) * 60 + int(match.group(2)))


def _load_v3_metadata(
    root: Path, official: pd.DataFrame, seasons: tuple[int, ...]
) -> tuple[dict[str, dict[str, object]], list[Path]]:
    """Return max period and unambiguous V3 team identities for each game."""
    metadata: dict[str, dict[str, object]] = {}
    paths: list[Path] = []
    requested = official.loc[official["project_season"].isin(seasons)]
    for (season, season_type), games in requested.groupby(["project_season", "season_type"], sort=True):
        path = root / f"project_season={int(season)}" / f"{season_type}.parquet"
        if not path.exists():
            continue
        paths.append(path)
        raw = pd.read_parquet(path, columns=list(_V3_COLUMNS))
        raw["game_id"] = raw["gameId"].map(_canonical_game_id)
        raw = raw.loc[raw["game_id"].isin(set(games["game_id"]))].copy()
        raw["teamId"] = pd.to_numeric(raw["teamId"], errors="coerce")
        raw["teamTricode"] = raw["teamTricode"].astype("string")
        for game_id, group in raw.groupby("game_id", sort=False):
            team_rows = group.loc[group["teamId"].gt(0) & group["teamTricode"].notna()]
            team_ids = team_rows.groupby("teamTricode")["teamId"].agg(
                lambda values: tuple(sorted({int(value) for value in values.dropna()}))
            )
            metadata[game_id] = {
                "max_period": int(pd.to_numeric(group["period"], errors="raise").max()),
                "team_ids": team_ids.to_dict(),
            }
    return metadata, paths


def build_historical_espn_player_games(
    espn_path: str | Path,
    official_scores_path: str | Path,
    v3_root: str | Path,
    destination: str | Path,
    quality_destination: str | Path,
    manifest_dir: str | Path,
    *,
    seasons: tuple[int, ...] = tuple(range(2017, 2024)),
    official_box_dir: str | Path | None = None,
) -> dict:
    """Emit only official/V3/reconciled player-games and a full QA ledger.

    An available official NBA boxscore always takes priority over ESPN. A failed
    official source rejects its game. It never falls back to ESPN silently.
    """
    if not seasons:
        raise ValueError("At least one project season is required.")
    espn_source = Path(espn_path)
    official_source = Path(official_scores_path)
    official = pd.read_parquet(official_source)
    missing = sorted(set(_OFFICIAL_COLUMNS) - set(official.columns))
    if missing:
        raise ValueError(f"Official-score input is missing columns: {missing}")
    official = official.loc[official["project_season"].isin(seasons), list(_OFFICIAL_COLUMNS)].copy()
    official["game_id"] = official["game_id"].map(_canonical_game_id)
    official["game_date"] = pd.to_datetime(official["game_date"], errors="raise")
    for column in ("project_season", "home_team_id", "away_team_id"):
        official[column] = pd.to_numeric(official[column], errors="raise").astype("int64")
    if official.duplicated("game_id", keep=False).any():
        raise ValueError("Official-score input has duplicate game IDs.")

    espn = pd.read_parquet(espn_source)
    missing = sorted(set(_ESPN_COLUMNS) - set(espn.columns))
    if missing:
        raise ValueError(f"ESPN player-box input is missing columns: {missing}")
    espn = espn.loc[espn["season"].isin(seasons), list(_ESPN_COLUMNS)].copy()
    espn["game_id"] = espn["game_id"].map(_canonical_game_id)
    espn["season"] = pd.to_numeric(espn["season"], errors="raise").astype("int64")
    espn["player_id"] = pd.to_numeric(espn["player_id"], errors="raise").astype("int64")
    espn["team"] = espn["team"].astype(str).str.strip().replace(_TRICODE_ALIASES)
    espn["home"] = pd.to_numeric(espn["home"], errors="coerce").astype("Int64")
    espn["starter"] = pd.to_numeric(espn["starter"], errors="coerce").fillna(0).astype(bool)
    espn["minutes_seconds"] = espn["minutes_played"].map(_minutes_to_seconds)
    metadata, v3_paths = _load_v3_metadata(Path(v3_root), official, seasons)
    espn_groups = {key: value.copy() for key, value in espn.groupby(["season", "game_id"], sort=False)}
    official_paths: list[Path] = []
    official_groups: dict[str, pd.DataFrame] = {}
    if official_box_dir is not None:
        official_boxes, official_paths = load_official_boxscore_rows(official_box_dir)
        if not official_boxes.empty:
            official_boxes = official_boxes.loc[official_boxes["game_id"].isin(set(official["game_id"]))].copy()
            official_boxes["team_id"] = pd.to_numeric(official_boxes["team_id"], errors="raise").astype("int64")
            official_boxes["player_id"] = pd.to_numeric(official_boxes["player_id"], errors="raise").astype("int64")
            official_boxes["starter_position"] = official_boxes["starter_position"].fillna("").astype(str).str.strip()
            official_boxes = infer_official_starters(
                official_boxes,
                first_five_fallback_game_ids=set(
                    official.loc[official["project_season"].eq(2017), "game_id"].astype(str)
                ),
            )
            official_boxes["minutes_seconds"] = official_boxes["minutes"].map(_minutes_to_seconds)
            official_groups = {
                game_id: group.copy() for game_id, group in official_boxes.groupby("game_id", sort=False)
            }

    rows: list[pd.DataFrame] = []
    quality: list[dict[str, object]] = []
    for record in official.itertuples(index=False):
        game_id = str(record.game_id)
        season = int(record.project_season)
        official_rows = official_groups.get(game_id)
        source_kind = "official_boxscore" if official_rows is not None else "espn_player_box"
        source = official_rows if official_rows is not None else espn_groups.get((season, game_id))
        issues: list[str] = []
        meta = metadata.get(game_id)
        if source is None:
            issues.append("missing_espn_player_box")
        if meta is None:
            issues.append("missing_v3_play_by_play")
        if source is not None and meta is not None:
            if source.duplicated(["player_id"], keep=False).any():
                issues.append("duplicate_game_player")
            expected_seconds = float((2880 + (max(4, int(meta["max_period"])) - 4) * 300) * 5)
            if source_kind == "official_boxscore":
                expected_ids = {int(record.home_team_id), int(record.away_team_id)}
                v3_ids = {team_id for ids in meta["team_ids"].values() for team_id in ids}
                if v3_ids != expected_ids:
                    issues.append("official_v3_team_identity_mismatch")
                for team_id, side in ((int(record.home_team_id), "home"), (int(record.away_team_id), "away")):
                    team_rows = source.loc[source["team_id"].eq(team_id)]
                    if team_rows.empty or source.loc[~source["team_id"].isin(expected_ids)].any(axis=None):
                        issues.append(f"{side}_team_id_mismatch")
                    if int(team_rows["starter"].sum()) != 5:
                        issues.append(f"{side}_starter_count")
                    if team_rows["minutes_seconds"].isna().any():
                        issues.append(f"{side}_minutes_parse")
                    elif abs(float(team_rows["minutes_seconds"].sum()) - expected_seconds) > 5.0:
                        issues.append(f"{side}_minutes_not_reconciled")
            else:
                if set(source["home"].dropna()) != {0, 1}:
                    issues.append("invalid_home_side")
                for home, side in ((1, "home"), (0, "away")):
                    team_rows = source.loc[source["home"].eq(home)]
                    tricodes = set(team_rows["team"])
                    if len(tricodes) != 1:
                        issues.append(f"{side}_nonunique_tricode")
                    else:
                        tricode = next(iter(tricodes))
                        actual_ids = meta["team_ids"].get(tricode, ())
                        expected_id = int(record.home_team_id if home else record.away_team_id)
                        if actual_ids != (expected_id,):
                            issues.append(f"{side}_team_id_mismatch")
                    if int(team_rows["starter"].sum()) != 5:
                        issues.append(f"{side}_starter_count")
                    if team_rows["minutes_seconds"].isna().any():
                        issues.append(f"{side}_minutes_parse")
                    elif abs(float(team_rows["minutes_seconds"].sum()) - expected_seconds) > 5.0:
                        issues.append(f"{side}_minutes_not_reconciled")
        quality.append(
            {
                "project_season": season,
                "season_type": record.season_type,
                "game_id": game_id,
                "game_date": record.game_date,
                "passed": not issues,
                "issues": ";".join(issues),
                "selected_source": source_kind,
                "official_box_available": official_rows is not None,
                "official_starter_inference_sources": (
                    "|".join(sorted(set(official_rows["starter_inference_source"].astype(str))))
                    if official_rows is not None else ""
                ),
                "source_rows": int(len(source)) if source is not None else 0,
                "v3_available": meta is not None,
                "max_period": int(meta["max_period"]) if meta is not None else pd.NA,
            }
        )
        if issues or source is None:
            continue
        frame = source.copy()
        frame["season_start"] = season - 1
        frame["season_end"] = season
        frame["season_label"] = _season_label(season)
        frame["season_type"] = record.season_type
        frame["game_date"] = record.game_date
        team_ids = meta["team_ids"]
        if source_kind == "official_boxscore":
            tricode_by_id = {ids[0]: tricode for tricode, ids in team_ids.items() if len(ids) == 1}
            frame["team_side"] = np.where(frame["team_id"].eq(record.home_team_id), "home", "away")
            frame["team_tricode"] = frame["team_id"].map(tricode_by_id)
            frame["player_name"] = (
                frame["first_name"].fillna("").str.strip() + " " + frame["family_name"].fillna("").str.strip()
            ).str.strip()
            frame["minutes"] = frame["minutes"]
            frame["espn_played"] = pd.NA
        else:
            frame["team_side"] = np.where(frame["home"].eq(1), "home", "away")
            frame["team_id"] = frame["team"].map(lambda tricode: int(team_ids[tricode][0]))
            frame["team_tricode"] = frame["team"]
            frame["player_name"] = frame["name"].fillna("").astype(str).str.strip()
            frame["starter_position"] = ""
            frame["starter_inference_source"] = "espn_native_starter"
            frame["minutes"] = frame["minutes_played"]
            frame["espn_played"] = pd.to_numeric(frame["played"], errors="raise").astype(bool)
            frame["player_game_source"] = "espn_player_box_historical"
        frame["played"] = frame["minutes_seconds"].gt(0)
        rows.append(frame)

    output = Path(destination)
    quality_output = Path(quality_destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality_output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "game_id", "season_start", "season_end", "season_label", "season_type", "game_date",
        "team_id", "team_tricode", "team_side", "player_game_source", "player_id", "player_name",
        "starter_position", "starter", "played", "minutes", "minutes_seconds", "espn_played",
        "starter_inference_source",
    ]
    output_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    output_rows = output_rows.loc[:, columns].sort_values(["game_id", "team_side", "starter", "player_id"])
    output_rows.to_parquet(output, index=False)
    quality_rows = pd.DataFrame(quality).sort_values(["project_season", "game_id"])
    quality_rows.to_parquet(quality_output, index=False)

    source_paths = [espn_source, official_source, *v3_paths, *official_paths]
    source_files = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ]
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_files]).encode("utf-8")
    ).hexdigest()[:16]
    rejected = quality_rows.loc[~quality_rows["passed"]]
    snapshot = {
        "snapshot_id": f"historical_espn_player_games_{identity}",
        "dataset": "historical_espn_player_games",
        "grain": "one player in one official NBA game",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": "separate historical ESPN-derived player-game table; not canonical player_games",
        "row_count": int(len(output_rows)),
        "game_count": int(output_rows["game_id"].nunique()),
        "requested_game_count": int(len(official)),
        "rejected_game_count": int(len(rejected)),
        "accepted_games_by_season": {
            str(season): int(group["game_id"].nunique())
            for season, group in output_rows.groupby("season_end")
        },
        "rejected_games_by_issue": rejected["issues"].str.split(";").explode().value_counts().to_dict(),
        "source_files": source_files,
        "path": str(output.resolve()),
        "quality_path": str(quality_output.resolve()),
        "license_note": (
            "The llimllib/nba_data repository declares no license. ESPN-derived rows are research-only "
            "and must not be redistributed until source rights are clarified."
        ),
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
