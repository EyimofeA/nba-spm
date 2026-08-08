"""Canonical post-action score states from NBA Stats V3 play-by-play."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


def parse_clock_seconds(clock: pd.Series) -> pd.Series:
    parts = clock.astype("string").str.extract(r"^PT(?:(\d+)M)?([0-9]+(?:\.[0-9]+)?)S$")
    minutes = pd.to_numeric(parts[0], errors="coerce").fillna(0.0)
    seconds = pd.to_numeric(parts[1], errors="coerce")
    return minutes * 60.0 + seconds


def _partition(path: Path) -> tuple[int, str]:
    return int(path.parent.name.split("=", maxsplit=1)[1]), path.stem


def _normalize_partition(path: Path, game_dim: pd.DataFrame) -> pd.DataFrame:
    season_start, season_type = _partition(path)
    frame = pd.read_parquet(
        path,
        columns=[
            "gameId",
            "actionId",
            "actionNumber",
            "clock",
            "period",
            "teamId",
            "teamTricode",
            "personId",
            "playerName",
            "location",
            "description",
            "actionType",
            "subType",
            "scoreHome",
            "scoreAway",
            "shotValue",
            "isFieldGoal",
        ],
    )
    frame["game_id"] = frame["gameId"].map(canonical_game_id)
    frame = frame.sort_values(["game_id", "actionId"], kind="stable").reset_index(drop=True)
    frame["seconds_remaining_period"] = parse_clock_seconds(frame["clock"])
    period = pd.to_numeric(frame["period"], errors="coerce")
    regulation = period <= 4
    frame["regulation_seconds_remaining"] = np.where(
        regulation,
        (4 - period) * 720.0 + frame["seconds_remaining_period"],
        0.0,
    )
    frame["seconds_elapsed_game"] = np.where(
        regulation,
        (period - 1) * 720.0 + (720.0 - frame["seconds_remaining_period"]),
        4 * 720.0 + (period - 5) * 300.0 + (300.0 - frame["seconds_remaining_period"]),
    )

    frame["source_score_home"] = pd.to_numeric(frame["scoreHome"], errors="coerce")
    frame["source_score_away"] = pd.to_numeric(frame["scoreAway"], errors="coerce")
    source_score_present = frame["source_score_home"].notna() & frame["source_score_away"].notna()
    made_field_goal = frame["actionType"].eq("Made Shot")
    made_free_throw = frame["actionType"].eq("Free Throw") & source_score_present
    event_points = np.where(
        made_field_goal,
        pd.to_numeric(frame["shotValue"], errors="coerce").fillna(0.0),
        np.where(made_free_throw, 1.0, 0.0),
    )
    frame["home_points_added"] = np.where(frame["location"].eq("h"), event_points, 0.0)
    frame["away_points_added"] = np.where(frame["location"].eq("v"), event_points, 0.0)
    frame["home_score_after"] = frame.groupby("game_id", sort=False)["home_points_added"].cumsum()
    frame["away_score_after"] = frame.groupby("game_id", sort=False)["away_points_added"].cumsum()
    frame["home_score_before"] = frame["home_score_after"] - frame["home_points_added"]
    frame["away_score_before"] = frame["away_score_after"] - frame["away_points_added"]
    frame["points_added"] = frame["home_points_added"] + frame["away_points_added"]
    frame["home_score_diff_after"] = frame["home_score_after"] - frame["away_score_after"]
    frame["home_score_diff_before"] = frame["home_score_before"] - frame["away_score_before"]
    frame["is_overtime"] = period > 4
    frame["event_team_side"] = frame["location"].where(frame["location"].isin(["h", "v"]))
    frame["event_id"] = frame["game_id"] + ":" + frame["actionId"].astype(str)
    frame["source_season"] = season_start
    frame["season_type"] = season_type
    frame["is_terminal_event"] = frame.groupby("game_id", sort=False).cumcount(ascending=False).eq(0)

    dimension = game_dim.loc[
        :,
        [
            "game_id",
            "season_start",
            "season_end",
            "season_label",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_win",
        ],
    ]
    frame = frame.merge(dimension, on="game_id", how="left", validate="many_to_one")
    return frame


def build_event_states(
    root: str | Path,
    game_dim_path: str | Path,
    destination: str | Path,
    manifest_dir: str | Path,
) -> dict:
    source_root = Path(root)
    game_dim = pd.read_parquet(game_dim_path)
    frames = [
        _normalize_partition(path, game_dim)
        for path in sorted((source_root / "nbastatsv3").rglob("*.parquet"))
    ]
    if not frames:
        raise ValueError(f"No NBA Stats V3 partitions found under {source_root}")
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["game_date", "game_id", "actionId"], kind="stable").reset_index(drop=True)

    previous_elapsed = events.groupby("game_id", sort=False)["seconds_elapsed_game"].shift()
    previous_action_number = events.groupby("game_id", sort=False)["actionNumber"].shift()
    terminal = events.loc[events["is_terminal_event"]].merge(
        game_dim[["game_id", "home_score", "away_score"]],
        on="game_id",
        suffixes=("_event", "_game"),
        validate="one_to_one",
    )
    critical_issues = {
        "duplicate_event_ids": int(events.duplicated("event_id", keep=False).sum()),
        "missing_game_dimension_rows": int(events["season_label"].isna().sum()),
        "invalid_clocks": int(events["seconds_remaining_period"].isna().sum()),
        "clock_order_violations": int(
            ((events["seconds_elapsed_game"] + 1e-9) < previous_elapsed).fillna(False).sum()
        ),
        "terminal_score_mismatch_games": int(
            (
                terminal["home_score_after"].ne(terminal["home_score"])
                | terminal["away_score_after"].ne(terminal["away_score"])
            ).sum()
        ),
        "terminal_nonzero_clock_games": int((terminal["seconds_remaining_period"].abs() > 1e-9).sum()),
    }
    warnings = {
        "source_score_snapshot_disagreement_rows": int(
            (
                events["source_score_home"].notna()
                & events["source_score_away"].notna()
                & (
                    events["source_score_home"].ne(events["home_score_after"])
                    | events["source_score_away"].ne(events["away_score_after"])
                )
            ).sum()
        ),
        "action_number_backtrack_rows": int(
            (events["actionNumber"] < previous_action_number).fillna(False).sum()
        ),
    }
    passed = not any(critical_issues.values())

    output_columns = [
        "event_id",
        "game_id",
        "source_season",
        "season_start",
        "season_end",
        "season_label",
        "season_type",
        "game_date",
        "actionId",
        "actionNumber",
        "period",
        "clock",
        "seconds_remaining_period",
        "regulation_seconds_remaining",
        "seconds_elapsed_game",
        "is_overtime",
        "teamId",
        "teamTricode",
        "personId",
        "playerName",
        "event_team_side",
        "description",
        "actionType",
        "subType",
        "shotValue",
        "isFieldGoal",
        "source_score_home",
        "source_score_away",
        "home_score_before",
        "away_score_before",
        "home_score_after",
        "away_score_after",
        "home_points_added",
        "away_points_added",
        "points_added",
        "home_score_diff_before",
        "home_score_diff_after",
        "home_team_id",
        "away_team_id",
        "home_win",
        "is_terminal_event",
    ]
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    events.loc[:, output_columns].to_parquet(temporary, index=False)
    temporary.replace(output)

    source_files = sorted((source_root / "nbastatsv3").rglob("*.parquet"))
    lineage = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    ]
    game_dim_hash = sha256_file(game_dim_path)
    identity = hashlib.sha256(
        json.dumps(
            {
                "events": [(item["path"], item["sha256"]) for item in lineage],
                "game_dim": game_dim_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"event_states_{identity}",
        "dataset": "event_states",
        "grain": "one NBA Stats V3 action with post-action score state",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "row_count": int(len(events)),
        "game_count": int(events["game_id"].nunique()),
        "season_labels": sorted(events["season_label"].dropna().unique().tolist()),
        "issues": critical_issues,
        "warnings": warnings,
        "path": str(output.resolve()),
        "game_dim": {"path": str(Path(game_dim_path).resolve()), "sha256": game_dim_hash},
        "source_files": lineage,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
