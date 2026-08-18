"""Complete a cut-off CDN playoff partition from the official NBA Live feed."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


LIVE_URL = (
    "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/"
    "NBA/liveData/playbyplay/playbyplay_{game_id}.json"
)
IDENTITY_COLUMNS = (
    "actionNumber", "period", "clock", "possession", "scoreHome", "scoreAway", "actionType"
)


def _read_live_game(path: Path, expected_game_id: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    game = payload.get("game", {})
    game_id = canonical_game_id(game.get("gameId"))
    if game_id != expected_game_id:
        raise ValueError(f"{path}: expected game {expected_game_id}, found {game_id}")
    actions = game.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{path}: missing nonempty game.actions")
    frame = pd.DataFrame(actions)
    required = {"actionNumber", "orderNumber", "period", "clock", "possession", "scoreHome", "scoreAway"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing action fields {missing}")
    frame["gameId"] = int(expected_game_id)
    frame["game_id"] = expected_game_id
    return frame


def _download_live_game(game_id: str, destination: Path, max_attempts: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    session = requests.Session()

    def fetch() -> None:
        with session.get(
            LIVE_URL.format(game_id=game_id),
            timeout=(30, 300),
            headers={"User-Agent": "nba-impact-lab/0.1 (research ingestion)"},
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        _read_live_game(partial, game_id)
        partial.replace(destination)

    for attempt in Retrying(
        retry=retry_if_exception_type((requests.RequestException, ValueError, json.JSONDecodeError)),
        wait=wait_exponential_jitter(initial=2, max=300),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    ):
        with attempt:
            fetch()


def _coerce_to_reference(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    output = frame.reindex(columns=schema.names).copy()
    for field in schema:
        name = field.name
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            output[name] = pd.to_numeric(output[name], errors="coerce")
        elif pa.types.is_boolean(field.type):
            output[name] = output[name].astype("boolean")
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            output[name] = output[name].map(
                lambda value: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
            )
    arrays = [pa.array(output[field.name], type=field.type, from_pandas=True) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def build_live_playoff_completion(
    raw_dir: str | Path,
    reference_cdn_path: str | Path,
    game_dim_path: str | Path,
    player_games_path: str | Path,
    destination: str | Path,
    scoped_game_dim_destination: str | Path,
    scoped_player_games_destination: str | Path,
    report_destination: str | Path,
    *,
    season_label: str = "2025-26",
    max_attempts: int = 20,
    download_missing: bool = True,
) -> dict:
    """Create one complete, schema-identical CDN partition and scoped dimensions."""
    raw_root = Path(raw_dir)
    reference_path = Path(reference_cdn_path)
    games = pd.read_parquet(game_dim_path)
    games["game_id"] = games["game_id"].map(canonical_game_id)
    scoped_games = games.loc[
        games["season_label"].eq(season_label) & games["season_type"].eq("playoffs")
    ].copy()
    expected_ids = set(scoped_games["game_id"])
    if not expected_ids:
        raise ValueError(f"No playoff games found for {season_label}")

    live_frames: list[pd.DataFrame] = []
    source_files: list[dict] = []
    for game_id in sorted(expected_ids):
        path = raw_root / f"live_{game_id}.json"
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(path)
            _download_live_game(game_id, path, max_attempts)
        frame = _read_live_game(path, game_id)
        live_frames.append(frame)
        source_files.append(
            {"game_id": game_id, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    live = pd.concat(live_frames, ignore_index=True)

    reference = pd.read_parquet(reference_path)
    reference["game_id"] = reference["gameId"].map(canonical_game_id)
    reference_ids = set(reference["game_id"])
    overlap_ids = expected_ids & reference_ids
    overlap_left = reference.loc[reference["game_id"].isin(overlap_ids), ["game_id", "orderNumber", *IDENTITY_COLUMNS]]
    overlap_right = live.loc[live["game_id"].isin(overlap_ids), ["game_id", "orderNumber", *IDENTITY_COLUMNS]]
    overlap = overlap_left.merge(
        overlap_right,
        on=["game_id", "orderNumber"],
        how="outer",
        suffixes=("_reference", "_live"),
        indicator=True,
        validate="one_to_one",
    )
    overlap_key_failures = int(overlap["_merge"].ne("both").sum())
    overlap_value_failures = sum(
        int(overlap[f"{column}_reference"].astype(str).ne(overlap[f"{column}_live"].astype(str)).sum())
        for column in IDENTITY_COLUMNS
    )
    if overlap_key_failures or overlap_value_failures:
        raise ValueError(
            f"Official Live overlap differs from pinned CDN: keys={overlap_key_failures}, values={overlap_value_failures}"
        )

    tail_ids = expected_ids - reference_ids
    tail = live.loc[live["game_id"].isin(tail_ids)].copy()
    tail["_season"] = int(scoped_games["season_start"].iloc[0])
    tail["_season_type"] = "playoffs"
    completed = pd.concat([reference.drop(columns="game_id"), tail.drop(columns="game_id")], ignore_index=True)
    completed["game_id"] = completed["gameId"].map(canonical_game_id)
    duplicate_order = int(completed.duplicated(["game_id", "orderNumber"], keep=False).sum())
    duplicate_action = int(completed.duplicated(["game_id", "actionNumber"], keep=False).sum())
    completed_ids = set(completed["game_id"])

    score = completed.groupby("game_id", as_index=False).agg(
        observed_home_score=("scoreHome", lambda values: pd.to_numeric(values, errors="raise").max()),
        observed_away_score=("scoreAway", lambda values: pd.to_numeric(values, errors="raise").max()),
    ).merge(
        scoped_games[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]],
        on="game_id",
        how="outer",
        validate="one_to_one",
    )
    score_failures = int(
        score["observed_home_score"].ne(score["home_score"]).sum()
        + score["observed_away_score"].ne(score["away_score"]).sum()
    )
    teams = scoped_games.set_index("game_id")[["home_team_id", "away_team_id"]]
    with_teams = completed.join(teams, on="game_id")
    owner = pd.to_numeric(with_teams["possession"], errors="raise")
    invalid_owner = int(
        (~owner.eq(0) & ~owner.eq(with_teams["home_team_id"]) & ~owner.eq(with_teams["away_team_id"])).sum()
    )
    issues = {
        "missing_games": len(expected_ids - completed_ids),
        "extra_games": len(completed_ids - expected_ids),
        "duplicate_order_keys": duplicate_order,
        "duplicate_action_keys": duplicate_action,
        "score_failures": score_failures,
        "invalid_possession_owner_rows": invalid_owner,
        "overlap_key_failures": overlap_key_failures,
        "overlap_value_failures": overlap_value_failures,
    }
    if any(issues.values()):
        raise ValueError(f"Completed playoff partition failed QA: {issues}")

    schema = pq.read_schema(reference_path)
    table = _coerce_to_reference(completed.drop(columns="game_id"), schema)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    pq.write_table(table, temporary)
    temporary.replace(output)

    players = pd.read_parquet(player_games_path)
    scoped_players = players.loc[players["game_id"].map(canonical_game_id).isin(expected_ids)].copy()
    for frame, path in (
        (scoped_games, Path(scoped_game_dim_destination)),
        (scoped_players, Path(scoped_player_games_destination)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        frame.to_parquet(partial, index=False)
        partial.replace(path)

    report = {
        "dataset": "nba_live_playoff_completion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season_label": season_label,
        "passed": True,
        "game_count": len(completed_ids),
        "reference_game_count": len(reference_ids),
        "tail_game_count": len(tail_ids),
        "action_row_count": len(completed),
        "issues": issues,
        "source_rights": "Official NBA Live endpoint; internal research use only. Do not include raw rows in a public release bundle.",
        "ordering_policy": "orderNumber only",
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "source_files": source_files,
    }
    write_json_atomic(report, report_destination)
    return report
