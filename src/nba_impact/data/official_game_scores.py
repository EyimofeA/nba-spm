"""Download minimal official NBA game logs and build verified final scores."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .manifest import sha256_file, write_json_atomic


Fetcher = Callable[[str, str, int], pd.DataFrame]


def season_label(project_season: int) -> str:
    return f"{project_season - 1}-{str(project_season)[-2:]}"


def fetch_team_game_log(season: str, season_type: str, timeout: int) -> pd.DataFrame:
    endpoint = leaguegamelog.LeagueGameLog(
        league_id="00",
        player_or_team_abbreviation="T",
        season=season,
        season_type_all_star=season_type,
        timeout=timeout,
    )
    return endpoint.get_data_frames()[0]


def normalize_game_scores(
    frame: pd.DataFrame,
    *,
    project_season: int,
    season_type: str,
) -> tuple[pd.DataFrame, dict]:
    required = {"GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "PTS"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Official game log is missing columns: {missing}")
    work = frame.loc[:, sorted(required)].copy()
    work["game_id"] = (
        pd.to_numeric(work["GAME_ID"], errors="raise")
        .astype("int64")
        .map(lambda value: f"{value:010d}")
    )
    work["team_id"] = pd.to_numeric(work["TEAM_ID"], errors="raise").astype("int64")
    work["pts"] = pd.to_numeric(work["PTS"], errors="raise")
    work["is_home"] = work["MATCHUP"].astype(str).str.contains(" vs. ", regex=False)
    work["is_away"] = work["MATCHUP"].astype(str).str.contains(" @ ", regex=False)

    duplicate_team_games = int(work.duplicated(["game_id", "team_id"]).sum())
    invalid_points = int((work["pts"].isna() | work["pts"].lt(0) | work["pts"].mod(1).ne(0)).sum())
    group_sizes = work.groupby("game_id").size()
    home_counts = work.groupby("game_id")["is_home"].sum()
    away_counts = work.groupby("game_id")["is_away"].sum()
    invalid_game_rows = int(
        (~group_sizes.eq(2)).sum()
        + (~home_counts.eq(1)).sum()
        + (~away_counts.eq(1)).sum()
    )

    home = work.loc[work["is_home"]].set_index("game_id")
    away = work.loc[work["is_away"]].set_index("game_id")
    games = home.join(away, how="outer", lsuffix="_home", rsuffix="_away")
    output = pd.DataFrame(
        {
            "project_season": project_season,
            "season_type": season_type,
            "game_id": games.index.to_numpy(),
            "game_date": pd.to_datetime(games["GAME_DATE_home"], errors="raise").dt.date.astype(str),
            "home_team_id": games["team_id_home"].astype("Int64"),
            "away_team_id": games["team_id_away"].astype("Int64"),
            "home_score": games["pts_home"].astype("Int64"),
            "away_score": games["pts_away"].astype("Int64"),
        }
    ).reset_index(drop=True).sort_values("game_id", kind="stable").reset_index(drop=True)
    metrics = {
        "project_season": project_season,
        "season_type": season_type,
        "team_rows": int(len(work)),
        "games": int(len(output)),
        "duplicate_team_games": duplicate_team_games,
        "invalid_points": invalid_points,
        "invalid_game_rows": invalid_game_rows,
    }
    metrics["passed"] = not any(
        metrics[key] for key in ("duplicate_team_games", "invalid_points", "invalid_game_rows")
    )
    return output, metrics


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(destination)


def scores_from_game_dimension(
    frame: pd.DataFrame,
    *,
    project_season: int,
    season_type: str,
) -> pd.DataFrame:
    """Select a minimal final-score reference from a verified game dimension."""
    required = {
        "game_id", "game_date", "season_end", "season_type",
        "home_team_id", "away_team_id", "home_score", "away_score",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Game dimension is missing columns: {missing}")
    selected = frame.loc[
        frame["season_end"].eq(project_season)
        & frame["season_type"].eq(season_type)
    ].copy()
    if selected.empty:
        raise ValueError(f"Game dimension has no {project_season} {season_type} rows")
    output = selected[
        ["game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].copy()
    output.insert(0, "season_type", season_type)
    output.insert(0, "project_season", project_season)
    output["game_id"] = output["game_id"].astype(str).str.zfill(10)
    output["game_date"] = pd.to_datetime(output["game_date"], errors="raise").dt.date.astype(str)
    numeric = ["home_team_id", "away_team_id", "home_score", "away_score"]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="raise").astype("Int64")
    if output["game_id"].duplicated().any() or output[numeric].isna().any().any():
        raise ValueError("Game-dimension score reference has duplicate games or null values")
    return output.sort_values("game_id", kind="stable").reset_index(drop=True)


def build_official_game_scores(
    output_root: str | Path,
    *,
    project_seasons: tuple[int, ...] = tuple(range(2017, 2027)),
    fetcher: Fetcher = fetch_team_game_log,
    max_attempts: int = 20,
    timeout_seconds: int = 30,
    request_delay_seconds: float = 0.6,
    fallback_game_dim_path: str | Path | None = None,
    existing_only: bool = False,
) -> dict:
    """Fetch, checkpoint, validate, and combine minimal official final scores."""
    root = Path(output_root)
    fallback_path = Path(fallback_game_dim_path) if fallback_game_dim_path else None
    fallback_frame = pd.read_parquet(fallback_path) if fallback_path and fallback_path.exists() else None
    partition_rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    for project_season in project_seasons:
        for kind, endpoint_kind in (("regular", "Regular Season"), ("playoffs", "Playoffs")):
            destination = root / f"project_season={project_season}" / f"{kind}.parquet"
            manifest_path = destination.with_suffix(".parquet.manifest.json")
            cached = None
            if destination.exists() and manifest_path.exists():
                candidate = json.loads(manifest_path.read_text())
                if candidate.get("output_sha256") == sha256_file(destination):
                    cached = candidate
            if cached is None:
                if existing_only:
                    if fallback_frame is None:
                        raise FileNotFoundError(
                            f"Missing cached {project_season} {kind} scores and no fallback game dimension"
                        )
                    scores = scores_from_game_dimension(
                        fallback_frame,
                        project_season=project_season,
                        season_type=kind,
                    )
                    metrics = {
                        "project_season": project_season,
                        "season_type": kind,
                        "team_rows": int(len(scores) * 2),
                        "games": int(len(scores)),
                        "duplicate_team_games": 0,
                        "invalid_points": 0,
                        "invalid_game_rows": 0,
                        "passed": True,
                    }
                    _write_parquet_atomic(scores, destination)
                    cached = {
                        **metrics,
                        "provider": "verified game_dim fallback",
                        "source_path": str(fallback_path),
                        "source_sha256": sha256_file(fallback_path),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "output_path": str(destination),
                        "output_sha256": sha256_file(destination),
                    }
                    write_json_atomic(cached, manifest_path)
                else:
                    retrying = Retrying(
                        retry=retry_if_exception_type(Exception),
                        wait=wait_exponential_jitter(initial=2, max=120),
                        stop=stop_after_attempt(max_attempts),
                        reraise=True,
                    )
                    for attempt in retrying:
                        with attempt:
                            raw = fetcher(season_label(project_season), endpoint_kind, timeout_seconds)
                            scores, metrics = normalize_game_scores(
                                raw,
                                project_season=project_season,
                                season_type=kind,
                            )
                            if not metrics["passed"]:
                                raise ValueError(f"Official game-log validation failed: {metrics}")
                    _write_parquet_atomic(scores, destination)
                    cached = {
                        **metrics,
                        "provider": "stats.nba.com LeagueGameLog",
                        "endpoint_season": season_label(project_season),
                        "endpoint_season_type": endpoint_kind,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "output_path": str(destination),
                        "output_sha256": sha256_file(destination),
                    }
                    write_json_atomic(cached, manifest_path)
                    if request_delay_seconds:
                        time.sleep(request_delay_seconds)
            frame = pd.read_parquet(destination)
            frames.append(frame)
            partition_rows.append(cached)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["project_season", "season_type", "game_id"], kind="stable"
    )
    combined_path = root / "official_game_scores.parquet"
    _write_parquet_atomic(combined, combined_path)
    coverage = pd.DataFrame(partition_rows)
    coverage_path = root / "coverage.parquet"
    _write_parquet_atomic(coverage, coverage_path)
    run = {
        "schema_version": "official_nba_game_scores_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "stats.nba.com LeagueGameLog with verified game_dim fallback",
        "project_seasons": list(project_seasons),
        "partitions": int(len(coverage)),
        "games": int(len(combined)),
        "passed": bool(coverage["passed"].all()),
        "output_path": str(combined_path),
        "output_sha256": sha256_file(combined_path),
        "coverage_path": str(coverage_path),
    }
    write_json_atomic(run, root / "run.json")
    return run
