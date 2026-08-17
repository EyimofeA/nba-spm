"""Build and verify compact score-change events from pinned NBA Stats V3 files."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .manifest import sha256_file, write_json_atomic


REQUIRED_COLUMNS = (
    "gameId",
    "actionId",
    "actionNumber",
    "period",
    "clock",
    "teamId",
    "personId",
    "actionType",
    "scoreHome",
    "scoreAway",
    "pointsTotal",
    "description",
    "_season",
    "_season_type",
)
OPTIONAL_COLUMNS = ("subType", "shotResult", "isFieldGoal")
PARTITION_PATTERN = re.compile(r"project_season=(\d{4})")
EXTRACTION_VERSION = "score_state_verified_final_v3"


def _game_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype("int64")
    return numeric.map(lambda value: f"{value:010d}")


def extract_scoring_events(
    frame: pd.DataFrame,
    *,
    project_season: int,
    season_type: str,
    expected_final_scores: dict[str, tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one row per official score change plus partition QA metrics."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Scoring source is missing columns: {missing}")
    expected_source_season = project_season - 1
    source_seasons = set(pd.to_numeric(frame["_season"], errors="raise").astype(int))
    expected_source_type = "rg" if season_type == "regular" else "po"
    source_types = set(frame["_season_type"].astype(str))
    if source_seasons != {expected_source_season}:
        raise ValueError(
            f"Expected source season {expected_source_season}, found {sorted(source_seasons)}"
        )
    if source_types != {expected_source_type}:
        raise ValueError(
            f"Expected source type {expected_source_type!r}, found {sorted(source_types)}"
        )

    work = frame.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in work:
            work[column] = None
    work["game_id"] = _game_id(work["gameId"])
    work["event_order"] = pd.to_numeric(work["actionId"], errors="coerce")
    work["action_number"] = pd.to_numeric(work["actionNumber"], errors="coerce")
    work["score_home"] = pd.to_numeric(work["scoreHome"], errors="coerce")
    work["score_away"] = pd.to_numeric(work["scoreAway"], errors="coerce")
    work["cumulative_points"] = pd.to_numeric(work["pointsTotal"], errors="coerce")
    duplicate_event_keys = int(work.duplicated(["game_id", "event_order"]).sum())
    null_event_keys = int(work[["game_id", "event_order", "period", "clock"]].isna().any(axis=1).sum())
    action_order_backtracks = int(
        work.groupby("game_id", sort=False)["event_order"].diff().lt(0).sum()
    )

    partial_score_rows = int(work["score_home"].isna().ne(work["score_away"].isna()).sum())
    score_field_rows = work["score_home"].notna() & work["score_away"].notna()
    zero_score_sentinel_rows = int(
        (score_field_rows & work["score_home"].eq(0) & work["score_away"].eq(0)).sum()
    )
    # Older NBA Stats exports use 0/0 on every non-scoring action instead of
    # null score fields.  Zero is therefore a sentinel, not a score-state reset.
    score_states = work.loc[
        score_field_rows & (work["score_home"] + work["score_away"]).gt(0)
    ].copy()
    score_states = score_states.sort_values(["game_id", "event_order"], kind="stable")
    source_tail_rows_removed = 0
    repaired_final_score_games = 0
    if expected_final_scores:
        retained: list[pd.DataFrame] = []
        for game_id, game_rows in score_states.groupby("game_id", sort=False):
            expected = expected_final_scores.get(game_id)
            if expected is None:
                retained.append(game_rows)
                continue
            last = game_rows.iloc[-1]
            observed = (int(last["score_home"]), int(last["score_away"]))
            if observed == expected:
                retained.append(game_rows)
                continue
            exact = game_rows.loc[
                game_rows["score_home"].eq(expected[0])
                & game_rows["score_away"].eq(expected[1])
            ]
            if exact.empty:
                retained.append(game_rows)
                continue
            last_valid_order = exact.iloc[-1]["event_order"]
            valid_rows = game_rows.loc[game_rows["event_order"].le(last_valid_order)]
            source_tail_rows_removed += int(len(game_rows) - len(valid_rows))
            repaired_final_score_games += 1
            retained.append(valid_rows)
        score_states = pd.concat(retained, ignore_index=True)
    noninteger_score_rows = int(
        (
            score_states[["score_home", "score_away", "cumulative_points"]]
            .mod(1)
            .abs()
            .gt(1e-9)
            .any(axis=1)
        ).sum()
    )
    points_total_mismatches = int(
        (
            score_states["cumulative_points"]
            .ne(score_states["score_home"] + score_states["score_away"])
        ).sum()
    )
    grouped = score_states.groupby("game_id", sort=False)
    score_states["home_points_delta"] = grouped["score_home"].diff().fillna(
        score_states["score_home"]
    )
    score_states["away_points_delta"] = grouped["score_away"].diff().fillna(
        score_states["score_away"]
    )
    score_states["points_delta"] = (
        score_states["home_points_delta"] + score_states["away_points_delta"]
    )
    scoring = score_states.loc[
        score_states["home_points_delta"].ne(0)
        | score_states["away_points_delta"].ne(0)
    ].copy()
    scoring["is_score_correction"] = (
        scoring["home_points_delta"].lt(0)
        | scoring["away_points_delta"].lt(0)
        | (
            scoring["home_points_delta"].ne(0)
            & scoring["away_points_delta"].ne(0)
        )
        | ~scoring["points_delta"].between(1, 3)
        | scoring["actionType"].fillna("").astype(str).str.contains("Replay", case=False)
    )

    source_games = set(work["game_id"])
    score_games = set(score_states["game_id"])
    final_scores = (
        score_states.sort_values(["game_id", "event_order"], kind="stable")
        .groupby("game_id", as_index=False, sort=False)
        .tail(1)
    )
    conservation = scoring.groupby("game_id", as_index=False)[
        ["home_points_delta", "away_points_delta"]
    ].sum()
    conservation = conservation.merge(
        final_scores[["game_id", "score_home", "score_away"]],
        on="game_id",
        how="outer",
        validate="one_to_one",
    )
    conservation_failures = int(
        (
            conservation["home_points_delta"].ne(conservation["score_home"])
            | conservation["away_points_delta"].ne(conservation["score_away"])
        ).sum()
    )

    output = pd.DataFrame(
        {
            "project_season": project_season,
            "season_type": season_type,
            "game_id": scoring["game_id"],
            "event_order": scoring["event_order"].astype("int64"),
            "action_number": scoring["action_number"].astype("Int64"),
            "period": pd.to_numeric(scoring["period"], errors="coerce").astype("Int64"),
            "clock": scoring["clock"].astype(str),
            "team_id": pd.to_numeric(scoring["teamId"], errors="coerce").astype("Int64"),
            "player_id": pd.to_numeric(scoring["personId"], errors="coerce").astype("Int64"),
            "action_type": scoring["actionType"].astype("string"),
            "sub_type": scoring["subType"].astype("string"),
            "description": scoring["description"].astype("string"),
            "shot_result": scoring["shotResult"].astype("string"),
            "is_field_goal": pd.to_numeric(scoring["isFieldGoal"], errors="coerce").astype("Int64"),
            "score_home": scoring["score_home"].astype("int64"),
            "score_away": scoring["score_away"].astype("int64"),
            "cumulative_points": scoring["cumulative_points"].astype("int64"),
            "home_points_delta": scoring["home_points_delta"].astype("int64"),
            "away_points_delta": scoring["away_points_delta"].astype("int64"),
            "points_delta": scoring["points_delta"].astype("int64"),
            "is_score_correction": scoring["is_score_correction"].astype(bool),
        }
    ).sort_values(["game_id", "event_order"], kind="stable").reset_index(drop=True)

    metrics = {
        "project_season": project_season,
        "season_type": season_type,
        "raw_rows": int(len(work)),
        "games": int(len(source_games)),
        "score_state_rows": int(len(score_states)),
        "score_change_rows": int(len(output)),
        "score_correction_rows": int(output["is_score_correction"].sum()),
        "duplicate_event_keys": duplicate_event_keys,
        "null_event_keys": null_event_keys,
        "action_order_backtracks": action_order_backtracks,
        "partial_score_rows": partial_score_rows,
        "zero_score_sentinel_rows": zero_score_sentinel_rows,
        "noninteger_score_rows": noninteger_score_rows,
        "points_total_mismatches": points_total_mismatches,
        "games_without_score_state": int(len(source_games - score_games)),
        "score_conservation_failures": conservation_failures,
        "source_tail_rows_removed": source_tail_rows_removed,
        "repaired_final_score_games": repaired_final_score_games,
    }
    metrics["structural_passed"] = not any(
        metrics[key]
        for key in (
            "duplicate_event_keys",
            "null_event_keys",
            "action_order_backtracks",
            "partial_score_rows",
            "noninteger_score_rows",
            "points_total_mismatches",
            "games_without_score_state",
            "score_conservation_failures",
        )
    )
    return output, metrics


def _reference_games(
    *,
    project_season: int,
    season_type: str,
    game_dim_path: Path | None,
    legacy_cache_dir: Path | None,
    official_game_scores_path: Path | None,
) -> tuple[set[str], dict[str, tuple[int, int]], str]:
    if official_game_scores_path and official_game_scores_path.exists():
        frame = pd.read_parquet(official_game_scores_path)
        selected = frame.loc[
            frame["project_season"].eq(project_season)
            & frame["season_type"].eq(season_type)
        ]
        ids = set(selected["game_id"].astype(str).str.zfill(10))
        scores = {
            str(row.game_id).zfill(10): (int(row.home_score), int(row.away_score))
            for row in selected.itertuples(index=False)
        }
        return ids, scores, "official_nba_game_scores"
    if project_season >= 2024 and game_dim_path and game_dim_path.exists():
        frame = pd.read_parquet(
            game_dim_path,
            columns=["game_id", "season_end", "season_type", "home_score", "away_score"],
        )
        selected = frame.loc[
            frame["season_end"].eq(project_season)
            & frame["season_type"].eq(season_type)
        ].copy()
        ids = set(selected["game_id"].astype(str).str.zfill(10))
        scores = {
            str(row.game_id).zfill(10): (int(row.home_score), int(row.away_score))
            for row in selected.itertuples(index=False)
        }
        return ids, scores, "game_dim"
    if legacy_cache_dir:
        path = legacy_cache_dir / f"matchups_{project_season}.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=["gameid"])
            ids = set(_game_id(frame["gameid"]))
            prefix = "002" if season_type == "regular" else "004"
            return {game for game in ids if game.startswith(prefix)}, {}, "legacy_rapm_cache"
    return set(), {}, "none"


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def build_scoring_event_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    project_seasons: tuple[int, ...] = tuple(range(2017, 2027)),
    game_dim_path: str | Path | None = None,
    legacy_cache_dir: str | Path | None = None,
    official_game_scores_path: str | Path | None = None,
    require_reference_coverage: bool = True,
) -> dict[str, Any]:
    """Build resumable score-change partitions and a coverage report."""
    source = Path(source_root)
    output = Path(output_root)
    game_dim = Path(game_dim_path) if game_dim_path else None
    legacy = Path(legacy_cache_dir) if legacy_cache_dir else None
    official_scores = Path(official_game_scores_path) if official_game_scores_path else None
    expected_pairs = {(season, kind) for season in project_seasons for kind in ("regular", "playoffs")}
    discovered: dict[tuple[int, str], Path] = {}
    for path in sorted(source.glob("project_season=*/*.parquet")):
        match = PARTITION_PATTERN.search(str(path.parent))
        if not match or int(match.group(1)) not in project_seasons:
            continue
        kind = path.stem
        if kind in {"regular", "playoffs"}:
            discovered[(int(match.group(1)), kind)] = path
    missing_pairs = sorted(expected_pairs - set(discovered))
    if missing_pairs:
        raise FileNotFoundError(f"Missing scoring source partitions: {missing_pairs}")

    coverage_rows: list[dict[str, Any]] = []
    for project_season, season_type in sorted(expected_pairs):
        path = discovered[(project_season, season_type)]
        source_sha256 = sha256_file(path)
        reference_games, reference_scores, reference_source = _reference_games(
            project_season=project_season,
            season_type=season_type,
            game_dim_path=game_dim,
            legacy_cache_dir=legacy,
            official_game_scores_path=official_scores,
        )
        reference_score_hash = None
        if reference_scores:
            reference_path = official_scores if reference_source == "official_nba_game_scores" else game_dim
            if reference_path and reference_path.exists():
                reference_score_hash = sha256_file(reference_path)
        destination = output / f"project_season={project_season}" / f"{season_type}.parquet"
        partition_manifest_path = destination.with_suffix(".parquet.manifest.json")
        cached = None
        if destination.exists() and partition_manifest_path.exists():
            candidate = json.loads(partition_manifest_path.read_text())
            if (
                candidate.get("extraction_version") == EXTRACTION_VERSION
                and candidate.get("source_sha256") == source_sha256
                and candidate.get("reference_score_hash") == reference_score_hash
                and candidate.get("output_sha256") == sha256_file(destination)
            ):
                cached = candidate
        if cached is None:
            available = set(pq.ParquetFile(path).schema_arrow.names)
            frame = pd.read_parquet(
                path,
                columns=[
                    column
                    for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
                    if column in available
                ],
            )
            scoring, metrics = extract_scoring_events(
                frame,
                project_season=project_season,
                season_type=season_type,
                expected_final_scores=reference_scores,
            )
            _write_parquet_atomic(scoring, destination)
            cached = {
                **metrics,
                "extraction_version": EXTRACTION_VERSION,
                "source_path": str(path),
                "source_sha256": source_sha256,
                "reference_score_hash": reference_score_hash,
                "output_path": str(destination),
                "output_sha256": sha256_file(destination),
            }

        scoring = pd.read_parquet(
            destination,
            columns=["game_id", "event_order", "score_home", "score_away"],
        )
        source_games = set(scoring["game_id"].astype(str))
        missing_reference_games = len(reference_games - source_games)
        unexpected_games = len(source_games - reference_games) if reference_games else 0
        final_score_mismatches = 0
        if reference_scores:
            final = (
                scoring.sort_values(["game_id", "event_order"], kind="stable")
                .groupby("game_id", sort=False)
                .tail(1)
                .set_index("game_id")
            )
            final_score_mismatches = sum(
                int(
                    game not in final.index
                    or (int(final.at[game, "score_home"]), int(final.at[game, "score_away"]))
                    != expected
                )
                for game, expected in reference_scores.items()
            )
        reference_available = bool(reference_games)
        source_only_allowed = reference_source == "legacy_rapm_cache"
        coverage = {
            **cached,
            "reference_games": len(reference_games),
            "reference_available": reference_available,
            "reference_source": reference_source,
            "source_only_allowed": source_only_allowed,
            "missing_reference_games": missing_reference_games,
            "unexpected_games": unexpected_games,
            "final_score_mismatches": final_score_mismatches,
        }
        coverage["passed"] = bool(
            coverage["structural_passed"]
            and (reference_available or not require_reference_coverage)
            and missing_reference_games == 0
            and (unexpected_games == 0 or source_only_allowed)
            and final_score_mismatches == 0
        )
        write_json_atomic(coverage, partition_manifest_path)
        coverage_rows.append(coverage)

    coverage_frame = pd.DataFrame(coverage_rows)
    _write_parquet_atomic(coverage_frame, output / "coverage.parquet")
    run = {
        "schema_version": "nba_scoring_events_v2",
        "extraction_version": EXTRACTION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_seasons": list(project_seasons),
        "season_types": ["regular", "playoffs"],
        "source": "cdechoch/nba-data-archive nbastatsv3",
        "license": "Apache-2.0",
        "ordering": "actionId",
        "grain": "official score-state change, including score corrections",
        "passed": bool(coverage_frame["passed"].all()),
        "partitions": int(len(coverage_frame)),
        "games": int(coverage_frame["games"].sum()),
        "raw_rows": int(coverage_frame["raw_rows"].sum()),
        "score_change_rows": int(coverage_frame["score_change_rows"].sum()),
        "score_correction_rows": int(coverage_frame["score_correction_rows"].sum()),
        "coverage_path": str(output / "coverage.parquet"),
    }
    write_json_atomic(run, output / "run.json")
    if not run["passed"]:
        failed = coverage_frame.loc[~coverage_frame["passed"], [
            "project_season", "season_type", "structural_passed",
            "reference_available", "missing_reference_games", "unexpected_games",
            "final_score_mismatches",
        ]]
        raise ValueError(f"Scoring event verification failed:\n{failed.to_string(index=False)}")
    return run
