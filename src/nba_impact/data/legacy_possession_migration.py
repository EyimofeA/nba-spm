"""Migrate validated legacy RAPM cache rows into the clean possession contract.

The legacy cache stores one terminal ten-player lineup and one point outcome per
historical row.  It does not preserve within-possession substitution timing.
This builder therefore creates exactly one segment per accepted legacy row and
labels it as a terminal-only assignment.  It never creates a longer observed
lineup stint or adds missing scoring rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic
from .normalize import classify_game_type
from .possessions import AWAY_LINEUP_COLUMNS, HOME_LINEUP_COLUMNS


LEGACY_REQUIRED = (
    "home_poss", "pts", "a1", "a2", "a3", "a4", "a5", "h1", "h2", "h3", "h4", "h5",
    "season", "date", "period", "num", "gameid",
)
OFFICIAL_REQUIRED = (
    "project_season", "season_type", "game_id", "game_date", "home_team_id", "away_team_id",
    "home_score", "away_score",
)
LEGACY_AWAY = tuple(f"a{number}" for number in range(1, 6))
LEGACY_HOME = tuple(f"h{number}" for number in range(1, 6))


def _canonical_game_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Game ID cannot be null.")
    return f"{int(value):010d}"


def _season_label(season_end: int) -> str:
    return f"{season_end - 1}-{str(season_end)[-2:]}"


def _load_official_scores(path: str | Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    scores = pd.read_parquet(path)
    missing = sorted(set(OFFICIAL_REQUIRED) - set(scores.columns))
    if missing:
        raise ValueError(f"Official-score input is missing columns: {missing}")
    scores = scores.loc[scores["project_season"].isin(seasons), list(OFFICIAL_REQUIRED)].copy()
    scores["game_id"] = scores["game_id"].map(_canonical_game_id)
    scores["game_date"] = pd.to_datetime(scores["game_date"], errors="raise")
    numeric = ("project_season", "home_team_id", "away_team_id", "home_score", "away_score")
    for column in numeric:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype("int64")
    if scores.duplicated("game_id", keep=False).any():
        raise ValueError("Official-score input has duplicate game identities.")
    if scores[["home_team_id", "away_team_id"]].le(0).any(axis=None) or scores["home_team_id"].eq(
        scores["away_team_id"]
    ).any():
        raise ValueError("Official-score input has invalid home/away team identities.")
    return scores


def _game_quality(cache: pd.DataFrame, official: pd.DataFrame, season: int) -> pd.DataFrame:
    work = cache.copy()
    official = official.copy()
    official["game_id"] = official["game_id"].map(_canonical_game_id)
    official["game_date"] = pd.to_datetime(official["game_date"], errors="raise")
    work["game_id"] = work["gameid"].map(_canonical_game_id)
    work["cache_date"] = pd.to_datetime(work["date"], errors="raise")
    work["legacy_season_type"] = work["game_id"].map(classify_game_type)
    player_columns = [*LEGACY_AWAY, *LEGACY_HOME]
    for column in [*player_columns, "home_poss", "pts", "period", "num", "season"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["valid_lineup"] = work[player_columns].notna().all(axis=1) & work[player_columns].nunique(axis=1).eq(10)
    work["valid_points"] = work["pts"].notna() & work["pts"].ge(0) & work["pts"].mod(1).eq(0)
    work["valid_home_poss"] = work["home_poss"].isin((0, 1))
    work["valid_period"] = work["period"].notna() & work["period"].ge(1) & work["period"].mod(1).eq(0)
    work["valid_num"] = work["num"].notna() & work["num"].ge(0) & work["num"].mod(1).eq(0)
    work["valid_legacy_key"] = ~work.duplicated(["game_id", "period", "num"], keep=False)

    def _side_points(group: pd.Series, side: int) -> int:
        return int(group[work.loc[group.index, "home_poss"].eq(side)].sum())

    grouped = work.groupby("game_id", as_index=False, sort=False).agg(
        cache_date=("cache_date", "first"),
        source_rows=("game_id", "size"),
        source_season=("season", "first"),
        source_season_type=("legacy_season_type", "first"),
        min_period=("period", "min"),
        max_period=("period", "max"),
        legacy_home_score=("pts", lambda points: _side_points(points, 1)),
        legacy_away_score=("pts", lambda points: _side_points(points, 0)),
        invalid_lineup_rows=("valid_lineup", lambda valid: int((~valid).sum())),
        invalid_point_rows=("valid_points", lambda valid: int((~valid).sum())),
        invalid_home_poss_rows=("valid_home_poss", lambda valid: int((~valid).sum())),
        invalid_period_rows=("valid_period", lambda valid: int((~valid).sum())),
        invalid_num_rows=("valid_num", lambda valid: int((~valid).sum())),
        duplicate_legacy_key_rows=("valid_legacy_key", lambda valid: int((~valid).sum())),
    )
    official = official.loc[official["project_season"].eq(season)].copy()
    quality = grouped.merge(official, on="game_id", how="outer", validate="one_to_one", indicator=True)
    quality["date_matches_official"] = quality["cache_date"].dt.normalize().eq(
        quality["game_date"].dt.normalize()
    )
    quality["season_matches_source"] = quality["source_season"].eq(season)
    quality["period_coverage_complete"] = quality["min_period"].eq(1) & quality["max_period"].ge(4)
    quality["score_conserved"] = quality["legacy_home_score"].eq(quality["home_score"]) & quality[
        "legacy_away_score"
    ].eq(quality["away_score"])
    matched_identity = quality["_merge"].eq("both")
    issue_columns = {
        "missing_official_game_identity": quality["_merge"].eq("left_only"),
        "missing_legacy_cache_game": quality["_merge"].eq("right_only"),
        "date_mismatch": matched_identity & ~quality["date_matches_official"],
        "season_mismatch": matched_identity & ~quality["season_matches_source"],
        "incomplete_period_coverage": matched_identity & ~quality["period_coverage_complete"],
        "score_not_conserved": matched_identity & ~quality["score_conserved"],
        "invalid_lineup": quality["invalid_lineup_rows"].fillna(1).gt(0),
        "invalid_points": quality["invalid_point_rows"].fillna(1).gt(0),
        "invalid_home_poss": quality["invalid_home_poss_rows"].fillna(1).gt(0),
        "invalid_period": quality["invalid_period_rows"].fillna(1).gt(0),
        "invalid_event_number": quality["invalid_num_rows"].fillna(1).gt(0),
        "duplicate_legacy_period_event": quality["duplicate_legacy_key_rows"].fillna(1).gt(0),
    }
    quality["issues"] = [
        ";".join(name for name, mask in issue_columns.items() if bool(mask.iloc[index]))
        for index in range(len(quality))
    ]
    quality["passed"] = quality["issues"].eq("")
    quality["project_season"] = season
    return quality


def _migrate_accepted_rows(cache: pd.DataFrame, identities: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = identities.loc[identities["passed"]].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    source = cache.copy().reset_index(names="legacy_row_index")
    source["game_id"] = source["gameid"].map(_canonical_game_id)
    source["cache_date"] = pd.to_datetime(source["date"], errors="raise")
    source["period"] = pd.to_numeric(source["period"], errors="raise").astype("int64")
    source["num"] = pd.to_numeric(source["num"], errors="raise").astype("int64")
    source["pts"] = pd.to_numeric(source["pts"], errors="raise").astype("int64")
    source["home_poss"] = pd.to_numeric(source["home_poss"], errors="raise").astype("int64")
    source = source.merge(
        accepted[
            [
                "game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score",
                "source_season_type",
            ]
        ],
        on="game_id",
        how="inner",
        validate="many_to_one",
    ).sort_values(["game_date", "game_id", "period", "num", "legacy_row_index"], kind="stable")
    source["possession_number"] = source.groupby("game_id", sort=False).cumcount().add(1).astype("int64")
    source["possession_id"] = source["game_id"] + ":legacy:" + source["possession_number"].astype(str).str.zfill(4)
    source["legacy_event_sequence"] = source["period"] * 1_000_000 + source["num"]
    source["season_start"] = season - 1
    source["season_end"] = season
    source["season_label"] = _season_label(season)
    source["season_type"] = source["source_season_type"].astype("string")
    source["offense_team_id"] = np.where(
        source["home_poss"].eq(1), source["home_team_id"], source["away_team_id"]
    ).astype("int64")
    source["defense_team_id"] = np.where(
        source["home_poss"].eq(1), source["away_team_id"], source["home_team_id"]
    ).astype("int64")
    source["offense_is_home"] = source["home_poss"].eq(1)
    source["home_points"] = np.where(source["offense_is_home"], source["pts"], 0).astype("int64")
    source["away_points"] = np.where(~source["offense_is_home"], source["pts"], 0).astype("int64")
    source["source_dataset"] = "legacy_matchups_cache"
    source["lineup_assignment"] = "legacy_terminal_lineup"
    source["legacy_source_file"] = f"matchups_{season}.parquet"
    source["ordinal_stint_id"] = (
        source["game_id"] + ":legacy_terminal:" + source["possession_number"].astype(str).str.zfill(4)
    )

    possessions = pd.DataFrame(
        {
            "possession_id": source["possession_id"], "game_id": source["game_id"],
            "possession_number": source["possession_number"], "season_start": source["season_start"],
            "season_end": source["season_end"], "season_label": source["season_label"],
            "season_type": source["season_type"], "game_date": source["game_date"], "period": source["period"],
            "start_order_number": source["legacy_event_sequence"], "end_order_number": source["legacy_event_sequence"],
            "start_action_number": source["num"], "end_action_number": source["num"],
            "start_seconds_elapsed": np.nan, "end_seconds_elapsed": np.nan,
            "offense_team_id": source["offense_team_id"], "home_team_id": source["home_team_id"],
            "away_team_id": source["away_team_id"], "points": source["pts"],
            "home_points": source["home_points"], "away_points": source["away_points"],
            "action_count": 1, "lineup_segment_count": 1, "defense_team_id": source["defense_team_id"],
            "offense_is_home": source["offense_is_home"], "lineup_ready": True,
            "source_dataset": source["source_dataset"], "legacy_source_file": source["legacy_source_file"],
            "legacy_row_index": source["legacy_row_index"], "legacy_event_num": source["num"],
            "lineup_assignment": source["lineup_assignment"],
        }
    )
    segments = pd.DataFrame(
        {
            "possession_id": source["possession_id"], "game_id": source["game_id"],
            "possession_number": source["possession_number"], "segment_number": 1,
            "period": source["period"], "start_order_number": source["legacy_event_sequence"],
            "end_order_number": source["legacy_event_sequence"], "start_action_number": source["num"],
            "end_action_number": source["num"], "start_seconds_elapsed": np.nan,
            "end_seconds_elapsed": np.nan, "ordinal_stint_id": source["ordinal_stint_id"],
            "offense_team_id": source["offense_team_id"], "points": source["pts"], "action_count": 1,
            **{f"home_player_{number}": source[f"h{number}"].astype("int64") for number in range(1, 6)},
            **{f"away_player_{number}": source[f"a{number}"].astype("int64") for number in range(1, 6)},
            "possession_segment_id": source["possession_id"] + ":s01",
            "source_dataset": source["source_dataset"], "legacy_source_file": source["legacy_source_file"],
            "legacy_row_index": source["legacy_row_index"], "legacy_event_num": source["num"],
            "lineup_assignment": source["lineup_assignment"],
        }
    )
    return possessions, segments


def _output_quality(possessions: pd.DataFrame, segments: pd.DataFrame, identities: pd.DataFrame) -> dict[str, int]:
    player_columns = [*HOME_LINEUP_COLUMNS, *AWAY_LINEUP_COLUMNS]
    accepted = identities.loc[identities["passed"], "game_id"].astype(str)
    possession_scores = possessions.groupby("game_id", as_index=False).agg(
        home_points=("home_points", "sum"), away_points=("away_points", "sum"), rows=("possession_id", "size")
    )
    expected = identities.loc[identities["passed"], ["game_id", "home_score", "away_score", "source_rows"]]
    score_check = possession_scores.merge(expected, on="game_id", validate="one_to_one")
    segment_points = segments.groupby("possession_id", as_index=False)["points"].sum()
    point_check = possessions[["possession_id", "points"]].merge(
        segment_points, on="possession_id", how="left", suffixes=("_possession", "_segments"), validate="one_to_one"
    )
    return {
        "accepted_games": int(len(accepted)),
        "output_possession_rows": int(len(possessions)),
        "output_segment_rows": int(len(segments)),
        "duplicate_possession_ids": int(possessions.duplicated("possession_id", keep=False).sum()),
        "duplicate_segment_ids": int(segments.duplicated("possession_segment_id", keep=False).sum()),
        "invalid_segment_lineups": int(segments.loc[:, player_columns].nunique(axis=1).ne(10).sum()),
        "unreconciled_legacy_rows": int((score_check["rows"] != score_check["source_rows"]).sum()),
        "score_conservation_failures": int(
            (score_check["home_points"].ne(score_check["home_score"]) | score_check["away_points"].ne(score_check["away_score"])).sum()
        ),
        "segment_point_mismatches": int(point_check["points_possession"].ne(point_check["points_segments"]).sum()),
        "unexpected_games": int(len(set(possessions["game_id"].astype(str)) - set(accepted))),
        "missing_accepted_games": int(len(set(accepted) - set(possessions["game_id"].astype(str)))),
    }


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(destination)


def migrate_legacy_possession_cache(
    cache_dir: str | Path,
    official_scores_path: str | Path,
    possessions_output: str | Path,
    segments_output: str | Path,
    game_identity_output: str | Path,
    quality_output: str | Path,
    report_path: str | Path,
    *,
    seasons: tuple[int, ...] = tuple(range(2017, 2024)),
) -> dict:
    """Write only legacy games that meet every independent identity and score gate."""
    requested = tuple(sorted({int(season) for season in seasons}))
    if not requested:
        raise ValueError("At least one legacy season is required.")
    cache_dir = Path(cache_dir)
    official_scores_path = Path(official_scores_path)
    official = _load_official_scores(official_scores_path, requested)
    all_possessions: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_quality: list[pd.DataFrame] = []
    source_files: list[dict[str, str | int]] = []
    for season in requested:
        cache_path = cache_dir / f"matchups_{season}.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing legacy cache partition: {cache_path}")
        cache = pd.read_parquet(cache_path)
        missing = sorted(set(LEGACY_REQUIRED) - set(cache.columns))
        if missing:
            raise ValueError(f"{cache_path.name} is missing legacy columns: {missing}")
        quality = _game_quality(cache, official, season)
        possessions, segments = _migrate_accepted_rows(cache, quality, season)
        all_possessions.append(possessions)
        all_segments.append(segments)
        all_quality.append(quality)
        source_files.append(
            {"path": cache_path.name, "bytes": cache_path.stat().st_size, "sha256": sha256_file(cache_path)}
        )
    possessions = pd.concat(all_possessions, ignore_index=True)
    segments = pd.concat(all_segments, ignore_index=True)
    quality = pd.concat(all_quality, ignore_index=True).sort_values(
        ["project_season", "game_id"], kind="stable"
    ).reset_index(drop=True)
    checks = _output_quality(possessions, segments, quality)
    failed_checks = {key: value for key, value in checks.items() if key not in {"accepted_games", "output_possession_rows", "output_segment_rows"} and value}
    if failed_checks:
        raise AssertionError(f"Migrated legacy output failed contract checks: {failed_checks}")
    identities = quality.loc[
        quality["passed"],
        [
            "game_id", "project_season", "source_season", "source_season_type", "game_date",
            "home_team_id", "away_team_id", "home_score", "away_score", "source_rows",
            "legacy_home_score", "legacy_away_score", "score_conserved",
        ],
    ].copy()
    for path, frame in (
        (Path(possessions_output), possessions), (Path(segments_output), segments),
        (Path(game_identity_output), identities), (Path(quality_output), quality),
    ):
        _write_parquet_atomic(frame, path)
    report_path = Path(report_path)
    report = {
        "dataset": "legacy_matchups_cache_migration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": (
            "One legacy cache row becomes one terminal-lineup possession and one segment. "
            "No within-possession substitution timing is inferred. Only games with independently verified "
            "official home/away identity, date, final score, and valid legacy keys/lineups are emitted."
        ),
        "requested_seasons": list(requested),
        "source_files": source_files + [{"path": official_scores_path.name, "bytes": official_scores_path.stat().st_size, "sha256": sha256_file(official_scores_path)}],
        "quality": checks,
        "accepted_games_by_season": {
            str(season): int(quality.loc[quality["project_season"].eq(season) & quality["passed"]].shape[0])
            for season in requested
        },
        "blocked_games_by_issue": {
            issue: int(quality["issues"].str.contains(issue, regex=False).sum())
            for issue in sorted({item for values in quality["issues"] for item in str(values).split(";") if item})
        },
        "complete": bool(quality["passed"].all()),
        "passed": True,
        "possessions_path": str(Path(possessions_output).resolve()),
        "segments_path": str(Path(segments_output).resolve()),
        "game_identity_path": str(Path(game_identity_output).resolve()),
        "quality_path": str(Path(quality_output).resolve()),
    }
    write_json_atomic(report, report_path)
    return report
