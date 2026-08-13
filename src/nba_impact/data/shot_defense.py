"""Build a leakage-safe event panel for observed-lineup shot-defense research."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


SHOT_DEFENSE_SCHEMA_VERSION = "observed_lineup_shot_defense_events_v1"
HOME_LINEUP_COLUMNS = tuple(f"home_player_{index}" for index in range(1, 6))
AWAY_LINEUP_COLUMNS = tuple(f"away_player_{index}" for index in range(1, 6))
LINEUP_COLUMNS = HOME_LINEUP_COLUMNS + AWAY_LINEUP_COLUMNS

ZONE_MAP = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "short_mid",
    "Mid-Range": "long_mid",
    "Left Corner 3": "corner_3",
    "Right Corner 3": "corner_3",
    "Above the Break 3": "above_break_3",
}


def _canonical_ids(values: pd.Series) -> pd.Series:
    return values.map(canonical_game_id)


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    count = int(frame.duplicated(columns, keep=False).sum())
    if count:
        raise ValueError(f"{label} has {count} rows on duplicate keys {columns}.")


def _attach_segments(shots: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    segment_groups = {
        str(game_id): group.sort_values("start_order_number", kind="stable")
        for game_id, group in segments.groupby("game_id", sort=False)
    }
    for game_id, group in shots.groupby("game_id", sort=False):
        local_segments = segment_groups.get(str(game_id))
        if local_segments is None or local_segments.empty:
            continue
        starts = local_segments["start_order_number"].to_numpy(dtype=np.int64)
        ends = local_segments["end_order_number"].to_numpy(dtype=np.int64)
        orders = group["orderNumber"].to_numpy(dtype=np.int64)
        indices = np.searchsorted(starts, orders, side="right") - 1
        safe_indices = np.maximum(indices, 0)
        valid = (indices >= 0) & (orders <= ends[safe_indices])
        if not valid.any():
            continue
        local = group.loc[valid].copy()
        indices = indices[valid]
        for column in ("ordinal_stint_id", "offense_team_id", *LINEUP_COLUMNS):
            local[column] = local_segments[column].to_numpy()[indices]
        outputs.append(local)
    if not outputs:
        return pd.DataFrame(columns=[*shots.columns, "ordinal_stint_id", "offense_team_id", *LINEUP_COLUMNS])
    return pd.concat(outputs, ignore_index=True)


def compute_shot_defense_events(
    v3_events: pd.DataFrame,
    shot_detail: pd.DataFrame,
    cdn_events: pd.DataFrame,
    event_states: pd.DataFrame,
    segments: pd.DataFrame,
    games: pd.DataFrame,
    *,
    heave_distance_feet: float = 35.0,
    heave_seconds_remaining: float = 3.0,
    minimum_join_coverage: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    """Join official FGAs to exact ordinal 5v5 lineups.

    The output identifies the observed defensive unit. It does not identify a
    primary defender and must not be used as if it did.
    """
    _require(
        v3_events,
        {"gameId", "actionId", "actionNumber", "period", "clock", "teamId", "personId", "isFieldGoal"},
        "V3 events",
    )
    _require(
        shot_detail,
        {
            "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PERIOD", "MINUTES_REMAINING",
            "SECONDS_REMAINING", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_DISTANCE",
            "LOC_X", "LOC_Y", "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG",
        },
        "shot detail",
    )
    _require(
        cdn_events,
        {"gameId", "actionNumber", "orderNumber", "period", "clock", "possession"},
        "CDN events",
    )
    _require(
        event_states,
        {
            "game_id", "actionId", "home_score_diff_before",
            "seconds_remaining_period", "regulation_seconds_remaining",
        },
        "event states",
    )
    _require(
        segments,
        {
            "game_id", "start_order_number", "end_order_number", "ordinal_stint_id",
            "offense_team_id", *LINEUP_COLUMNS,
        },
        "possession lineup segments",
    )
    _require(
        games,
        {
            "game_id", "season_start", "season_end", "season_label", "season_type",
            "game_date", "home_team_id", "away_team_id",
        },
        "game dimension",
    )

    v3 = v3_events.loc[
        pd.to_numeric(v3_events["isFieldGoal"], errors="coerce").fillna(0).astype(bool)
    ].copy()
    v3["game_id"] = _canonical_ids(v3["gameId"])
    v3["actionId"] = pd.to_numeric(v3["actionId"], errors="raise").astype(int)
    v3["actionNumber"] = pd.to_numeric(v3["actionNumber"], errors="raise").astype(int)
    _unique(v3, ["game_id", "actionNumber"], "V3 field goals")
    _unique(v3, ["game_id", "actionId"], "V3 field-goal action IDs")

    official = shot_detail.loc[
        pd.to_numeric(shot_detail["SHOT_ATTEMPTED_FLAG"], errors="coerce").eq(1)
    ].copy()
    official["game_id"] = _canonical_ids(official["GAME_ID"])
    official["actionNumber"] = pd.to_numeric(
        official["GAME_EVENT_ID"], errors="raise"
    ).astype(int)
    _unique(official, ["game_id", "actionNumber"], "official shot detail")

    joined = v3.merge(
        official,
        on=["game_id", "actionNumber"],
        how="inner",
        suffixes=("_v3", "_shot"),
        validate="one_to_one",
    )
    v3_official_coverage = len(joined) / max(len(v3), len(official), 1)
    player_identity = pd.to_numeric(joined["personId"], errors="coerce").eq(
        pd.to_numeric(joined["PLAYER_ID"], errors="coerce")
    )
    period_identity = pd.to_numeric(joined["period"], errors="coerce").eq(
        pd.to_numeric(joined["PERIOD"], errors="coerce")
    )
    v3_official_identity_mismatch_rows = int((~(player_identity & period_identity)).sum())
    joined = joined.loc[player_identity & period_identity].copy()

    cdn = cdn_events.copy()
    cdn["game_id"] = _canonical_ids(cdn["gameId"])
    cdn["actionNumber"] = pd.to_numeric(cdn["actionNumber"], errors="raise").astype(int)
    _unique(cdn, ["game_id", "actionNumber"], "CDN events")
    joined = joined.merge(
        cdn[["game_id", "actionNumber", "orderNumber", "period", "clock", "possession"]],
        on=["game_id", "actionNumber"],
        how="left",
        suffixes=("_v3", "_cdn"),
        validate="one_to_one",
    )
    cdn_aligned = (
        joined["orderNumber"].notna()
        & pd.to_numeric(joined["period_v3"], errors="coerce").eq(
            pd.to_numeric(joined["period_cdn"], errors="coerce")
        )
        & joined["clock_v3"].eq(joined["clock_cdn"])
    )
    cdn_alignment_rate = float(cdn_aligned.mean()) if len(joined) else 0.0
    joined = joined.loc[cdn_aligned].copy()
    joined["orderNumber"] = pd.to_numeric(joined["orderNumber"], errors="raise").astype(int)

    states = event_states.copy()
    states["game_id"] = states["game_id"].astype(str).str.zfill(10)
    states["actionId"] = pd.to_numeric(states["actionId"], errors="raise").astype(int)
    _unique(states, ["game_id", "actionId"], "event states")
    state_columns = [
        "game_id", "actionId", "home_score_diff_before",
        "seconds_remaining_period", "regulation_seconds_remaining",
    ]
    joined = joined.merge(states[state_columns], on=["game_id", "actionId"], how="left", validate="one_to_one")
    state_complete = joined[state_columns[2:]].notna().all(axis=1)
    state_coverage = float(state_complete.mean()) if len(joined) else 0.0
    joined = joined.loc[state_complete].copy()

    segment_input_rows = len(joined)
    joined = _attach_segments(joined, segments)
    segment_coverage = len(joined) / max(segment_input_rows, 1)
    dimension_columns = [
        "game_id", "season_start", "season_end", "season_label", "season_type",
        "game_date", "home_team_id", "away_team_id",
    ]
    joined = joined.merge(games[dimension_columns], on="game_id", validate="many_to_one")

    joined["shooter_id"] = pd.to_numeric(joined["personId"], errors="raise").astype(int)
    joined["shot_team_id"] = pd.to_numeric(joined["teamId"], errors="raise").astype(int)
    joined["offense_team_id"] = pd.to_numeric(joined["offense_team_id"], errors="raise").astype(int)
    joined["home_team_id"] = pd.to_numeric(joined["home_team_id"], errors="raise").astype(int)
    joined["away_team_id"] = pd.to_numeric(joined["away_team_id"], errors="raise").astype(int)
    joined["offense_is_home"] = joined["offense_team_id"].eq(joined["home_team_id"])
    joined["defense_team_id"] = np.where(
        joined["offense_is_home"], joined["away_team_id"], joined["home_team_id"]
    ).astype(int)
    joined["possession"] = pd.to_numeric(joined["possession"], errors="coerce")
    ownership_valid = joined["shot_team_id"].eq(joined["offense_team_id"]) & joined[
        "possession"
    ].eq(joined["offense_team_id"])

    for index in range(1, 6):
        joined[f"offense_player_{index}"] = np.where(
            joined["offense_is_home"],
            joined[f"home_player_{index}"],
            joined[f"away_player_{index}"],
        ).astype(int)
        joined[f"defense_player_{index}"] = np.where(
            joined["offense_is_home"],
            joined[f"away_player_{index}"],
            joined[f"home_player_{index}"],
        ).astype(int)
    offense_columns = [f"offense_player_{index}" for index in range(1, 6)]
    defense_columns = [f"defense_player_{index}" for index in range(1, 6)]
    offense_values = joined[offense_columns].to_numpy(dtype=np.int64)
    defense_values = joined[defense_columns].to_numpy(dtype=np.int64)
    shooters = joined["shooter_id"].to_numpy(dtype=np.int64)
    shooter_in_offense = (offense_values == shooters[:, None]).any(axis=1)
    valid_lineups = (
        np.array([len(set(row)) == 5 for row in offense_values])
        & np.array([len(set(row)) == 5 for row in defense_values])
        & np.array([not set(left).intersection(right) for left, right in zip(offense_values, defense_values)])
    )
    identity_valid = ownership_valid.to_numpy() & shooter_in_offense & valid_lineups
    identity_failure_rows = int((~identity_valid).sum())
    joined = joined.loc[identity_valid].copy()

    joined["shot_zone"] = joined["SHOT_ZONE_BASIC"].map(ZONE_MAP)
    backcourt_rows = int(joined["SHOT_ZONE_BASIC"].eq("Backcourt").sum())
    unknown_zone_rows = int(joined["shot_zone"].isna().sum() - backcourt_rows)
    seconds_from_official = (
        60.0 * pd.to_numeric(joined["MINUTES_REMAINING"], errors="coerce")
        + pd.to_numeric(joined["SECONDS_REMAINING"], errors="coerce")
    )
    distance = pd.to_numeric(joined["SHOT_DISTANCE"], errors="coerce")
    heave = (
        joined["shot_zone"].notna()
        & distance.gt(heave_distance_feet)
        & seconds_from_official.le(heave_seconds_remaining)
    )
    heave_rows = int(heave.sum())
    joined = joined.loc[joined["shot_zone"].notna() & ~heave].copy()

    joined["shot_made"] = pd.to_numeric(joined["SHOT_MADE_FLAG"], errors="raise").astype(int)
    joined["shot_value"] = np.where(joined["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2).astype(int)
    joined["shot_distance_feet"] = pd.to_numeric(joined["SHOT_DISTANCE"], errors="raise")
    joined["location_x"] = pd.to_numeric(joined["LOC_X"], errors="raise")
    joined["location_y"] = pd.to_numeric(joined["LOC_Y"], errors="raise")
    joined["shot_angle_radians"] = np.arctan2(joined["location_x"], joined["location_y"])
    joined["offense_score_diff_before"] = np.where(
        joined["offense_is_home"],
        joined["home_score_diff_before"],
        -joined["home_score_diff_before"],
    )
    joined["shot_id"] = joined["game_id"] + ":" + joined["actionNumber"].astype(str)

    output_columns = [
        "shot_id", "game_id", "actionId", "actionNumber", "orderNumber", "ordinal_stint_id",
        "season_start", "season_end", "season_label", "season_type", "game_date",
        "period_v3", "seconds_remaining_period", "regulation_seconds_remaining",
        "offense_score_diff_before", "offense_is_home", "shooter_id",
        "offense_team_id", "defense_team_id", "shot_zone", "shot_value", "shot_made",
        "shot_distance_feet", "location_x", "location_y", "shot_angle_radians",
        *offense_columns, *defense_columns,
    ]
    output = joined[output_columns].rename(columns={"period_v3": "period"}).sort_values(
        ["game_id", "orderNumber"], kind="stable"
    ).reset_index(drop=True)
    duplicate_shot_ids = int(output.duplicated("shot_id", keep=False).sum())
    nonfinite_numeric = int(
        (~np.isfinite(output.select_dtypes(include="number"))).sum().sum()
    )
    passed = (
        v3_official_coverage >= minimum_join_coverage
        and cdn_alignment_rate >= minimum_join_coverage
        and state_coverage >= minimum_join_coverage
        and segment_coverage >= minimum_join_coverage
        and identity_failure_rows == 0
        and unknown_zone_rows == 0
        and duplicate_shot_ids == 0
        and nonfinite_numeric == 0
    )
    quality = {
        "passed": passed,
        "v3_field_goal_rows": int(len(v3)),
        "official_shot_rows": int(len(official)),
        "v3_official_join_rows": int(len(v3.merge(official, on=["game_id", "actionNumber"], how="inner"))),
        "v3_official_join_coverage": float(v3_official_coverage),
        "v3_official_identity_mismatch_rows": v3_official_identity_mismatch_rows,
        "cdn_alignment_rate": cdn_alignment_rate,
        "event_state_coverage": state_coverage,
        "ordinal_segment_coverage": float(segment_coverage),
        "identity_failure_rows": identity_failure_rows,
        "backcourt_rows_excluded": backcourt_rows,
        "heave_rows_excluded": heave_rows,
        "unknown_zone_rows": unknown_zone_rows,
        "output_rows": int(len(output)),
        "output_games": int(output["game_id"].nunique()),
        "duplicate_shot_ids": duplicate_shot_ids,
        "nonfinite_numeric_values": nonfinite_numeric,
        "zone_counts": {str(key): int(value) for key, value in output["shot_zone"].value_counts().sort_index().items()},
    }
    if not passed:
        raise ValueError(f"Shot-defense event panel failed quality gates: {quality}.")
    return output, quality


def build_shot_defense_events(
    event_root: str | Path,
    event_states_path: str | Path,
    game_dim_path: str | Path,
    segments_path: str | Path,
    destination: str | Path,
    manifest_dir: str | Path,
    *,
    seasons: tuple[int, ...] = (2023, 2024, 2025),
) -> dict:
    """Build and atomically write the regular-season shot-defense panel."""
    root = Path(event_root)
    games = pd.read_parquet(game_dim_path)
    games = games.loc[
        games["season_start"].isin(seasons) & games["season_type"].eq("regular")
    ].copy()
    game_ids = set(games["game_id"].astype(str))
    states = pd.read_parquet(event_states_path)
    states = states.loc[states["game_id"].astype(str).isin(game_ids)].copy()
    segments = pd.read_parquet(segments_path)
    segments = segments.loc[segments["game_id"].astype(str).isin(game_ids)].copy()

    outputs: list[pd.DataFrame] = []
    season_quality: dict[str, dict] = {}
    source_paths = [Path(event_states_path), Path(game_dim_path), Path(segments_path)]
    for season in seasons:
        paths = {
            source: root / source / f"season={season}" / "regular.parquet"
            for source in ("nbastatsv3", "shotdetail", "cdnnba")
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise ValueError(f"Season {season} is missing shot-defense sources: {missing}.")
        source_paths.extend(paths.values())
        season_games = games.loc[games["season_start"].eq(season)]
        season_ids = set(season_games["game_id"].astype(str))
        panel, quality = compute_shot_defense_events(
            pd.read_parquet(paths["nbastatsv3"]),
            pd.read_parquet(paths["shotdetail"]),
            pd.read_parquet(paths["cdnnba"]),
            states.loc[states["game_id"].astype(str).isin(season_ids)],
            segments.loc[segments["game_id"].astype(str).isin(season_ids)],
            season_games,
        )
        outputs.append(panel)
        season_quality[str(season)] = quality

    output = pd.concat(outputs, ignore_index=True).sort_values(
        ["game_id", "orderNumber"], kind="stable"
    ).reset_index(drop=True)
    source_records = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ]
    identity = hashlib.sha256(
        json.dumps([(record["path"], record["sha256"]) for record in source_records]).encode()
    ).hexdigest()[:16]
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    output.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    snapshot = {
        "snapshot_id": f"shot_defense_events_{identity}",
        "dataset": SHOT_DEFENSE_SCHEMA_VERSION,
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in season_quality.values()),
        "row_count": int(len(output)),
        "game_count": int(output["game_id"].nunique()),
        "seasons": list(seasons),
        "season_quality": season_quality,
        "grain": "one official field-goal attempt with exact shooter, zone, result, and ordinal 5v5 lineups",
        "estimand_boundary": "Observed defensive-unit association with zone mix and conversion conditional on an FGA.",
        "forbidden_interpretation": "Primary-defender impact, causal player defense, shot suppression, or possession defense.",
        "lineup_policy": "Exact CDN orderNumber segment; actionNumber is only an equality join key.",
        "exclusions": {
            "season_type": "regular only",
            "backcourt": True,
            "heave": "shot distance >35 feet with <=3 seconds left in period",
        },
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "source_files": source_records,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
