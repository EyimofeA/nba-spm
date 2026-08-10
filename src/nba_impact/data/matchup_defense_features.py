"""Opponent-adjusted annual defense features from NBA matchup assignments."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


RAW_COLUMNS = (
    "game_id",
    "person_id",
    "matchups_person_id",
    "partial_possessions",
    "player_points",
    "matchup_assists",
    "matchup_turnovers",
    "matchup_field_goals_made",
    "matchup_field_goals_attempted",
    "matchup_three_pointers_made",
    "matchup_three_pointers_attempted",
    "matchup_free_throws_made",
    "shooting_fouls",
)

MATCHUP_DEFENSE_FEATURES = (
    "matchup_points_allowed_p100",
    "matchup_opponent_adjusted_points_saved_p100",
    "matchup_opponent_adjusted_points_saved_p100_eb",
    "matchup_efg_pct_allowed_eb",
    "matchup_fga_p100",
    "matchup_3pa_share",
    "matchup_turnovers_forced_p100",
    "matchup_assists_allowed_p100",
    "matchup_shooting_fouls_committed_p100",
)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def compute_matchup_defense_features(
    frame: pd.DataFrame,
    season: int,
    *,
    defender_prior_possessions: float = 500.0,
    shooting_prior_attempts: float = 200.0,
) -> tuple[pd.DataFrame, dict]:
    """Build defender-season rates; positive points saved means better defense."""
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Matchup source is missing columns: {missing}.")
    work = _numeric(frame[list(RAW_COLUMNS)], tuple(RAW_COLUMNS[1:]))
    required = ("game_id", "person_id", "matchups_person_id", "partial_possessions")
    null_rows = int(work[list(required)].isna().any(axis=1).sum())
    negative_possessions = int(work["partial_possessions"].lt(0).sum())
    duplicate_rows = int(
        work.duplicated(["game_id", "person_id", "matchups_person_id"]).sum()
    )
    reconstructed_points = (
        2.0
        * (
            work["matchup_field_goals_made"]
            - work["matchup_three_pointers_made"]
        )
        + 3.0 * work["matchup_three_pointers_made"]
        + work["matchup_free_throws_made"]
    )
    point_mismatches = int(
        (~np.isclose(reconstructed_points, work["player_points"], atol=1e-9)).sum()
    )
    if null_rows or negative_possessions or duplicate_rows or point_mismatches:
        raise ValueError(
            "Matchup source failed identity or conservation checks: "
            f"null_rows={null_rows}, negative_possessions={negative_possessions}, "
            f"duplicate_rows={duplicate_rows}, point_mismatches={point_mismatches}."
        )

    work = work.loc[work["partial_possessions"].gt(0)].copy()
    pair = work.groupby(["person_id", "matchups_person_id"], as_index=False).agg(
        pair_possessions=("partial_possessions", "sum"),
        pair_points=("player_points", "sum"),
        fgm=("matchup_field_goals_made", "sum"),
        fga=("matchup_field_goals_attempted", "sum"),
        three_pm=("matchup_three_pointers_made", "sum"),
        three_pa=("matchup_three_pointers_attempted", "sum"),
        turnovers=("matchup_turnovers", "sum"),
        assists=("matchup_assists", "sum"),
        shooting_fouls=("shooting_fouls", "sum"),
    )
    offense = pair.groupby("person_id", as_index=False).agg(
        offense_possessions=("pair_possessions", "sum"),
        offense_points=("pair_points", "sum"),
    )
    league_ppp = float(pair["pair_points"].sum() / pair["pair_possessions"].sum())
    pair = pair.merge(offense, on="person_id", validate="many_to_one")
    other_possessions = pair["offense_possessions"] - pair["pair_possessions"]
    other_points = pair["offense_points"] - pair["pair_points"]
    pair["expected_ppp"] = (other_points / other_possessions.where(other_possessions.gt(0))).fillna(
        league_ppp
    )
    pair["expected_points"] = pair["expected_ppp"] * pair["pair_possessions"]
    pair["points_saved"] = pair["expected_points"] - pair["pair_points"]

    defense = pair.groupby("matchups_person_id", as_index=False).agg(
        matchup_possessions=("pair_possessions", "sum"),
        matchup_points=("pair_points", "sum"),
        matchup_expected_points=("expected_points", "sum"),
        matchup_points_saved=("points_saved", "sum"),
        fgm=("fgm", "sum"),
        fga=("fga", "sum"),
        three_pm=("three_pm", "sum"),
        three_pa=("three_pa", "sum"),
        turnovers=("turnovers", "sum"),
        assists=("assists", "sum"),
        shooting_fouls=("shooting_fouls", "sum"),
    ).rename(columns={"matchups_person_id": "PLAYER_ID"})

    # Leave-one-defender-out expectations are not guaranteed to sum to actual
    # points. Remove the season-wide exposure-weighted offset before shrinkage.
    offset = float(defense["matchup_points_saved"].sum() / defense["matchup_possessions"].sum())
    defense["matchup_points_saved_centered"] = (
        defense["matchup_points_saved"] - offset * defense["matchup_possessions"]
    )
    possessions = defense["matchup_possessions"]
    reliability = possessions / (possessions + defender_prior_possessions)
    league_efg = float((defense["fgm"].sum() + 0.5 * defense["three_pm"].sum()) / defense["fga"].sum())

    defense["Season"] = int(season)
    defense["matchup_points_allowed_p100"] = 100.0 * defense["matchup_points"] / possessions
    defense["matchup_opponent_adjusted_points_saved_p100"] = (
        100.0 * defense["matchup_points_saved_centered"] / possessions
    )
    defense["matchup_opponent_adjusted_points_saved_p100_eb"] = (
        reliability * defense["matchup_opponent_adjusted_points_saved_p100"]
    )
    defense["matchup_efg_pct_allowed_eb"] = 100.0 * (
        defense["fgm"] + 0.5 * defense["three_pm"] + shooting_prior_attempts * league_efg
    ) / (defense["fga"] + shooting_prior_attempts)
    defense["matchup_fga_p100"] = 100.0 * defense["fga"] / possessions
    league_three_pa_share = float(defense["three_pa"].sum() / defense["fga"].sum())
    defense["matchup_3pa_share"] = (
        defense["three_pa"] / defense["fga"].where(defense["fga"].gt(0))
    ).fillna(league_three_pa_share)
    defense["matchup_turnovers_forced_p100"] = 100.0 * defense["turnovers"] / possessions
    defense["matchup_assists_allowed_p100"] = 100.0 * defense["assists"] / possessions
    defense["matchup_shooting_fouls_committed_p100"] = 100.0 * defense["shooting_fouls"] / possessions

    output = defense[["PLAYER_ID", "Season", "matchup_possessions", *MATCHUP_DEFENSE_FEATURES]].copy()
    output["PLAYER_ID"] = output["PLAYER_ID"].astype(int)
    if output[list(MATCHUP_DEFENSE_FEATURES)].isna().any().any():
        raise ValueError("Matchup features contain null values.")
    audit = {
        "source_rows": int(len(frame)),
        "positive_possession_rows": int(len(work)),
        "offensive_players": int(pair["person_id"].nunique()),
        "defenders": int(len(output)),
        "games": int(frame["game_id"].nunique()),
        "zero_possession_rows": int(len(frame) - len(work)),
        "duplicate_source_keys": duplicate_rows,
        "required_null_rows": null_rows,
        "negative_possession_rows": negative_possessions,
        "point_reconstruction_mismatches": point_mismatches,
        "league_points_per_matchup_possession": league_ppp,
        "league_efg_pct": 100.0 * league_efg,
        "centered_points_saved_sum": float(defense["matchup_points_saved_centered"].sum()),
    }
    return output, audit


def _read_archive(path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = Path(f"{path}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    member_name = manifest["archive_member"]
    with tarfile.open(path, mode="r:xz") as archive:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Could not read {member_name} from {path}.")
        frame = pd.read_csv(extracted, low_memory=False)
    return frame, manifest


def build_matchup_defense_features(
    archive_root: str | Path,
    box_source_dir: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2018, 2026)),
    defender_prior_possessions: float = 500.0,
    shooting_prior_attempts: float = 200.0,
) -> dict:
    """Build a content-addressed annual matchup-defense feature artifact."""
    archive_root = Path(archive_root)
    box_source_dir = Path(box_source_dir)
    outputs: list[pd.DataFrame] = []
    season_quality: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}
    box_hashes: dict[str, str] = {}

    for season in seasons:
        archives = sorted((archive_root / f"season={season}").glob("*.tar.xz"))
        if len(archives) != 1:
            raise ValueError(f"Expected one matchup archive for season {season}; found {len(archives)}.")
        archive = archives[0]
        frame, manifest = _read_archive(archive)
        features, quality = compute_matchup_defense_features(
            frame,
            season,
            defender_prior_possessions=defender_prior_possessions,
            shooting_prior_attempts=shooting_prior_attempts,
        )
        box_path = box_source_dir / f"{season}.csv"
        box = pd.read_csv(box_path, usecols=["PLAYER_ID", "DefPoss"])
        box["PLAYER_ID"] = pd.to_numeric(box["PLAYER_ID"], errors="coerce")
        box["DefPoss"] = pd.to_numeric(box["DefPoss"], errors="coerce")
        box = box.dropna().groupby("PLAYER_ID", as_index=False)["DefPoss"].sum()
        joined = features.merge(box, on="PLAYER_ID", how="left", validate="one_to_one")
        matched = joined["DefPoss"].notna()
        quality["box_id_row_match_rate"] = float(matched.mean())
        quality["box_id_exposure_match_rate"] = float(
            joined.loc[matched, "matchup_possessions"].sum()
            / joined["matchup_possessions"].sum()
        )
        quality["box_exposure_correlation"] = float(
            joined.loc[matched, ["matchup_possessions", "DefPoss"]].corr().iloc[0, 1]
        )
        quality["source_manifest_season"] = int(manifest["season"])
        if (
            quality["box_id_exposure_match_rate"] < 0.99
            or quality["box_exposure_correlation"] < 0.98
        ):
            raise ValueError(f"Season {season} matchup-to-box quality gate failed: {quality}.")
        outputs.append(features)
        season_quality[str(season)] = quality
        source_hashes[str(archive.resolve())] = sha256_file(archive)
        box_hashes[str(box_path.resolve())] = sha256_file(box_path)

    panel = pd.concat(outputs, ignore_index=True)
    duplicate_keys = int(panel.duplicated(["PLAYER_ID", "Season"]).sum())
    nonfinite_values = int((~np.isfinite(panel[["matchup_possessions", *MATCHUP_DEFENSE_FEATURES]])).sum().sum())
    if duplicate_keys or nonfinite_values:
        raise ValueError(
            f"Matchup feature panel failed: duplicate_keys={duplicate_keys}, "
            f"nonfinite_values={nonfinite_values}."
        )
    config = {
        "seasons": list(seasons),
        "defender_prior_possessions": defender_prior_possessions,
        "shooting_prior_attempts": shooting_prior_attempts,
        "source_hashes": source_hashes,
        "box_hashes": box_hashes,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"matchup_defense_features_v1_{identity}"
    output_dir = Path(artifact_root) / "features" / "matchup_defense" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    features_path = output_dir / "features.parquet"
    panel.to_parquet(features_path, index=False)
    run = {
        "run_id": run_id,
        "dataset": "annual_matchup_defense_features_v1",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": (
            "Defender points saved per 100 assigned matchup possessions versus each "
            "scorer's leave-one-defender-out scoring rate; season-centered and empirical-Bayes shrunk."
        ),
        "config": config,
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_keys": duplicate_keys,
            "nonfinite_values": nonfinite_values,
            "season_quality": season_quality,
        },
        "feature_names": list(MATCHUP_DEFENSE_FEATURES),
        "exposure_column": "matchup_possessions",
        "features_path": str(features_path.resolve()),
        "artifact_path": str(output_dir.resolve()),
        "caveat": (
            "Primary-defender assignment is observational. Opponent adjustment controls scorer "
            "quality, not teammate, scheme, help-defense, or full lineup context."
        ),
    }
    write_json_atomic(run, output_dir / "run.json")
    return run
