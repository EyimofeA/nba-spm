"""Audit the versioned 1997-2026 CourtSignal data contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic


_LEGACY_PLAYERS = tuple(
    [f"a{index}" for index in range(1, 6)] + [f"h{index}" for index in range(1, 6)]
)
_CURRENT_PLAYERS = tuple(
    [f"away_player_{index}" for index in range(1, 6)]
    + [f"home_player_{index}" for index in range(1, 6)]
)


def _season_range(value: str) -> tuple[int, ...]:
    start, end = (int(item) for item in value.split("-", maxsplit=1))
    return tuple(range(start, end + 1))


def _valid_lineup(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return frame.loc[:, columns].notna().all(axis=1) & frame.loc[:, columns].nunique(
        axis=1
    ).eq(10)


def _normalize_game_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def _game_reconciliation(
    official: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    season: int,
    source_kind: str,
) -> pd.DataFrame:
    joined = official.merge(observed, on="game_id", how="left", validate="one_to_one")
    for side in ("home", "away"):
        joined[f"{side}_point_deficit"] = (
            joined[f"{side}_score"] - joined[f"{side}_points"]
        )
        native = f"native_{side}_points"
        if native not in joined:
            joined[native] = np.nan
        legacy_excluded = joined[f"{side}_point_deficit"].where(
            joined[f"{side}_point_deficit"].between(0, 8), np.nan
        )
        joined[f"excluded_{side}_points"] = (
            joined[native] - joined[f"{side}_points"]
        ).where(joined[native].notna(), legacy_excluded)
        joined[f"accounted_{side}_points"] = (
            joined[f"{side}_points"] + joined[f"excluded_{side}_points"]
        )
    joined["matched"] = joined["home_points"].notna() & joined["away_points"].notna()
    joined["model_score_reconciled"] = (
        joined["matched"]
        & joined["home_point_deficit"].eq(0)
        & joined["away_point_deficit"].eq(0)
    )
    joined["score_reconciled"] = (
        joined["matched"]
        & joined["accounted_home_points"].eq(joined["home_score"])
        & joined["accounted_away_points"].eq(joined["away_score"])
    ).fillna(False).astype(bool)
    native_missing = (
        joined["native_home_points"].isna() | joined["native_away_points"].isna()
    )
    joined["repair_required"] = (
        ~joined["score_reconciled"]
        | joined["max_period"].fillna(0).lt(4)
    ).fillna(True).astype(bool)
    joined["repair_reason"] = np.select(
        [
            ~joined["matched"],
            joined["max_period"].fillna(0).lt(4),
            native_missing
            & (
                joined["home_point_deficit"].fillna(np.inf).lt(0)
                | joined["away_point_deficit"].fillna(np.inf).lt(0)
            ),
            joined["home_point_deficit"].fillna(np.inf).abs().gt(8)
            | joined["away_point_deficit"].fillna(np.inf).abs().gt(8),
            ~joined["score_reconciled"].fillna(False).astype(bool),
        ],
        [
            "missing_game",
            "truncated_before_fourth_period",
            "observed_points_exceed_official_score",
            "large_score_deficit",
            "unreconciled_score",
        ],
        default="none",
    )
    joined.insert(0, "season", season)
    joined.insert(1, "source_kind", source_kind)
    return joined


def _legacy_season(
    path: Path, scores: pd.DataFrame, season: int, *, source_label: str
) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["game_id"] = _normalize_game_id(frame["gameid"])
    valid = _valid_lineup(frame, _LEGACY_PLAYERS)
    game = (
        frame.assign(
            home_points=np.where(frame["home_poss"], frame["pts"], 0.0),
            away_points=np.where(frame["home_poss"], 0.0, frame["pts"]),
        )
        .groupby("game_id", as_index=False)
        .agg(
            home_points=("home_points", "sum"),
            away_points=("away_points", "sum"),
            max_period=("period", "max"),
            possession_rows=("pts", "size"),
        )
    )
    official = scores.loc[
        scores["project_season"].eq(season) & scores["season_type"].eq("regular"),
        ["game_id", "home_score", "away_score"],
    ].copy()
    official["game_id"] = _normalize_game_id(official["game_id"])
    joined = _game_reconciliation(
        official, game, season=season, source_kind="legacy_terminal_possession"
    )
    row = {
        "season": season,
        "possession_source": source_label,
        "possession_rows": len(frame),
        "possession_games": int(frame["game_id"].nunique()),
        "official_games": len(official),
        "matched_games": int(joined["matched"].sum()),
        "score_reconciled_games": int(joined["score_reconciled"].sum()),
        "score_reconciliation_rate": float(joined["score_reconciled"].mean())
        if len(joined)
        else 0.0,
        "repair_required_games": int(joined["repair_required"].sum()),
        "valid_lineup_fraction": float(valid.mean()),
        "event_rows": 0,
        "event_games": 0,
        "event_scope": "terminal_possession_only",
    }
    return row, joined


def _current_seasons(
    possessions_path: Path,
    lineups_path: Path,
    events_path: Path,
    scores: pd.DataFrame,
    seasons: tuple[int, ...],
) -> tuple[list[dict], pd.DataFrame]:
    possessions = pd.read_parquet(possessions_path)
    possessions = possessions.loc[possessions["season_type"].eq("regular")].copy()
    possessions["game_id"] = _normalize_game_id(possessions["game_id"])
    segments = pd.read_parquet(lineups_path)
    segments["game_id"] = _normalize_game_id(segments["game_id"])
    segment_seasons = segments.merge(
        possessions[["possession_id", "season_end"]].drop_duplicates("possession_id"),
        on="possession_id",
        how="left",
        validate="many_to_one",
    )
    segment_seasons["valid_lineup"] = _valid_lineup(segment_seasons, _CURRENT_PLAYERS)
    events = pd.read_parquet(
        events_path, columns=["game_id", "season_end", "season_type", "event_id"]
    )
    events = events.loc[events["season_type"].eq("regular")].copy()
    events["game_id"] = _normalize_game_id(events["game_id"])
    rows = []
    reconciliation = []
    for season in seasons:
        source = possessions.loc[possessions["season_end"].eq(season)].copy()
        game = (
            source.assign(
                scored_home=source["points"].where(source["offense_is_home"], 0.0),
                scored_away=source["points"].where(~source["offense_is_home"], 0.0),
            )
            .groupby("game_id", as_index=False)
            .agg(
                home_points=("scored_home", "sum"),
                away_points=("scored_away", "sum"),
                native_home_points=("home_points", "sum"),
                native_away_points=("away_points", "sum"),
                max_period=("period", "max"),
                possession_rows=("points", "size"),
            )
        )
        official = scores.loc[
            scores["project_season"].eq(season) & scores["season_type"].eq("regular"),
            ["game_id", "home_score", "away_score"],
        ].copy()
        official["game_id"] = _normalize_game_id(official["game_id"])
        joined = _game_reconciliation(
            official, game, season=season, source_kind="current_event_possession"
        )
        season_segments = segment_seasons.loc[segment_seasons["season_end"].eq(season)]
        season_events = events.loc[events["season_end"].eq(season)]
        rows.append(
            {
                "season": season,
                "possession_source": "data/lake/silver/possessions.parquet",
                "possession_rows": len(source),
                "possession_games": int(source["game_id"].nunique()),
                "official_games": len(official),
                "matched_games": int(joined["matched"].sum()),
                "score_reconciled_games": int(joined["score_reconciled"].sum()),
                "score_reconciliation_rate": float(joined["score_reconciled"].mean())
                if len(joined)
                else 0.0,
                "repair_required_games": int(joined["repair_required"].sum()),
                "valid_lineup_fraction": float(season_segments["valid_lineup"].mean()),
                "event_rows": len(season_events),
                "event_games": int(season_events["game_id"].nunique()),
                "event_scope": "observed_event_state",
            }
        )
        reconciliation.append(joined)
    return rows, pd.concat(reconciliation, ignore_index=True)


def build_canonical_impact_contract(
    config_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Write the data coverage ledger and fail closed on publication gates."""
    config_path = Path(config_path)
    root = config_path.resolve().parents[2]
    config = yaml.safe_load(config_path.read_text())
    seasons = _season_range(config["project_seasons"])
    source = {name: root / value for name, value in config["sources"].items()}
    required = (
        source["official_scores"],
        source["current_possessions"],
        source["current_lineups"],
        source["current_events"],
        source["annual_player_features"],
    )
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError(f"Canonical contract sources are missing: {missing}")
    scores = pd.read_parquet(source["official_scores"])
    scores["game_id"] = _normalize_game_id(scores["game_id"])
    rows = []
    reconciliation = []
    current_start = int(config["transitions"]["current_event_start"])
    legacy_pattern = config["sources"]["legacy_terminal_possessions"]
    for season in seasons:
        if season >= current_start:
            continue
        path = root / legacy_pattern.format(season=season)
        if not path.exists():
            raise FileNotFoundError(path)
        row, game_reconciliation = _legacy_season(
            path,
            scores,
            season,
            source_label=legacy_pattern.format(season=season),
        )
        rows.append(row)
        reconciliation.append(game_reconciliation)
    current_rows, current_reconciliation = _current_seasons(
        source["current_possessions"],
        source["current_lineups"],
        source["current_events"],
        scores,
        tuple(season for season in seasons if season >= current_start),
    )
    rows.extend(current_rows)
    reconciliation.append(current_reconciliation)
    coverage = pd.DataFrame(rows).sort_values("season")
    game_reconciliation = pd.concat(reconciliation, ignore_index=True)
    features = pd.read_parquet(
        source["annual_player_features"], columns=["PLAYER_ID", "Window_End"]
    )
    feature_seasons = set(features["Window_End"].astype(int))
    duplicate_features = int(features.duplicated(["PLAYER_ID", "Window_End"]).sum())
    gates = config["gates"]
    coverage["score_gate_passed"] = coverage["score_reconciliation_rate"].ge(
        float(gates["minimum_game_score_reconciliation"])
    )
    coverage["lineup_gate_passed"] = coverage["valid_lineup_fraction"].ge(
        float(gates["minimum_valid_lineup_fraction"])
    )
    coverage["player_feature_coverage"] = coverage["season"].isin(feature_seasons)
    passed = bool(
        coverage["score_gate_passed"].all()
        and coverage["lineup_gate_passed"].all()
        and coverage["player_feature_coverage"].all()
        and duplicate_features == 0
    )
    hashes = {
        "config": sha256_file(config_path),
        **{
            name: sha256_file(path)
            for name, path in source.items()
            if path.exists() and path.is_file()
        },
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output = Path(artifact_root) / f"{config['contract_id']}_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(output / "season_coverage.parquet", index=False)
    game_reconciliation.to_parquet(output / "game_reconciliation.parquet", index=False)
    repair_queue = game_reconciliation.loc[
        game_reconciliation["repair_required"]
    ].copy()
    repair_queue.to_parquet(output / "repair_queue.parquet", index=False)
    run = {
        "run_id": output.name,
        "contract_id": config["contract_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "source_hashes": hashes,
        "quality": {
            "seasons": len(coverage),
            "minimum_score_reconciliation": float(
                coverage["score_reconciliation_rate"].min()
            ),
            "minimum_valid_lineup_fraction": float(
                coverage["valid_lineup_fraction"].min()
            ),
            "duplicate_player_season_keys": duplicate_features,
            "repair_required_games": len(repair_queue),
            "detailed_event_start": int(
                config["transitions"]["detailed_factor_event_start"]
            ),
        },
        "technical_free_throw_policy": config["technical_free_throws"],
        "forbidden_interpretation": config["forbidden_interpretation"],
        "relative_paths": {
            "season_coverage": "season_coverage.parquet",
            "game_reconciliation": "game_reconciliation.parquet",
            "repair_queue": "repair_queue.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    if not passed:
        raise ValueError(f"Canonical data contract failed; inspect {output}")
    return run
