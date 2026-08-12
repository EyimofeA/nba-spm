"""Reproducible annual Synergy playtype features, including project zTS."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

FT_FACTOR = 0.44


def _load_playtypes(path: str | Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={"year": "Season"})
    required = {"PLAYER_ID", "Season", "playtype", "Poss", "Points", "FGA"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Playtype source is missing {missing}.")
    old_ft = pd.to_numeric(frame.get("FTFreq%"), errors="coerce")
    new_ft = pd.to_numeric(frame.get("%FT"), errors="coerce") / 100.0
    frame["ft_frequency"] = old_ft.fillna(new_ft)
    for column in ("PLAYER_ID", "Season", "Poss", "Points", "FGA", "ft_frequency"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["Season"].isin(seasons)].dropna(
        subset=["PLAYER_ID", "Season", "playtype", "Poss", "Points"]
    )
    frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
    frame["Season"] = frame["Season"].astype(int)
    frame["playtype"] = frame["playtype"].astype(str).str.strip().str.lower()
    frame["fta_estimate"] = frame["ft_frequency"].fillna(0) * frame["Poss"] * 2.0
    return frame


def _load_box(source_dir: str | Path, seasons: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    hashes = {}
    for season in seasons:
        path = Path(source_dir) / f"{season}.csv"
        frame = pd.read_csv(path, low_memory=False)
        required = {"PLAYER_ID", "PTS", "FGA", "FTA"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Box source {path} is missing {missing}.")
        minutes = "Minutes" if "Minutes" in frame else "MIN"
        keep = frame[["PLAYER_ID", "PTS", "FGA", "FTA", minutes]].copy()
        keep = keep.rename(columns={minutes: "Minutes"})
        keep["Season"] = season
        for column in ("PLAYER_ID", "PTS", "FGA", "FTA", "Minutes"):
            keep[column] = pd.to_numeric(keep[column], errors="coerce")
        keep = keep.dropna(subset=["PLAYER_ID", "PTS", "FGA", "FTA"])
        keep["PLAYER_ID"] = keep["PLAYER_ID"].astype(int)
        rows.append(
            keep.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
                PTS=("PTS", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"),
                Minutes=("Minutes", "sum"),
            )
        )
        hashes[str(path.resolve())] = sha256_file(path)
    return pd.concat(rows, ignore_index=True), hashes


def compute_playtype_features(
    box: pd.DataFrame,
    playtypes: pd.DataFrame,
    *,
    minimum_minutes: float = 250.0,
    minimum_player_playtype_possessions: float = 50.0,
    minimum_league_row_possessions: float = 20.0,
) -> pd.DataFrame:
    """Compute zTS and points-over-expectation at player-season grain.

    zTS is in percentage points. POE rates are points per 75 Synergy possessions.
    Transition POE contribution uses all Synergy possessions as its denominator;
    transition share is supplied separately so role and efficiency remain inspectable.
    """
    box = box.copy()
    playtypes = playtypes.copy()
    box_denominator = 2.0 * (box["FGA"] + FT_FACTOR * box["FTA"])
    box["player_ts_pct"] = np.where(
        box_denominator > 0, 100.0 * box["PTS"] / box_denominator, np.nan
    )
    league_box = box.groupby("Season", as_index=False).agg(
        PTS=("PTS", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum")
    )
    league_box["league_ts_pct"] = 100.0 * league_box["PTS"] / (
        2.0 * (league_box["FGA"] + FT_FACTOR * league_box["FTA"])
    )

    league_rows = playtypes.loc[
        playtypes["Poss"].ge(minimum_league_row_possessions)
    ].copy()
    league = league_rows.groupby(["Season", "playtype"], as_index=False).agg(
        Points=("Points", "sum"), Poss=("Poss", "sum"), FGA=("FGA", "sum"),
        fta_estimate=("fta_estimate", "sum"),
    )
    league["league_playtype_ppp"] = league["Points"] / league["Poss"].where(league["Poss"].gt(0))
    league["league_playtype_ts_pct"] = 100.0 * league["Points"] / (
        2.0 * (league["FGA"] + FT_FACTOR * league["fta_estimate"])
    ).where(lambda value: value.gt(0))
    detail = playtypes.merge(
        league[["Season", "playtype", "league_playtype_ppp", "league_playtype_ts_pct"]],
        on=["Season", "playtype"], how="left", validate="many_to_one",
    )
    totals = detail.groupby(["PLAYER_ID", "Season"])["Poss"].transform("sum")
    detail["playtype_share"] = detail["Poss"] / totals.where(totals.gt(0))
    detail["expected_ts_contribution"] = (
        detail["playtype_share"] * detail["league_playtype_ts_pct"]
    )
    detail["poe"] = detail["Points"] - detail["Poss"] * detail["league_playtype_ppp"]
    detail["transition_poss"] = detail["Poss"].where(detail["playtype"].eq("tran"), 0.0)
    detail["transition_poe"] = detail["poe"].where(detail["playtype"].eq("tran"), 0.0)
    player = detail.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
        synergy_possessions=("Poss", "sum"),
        playtype_expected_ts_pct=("expected_ts_contribution", "sum"),
        playtype_poe=("poe", "sum"),
        transition_possessions=("transition_poss", "sum"),
        transition_poe=("transition_poe", "sum"),
    )
    player["playtype_poe_per_75"] = 75.0 * player["playtype_poe"] / player["synergy_possessions"]
    player["transition_share"] = player["transition_possessions"] / player["synergy_possessions"]
    player["transition_poe_per_75"] = 75.0 * player["transition_poe"] / player["synergy_possessions"]
    result = box.merge(player, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one")
    result = result.merge(
        league_box[["Season", "league_ts_pct"]], on="Season", validate="many_to_one"
    )
    result["relative_ts_pct_points"] = result["player_ts_pct"] - result["league_ts_pct"]
    result["playtype_difficulty_pct_points"] = (
        result["league_ts_pct"] - result["playtype_expected_ts_pct"]
    )
    result["zts_pct_points"] = result["player_ts_pct"] - result["playtype_expected_ts_pct"]
    result = result.loc[
        result["Minutes"].ge(minimum_minutes)
        & result["synergy_possessions"].ge(minimum_player_playtype_possessions)
    ].copy()
    columns = [
        "PLAYER_ID", "Season", "player_ts_pct", "league_ts_pct",
        "relative_ts_pct_points", "playtype_expected_ts_pct",
        "playtype_difficulty_pct_points", "zts_pct_points",
        "playtype_poe_per_75", "transition_share", "transition_poe_per_75",
        "synergy_possessions",
    ]
    return result[columns].sort_values(["Season", "PLAYER_ID"]).reset_index(drop=True)


def build_playtype_features(
    playtype_source: str | Path,
    box_source_dir: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2025)),
    minimum_minutes: float = 250.0,
    minimum_player_playtype_possessions: float = 50.0,
    minimum_league_row_possessions: float = 20.0,
) -> dict:
    playtypes = _load_playtypes(playtype_source, seasons)
    box, box_hashes = _load_box(box_source_dir, seasons)
    features = compute_playtype_features(
        box, playtypes, minimum_minutes=minimum_minutes,
        minimum_player_playtype_possessions=minimum_player_playtype_possessions,
        minimum_league_row_possessions=minimum_league_row_possessions,
    )
    if features.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Playtype features have duplicate player-season keys.")
    config = {
        "seasons": list(seasons), "minimum_minutes": minimum_minutes,
        "minimum_player_playtype_possessions": minimum_player_playtype_possessions,
        "minimum_league_row_possessions": minimum_league_row_possessions,
        "playtype_source_sha256": sha256_file(playtype_source),
        "box_source_hashes": box_hashes, "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"playtype_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "playtype_impact" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "features.parquet"
    features.to_parquet(path, index=False)
    run = {
        "run_id": run_id, "dataset": "annual_playtype_impact_features_v1",
        "status": "validated", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {"rows": len(features), "players": int(features["PLAYER_ID"].nunique()),
                    "duplicate_keys": 0, "nonfinite_values": int((~np.isfinite(features.select_dtypes("number"))).sum().sum())},
        "definitions": {
            "zts_pct_points": "player TS% minus expected TS% from the player's playtype mix",
            "playtype_poe_per_75": "points above season-playtype league PPP per 75 Synergy possessions",
            "transition_poe_per_75": "transition POE contribution per 75 total Synergy possessions",
        },
        "features_path": str(path.resolve()), "artifact_path": str(output.resolve()),
    }
    if run["quality"]["nonfinite_values"]:
        raise ValueError("Playtype features contain non-finite values.")
    write_json_atomic(run, output / "run.json")
    return run
