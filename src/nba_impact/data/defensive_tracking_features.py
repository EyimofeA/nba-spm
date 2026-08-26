"""Canonical annual defensive tracking and hustle features."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

DEFENSIVE_TRACKING_FEATURES = (
    "dfg_attempts_p100",
    "dfg_diff_pct_eb",
    "dfg_two_point_equivalent_saved_p100",
    "rim_dfga_p100",
    "rim_diff_pct_eb",
    "rim_points_saved_p100_raw",
    "rim_points_saved_p100",
    "rim_matchup_attempt_share",
    "deflections_p100",
    "charges_drawn_p100",
    "contested_2pt_p100",
    "contested_3pt_p100",
    "contested_3pt_share",
    "def_loose_balls_recovered_p100",
)

HUSTLE_FEATURES = (
    "deflections_p100",
    "charges_drawn_p100",
    "contested_2pt_p100",
    "contested_3pt_p100",
    "contested_3pt_share",
    "def_loose_balls_recovered_p100",
)


def _name_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower().strip())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", text)


def _load_box(
    source_dir: str | Path,
    seasons: tuple[int, ...],
    *,
    source_overrides: dict[int, str | Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    outputs = []
    hashes = {}
    source_overrides = source_overrides or {}
    for season in seasons:
        path = Path(source_overrides.get(season, Path(source_dir) / f"{season}.csv"))
        frame = (
            pd.read_parquet(path)
            if path.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(path, low_memory=False)
        )
        required = {"PLAYER_ID", "PLAYER_NAME", "DefPoss"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Box source {path} is missing {missing}.")
        frame = frame[["PLAYER_ID", "PLAYER_NAME", "DefPoss"]].copy()
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce")
        frame["DefPoss"] = pd.to_numeric(frame["DefPoss"], errors="coerce")
        frame = frame.dropna(subset=["PLAYER_ID", "PLAYER_NAME"]).copy()
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frame["Season"] = season
        frame["_name_key"] = frame["PLAYER_NAME"].map(_name_key)
        frame = frame.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
            PLAYER_NAME=("PLAYER_NAME", "first"),
            _name_key=("_name_key", "first"),
            DefPoss=("DefPoss", "sum"),
        )
        outputs.append(frame)
        hashes[str(path.resolve())] = sha256_file(path)
    box = pd.concat(outputs, ignore_index=True)
    ambiguous = box.groupby(["Season", "_name_key"])["PLAYER_ID"].nunique()
    ambiguous = set(ambiguous.loc[ambiguous.gt(1)].index)
    box["_name_ambiguous"] = [
        (season, name) in ambiguous
        for season, name in zip(box["Season"], box["_name_key"], strict=True)
    ]
    return box, hashes


def _attach_player_id(
    source: pd.DataFrame,
    box: pd.DataFrame,
    *,
    name_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    frame = source.copy().rename(columns={"year": "Season"})
    name_column = next((column for column in name_columns if column in frame), None)
    if name_column is None:
        raise ValueError(f"Source has none of the player-name columns {name_columns}.")
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["_name_key"] = frame[name_column].map(_name_key)
    if "PLAYER_ID" not in frame:
        frame["PLAYER_ID"] = np.nan
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce")
    resolver = box.loc[~box["_name_ambiguous"], ["Season", "_name_key", "PLAYER_ID"]]
    resolver = resolver.drop_duplicates(["Season", "_name_key"])
    frame = frame.merge(
        resolver.rename(columns={"PLAYER_ID": "_resolved_player_id"}),
        on=["Season", "_name_key"], how="left", validate="many_to_one",
    )
    frame["PLAYER_ID"] = frame["PLAYER_ID"].fillna(frame["_resolved_player_id"])
    valid = frame["PLAYER_ID"].notna()
    quality = {
        "source_rows": len(frame),
        "matched_rows": int(valid.sum()),
        "match_rate": float(valid.mean()) if len(frame) else 0.0,
        "unmatched_names": sorted(frame.loc[~valid, name_column].dropna().astype(str).unique()),
    }
    frame = frame.loc[valid].copy()
    frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
    frame["Season"] = frame["Season"].astype(int)
    return frame, quality


def _coalesce_numeric(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in aliases:
        if column in frame:
            output = output.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return output


def compute_defensive_tracking_features(
    box: pd.DataFrame,
    dfg: pd.DataFrame,
    rim_dfg: pd.DataFrame,
    hustle: pd.DataFrame,
    *,
    dfg_prior_attempts: float = 200.0,
    rim_prior_attempts: float = 100.0,
) -> tuple[pd.DataFrame, dict]:
    """Return ten predeclared, positive-good where applicable, annual features."""
    dfg, dfg_quality = _attach_player_id(dfg, box, name_columns=("PLAYER", "PLAYER_NAME"))
    rim, rim_quality = _attach_player_id(
        rim_dfg, box, name_columns=("PLAYER", "PLAYER_NAME")
    )
    hustle, hustle_quality = _attach_player_id(
        hustle, box, name_columns=("PLAYER_NAME", "Player", "PLAYER")
    )
    key = box[["PLAYER_ID", "Season", "DefPoss"]].copy()

    def defense_shooting(frame: pd.DataFrame, prefix: str, prior: float) -> pd.DataFrame:
        work = frame.copy()
        work["_dfga"] = _coalesce_numeric(work, ("DFGA",))
        work["_diff"] = _coalesce_numeric(work, ("DIFF%", "Diff%"))
        work = work.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
            _dfga=("_dfga", "sum"), _diff=("_diff", "mean")
        )
        work = work.merge(key, on=["PLAYER_ID", "Season"], validate="one_to_one")
        reliability = work["_dfga"].clip(lower=0) / (work["_dfga"].clip(lower=0) + prior)
        work[f"{prefix}_dfga_p100"] = 100.0 * work["_dfga"] / work["DefPoss"].where(work["DefPoss"].gt(0))
        work[f"{prefix}_diff_pct_raw"] = work["_diff"]
        work[f"{prefix}_diff_pct_eb"] = reliability * work["_diff"]
        return work[[
            "PLAYER_ID", "Season", f"{prefix}_dfga_p100",
            f"{prefix}_diff_pct_raw", f"{prefix}_diff_pct_eb",
        ]]

    overall = defense_shooting(dfg, "dfg", dfg_prior_attempts).rename(
        columns={"dfg_dfga_p100": "dfg_attempts_p100"}
    )
    rim_features = defense_shooting(rim, "rim", rim_prior_attempts)
    rim_features["rim_points_saved_p100_raw"] = (
        -2.0 * rim_features["rim_dfga_p100"] * rim_features["rim_diff_pct_raw"] / 100.0
    )
    rim_features["rim_points_saved_p100"] = (
        -2.0 * rim_features["rim_dfga_p100"] * rim_features["rim_diff_pct_eb"] / 100.0
    )

    hustle = hustle.copy()
    hustle["deflections"] = _coalesce_numeric(hustle, ("DEFLECTIONS", "Deflections"))
    hustle["charges"] = _coalesce_numeric(hustle, ("CHARGES_DRAWN", "ChargesDrawn"))
    hustle["contested_2pt"] = _coalesce_numeric(
        hustle, ("CONTESTED_SHOTS_2PT", "Contested2PT Shots", "Contested2PT\u00a0Shots")
    )
    hustle["contested_3pt"] = _coalesce_numeric(
        hustle, ("CONTESTED_SHOTS_3PT", "Contested3PT Shots", "Contested3PT\u00a0Shots")
    )
    hustle["def_loose_balls"] = _coalesce_numeric(
        hustle, ("DEF_LOOSE_BALLS_RECOVERED", "DEF Loose BallsRecovered")
    )
    hustle = hustle.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
        deflections=("deflections", "sum"), charges=("charges", "sum"),
        contested_2pt=("contested_2pt", "sum"), contested_3pt=("contested_3pt", "sum"),
        def_loose_balls=("def_loose_balls", "sum"),
    ).merge(key, on=["PLAYER_ID", "Season"], validate="one_to_one")
    hustle["contested_3pt_share"] = hustle["contested_3pt"] / (
        hustle["contested_2pt"] + hustle["contested_3pt"]
    ).where((hustle["contested_2pt"] + hustle["contested_3pt"]).gt(0))
    for source, destination in (
        ("deflections", "deflections_p100"), ("charges", "charges_drawn_p100"),
        ("contested_2pt", "contested_2pt_p100"),
        ("contested_3pt", "contested_3pt_p100"),
        ("def_loose_balls", "def_loose_balls_recovered_p100"),
    ):
        hustle[destination] = 100.0 * hustle[source] / hustle["DefPoss"].where(hustle["DefPoss"].gt(0))

    output = key[["PLAYER_ID", "Season"]].merge(
        overall, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    ).merge(
        rim_features, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    ).merge(
        hustle[["PLAYER_ID", "Season", *HUSTLE_FEATURES]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    )
    output["dfg_two_point_equivalent_saved_p100"] = (
        -2.0 * output["dfg_attempts_p100"] * output["dfg_diff_pct_eb"] / 100.0
    )
    output = output.drop(columns=["dfg_diff_pct_raw", "rim_diff_pct_raw"])
    output["rim_matchup_attempt_share"] = (
        output["rim_dfga_p100"] / output["dfg_attempts_p100"].where(
            output["dfg_attempts_p100"].gt(0)
        )
    ).where(output["rim_dfga_p100"].le(output["dfg_attempts_p100"]))
    missing_before = {name: float(output[name].isna().mean()) for name in DEFENSIVE_TRACKING_FEATURES}
    for feature in DEFENSIVE_TRACKING_FEATURES:
        # A player-season can borrow the center of its own season, but it must
        # never borrow another season's distribution.  When a source family is
        # unavailable for an entire season, zero is the explicit neutral value.
        season_median = output.groupby("Season")[feature].transform("median")
        output[feature] = output[feature].fillna(season_median).fillna(0.0)
    quality = {
        "source_join_quality": {"dfg": dfg_quality, "rim_dfg": rim_quality, "hustle": hustle_quality},
        "missing_fraction_before_neutral_fill": missing_before,
        "neutral_fill_policy": "same_season_median_then_zero",
    }
    return output, quality


def build_defensive_tracking_features(
    dfg_source: str | Path,
    rim_dfg_source: str | Path,
    hustle_source: str | Path,
    box_source_dir: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2025)),
    box_source_overrides: dict[int, str | Path] | None = None,
) -> dict:
    box, box_hashes = _load_box(
        box_source_dir, seasons, source_overrides=box_source_overrides
    )
    dfg = pd.read_csv(dfg_source, low_memory=False)
    rim = pd.read_csv(rim_dfg_source, low_memory=False)
    hustle = pd.read_csv(hustle_source, low_memory=False)
    dfg = dfg.loc[pd.to_numeric(dfg["year"], errors="coerce").isin(seasons)]
    rim = rim.loc[pd.to_numeric(rim["year"], errors="coerce").isin(seasons)]
    hustle = hustle.loc[pd.to_numeric(hustle["year"], errors="coerce").isin(seasons)]
    features, audit = compute_defensive_tracking_features(box, dfg, rim, hustle)
    if features.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Defensive tracking feature keys are not unique.")
    config = {
        "seasons": list(seasons), "dfg_prior_attempts": 200.0, "rim_prior_attempts": 100.0,
        "box_source_overrides": {
            str(season): str(Path(path).resolve())
            for season, path in sorted((box_source_overrides or {}).items())
        },
        "source_hashes": {"dfg": sha256_file(dfg_source), "rim_dfg": sha256_file(rim_dfg_source),
                          "hustle": sha256_file(hustle_source), "box": box_hashes},
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"defensive_tracking_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "defensive_tracking" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "features.parquet"
    features.to_parquet(path, index=False)
    run = {
        "run_id": run_id, "dataset": "annual_defensive_tracking_features_v1",
        "status": "validated", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {"rows": len(features), "players": int(features["PLAYER_ID"].nunique()),
                    "duplicate_keys": 0, "nonfinite_values": int((~np.isfinite(features[list(DEFENSIVE_TRACKING_FEATURES)])).sum().sum()), **audit},
        "feature_names": list(DEFENSIVE_TRACKING_FEATURES),
        "features_path": str(path.resolve()), "artifact_path": str(output.resolve()),
    }
    if run["quality"]["nonfinite_values"]:
        raise ValueError("Defensive tracking features contain non-finite values.")
    write_json_atomic(run, output / "run.json")
    return run
