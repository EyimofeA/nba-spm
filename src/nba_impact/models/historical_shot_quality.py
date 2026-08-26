"""Historical row-level expected-shot prototype with defender and PBP context."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nba_impact.data.manifest import sha256_file, write_json_atomic


EXPERIMENT_ID = "historical_shot_quality_2015_v1"
PBP_COLUMNS = (
    "game_id", "period", "clock_display", "person_id", "actionType",
    "shotResult", "isFieldGoal", "assisted", "qualifier", "xLegacy", "yLegacy",
)
FEATURE_ARMS = {
    "location_only": ("x", "y", "shot_distance", "points_type"),
    "location_defender": (
        "x", "y", "shot_distance", "points_type", "nearest_defender_distance",
    ),
    "full_context": (
        "x", "y", "shot_distance", "points_type", "nearest_defender_distance",
        "shot_clock", "dribbles", "touch_time", "period", "home", "fast_break",
    ),
    "kobe_inspired_context": (
        "x", "y", "shot_distance", "points_type", "nearest_defender_distance",
        "shot_clock", "dribbles", "touch_time", "period", "clock_seconds",
        "home", "fast_break", "height_difference_inches",
    ),
}


def _clock_seconds(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(r"^(\d+):(\d+)$").astype(float)
    return 60.0 * parsed[0] + parsed[1]


def load_pbp_shot_context(paths: tuple[str | Path, ...]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=list(PBP_COLUMNS), low_memory=False)
        frame = frame.loc[
            frame["isFieldGoal"].astype(str).str.lower().isin({"true", "1"})
        ].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No field-goal rows were found in the PBP inputs.")
    output = pd.concat(frames, ignore_index=True)
    output["points_type"] = output["actionType"].map({"2pt": 2, "3pt": 3})
    output["shot_result"] = output["shotResult"].astype(str).str.lower()
    output["game_id"] = pd.to_numeric(output["game_id"], errors="coerce")
    output["period"] = pd.to_numeric(output["period"], errors="coerce")
    output["player_id"] = pd.to_numeric(output["person_id"], errors="coerce")
    output["clock_seconds"] = _clock_seconds(output["clock_display"])
    output["assisted"] = output["assisted"].astype(str).str.lower().isin({"true", "1"}).astype(float)
    output["fast_break"] = output["qualifier"].astype(str).str.lower().str.contains(
        "fastbreak|fast break", regex=True
    ).astype(float)
    output = output.rename(columns={"xLegacy": "x", "yLegacy": "y"})
    return output[
        [
            "game_id", "period", "clock_seconds", "player_id", "points_type",
            "shot_result", "assisted", "fast_break", "x", "y",
        ]
    ].dropna(subset=["game_id", "period", "clock_seconds", "player_id", "points_type"])


def attach_pbp_context(shots: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fuzzy-match tracking and PBP clocks within five seconds."""
    left = shots.copy()
    left["game_id"] = pd.to_numeric(left["GAME_ID"], errors="coerce")
    left["period"] = pd.to_numeric(left["PERIOD"], errors="coerce")
    left["player_id"] = pd.to_numeric(left["player_id"], errors="coerce")
    left["points_type"] = pd.to_numeric(left["PTS_TYPE"], errors="coerce")
    left["shot_result"] = left["SHOT_RESULT"].astype(str).str.lower()
    left["clock_seconds"] = _clock_seconds(left["GAME_CLOCK"])
    left = left.reset_index(names="source_row")

    games = sorted(
        set(left["game_id"].dropna().astype(int)) | set(pbp["game_id"].dropna().astype(int))
    )
    game_order = {game_id: index for index, game_id in enumerate(games)}
    for frame in (left, pbp):
        frame["match_time"] = (
            frame["game_id"].map(game_order) * 10000
            + frame["period"] * 1000
            + frame["clock_seconds"]
        ).astype(float)
        frame["player_id"] = frame["player_id"].astype("Int64")
        frame["points_type"] = frame["points_type"].astype("Int64")
    # Outcome must not help select the context row.
    by = ["player_id", "points_type"]
    valid_left = left.dropna(subset=["match_time", *by]).sort_values("match_time")
    valid_pbp = pbp.dropna(subset=["match_time", *by]).sort_values("match_time")
    matched = pd.merge_asof(
        valid_left,
        valid_pbp[["match_time", *by, "fast_break", "x", "y"]],
        on="match_time",
        by=by,
        direction="nearest",
        tolerance=5.0,
    )
    output = left.drop(columns=["assisted", "fast_break", "x", "y"], errors="ignore").merge(
        matched[["source_row", "fast_break", "x", "y"]],
        on="source_row",
        how="left",
        validate="one_to_one",
    )
    match_rate = float(output["fast_break"].notna().mean())
    if match_rate < 0.98:
        raise ValueError(f"PBP shot-context match rate is only {match_rate:.3%}.")
    return output, {"rows": int(len(output)), "pbp_match_rate": match_rate, "clock_tolerance_seconds": 5.0}


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=10.0,
        random_state=7,
    )


def attach_player_heights(
    shots: pd.DataFrame, player_sheet_path: str | Path
) -> tuple[pd.DataFrame, dict]:
    """Attach shooter-minus-nearest-defender height from one season's roster rows."""
    path = Path(player_sheet_path)
    source = (
        pd.read_parquet(path, columns=["PLAYER_ID", "PLAYER_HEIGHT_INCHES"])
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(
            path, usecols=["PLAYER_ID", "PLAYER_HEIGHT_INCHES"], low_memory=False
        )
    )
    source["PLAYER_ID"] = pd.to_numeric(source["PLAYER_ID"], errors="coerce")
    source["PLAYER_HEIGHT_INCHES"] = pd.to_numeric(
        source["PLAYER_HEIGHT_INCHES"], errors="coerce"
    )
    heights = source.dropna().groupby("PLAYER_ID")["PLAYER_HEIGHT_INCHES"].median()
    output = shots.copy()
    output["shooter_height_inches"] = pd.to_numeric(
        output["player_id"], errors="coerce"
    ).map(heights)
    output["defender_height_inches"] = pd.to_numeric(
        output["CLOSEST_DEFENDER_PLAYER_ID"], errors="coerce"
    ).map(heights)
    output["height_difference_inches"] = (
        output["shooter_height_inches"] - output["defender_height_inches"]
    )
    return output, {
        "height_source_players": int(heights.size),
        "shooter_height_coverage": float(output["shooter_height_inches"].notna().mean()),
        "defender_height_coverage": float(output["defender_height_inches"].notna().mean()),
        "height_difference_coverage": float(output["height_difference_inches"].notna().mean()),
    }


def run_historical_shot_quality(
    shot_log_path: str | Path,
    pbp_paths: tuple[str | Path, ...],
    *,
    player_sheet_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    shot_log_path = Path(shot_log_path)
    shots = pd.read_csv(shot_log_path, low_memory=False)
    required = {
        "GAME_ID", "MATCHUP", "LOCATION", "PERIOD", "GAME_CLOCK", "SHOT_CLOCK",
        "DRIBBLES", "TOUCH_TIME", "SHOT_DIST", "PTS_TYPE", "SHOT_RESULT",
        "CLOSEST_DEFENDER_PLAYER_ID", "CLOSE_DEF_DIST", "FGM", "player_id",
    }
    if missing := sorted(required - set(shots.columns)):
        raise ValueError(f"Shot log is missing {missing}.")
    pbp = load_pbp_shot_context(pbp_paths)
    panel, join_quality = attach_pbp_context(shots, pbp)
    panel, height_quality = attach_player_heights(panel, player_sheet_path)
    panel = panel.rename(
        columns={
            "SHOT_CLOCK": "shot_clock",
            "DRIBBLES": "dribbles",
            "TOUCH_TIME": "touch_time",
            "SHOT_DIST": "shot_distance",
            "CLOSE_DEF_DIST": "nearest_defender_distance",
        }
    )
    panel["home"] = panel["LOCATION"].eq("H").astype(float)
    panel["made"] = pd.to_numeric(panel["FGM"], errors="raise").astype(int)
    panel["shot_date"] = pd.to_datetime(
        panel["MATCHUP"].astype(str).str.extract(r"^([A-Z]{3} \d{2}, \d{4})")[0],
        format="%b %d, %Y",
        errors="raise",
    )
    dates = sorted(panel["shot_date"].unique())
    cutoff = dates[int(0.75 * len(dates))]
    train = panel["shot_date"].lt(cutoff)
    test = ~train
    if panel.loc[test, "GAME_ID"].isin(panel.loc[train, "GAME_ID"]).any():
        raise ValueError("Temporal shot split leaks a game across train and test.")

    metric_rows = []
    prediction = panel.loc[test, [
        "player_id", "CLOSEST_DEFENDER_PLAYER_ID", "points_type", "made"
    ]].copy()
    for arm, features in FEATURE_ARMS.items():
        model = _model().fit(panel.loc[train, list(features)], panel.loc[train, "made"])
        probability = model.predict_proba(panel.loc[test, list(features)])[:, 1]
        prediction[f"expected_make_{arm}"] = probability
        metric_rows.append(
            {
                "arm": arm,
                "train_shots": int(train.sum()),
                "test_shots": int(test.sum()),
                "log_loss": float(log_loss(panel.loc[test, "made"], probability)),
                "brier": float(brier_score_loss(panel.loc[test, "made"], probability)),
                "auc": float(roc_auc_score(panel.loc[test, "made"], probability)),
                "mean_prediction": float(probability.mean()),
                "actual_make_rate": float(panel.loc[test, "made"].mean()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    baseline_log_loss = float(metrics.loc[metrics["arm"].eq("location_only"), "log_loss"].iloc[0])
    metrics["log_loss_improvement_vs_location_only"] = baseline_log_loss - metrics["log_loss"]

    probability = prediction["expected_make_kobe_inspired_context"]
    prediction["actual_points"] = prediction["points_type"] * prediction["made"]
    prediction["expected_points"] = prediction["points_type"] * probability
    prediction["points_above_expected"] = prediction["actual_points"] - prediction["expected_points"]
    shotmaking = prediction.groupby("player_id", as_index=False).agg(
        shots=("made", "size"),
        actual_points=("actual_points", "sum"),
        expected_points=("expected_points", "sum"),
        points_above_expected=("points_above_expected", "sum"),
    )
    shotmaking["points_above_expected_per_100_shots"] = (
        100.0 * shotmaking["points_above_expected"] / shotmaking["shots"]
    )
    defender = prediction.groupby("CLOSEST_DEFENDER_PLAYER_ID", as_index=False).agg(
        shots=("made", "size"),
        actual_points=("actual_points", "sum"),
        expected_points=("expected_points", "sum"),
    )
    defender["closest_defender_points_saved_per_100_shots"] = (
        100.0 * (defender["expected_points"] - defender["actual_points"]) / defender["shots"]
    )

    pbp_hash = hashlib.sha256(
        "".join(sha256_file(path) for path in sorted(map(Path, pbp_paths))).encode()
    ).hexdigest()
    config = {
        "season": 2015,
        "temporal_split": {"train_before": str(pd.Timestamp(cutoff).date()), "test_on_or_after": str(pd.Timestamp(cutoff).date())},
        "feature_arms": {name: list(values) for name, values in FEATURE_ARMS.items()},
        "source_hashes": {
            "shot_log": sha256_file(shot_log_path),
            "player_sheet": sha256_file(player_sheet_path),
            "pbp_files_combined": pbp_hash,
        },
        "pbp_file_count": len(pbp_paths),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "research" / "historical_shot_quality" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    shotmaking.to_parquet(output / "shotmaking.parquet", index=False)
    defender.to_parquet(output / "closest_defender_residuals.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "historical_research_prototype",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            **join_quality,
            **height_quality,
            "train_games": int(panel.loc[train, "GAME_ID"].nunique()),
            "test_games": int(panel.loc[test, "GAME_ID"].nunique()),
        },
        "metrics": metrics.to_dict(orient="records"),
        "caveats": [
            "The public source covers only 2014-15 and has no declared data license; outputs are research-only.",
            "Closest defender is not necessarily the causally responsible defender.",
            "The test is one within-season temporal split, not a modern-season validation.",
            "Fast-break context is fuzzy-matched to play-by-play within five seconds.",
            "The PBP assisted flag is excluded because assists are recorded only after made shots; zero dribbles and short touch time are the pre-shot pass proxies.",
            "The KOBE-inspired arm adds shooter-minus-defender height and period clock, but it is a histogram GBM fit in one pooled shot model rather than Narsu's separate close-shot and long-shot logistic regressions.",
        ],
        "paths": {"metrics": "metrics.parquet", "shotmaking": "shotmaking.parquet", "closest_defender_residuals": "closest_defender_residuals.parquet"},
    }
    write_json_atomic(run, output / "run.json")
    return run
