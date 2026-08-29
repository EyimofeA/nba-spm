"""Player-neutral expected-shot baseline for the shot-quality research project.

This is intentionally a shot-outcome model, not a player-impact model.  It
does not read shooter, defender, team, lineup, or post-shot fields, so its
residuals describe shotmaking above the opportunity represented by location and
pre-shot game context.  It cannot assign a shot to a primary defender.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic


MODEL_VERSION = "expected_shot_quality_v1"
ZONE_ORDER = ("rim", "short_mid", "long_mid", "corner_3", "above_break_3")
FORBIDDEN_FEATURES = {
    "shooter_id", "offense_team_id", "defense_team_id", "offense_player_1",
    "offense_player_2", "offense_player_3", "offense_player_4", "offense_player_5",
    "defense_player_1", "defense_player_2", "defense_player_3", "defense_player_4",
    "defense_player_5", "shot_made", "shot_value", "game_id", "actionId",
    "actionNumber", "orderNumber",
}


def _feature_frame(
    panel: pd.DataFrame, *, include_possession_context: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Make a fixed, player-neutral pre-shot design matrix.

    Smooth location terms allow expected make probability to vary within the
    NBA's discrete shot zones.  Period, clock, score state, and home status are
    all known at release.  No identity feature is permitted.
    """
    required = {
        "shot_zone", "location_x", "location_y", "shot_distance_feet", "period",
        "regulation_seconds_remaining", "offense_score_diff_before", "offense_is_home",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Shot panel lacks expected-shot features: {missing}.")

    x = pd.to_numeric(panel["location_x"], errors="raise").to_numpy(dtype=float) / 250.0
    y = pd.to_numeric(panel["location_y"], errors="raise").to_numpy(dtype=float) / 420.0
    distance = np.clip(
        pd.to_numeric(panel["shot_distance_feet"], errors="raise").to_numpy(dtype=float),
        0.0,
        40.0,
    ) / 40.0
    angle = np.arctan2(x, y)
    clock = np.clip(
        pd.to_numeric(panel["regulation_seconds_remaining"], errors="raise").to_numpy(dtype=float),
        0.0,
        2880.0,
    ) / 2880.0
    score = np.clip(
        pd.to_numeric(panel["offense_score_diff_before"], errors="raise").to_numpy(dtype=float),
        -25.0,
        25.0,
    ) / 25.0
    home = panel["offense_is_home"].astype(float).to_numpy()
    is_three = pd.to_numeric(panel["shot_value"], errors="raise").eq(3).astype(float).to_numpy()
    period = pd.to_numeric(panel["period"], errors="raise").clip(upper=5).astype(int)

    numeric = np.column_stack(
        [
            x, y, distance, x * x, y * y, distance * distance, x * y,
            np.sin(angle), np.cos(angle), clock, score, home, is_three,
        ]
    )
    names = [
        "location_x", "location_y", "distance", "location_x_sq", "location_y_sq",
        "distance_sq", "location_xy", "angle_sin", "angle_cos", "clock",
        "offense_score_diff", "offense_is_home", "is_three",
    ]
    zone = panel["shot_zone"].astype(str)
    zone_columns = [(zone.eq(value)).astype(float).to_numpy() for value in ZONE_ORDER]
    period_columns = [(period.eq(value)).astype(float).to_numpy() for value in range(1, 6)]
    columns = [numeric, *zone_columns, *period_columns]
    names += [
        *(f"zone_{value}" for value in ZONE_ORDER),
        *(f"period_{value}" for value in range(1, 6)),
    ]
    if include_possession_context:
        context = {
            "seconds_since_possession_start",
            "is_transition",
            "is_putback",
            "is_second_chance",
            "is_from_turnover",
            "shot_finish",
        }
        if missing := sorted(context - set(panel.columns)):
            raise ValueError(f"Shot panel lacks possession context: {missing}.")
        seconds = np.clip(
            pd.to_numeric(
                panel["seconds_since_possession_start"], errors="raise"
            ).to_numpy(dtype=float),
            0.0,
            30.0,
        ) / 30.0
        flags = panel[
            [
                "is_transition",
                "is_putback",
                "is_second_chance",
                "is_from_turnover",
            ]
        ].astype(float).to_numpy()
        finish = panel["shot_finish"].astype(str)
        finish_names = (
            "transition",
            "putback",
            "cut",
            "drive",
            "pullup",
            "post",
            "spotup",
            "other",
        )
        columns.extend(
            [
                seconds,
                flags,
                *(finish.eq(value).astype(float).to_numpy() for value in finish_names),
            ]
        )
        names.extend(
            [
                "seconds_since_possession_start",
                "is_transition",
                "is_putback",
                "is_second_chance",
                "is_from_turnover",
                *(f"finish_{value}" for value in finish_names),
            ]
        )
    return np.column_stack(columns), names


def _calibration(prediction: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    bucket = pd.qcut(prediction, q=10, duplicates="drop")
    frame = pd.DataFrame({"prediction": prediction, "outcome": outcome, "bucket": bucket})
    return (
        frame.groupby("bucket", observed=True)
        .agg(shots=("outcome", "size"), mean_prediction=("prediction", "mean"), make_rate=("outcome", "mean"))
        .reset_index(drop=True)
    )


def _metric_row(frame: pd.DataFrame, prediction: np.ndarray, label: str) -> dict:
    outcome = frame["shot_made"].to_numpy(dtype=int)
    return {
        "split": label,
        "shots": int(len(frame)),
        "make_rate": float(outcome.mean()),
        "mean_prediction": float(prediction.mean()),
        "brier": float(brier_score_loss(outcome, prediction)),
        "log_loss": float(log_loss(outcome, prediction, labels=[0, 1])),
    }


def _player_aggregates(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    output = frame[["shooter_id", "season_end", "shot_zone", "shot_value", "shot_made"]].copy()
    output["shot_class"] = np.where(output["shot_zone"].eq("rim"), "rim", "non_rim")
    output["expected_points"] = output["shot_value"].to_numpy(dtype=float) * prediction
    output["actual_points"] = output["shot_value"].to_numpy(dtype=float) * output["shot_made"].to_numpy(dtype=float)
    pieces = []
    for grouping in (("shooter_id", "season_end", "shot_class"), ("shooter_id", "season_end")):
        grouped = (
            output.groupby(list(grouping), as_index=False)
            .agg(
                attempts=("shot_made", "size"),
                expected_points=("expected_points", "sum"),
                actual_points=("actual_points", "sum"),
            )
        )
        if "shot_class" not in grouped:
            grouped["shot_class"] = "all"
        grouped["expected_points_per_attempt"] = grouped["expected_points"] / grouped["attempts"]
        grouped["actual_minus_expected"] = grouped["actual_points"] - grouped["expected_points"]
        grouped["actual_minus_expected_per_attempt"] = grouped["actual_minus_expected"] / grouped["attempts"]
        league = grouped.groupby(["season_end", "shot_class"])["expected_points_per_attempt"].transform("mean")
        grouped["shot_quality_vs_league_per_attempt"] = grouped["expected_points_per_attempt"] - league
        pieces.append(grouped)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["season_end", "shot_class", "actual_minus_expected"], ascending=[True, True, False], kind="stable"
    ).reset_index(drop=True)


def fit_and_predict_expected_shots(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    score: pd.DataFrame,
    *,
    c: float = 0.2,
    max_iter: int = 300,
    include_possession_context: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the player-neutral model and return base and calibrated predictions.

    This small public helper keeps every downstream shot-quality experiment on
    the same identity-free feature contract.  ``score`` must be later than the
    training and calibration data in any longitudinal use; this function does
    not infer temporal ordering itself.
    """
    if train.empty or calibration.empty or score.empty:
        raise ValueError("Expected-shot fitting requires nonempty train, calibration, and score frames.")
    combined = pd.concat([train, calibration, score], ignore_index=True)
    features, feature_names = _feature_frame(
        combined, include_possession_context=include_possession_context
    )
    if set(feature_names) & FORBIDDEN_FEATURES:
        raise AssertionError("Expected-shot design contains a forbidden identity or outcome feature.")
    train_features = features[: len(train)]
    calibration_features = features[len(train) : len(train) + len(calibration)]
    score_features = features[len(train) + len(calibration) :]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    calibration_features = scaler.transform(calibration_features)
    score_features = scaler.transform(score_features)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
    model.fit(train_features, train["shot_made"].to_numpy(dtype=int))
    calibration_base = model.predict_proba(calibration_features)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_base, calibration["shot_made"].to_numpy(dtype=int))
    base_prediction = model.predict_proba(score_features)[:, 1]
    return base_prediction, calibrator.predict(base_prediction)


def run_expected_shot_quality(
    panel_path: str | Path,
    *,
    artifact_root: str | Path,
    train_season_ends: tuple[int, ...] = (2024,),
    calibration_season_ends: tuple[int, ...] = (2025,),
    test_season_end: int = 2026,
    c: float = 0.2,
    max_iter: int = 300,
) -> dict:
    """Fit, calibrate on later development data, then score a later year."""
    panel_path = Path(panel_path)
    panel = pd.read_parquet(panel_path)
    required = {"season_end", "shooter_id", "shot_zone", "shot_value", "shot_made"}
    if missing := sorted(required - set(panel.columns)):
        raise ValueError(f"Shot panel is missing required columns: {missing}.")
    if any(value in set(panel.columns) for value in FORBIDDEN_FEATURES):
        # Presence is expected in the panel; feature construction below is the guard.
        pass
    train = panel.loc[panel["season_end"].isin(train_season_ends)].copy()
    calibration_frame = panel.loc[
        panel["season_end"].isin(calibration_season_ends)
    ].copy()
    test = panel.loc[panel["season_end"].eq(test_season_end)].copy()
    if train.empty or calibration_frame.empty or test.empty:
        raise ValueError("Expected-shot split requires nonempty train, calibration, and test seasons.")
    if set(train_season_ends) & set(calibration_season_ends):
        raise ValueError("Expected-shot training and calibration seasons must not overlap.")

    _, feature_names = _feature_frame(pd.concat([train, calibration_frame, test], ignore_index=True))
    if set(feature_names) & FORBIDDEN_FEATURES:
        raise AssertionError("Expected-shot design contains a forbidden identity or outcome feature.")
    base_prediction, prediction = fit_and_predict_expected_shots(
        train,
        calibration_frame,
        test,
        c=c,
        max_iter=max_iter,
    )

    metric_rows = [
        {**_metric_row(test, base_prediction, "all_base"), "calibrated": False},
        {**_metric_row(test, prediction, "all"), "calibrated": True},
    ]
    for zone in ("rim", "non_rim"):
        mask = test["shot_zone"].eq("rim").to_numpy() if zone == "rim" else ~test["shot_zone"].eq("rim").to_numpy()
        metric_rows.append({**_metric_row(test.loc[mask], prediction[mask], zone), "calibrated": True})
    calibration_table = _calibration(prediction, test["shot_made"].to_numpy(dtype=int))
    players = _player_aggregates(test, prediction)

    config = {
        "model_version": MODEL_VERSION,
        "train_season_ends": list(train_season_ends),
        "calibration_season_ends": list(calibration_season_ends),
        "test_season_end": test_season_end,
        "c": c,
        "max_iter": max_iter,
        "feature_names": feature_names,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "panel_sha256": sha256_file(panel_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "models" / "expected_shot_quality" / f"{MODEL_VERSION}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    players.to_parquet(output / "player_shot_quality.parquet", index=False)
    pd.DataFrame(metric_rows).to_parquet(output / "test_metrics.parquet", index=False)
    calibration_table.to_parquet(output / "calibration.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "research_baseline",
        "estimand": "Player-neutral pre-shot expected field-goal points and descriptive shooter residuals.",
        "evidence_status": "reused_2026_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "training": {"shots": int(len(train)), "season_ends": list(train_season_ends)},
        "calibration": {"shots": int(len(calibration_frame)), "season_ends": list(calibration_season_ends)},
        "test": {"shots": int(len(test)), "season_end": test_season_end, "metrics": metric_rows},
        "artifact_path": str(output.resolve()),
        "forbidden_interpretation": (
            "Player impact, defender impact, primary-defender credit, causal shot creation, or RAPM/SPM/AIO input."
        ),
        "next_gate": "Validate a permitted shot-level defender assignment before any defender-specific matchup residual.",
    }
    write_json_atomic(run, output / "run.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the player-neutral expected-shot research baseline.")
    parser.add_argument("--panel", type=Path, default=Path("data/lake/silver/shot_defense_events.parquet"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run_expected_shot_quality(args.panel, artifact_root=args.artifact_root, c=args.c, max_iter=args.max_iter), indent=2))


if __name__ == "__main__":
    main()
