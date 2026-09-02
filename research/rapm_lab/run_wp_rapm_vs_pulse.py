"""Score WP-RAPM, ordinary RAPM, and PULSE on identical next-season games."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WP_GAMES = ROOT / "research/rapm_lab/outputs/wp_spm_aio/wp_spm_aio_v1_5d7272a48f/game_predictions.parquet"
PULSE_GAMES = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a/validation_games.parquet"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/wp_rapm_vs_pulse"
OUTCOME_SEASONS = range(2023, 2027)
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260902


def affine(train: pd.DataFrame, column: str) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(train)), train[column].to_numpy(float)])
    intercept, slope = np.linalg.lstsq(design, train["actual_margin"].to_numpy(float), rcond=None)[0]
    return float(intercept), float(slope)


def calibrated_predictions(games: pd.DataFrame):
    for season in OUTCOME_SEASONS:
        train = games.loc[games["outcome_season"].lt(season)]
        test = games.loc[games["outcome_season"].eq(season)].copy()
        for model, column in (
            ("PULSE", "pulse_raw"),
            ("RAPM", "rapm_raw"),
            ("WP-RAPM", "wp_raw"),
        ):
            intercept, slope = affine(train, column)
            predicted = intercept + slope * test[column]
            test[f"{model}_predicted_margin"] = predicted
        yield test


def paired_bootstrap(
    predictions: pd.DataFrame, left: str, right: str
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_DRAWS)
    seasons = sorted(predictions["outcome_season"].unique())
    for draw in range(BOOTSTRAP_DRAWS):
        fold_deltas = []
        for season in seasons:
            fold = predictions.loc[predictions["outcome_season"].eq(season)]
            take = rng.integers(0, len(fold), len(fold))
            actual = fold["actual_margin"].to_numpy()[take]
            left_error = actual - fold[f"{left}_predicted_margin"].to_numpy()[take]
            right_error = actual - fold[f"{right}_predicted_margin"].to_numpy()[take]
            fold_deltas.append(np.mean(left_error**2 - right_error**2))
        deltas[draw] = np.mean(fold_deltas)
    return {
        "draws": BOOTSTRAP_DRAWS,
        "left": left,
        "right": right,
        "mean_mse_delta": float(deltas.mean()),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_left_better": float(np.mean(deltas < 0)),
    }


def main() -> int:
    wp = pd.read_parquet(WP_GAMES)
    wp = wp.loc[wp["candidate"].eq("zero_wp_rapm"), ["outcome_season", "game_id", "predicted"]]
    wp = wp.rename(columns={"predicted": "wp_raw"})
    pulse_games = pd.read_parquet(PULSE_GAMES)
    pulse = pulse_games.loc[pulse_games["candidate"].eq("pulse"), [
        "outcome_season", "game_id", "actual_margin", "predicted_margin",
    ]].rename(columns={"predicted_margin": "pulse_raw"})
    rapm = pulse_games.loc[pulse_games["candidate"].eq("rapm"), [
        "outcome_season", "game_id", "predicted_margin",
    ]].rename(columns={"predicted_margin": "rapm_raw"})
    games = pulse.merge(rapm, on=["outcome_season", "game_id"], validate="one_to_one")
    games = games.merge(wp, on=["outcome_season", "game_id"], validate="one_to_one")
    if games.groupby("outcome_season").size().min() < 1_200:
        raise ValueError("Common next-season game coverage fell below 1,200 games")

    predictions = pd.concat(list(calibrated_predictions(games)), ignore_index=True)
    metrics = []
    for model in ("PULSE", "RAPM", "WP-RAPM"):
        folds = []
        for season, frame in predictions.groupby("outcome_season"):
            error = frame["actual_margin"] - frame[f"{model}_predicted_margin"]
            folds.append({
                "outcome_season": int(season), "model": model, "games": len(frame),
                "mse": float(np.mean(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
                "correlation": float(frame["actual_margin"].corr(frame[f"{model}_predicted_margin"])),
            })
        metrics.extend(folds)
    metrics_frame = pd.DataFrame(metrics)
    summary = (
        metrics_frame.groupby("model", as_index=False)
        .agg(folds=("outcome_season", "nunique"), games=("games", "sum"), mean_mse=("mse", "mean"), mean_correlation=("correlation", "mean"))
    )
    summary["equal_season_rmse"] = np.sqrt(summary["mean_mse"])
    comparisons = [
        paired_bootstrap(predictions, "WP-RAPM", "RAPM"),
        paired_bootstrap(predictions, "WP-RAPM", "PULSE"),
        paired_bootstrap(predictions, "PULSE", "RAPM"),
    ]

    digest = hashlib.sha256(
        json.dumps({"wp": str(WP_GAMES), "pulse": str(PULSE_GAMES), "seasons": list(OUTCOME_SEASONS)}, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"wp_rapm_vs_pulse_v1_{digest}"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    metrics_frame.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "reused_historical_diagnostic",
        "comparison": "one-season zero-prior WP-RAPM, ordinary RAPM, and PULSE",
        "target": "next-season whole-game point margin",
        "calibration": "expanding affine map trained only on earlier common outcome seasons for both models",
        "outcome_seasons": list(OUTCOME_SEASONS),
        "games": int(len(predictions)),
        "summary": summary.to_dict("records"),
        "paired_comparisons": comparisons,
        "warning": "Uses observed next-season lineups and reused historical folds; this is retrodiction, not a deployable forecast.",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
