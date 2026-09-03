#!/usr/bin/env python3
"""Build the frozen PULSE next-season validation release."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.pulse_validation import load_pulse_validation


ROOT = Path(__file__).resolve().parents[1]
PULSE_RUN = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a"
OUTPUT_ROOT = ROOT / "artifacts/research/pulse_validation"
DRAW_COUNT = 5_000
SEED = 20260902


def metric_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Return fold and equal-season metrics without pooling seasons by game count."""
    rows: list[dict] = []
    for (candidate, season), group in frame.groupby(["candidate", "test_season"], sort=True):
        actual = group["actual_margin"].to_numpy(dtype=float)
        predicted = group["predicted_margin"].to_numpy(dtype=float)
        error = predicted - actual
        variance = float(np.var(predicted))
        rows.append({
            "scope": scope,
            "candidate": candidate,
            "outcome_season": int(season),
            "games": int(len(group)),
            "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
            "calibration_slope": (
                float(np.cov(actual, predicted, ddof=0)[0, 1] / variance)
                if variance > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    scope: str,
    draws: int = DRAW_COUNT,
    seed: int = SEED,
) -> dict:
    """Bootstrap games within season, then average season MSE differences equally."""
    left_rows = frame.loc[frame["candidate"].eq(left)]
    right_rows = frame.loc[frame["candidate"].eq(right)]
    keys = ["rating_season", "test_season", "game_id"]
    merged = left_rows[keys + ["actual_margin", "predicted_margin"]].merge(
        right_rows[keys + ["actual_margin", "predicted_margin"]],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left_rows) or len(merged) != len(right_rows):
        raise ValueError(f"{left} and {right} do not score identical games in {scope}.")
    if not np.allclose(merged["actual_margin_left"], merged["actual_margin_right"]):
        raise ValueError("Paired candidates disagree on actual game margins.")
    merged["delta"] = (
        (merged["predicted_margin_left"] - merged["actual_margin_left"]) ** 2
        - (merged["predicted_margin_right"] - merged["actual_margin_left"]) ** 2
    )
    seasons = [group["delta"].to_numpy(dtype=float) for _, group in merged.groupby("test_season")]
    observed = float(np.mean([values.mean() for values in seasons]))
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws, dtype=float)
    for draw in range(draws):
        distribution[draw] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean() for values in seasons
        ])
    return {
        "scope": scope,
        "left_candidate": left,
        "right_candidate": right,
        "seasons": len(seasons),
        "games": int(len(merged)),
        "mean_mse_delta_left_minus_right": observed,
        "bootstrap_95_low": float(np.quantile(distribution, 0.025)),
        "bootstrap_95_high": float(np.quantile(distribution, 0.975)),
        "probability_left_better": float(np.mean(distribution < 0)),
        "bootstrap_draws": draws,
        "seed": seed,
    }


def main() -> int:
    pulse_games, _ = load_pulse_validation(PULSE_RUN)
    combined = pulse_games.rename(columns={"outcome_season": "test_season"}).copy()
    combined["candidate"] = combined["candidate"].replace({"prior": "pulse_prior"})
    maximal = combined
    # The loader requires all three candidates to score exactly the same games.
    strict = maximal.copy()

    fold_metrics = pd.concat([
        metric_rows(maximal, "maximal_pulse_coverage"),
        metric_rows(strict, "strict_common_coverage"),
    ], ignore_index=True)
    aggregate = (
        fold_metrics.groupby(["scope", "candidate"], as_index=False)
        .agg(
            folds=("outcome_season", "nunique"),
            games=("games", "sum"),
            mean_mse=("mse", "mean"),
            mean_correlation=("correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
    )
    aggregate["rmse"] = np.sqrt(aggregate["mean_mse"])

    intervals: list[dict] = []
    for scope, frame in (("maximal_pulse_coverage", maximal), ("strict_common_coverage", strict)):
        candidates = sorted(frame["candidate"].unique())
        for right in candidates:
            if right == "pulse":
                continue
            intervals.append(paired_bootstrap(frame, "pulse", right, scope=scope))

    payload = {
        "inputs": {
            "pulse_validation_games": sha256_file(PULSE_RUN / "validation_games.parquet"),
            "pulse_validation_folds": sha256_file(PULSE_RUN / "validation_folds.parquet"),
            "pulse_validation_priors": sha256_file(PULSE_RUN / "validation_priors.parquet"),
            "pulse_manifest": sha256_file(PULSE_RUN / "run.json"),
        },
        "draws": DRAW_COUNT,
        "seed": SEED,
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"pulse_validation_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    aggregate.to_parquet(output / "aggregate_metrics.parquet", index=False)
    pd.DataFrame(intervals).to_parquet(output / "paired_intervals.parquet", index=False)
    maximal.to_parquet(output / "maximal_game_predictions.parquet", index=False)
    strict.to_parquet(output / "strict_game_predictions.parquet", index=False)
    quality = {
        "maximal_seasons": int(maximal["test_season"].nunique()),
        "strict_seasons": int(strict["test_season"].nunique()),
        "identical_games_within_scope": True,
        "bootstrap_draws": DRAW_COUNT,
    }
    write_json_atomic({
        "run_id": output.name,
        "status": "reused_evidence_validation_release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season whole-game margin prediction from prior-season ratings",
        "primary_metric": "equal-season mean whole-game margin MSE",
        "payload": payload,
        "quality": quality,
        "forbidden_interpretation": "This is retrospective validation with observed future lineups, not a preseason forecast.",
        "files": {
            "fold_metrics": "fold_metrics.parquet",
            "aggregate_metrics": "aggregate_metrics.parquet",
            "paired_intervals": "paired_intervals.parquet",
            "maximal_game_predictions": "maximal_game_predictions.parquet",
            "strict_game_predictions": "strict_game_predictions.parquet",
        },
    }, output / "run.json")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
