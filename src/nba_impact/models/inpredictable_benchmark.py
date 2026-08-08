"""Neutral-state surface comparison with Inpredictable's public NBA calculator."""
from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability_lineup import make_team_context_features


CALCULATOR_URL = "https://stats.inpredictable.com/nba/wpCalc.php"
CHECKPOINTS = (
    ("q1_11m59", "Q1", 11, 59, 2879.0),
    ("q2_11m59", "Q2", 11, 59, 2159.0),
    ("q3_11m59", "Q3", 11, 59, 1439.0),
    ("q4_11m59", "Q4", 11, 59, 719.0),
    ("q4_6m", "Q4", 6, 0, 360.0),
    ("q4_2m", "Q4", 2, 0, 120.0),
    ("q4_1m", "Q4", 1, 0, 60.0),
    ("q4_10s", "Q4", 0, 10, 10.0),
)


def parse_calculator_probability(html: str) -> float:
    match = re.search(r"Win Probability:\s*([0-9]+(?:\.[0-9]+)?)%", html)
    if match is None:
        raise ValueError("Inpredictable response did not contain a win probability.")
    return float(match.group(1)) / 100.0


def neutralize_home_probability(probability: np.ndarray, mirrored_probability: np.ndarray) -> np.ndarray:
    """Remove the fitted home/intercept advantage by averaging focal-team orientations."""
    return 0.5 * (probability + 1.0 - mirrored_probability)


def _calculator_query(row: dict) -> dict:
    response = requests.post(
        CALCULATOR_URL,
        data={
            "qtr": row["quarter"], "mintm": str(row["minutes"]),
            "sectm": str(row["seconds"]), "scr": str(row["margin"]),
            "poss": row["possession"],
        },
        timeout=(10, 45),
    )
    response.raise_for_status()
    return {**row, "inpredictable_probability": parse_calculator_probability(response.text)}


def _model_states(rows: pd.DataFrame, margin: pd.Series) -> pd.DataFrame:
    seconds_period = rows["minutes"].astype(float) * 60.0 + rows["seconds"].astype(float)
    return pd.DataFrame(
        {
            "home_score_diff_after": margin.astype(float),
            "regulation_seconds_remaining": rows["regulation_seconds_remaining"].astype(float),
            "seconds_remaining_period": seconds_period,
            "seconds_elapsed_game": 2880.0 - rows["regulation_seconds_remaining"].astype(float),
            "is_overtime": False,
            "pregame_elo_diff": 0.0,
            "pregame_starter_net_diff": 0.0,
            "pregame_rolling_margin_diff": 0.0,
            "pregame_rest_advantage_days": 0.0,
        }
    )


def run_inpredictable_surface_benchmark(
    model_run_path: str | Path,
    *,
    artifact_root: str | Path,
    max_workers: int = 4,
) -> dict:
    model_run_path = Path(model_run_path)
    model = joblib.load(model_run_path / "elo_plus_starters_team_context.joblib")
    inputs = []
    for checkpoint, quarter, minutes, seconds, regulation_remaining in CHECKPOINTS:
        for margin in range(-15, 16, 3):
            for possession in ("Y", "N"):
                inputs.append(
                    {
                        "checkpoint": checkpoint, "quarter": quarter, "minutes": minutes,
                        "seconds": seconds, "regulation_seconds_remaining": regulation_remaining,
                        "margin": margin, "possession": possession,
                    }
                )
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_calculator_query, row) for row in inputs]
        for future in as_completed(futures):
            rows.append(future.result())
    frame = pd.DataFrame(rows).sort_values(["regulation_seconds_remaining", "margin", "possession"])
    focal = frame.drop_duplicates(["checkpoint", "margin"])[
        ["checkpoint", "quarter", "minutes", "seconds", "regulation_seconds_remaining", "margin"]
    ].copy()
    probability = model.predict_proba(make_team_context_features(_model_states(focal, focal["margin"])))[:, 1]
    mirrored = model.predict_proba(make_team_context_features(_model_states(focal, -focal["margin"])))[:, 1]
    focal["local_neutral_probability"] = neutralize_home_probability(probability, mirrored)
    possession = frame.pivot_table(
        index=["checkpoint", "margin"], columns="possession",
        values="inpredictable_probability", aggfunc="first",
    ).reset_index().rename(columns={"Y": "inpredictable_possession", "N": "inpredictable_no_possession"})
    comparison = focal.merge(possession, on=["checkpoint", "margin"], validate="one_to_one")
    comparison["inpredictable_midpoint"] = 0.5 * (
        comparison["inpredictable_possession"] + comparison["inpredictable_no_possession"]
    )
    comparison["local_minus_inpredictable"] = (
        comparison["local_neutral_probability"] - comparison["inpredictable_midpoint"]
    )
    comparison["possession_swing"] = (
        comparison["inpredictable_possession"] - comparison["inpredictable_no_possession"]
    )
    error = comparison["local_minus_inpredictable"].to_numpy()
    run_id = f"wp_inpredictable_surface_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_benchmark" / run_id
    output.mkdir(parents=True, exist_ok=False)
    comparison.to_parquet(output / "surface_comparison.parquet", index=False)
    metrics = {
        "states": int(len(comparison)),
        "mean_absolute_probability_difference": float(np.abs(error).mean()),
        "root_mean_squared_probability_difference": float(np.sqrt(np.mean(error**2))),
        "correlation": float(comparison[["local_neutral_probability", "inpredictable_midpoint"]].corr().iloc[0, 1]),
        "max_absolute_probability_difference": float(np.abs(error).max()),
        "mean_absolute_inpredictable_possession_swing": float(np.abs(comparison["possession_swing"]).mean()),
    }
    run = {
        "run_id": run_id,
        "model_family": "neutral_wp_surface_comparison",
        "estimand": "neutral_team_win_probability_by_score_clock_and_possession_midpoint",
        "status": "reference_surface_comparison_not_outcome_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {"margins": list(range(-15, 16, 3)), "checkpoints": [row[0] for row in CHECKPOINTS], "max_workers": max_workers,
                   "model_run_path": str(model_run_path.resolve()), "model_run_sha256": sha256_file(model_run_path / "run.json"),
                   "source_code_sha256": sha256_file(Path(__file__)), "calculator_url": CALCULATOR_URL},
        "metrics": metrics,
        "caveats": [
            "This compares model surfaces, not predictive accuracy against game outcomes.",
            "Inpredictable is averaged across possession/no-possession because the local model lacks possession.",
            "The local home intercept is removed by averaging mirrored focal-team orientations.",
            "Inpredictable was trained on 2000-2012 games and assumes a zero point spread in the public calculator.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
