"""Tune WP-RAPM ridge penalties on next-season conserved WP changes."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, ratings_table
from nba_impact.models.win_probability_rapm import build_conserved_wp_target


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/wp_rapm_lambda_v1.json"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
WP = {
    2025: ROOT / "artifacts/models/win_probability_possession/wp_possession_start_v2_1db472e450/test_predictions.parquet",
    2026: ROOT / "artifacts/models/win_probability_possession/wp_possession_start_v2_0a5d626234/test_predictions.parquet",
}
OUTPUT = ROOT / "research/rapm_lab/outputs/wp_rapm_lambda"


def _metrics(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    games = pd.DataFrame(
        {"gameid": frame["gameid"].to_numpy(), "actual": actual, "predicted": predicted}
    ).groupby("gameid", as_index=False).sum()
    return {
        "possession_rmse": float(math.sqrt(np.mean(residual**2))),
        "possession_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "game_total_rmse": float(math.sqrt(np.mean((games["actual"] - games["predicted"]) ** 2))),
        "game_total_correlation": float(games[["actual", "predicted"]].corr().iloc[0, 1]),
        "games": int(len(games)),
    }


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    base = canonical_terminal_frame(
        pd.read_parquet(POSSESSIONS),
        pd.read_parquet(SEGMENTS),
        seasons=(2025, 2026),
    )
    probability_rows = []
    for season, path in WP.items():
        frame = pd.read_parquet(
            path,
            columns=["possession_id", "season_type", "home_win", "probability_context"],
        )
        probability_rows.append(frame.loc[frame["season_type"].eq("regular")].assign(_season=season))
    probabilities = pd.concat(probability_rows, ignore_index=True)
    frame = base.merge(
        probabilities[["possession_id", "home_win", "probability_context"]],
        on="possession_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(base):
        raise ValueError("Every canonical possession must have a cross-fitted WP state.")
    target, conservation = build_conserved_wp_target(frame)
    target["pts"] = target["offense_wp_change"]
    design = build_design(target, include_home=True)
    train = design.seasons == int(contract["selection_train_season"])
    test = design.seasons == int(contract["reused_diagnostic_season"])
    rows = []
    fits = {}
    for lambda_off, lambda_def in contract["lambda_candidates"]:
        config = RapmConfig(
            seasons=(2025,),
            lambda_off=float(lambda_off),
            lambda_def=float(lambda_def),
            lambda_home=float(contract["lambda_home"]),
            data_scope="next_season_wp_rapm_lambda_diagnostic",
        )
        beta, intercept = fit_coefficients(design, config, row_mask=train)
        predicted = intercept + np.asarray(design.X[test] @ beta).ravel()
        row = {
            "lambda_off": float(lambda_off),
            "lambda_def": float(lambda_def),
            **_metrics(target.loc[test], design.y[test], predicted),
        }
        rows.append(row)
        fits[(float(lambda_off), float(lambda_def))] = beta
    evaluation = pd.DataFrame(rows).sort_values(
        ["possession_rmse", "game_total_rmse"], kind="stable"
    )
    winner = evaluation.iloc[0]
    evaluation.to_parquet(OUTPUT / "evaluation.parquet", index=False) if OUTPUT.exists() else None
    selected = (float(winner["lambda_off"]), float(winner["lambda_def"]))
    full_config = RapmConfig(
        seasons=(2025, 2026),
        lambda_off=selected[0],
        lambda_def=selected[1],
        lambda_home=float(contract["lambda_home"]),
        data_scope="selected_conserved_wp_rapm",
    )
    full_beta, _ = fit_coefficients(design, full_config)
    ratings = ratings_table(design, full_beta)
    for component in ("offense", "defense", "net"):
        ratings[f"{component}_wp_percentage_points_per_100"] = 100.0 * ratings[f"{component}_per_100"]
    identity = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(CONTRACT),
                "possessions": sha256_file(POSSESSIONS),
                "segments": sha256_file(SEGMENTS),
                "wp": {str(k): sha256_file(v) for k, v in WP.items()},
                "runner": sha256_file(Path(__file__)),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"wp_rapm_lambda_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(output / "evaluation.parquet", index=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    conservation.to_parquet(output / "game_conservation.parquet", index=False)
    baseline = evaluation.loc[
        evaluation["lambda_off"].eq(3000) & evaluation["lambda_def"].eq(3000)
    ].iloc[0]
    run = {
        "run_id": output.name,
        "status": "reused_diagnostic_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": contract,
        "selected": winner.to_dict(),
        "baseline_3000_3000": baseline.to_dict(),
        "delta_selected_minus_baseline": {
            "possession_rmse": float(winner["possession_rmse"] - baseline["possession_rmse"]),
            "game_total_rmse": float(winner["game_total_rmse"] - baseline["game_total_rmse"]),
        },
        "quality": {
            "possessions": int(len(target)),
            "games": int(target["gameid"].nunique()),
            "maximum_conservation_error": float(conservation["conservation_error"].abs().max()),
        },
        "decision": "Use the selected penalty only for WP-credit research; do not alter points RAPM.",
        "artifacts": {"evaluation": "evaluation.parquet", "ratings": "ratings.parquet"},
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
