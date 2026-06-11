#!/usr/bin/env python3
"""Compare simple RAPM runs by next-season team prediction.

For each player-level RAPM output, use the next season's team minutes to build
a team predicted rating, then compare to actual team net rating and wins.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLAYER_SHEET = PROJECT_ROOT / "data" / "raw" / "playersheets" / "year_totals" / "2025.csv"
TEAM_RATINGS = PROJECT_ROOT / "zts" / "data" / "processed" / "team_ratings.csv"
TEST_SEASON = 2025
TEAM_ABBREVIATION_MAP = {
    "PHX": "PHO",
}


def simple_player_outputs() -> list[Path]:
    return sorted(
        p for p in OUTPUT_DIR.glob("*.csv")
        if "players" in p.name and "retrodict" not in p.name
    )


def load_run_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in OUTPUT_DIR.glob("*_run_2022_2024.json"):
        with open(path) as handle:
            run_meta = json.load(handle)
        players_path = run_meta.get("outputs", {}).get("players")
        if not players_path:
            continue
        metadata[Path(players_path).stem.replace("_players_2022_2024", "")] = run_meta
    return metadata


def load_team_minutes() -> pd.DataFrame:
    df = pd.read_csv(
        PLAYER_SHEET,
        usecols=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "Minutes"],
    )
    minute_col = "MIN" if "MIN" in df.columns and df["MIN"].notna().any() else "Minutes"
    df = df.rename(columns={minute_col: "minutes", "TEAM_ABBREVIATION": "team"})
    df = df.dropna(subset=["PLAYER_ID", "team", "minutes"])
    df["team"] = df["team"].replace(TEAM_ABBREVIATION_MAP)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["minutes"] = df["minutes"].astype(float)
    return (
        df.groupby(["team", "PLAYER_ID"], as_index=False)
        .agg(minutes=("minutes", "sum"), player_name=("PLAYER_NAME", "first"))
    )


def load_actual_team_results() -> pd.DataFrame:
    df = pd.read_csv(TEAM_RATINGS)
    df = df[df["Season"] == TEST_SEASON].copy()
    df["actual_net_rating"] = df["team_ortg"] - df["team_drtg"]
    df = df.rename(columns={"Team": "team", "team_wins": "actual_wins"})
    return df[["team", "actual_net_rating", "actual_wins", "team_ortg", "team_drtg"]]


def weighted_team_predictions(player_path: Path, team_minutes: pd.DataFrame) -> pd.DataFrame:
    ratings = pd.read_csv(player_path, usecols=["PLAYER_ID", "Name", "RAPM", "Off", "Def"])
    ratings["PLAYER_ID"] = ratings["PLAYER_ID"].astype(int)
    merged = team_minutes.merge(ratings, on="PLAYER_ID", how="left", indicator=True)
    merged["rating_matched"] = merged["_merge"].eq("both")
    merged[["RAPM", "Off", "Def"]] = merged[["RAPM", "Off", "Def"]].fillna(0.0)
    merged["matched_minutes"] = np.where(merged["rating_matched"], merged["minutes"], 0.0)

    def summarize(group: pd.DataFrame) -> pd.Series:
        minutes = group["minutes"].to_numpy(dtype=float)
        total_minutes = float(minutes.sum())
        if total_minutes <= 0:
            return pd.Series({
                "pred_net_rating": np.nan,
                "pred_off": np.nan,
                "pred_def": np.nan,
                "total_minutes": 0.0,
                "matched_minutes": 0.0,
                "minute_coverage": 0.0,
            })
        return pd.Series({
            "pred_net_rating": float(np.average(group["RAPM"], weights=minutes)),
            "pred_off": float(np.average(group["Off"], weights=minutes)),
            "pred_def": float(np.average(group["Def"], weights=minutes)),
            "total_minutes": total_minutes,
            "matched_minutes": float(group["matched_minutes"].sum()),
            "minute_coverage": float(group["matched_minutes"].sum() / total_minutes),
        })

    out = team_minutes[["team"]].drop_duplicates().merge(
        merged.groupby("team").apply(summarize, include_groups=False).reset_index(),
        on="team",
        how="left",
    )
    out["run"] = player_path.stem.replace("_players_2022_2024", "")
    out["source_file"] = str(player_path)
    return out


def corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def affine_calibrated_prediction(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(x, np.nan, dtype=float), float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    return slope * x + intercept, float(slope), float(intercept)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    centered = y_true[mask] - y_true[mask].mean()
    total = float(np.sum(centered**2))
    if total <= 0:
        return float("nan")
    residual = float(np.sum((y_pred[mask] - y_true[mask]) ** 2))
    return 1.0 - residual / total


def season_weight_label(run_meta: dict[str, Any]) -> str:
    weights = run_meta.get("season_weights", {})
    if not weights:
        return "unknown"
    normalized = {str(k): float(v) for k, v in weights.items()}
    if len(set(normalized.values())) == 1:
        return "uniform"
    return "time_decay"


def metrics_for_run(df: pd.DataFrame, run_meta: dict[str, Any]) -> dict:
    x = df["pred_net_rating"].to_numpy(dtype=float)
    y = df["actual_net_rating"].to_numpy(dtype=float)
    calibrated, slope, intercept = affine_calibrated_prediction(x, y)
    raw_error = x - y
    calibrated_error = calibrated - y

    wins = df["actual_wins"]
    return {
        "run": df["run"].iloc[0],
        "n_teams": int(len(df)),
        "mean_minute_coverage": float(df["minute_coverage"].mean()),
        "season_weighting": season_weight_label(run_meta),
        "season_weights": json.dumps(run_meta.get("season_weights", {}), sort_keys=True),
        "has_season_effects": bool(run_meta.get("season_effects")),
        "raw_net_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "raw_net_mae": float(np.mean(np.abs(raw_error))),
        "raw_net_r2": r_squared(y, x),
        "calibrated_net_rmse": float(np.sqrt(np.mean(calibrated_error**2))),
        "calibrated_net_mae": float(np.mean(np.abs(calibrated_error))),
        "calibrated_net_r2": r_squared(y, calibrated),
        "net_rating_corr": corr(df["pred_net_rating"], df["actual_net_rating"]),
        "wins_corr": corr(df["pred_net_rating"], wins),
        "wins_r2_linear": corr(df["pred_net_rating"], wins) ** 2,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    player_outputs = simple_player_outputs()
    if not player_outputs:
        raise RuntimeError(f"No simple player output files found under {OUTPUT_DIR}")

    run_metadata = load_run_metadata()
    team_minutes = load_team_minutes()
    actual = load_actual_team_results()
    all_team_rows = []
    summary_rows = []
    for player_path in player_outputs:
        team_pred = weighted_team_predictions(player_path, team_minutes)
        merged = team_pred.merge(actual, on="team", how="inner")
        all_team_rows.append(merged)
        run_name = player_path.stem.replace("_players_2022_2024", "")
        summary_rows.append(metrics_for_run(merged, run_metadata.get(run_name, {})))

    team_predictions = pd.concat(all_team_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["calibrated_net_rmse", "net_rating_corr"],
        ascending=[True, False],
    )

    team_path = OUTPUT_DIR / "team_retrodiction_2025_by_run.csv"
    summary_path = OUTPUT_DIR / "team_retrodiction_2025_summary.csv"
    meta_path = OUTPUT_DIR / "team_retrodiction_2025_summary.json"
    team_predictions.to_csv(team_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "test_season": TEST_SEASON,
                "player_outputs": [str(p) for p in player_outputs],
                "team_minutes": str(PLAYER_SHEET),
                "actual_team_ratings": str(TEAM_RATINGS),
                "outputs": {
                    "team_predictions": str(team_path),
                    "summary": str(summary_path),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(summary.to_string(index=False))
    print(f"Team predictions -> {team_path}")
    print(f"Summary -> {summary_path}")
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
