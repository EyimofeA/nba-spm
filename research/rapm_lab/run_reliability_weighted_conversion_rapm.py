"""Test partial and history-weighted shooting-luck targets in one-season RAPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.luck_adjusted_rapm import _game_metrics
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, _game_margin_frame, build_design, fit_coefficients
from research.rapm_lab.run_single_season_expected_outcome_rapm import _paired_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/reliability_weighted_conversion_rapm_v1.yml"
SOURCE = ROOT / "artifacts/models/luck_adjusted_rapm/luck_adjusted_rapm_v1_8580bb30e9"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/reliability_weighted_conversion_rapm"


def build_target_frames(
    base: pd.DataFrame,
    ledger: pd.DataFrame,
    selected: pd.DataFrame,
    residual_weights: tuple[float, ...],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build possession targets while preserving observed nonconversion points."""
    priors = selected[["category", "prior_attempts"]].drop_duplicates("category")
    if set(ledger["category"]) - set(priors["category"]):
        raise ValueError("Conversion ledger contains a category without a frozen prior.")
    work = ledger.merge(priors, on="category", how="left", validate="many_to_one")
    attempts = work.groupby(["season_end", "shooter_id", "category"])["actual_points"].transform("size")
    work["reliability"] = attempts / (attempts + work["prior_attempts"])
    work["history_hybrid_points"] = work["skill_expected_points"] + work["reliability"] * (
        work["actual_points"] - work["skill_expected_points"]
    )
    value_columns = {
        "actual_conversion_points": "actual_points",
        "neutral_conversion_points": "neutral_expected_points",
        "skill_conversion_points": "skill_expected_points",
        "history_hybrid_conversion_points": "history_hybrid_points",
    }
    for weight in residual_weights:
        label = f"neutral_residual_{int(round(100 * weight))}pct"
        column = f"{label}_points"
        work[column] = work["neutral_expected_points"] + weight * (
            work["actual_points"] - work["neutral_expected_points"]
        )
        value_columns[f"{label}_conversion_points"] = column
    conversion = work.groupby("possession_id", as_index=False).agg(
        **{name: (column, "sum") for name, column in value_columns.items()}
    )
    merged = base.merge(conversion, on="possession_id", how="left", validate="one_to_one")
    merged[list(value_columns)] = merged[list(value_columns)].fillna(0.0)
    frames = {"normal_realized_points": base.copy()}
    replacements = {
        "preseason_skill_expected": "skill_conversion_points",
        "history_reliability_hybrid": "history_hybrid_conversion_points",
        **{
            f"neutral_residual_{int(round(100 * weight))}pct":
                f"neutral_residual_{int(round(100 * weight))}pct_conversion_points"
            for weight in residual_weights
        },
    }
    for arm, replacement in replacements.items():
        target = merged.copy()
        target["pts"] = (
            target["pts"] - target["actual_conversion_points"] + target[replacement]
        )
        if not np.isfinite(target["pts"]).all():
            raise ValueError(f"Nonfinite possession target in {arm}.")
        frames[arm] = target[base.columns].copy()
    reliability = work.groupby(["season_end", "category"], as_index=False).agg(
        attempts=("actual_points", "size"),
        players=("shooter_id", "nunique"),
        mean_reliability=("reliability", "mean"),
    )
    return frames, reliability


def run() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["experiment_id"] != "reliability_weighted_conversion_rapm_v1":
        raise ValueError("Unexpected experiment contract.")
    if int(contract["season_policy"]["forbidden_season"]) != 2027:
        raise ValueError("Season 2027 must remain forbidden.")
    rating_seasons = tuple(map(int, contract["season_policy"]["rating_seasons"]))
    test_seasons = tuple(map(int, contract["season_policy"]["test_seasons"]))
    if rating_seasons != (2024, 2025) or test_seasons != (2025, 2026):
        raise ValueError("The frozen one-season folds changed.")
    residual_weights = tuple(map(float, contract["conversion_source"]["residual_retention_grid"]))
    expected_arms = tuple(contract["arms"])
    contract = json.loads(json.dumps(contract, default=str))

    paths = {
        "possessions": ROOT / "data/lake/silver/possessions.parquet",
        "segments": ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        "source_run": SOURCE / "run.json",
        "ledger": SOURCE / "checkpoints/conversion_ledger.parquet",
        "selected": SOURCE / "checkpoints/shooting_history_selected.parquet",
    }
    source_run = json.loads(paths["source_run"].read_text())
    if source_run.get("run_id") != contract["conversion_source"]["run_id"]:
        raise ValueError("The conversion source run changed.")
    possessions = pd.read_parquet(paths["possessions"])
    segments = pd.read_parquet(paths["segments"])
    base = canonical_terminal_frame(possessions, segments, seasons=(2024, 2025, 2026))
    ledger = pd.read_parquet(paths["ledger"])
    ledger = ledger.loc[ledger["season_end"].isin(rating_seasons)].copy()
    selected = pd.read_parquet(paths["selected"])
    frames, reliability = build_target_frames(base, ledger, selected, residual_weights)
    if set(frames) != set(expected_arms):
        raise ValueError("Target arms do not match the frozen contract.")
    frames = {arm: frames[arm] for arm in expected_arms}

    designs = {arm: build_design(frame, include_home=True) for arm, frame in frames.items()}
    normal_design = designs["normal_realized_points"]
    if any(not np.array_equal(design.players, normal_design.players) for design in designs.values()):
        raise ValueError("Every arm must use the same player design.")
    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=float(contract["rapm"]["lambda_off"]),
        lambda_def=float(contract["rapm"]["lambda_def"]),
        lambda_home=float(contract["rapm"]["lambda_home"]),
        data_scope=contract["experiment_id"],
    )
    game_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for test_season in test_seasons:
        train = normal_design.seasons == test_season - 1
        test = normal_design.seasons == test_season
        for arm, design in designs.items():
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            games = _game_margin_frame(normal_design, beta, intercept, test, train)
            games["arm"] = arm
            games["test_season"] = test_season
            game_parts.append(games)
            metric_rows.append({"arm": arm, "test_season": test_season, **_game_metrics(games)})
    games = pd.concat(game_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    grid_arms = [f"neutral_residual_{int(round(100 * weight))}pct" for weight in residual_weights]
    selection_season = int(contract["season_policy"]["residual_weight_selection_season"])
    selected_arm = str(
        metrics.loc[
            metrics["test_season"].eq(selection_season) & metrics["arm"].isin(grid_arms)
        ].sort_values(["margin_rmse", "arm"], kind="stable").iloc[0]["arm"]
    )
    challengers = [selected_arm, "preseason_skill_expected", "history_reliability_hybrid"]
    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap(games, challenger=arm, season=season, draws=int(contract["evaluation"]["bootstrap_draws"]), seed=20260903)
            for season in test_seasons
            for arm in challengers
        ]
    )

    identity_sources = [CONTRACT, Path(__file__), *paths.values()]
    identity = hashlib.sha256(
        json.dumps(
            {str(path.relative_to(ROOT)): sha256_file(path) for path in identity_sources},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"reliability_weighted_conversion_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / "future_game_metrics.parquet", index=False)
    games.to_parquet(output / "future_game_predictions.parquet", index=False)
    bootstrap.to_parquet(output / "paired_game_bootstrap.parquet", index=False)
    reliability.to_parquet(output / "reliability_summary.parquet", index=False)
    run_record = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": contract["estimand_id"],
        "config": contract,
        "source_run_id": source_run["run_id"],
        "selected_neutral_residual_arm_on_2025": selected_arm,
        "quality": {
            "possessions": len(base),
            "games": int(base["gameid"].nunique()),
            "rating_seasons": list(rating_seasons),
            "test_seasons": list(test_seasons),
            "season_2027_loaded": False,
            "conversion_rows": len(ledger),
        },
        "diagnostic_metrics": metrics.to_dict(orient="records"),
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "decision_rule": "retain_normal_unless_a_frozen_challenger_improves_both_reused_folds",
        "forbidden_interpretation": "Production rating, independent confirmation, causal shooting credit, or literal qSQ.",
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run_record, output / "run.json")
    return run_record


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
