"""Chronological age and opportunity ablations for predictive SPM.

The frozen predictive SPM remains the basketball-information arm.  This module
fits small residual corrections using only earlier forecast seasons.  Age and
lagged opportunity stay named ablations rather than silently entering the SPM.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic


METHODS = ("raw", "shared_age", "side_age", "side_age_opportunity")
SIDES = ("offense", "defense")
AGE_KNOTS = (22.0, 25.0, 28.0, 31.0, 34.0)
REQUIRED_PREDICTION_COLUMNS = {
    "PLAYER_ID",
    "Target_Season",
    "Window_End",
    "raw_offense",
    "raw_defense",
    "raw_net",
    "target_offense",
    "target_defense",
    "target_net",
    "sample_weight",
}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": "predictive_spm_trajectory_ablation_v1",
        "status": "preregistered_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    development = tuple(int(value) for value in contract["development_seasons"])
    diagnostics = tuple(int(value) for value in contract["reused_diagnostic_seasons"])
    if development != (2020, 2021, 2022, 2023, 2024):
        raise ValueError("Development seasons must remain 2020 through 2024.")
    if diagnostics != (2025, 2026):
        raise ValueError("Reused diagnostics must remain 2025 and 2026.")
    requested = {*development, *diagnostics}
    if 2027 in requested or max(requested) > 2026:
        raise ValueError("Season 2027 must be rejected before data are read.")
    if tuple(contract["methods"]) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS}.")
    return contract


def _age_basis(frame: pd.DataFrame) -> np.ndarray:
    age = frame["forecast_age"].to_numpy(dtype=float)
    centered = age - 27.0
    columns = [centered, centered**2 / 25.0]
    columns.extend(np.maximum(0.0, age - knot) for knot in AGE_KNOTS)
    return np.column_stack(columns)


def _opportunity_basis(frame: pd.DataFrame) -> np.ndarray:
    age = _age_basis(frame)
    minutes = np.log1p(frame["origin_minutes"].clip(lower=0).to_numpy(dtype=float))
    games = np.log1p(frame["origin_games"].clip(lower=0).to_numpy(dtype=float))
    minutes_per_game = np.divide(
        frame["origin_minutes"].to_numpy(dtype=float),
        frame["origin_games"].clip(lower=1).to_numpy(dtype=float),
    )
    return np.column_stack(
        [
            age,
            minutes,
            games,
            minutes_per_game,
            (frame["forecast_age"] - 27.0) * minutes,
        ]
    )


def _load_metadata(player_sheets_dir: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        path = player_sheets_dir / f"{season}.csv"
        frame = pd.read_csv(
            path,
            usecols=[
                "PLAYER_ID",
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "AGE",
                "MIN",
                "GP",
            ],
        )
        frame["Season"] = int(season)
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(
            int
        )
        frame = frame.sort_values(["PLAYER_ID", "MIN"], kind="stable").drop_duplicates(
            "PLAYER_ID", keep="last"
        )
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Player metadata contains duplicate player-season keys.")
    return output


def build_model_rows(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    scored_seasons: tuple[int, ...],
) -> tuple[pd.DataFrame, dict]:
    if missing := sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)):
        raise ValueError(f"Predictive SPM rows are missing columns: {missing}.")
    numeric_target = pd.to_numeric(predictions["Target_Season"], errors="raise").astype(
        int
    )
    frame = predictions.loc[numeric_target.le(max(scored_seasons))].copy()
    if frame.empty:
        raise ValueError("No predictive SPM rows match the configured seasons.")
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    frame["Target_Season"] = pd.to_numeric(
        frame["Target_Season"], errors="raise"
    ).astype(int)
    frame["Window_End"] = pd.to_numeric(frame["Window_End"], errors="raise").astype(int)
    if not frame["Window_End"].eq(frame["Target_Season"]).all():
        raise ValueError(
            "The frozen predictive artifact must label Window_End with its forecast season."
        )
    frame["Origin_Season"] = frame["Target_Season"] - 1
    if frame.duplicated(["PLAYER_ID", "Target_Season"]).any():
        raise ValueError("Predictive SPM rows contain duplicate player-season keys.")

    origin = metadata.rename(
        columns={
            "Season": "Origin_Season",
            "AGE": "origin_age",
            "MIN": "origin_minutes",
            "GP": "origin_games",
            "TEAM_ABBREVIATION": "origin_team",
            "PLAYER_NAME": "PLAYER_NAME",
        }
    )[
        [
            "PLAYER_ID",
            "Origin_Season",
            "PLAYER_NAME",
            "origin_team",
            "origin_age",
            "origin_minutes",
            "origin_games",
        ]
    ]
    target = metadata.rename(
        columns={"Season": "Target_Season", "TEAM_ABBREVIATION": "target_team"}
    )[["PLAYER_ID", "Target_Season", "target_team"]]
    merged = frame.merge(
        origin, on=["PLAYER_ID", "Origin_Season"], how="left", validate="one_to_one"
    )
    merged = merged.merge(
        target, on=["PLAYER_ID", "Target_Season"], how="left", validate="one_to_one"
    )
    required_metadata = ["origin_age", "origin_minutes", "origin_games"]
    complete = merged[required_metadata].notna().all(axis=1)
    quality = {
        "prediction_rows": int(len(frame)),
        "complete_origin_metadata_rows": int(complete.sum()),
        "excluded_missing_origin_metadata": int((~complete).sum()),
    }
    merged = merged.loc[complete].copy()
    merged[required_metadata] = merged[required_metadata].astype(float)
    merged["forecast_age"] = merged["origin_age"] + 1.0
    merged["evaluation_weight"] = pd.to_numeric(
        merged["sample_weight"], errors="raise"
    ).astype(float)
    merged["team_changer"] = (
        merged["origin_team"].notna()
        & merged["target_team"].notna()
        & merged["origin_team"].ne(merged["target_team"])
    )
    numeric = [
        "raw_offense",
        "raw_defense",
        "raw_net",
        "target_offense",
        "target_defense",
        "target_net",
        "evaluation_weight",
        "forecast_age",
    ]
    if not np.isfinite(merged[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Predictive trajectory rows contain non-finite values.")
    return merged.sort_values(
        ["Target_Season", "PLAYER_ID"], kind="stable"
    ).reset_index(drop=True), quality


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y, ridge__sample_weight=train_weight)
    return model.predict(test_x)


def predict_method(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    *,
    alpha: float,
) -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"Unknown trajectory ablation method {method!r}.")
    result = test[["PLAYER_ID", "Target_Season"]].copy()
    if method == "raw":
        for side in SIDES:
            result[f"predicted_{side}"] = test[f"raw_{side}"].to_numpy(dtype=float)
    elif method == "shared_age":
        train_age = _age_basis(train)
        test_age = _age_basis(test)
        train_side = np.concatenate(
            [np.full((len(train), 1), -0.5), np.full((len(train), 1), 0.5)]
        )
        test_off = np.column_stack([np.full(len(test), -0.5), test_age])
        test_def = np.column_stack([np.full(len(test), 0.5), test_age])
        stacked_x = np.column_stack([train_side, np.vstack([train_age, train_age])])
        stacked_y = np.concatenate(
            [
                train["target_offense"].to_numpy(dtype=float)
                - train["raw_offense"].to_numpy(dtype=float),
                train["target_defense"].to_numpy(dtype=float)
                - train["raw_defense"].to_numpy(dtype=float),
            ]
        )
        stacked_weight = np.tile(train["evaluation_weight"].to_numpy(dtype=float), 2)
        result["predicted_offense"] = test["raw_offense"].to_numpy(
            dtype=float
        ) + _fit_ridge(stacked_x, stacked_y, stacked_weight, test_off, alpha=alpha)
        result["predicted_defense"] = test["raw_defense"].to_numpy(
            dtype=float
        ) + _fit_ridge(stacked_x, stacked_y, stacked_weight, test_def, alpha=alpha)
    else:
        design = _age_basis if method == "side_age" else _opportunity_basis
        train_x = design(train)
        test_x = design(test)
        for side in SIDES:
            residual = train[f"target_{side}"].to_numpy(dtype=float) - train[
                f"raw_{side}"
            ].to_numpy(dtype=float)
            adjustment = _fit_ridge(
                train_x,
                residual,
                train["evaluation_weight"].to_numpy(dtype=float),
                test_x,
                alpha=alpha,
            )
            result[f"predicted_{side}"] = (
                test[f"raw_{side}"].to_numpy(dtype=float) + adjustment
            )
    result["predicted_net"] = result["predicted_offense"] + result["predicted_defense"]
    result["method"] = method
    return result


def _weighted_metrics(
    actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict:
    total = float(weight.sum())
    if total <= 0 or len(actual) == 0:
        raise ValueError("Metrics require positive weight and at least one row.")
    mean_actual = float(np.sum(weight * actual) / total)
    mean_prediction = float(np.sum(weight * prediction) / total)
    residual = actual - prediction
    covariance = float(
        np.sum(weight * (actual - mean_actual) * (prediction - mean_prediction)) / total
    )
    variance_actual = float(np.sum(weight * (actual - mean_actual) ** 2) / total)
    variance_prediction = float(
        np.sum(weight * (prediction - mean_prediction) ** 2) / total
    )
    correlation = (
        covariance / np.sqrt(variance_actual * variance_prediction)
        if variance_actual > 0 and variance_prediction > 0
        else np.nan
    )
    slope = covariance / variance_prediction if variance_prediction > 0 else np.nan
    return {
        "weighted_rmse": float(np.sqrt(np.sum(weight * residual**2) / total)),
        "weighted_correlation": float(correlation),
        "calibration_slope": float(slope),
        "calibration_intercept": float(mean_actual - slope * mean_prediction),
        "dispersion_ratio": float(np.sqrt(variance_prediction / variance_actual))
        if variance_actual > 0
        else np.nan,
    }


def score_predictions(scored: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (season, method), group in scored.groupby(
        ["Target_Season", "method"], sort=True
    ):
        for side in (*SIDES, "net"):
            records.append(
                {
                    "target_season": int(season),
                    "method": str(method),
                    "component": side,
                    "rows": int(len(group)),
                    **_weighted_metrics(
                        group[f"target_{side}"].to_numpy(dtype=float),
                        group[f"predicted_{side}"].to_numpy(dtype=float),
                        group["evaluation_weight"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(records)


def run_walk_forward(
    rows: pd.DataFrame,
    *,
    scored_seasons: tuple[int, ...],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for season in scored_seasons:
        train = rows.loc[rows["Target_Season"].lt(season)].copy()
        test = rows.loc[rows["Target_Season"].eq(season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Forecast season {season} lacks train or test rows.")
        for method in METHODS:
            prediction = predict_method(train, test, method, alpha=alpha)
            merged = test.merge(
                prediction,
                on=["PLAYER_ID", "Target_Season"],
                how="left",
                validate="one_to_one",
            )
            merged["method"] = method
            merged["training_target_end"] = int(train["Target_Season"].max())
            merged["training_target_seasons"] = int(train["Target_Season"].nunique())
            outputs.append(merged)
    scored = pd.concat(outputs, ignore_index=True)
    identity_error = np.abs(
        scored["predicted_net"]
        - scored["predicted_offense"]
        - scored["predicted_defense"]
    )
    if float(identity_error.max()) > 1e-10:
        raise AssertionError("Predictive trajectory components must sum to net.")
    return scored, score_predictions(scored)


def build_predictive_trajectory_ablation(
    contract_path: str | Path,
    predictions_path: str | Path,
    player_sheets_dir: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    contract_file = Path(contract_path)
    contract = _load_contract(contract_file)
    development = tuple(int(value) for value in contract["development_seasons"])
    diagnostics = tuple(int(value) for value in contract["reused_diagnostic_seasons"])
    scored_seasons = development + diagnostics

    predictions = pd.read_parquet(predictions_path)
    origin_seasons = tuple(
        sorted({int(value) - 1 for value in predictions["Target_Season"].unique()})
    )
    target_seasons = tuple(
        sorted(int(value) for value in predictions["Target_Season"].unique())
    )
    metadata_seasons = tuple(sorted(set(origin_seasons) | set(target_seasons)))
    if max(metadata_seasons) > 2026:
        raise ValueError("Season 2027 metadata is forbidden.")
    metadata = _load_metadata(Path(player_sheets_dir), metadata_seasons)
    rows, quality = build_model_rows(
        predictions, metadata, scored_seasons=scored_seasons
    )
    scored, metrics = run_walk_forward(
        rows,
        scored_seasons=scored_seasons,
        alpha=float(contract["ridge_alpha"]),
    )
    development_net = metrics.loc[
        metrics["target_season"].isin(development) & metrics["component"].eq("net")
    ]
    summary = (
        development_net.groupby("method", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_correlation=("weighted_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_dispersion_ratio=("dispersion_ratio", "mean"),
            folds=("target_season", "nunique"),
        )
        .sort_values(["mean_weighted_rmse", "method"], kind="stable")
        .reset_index(drop=True)
    )
    selected_method = str(summary.iloc[0]["method"])
    selected = scored.loc[scored["method"].eq(selected_method)].copy()
    selected["evidence_status"] = np.where(
        selected["Target_Season"].isin(development),
        "development_reused",
        "diagnostic_reused",
    )

    source_hashes = {
        "contract": sha256_file(contract_file),
        "predictions": sha256_file(predictions_path),
        "source_code": sha256_file(Path(__file__)),
        "player_sheets": {
            str(season): sha256_file(Path(player_sheets_dir) / f"{season}.csv")
            for season in metadata_seasons
        },
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"predictive_spm_trajectory_ablation_v1_{identity}"
    output = (
        Path(artifact_root) / "models" / "predictive_spm_trajectory_ablation" / run_id
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(metrics, output / "fold_metrics.parquet")
    _atomic_parquet(summary, output / "development_summary.parquet")
    _atomic_parquet(selected, output / "selected_predictions.parquet")
    _atomic_parquet(
        scored.loc[scored["Target_Season"].eq(2026)].copy(),
        output / "all_2026_predictions.parquet",
    )
    run = {
        "run_id": run_id,
        "experiment_id": contract["experiment_id"],
        "estimand_id": contract["estimand_id"],
        "status": "research_ablation_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_method": selected_method,
        "development_seasons": list(development),
        "reused_diagnostic_seasons": list(diagnostics),
        "methods": list(METHODS),
        "ridge_alpha": float(contract["ridge_alpha"]),
        "source_hashes": source_hashes,
        "quality": {
            **quality,
            "scored_rows": int(len(scored)),
            "selected_rows": int(len(selected)),
            "maximum_component_identity_error": float(
                np.abs(
                    selected["predicted_net"]
                    - selected["predicted_offense"]
                    - selected["predicted_defense"]
                ).max()
            ),
            "role_stratified_status": "not_run_no_frozen_time_safe_role_panel_through_2025",
        },
        "paths": {
            "fold_metrics": "fold_metrics.parquet",
            "development_summary": "development_summary.parquet",
            "selected_predictions": "selected_predictions.parquet",
            "all_2026_predictions": "all_2026_predictions.parquet",
        },
        "forbidden_interpretation": (
            "This is a reused next-season player-impact diagnostic, not an untouched "
            "confirmation, availability model, or retrospective season rating."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run
