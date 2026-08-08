"""First chronological statistical model for three-season normal RAPM targets."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic

BOX_FEATURES = (
    "PTS_p100",
    "AST_p100",
    "TOV_p100",
    "STL_p100",
    "BLK_p100",
    "OREB_p100",
    "DREB_p100",
    "PF_p100",
    "PFD_p100",
    "FTA_p100",
    "FTM_p100",
    "FG2A_p100",
    "FG2M_p100",
    "FG3A_p100",
    "FG3M_p100",
)

FORBIDDEN_PRIMARY_FEATURES = {
    "PLAYER_ID",
    "Window_End",
    "MIN",
    "GP",
    "OffPoss",
    "DefPoss",
    "AGE",
    "OnDefRtg",
    "OnOffRtg",
}


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _fit(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha: float,
) -> Pipeline:
    model = _pipeline(alpha)
    model.fit(
        frame.loc[:, features],
        frame[target],
        ridge__sample_weight=frame["sample_weight"],
    )
    return model


def _metrics(actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    return {
        "weighted_rmse": float(
            mean_squared_error(actual, prediction, sample_weight=weight) ** 0.5
        ),
        "correlation": float(np.corrcoef(actual, prediction)[0, 1]),
    }


def _load_panel(features_path: str | Path, targets_path: str | Path) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    targets = pd.read_csv(targets_path)
    keys = ["PLAYER_ID", "Window_End"]
    if features.duplicated(keys).any() or targets.duplicated(keys).any():
        raise ValueError("Statistical-impact input keys must be unique.")
    target_columns = [*keys, "Off", "Def", "Poss_Off", "Poss_Def"]
    panel = features.merge(
        targets[target_columns], on=keys, how="inner", validate="one_to_one"
    )
    panel["target_offense"] = pd.to_numeric(panel["Off"], errors="raise")
    panel["target_defense"] = -pd.to_numeric(panel["Def"], errors="raise")
    panel["target_net"] = panel["target_offense"] + panel["target_defense"]
    reliability = np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    panel["sample_weight"] = np.sqrt(reliability)
    return panel


def run_statistical_impact_baseline(
    features_path: str | Path,
    targets_path: str | Path,
    *,
    artifact_root: str | Path,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    test_window_ends: tuple[int, ...] = (2022, 2023, 2024),
    first_complete_tracking_window: int = 2016,
    target_window_seasons: int = 3,
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
) -> dict:
    """Fit purged chronological ridge baselines for offense and defense."""
    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"] >= first_complete_tracking_window].copy()
    if feature_sets is None:
        advanced = tuple(
            column
            for column in pd.read_parquet(features_path).columns
            if column not in FORBIDDEN_PRIMARY_FEATURES
        )
        feature_sets = {
            "box_rates": BOX_FEATURES,
            "advanced": advanced,
            "advanced_plus_onoff": (*advanced, "OnDefRtg", "OnOffRtg"),
        }
    required_features = {
        feature for values in feature_sets.values() for feature in values
    }
    if missing := sorted(required_features - set(panel.columns)):
        raise ValueError(f"Statistical-impact panel is missing features {missing}.")
    if any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("Ridge alphas must be positive.")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    selected_alphas: dict[tuple[int, str, str], float] = {}
    target_columns = {
        "offense": "target_offense",
        "defense": "target_defense",
    }
    for test_end in test_window_ends:
        train_max = test_end - target_window_seasons
        inner_validation_end = train_max
        inner_train_max = inner_validation_end - target_window_seasons
        outer_train = panel.loc[panel["Window_End"] <= train_max]
        test = panel.loc[panel["Window_End"] == test_end]
        inner_train = outer_train.loc[outer_train["Window_End"] <= inner_train_max]
        inner_validation = outer_train.loc[
            outer_train["Window_End"] == inner_validation_end
        ]
        if (
            min(len(outer_train), len(test), len(inner_train), len(inner_validation))
            == 0
        ):
            raise ValueError(f"Purged fold ending {test_end} has an empty partition.")

        fold_predictions = test[
            [
                "PLAYER_ID",
                "Window_End",
                "target_offense",
                "target_defense",
                "target_net",
                "sample_weight",
            ]
        ].copy()
        for feature_set, columns in feature_sets.items():
            for target_name, target_column in target_columns.items():
                best_alpha = None
                best_rmse = float("inf")
                for alpha in alpha_grid:
                    model = _fit(inner_train, columns, target_column, alpha)
                    prediction = model.predict(inner_validation.loc[:, columns])
                    rmse = _metrics(
                        inner_validation[target_column].to_numpy(),
                        prediction,
                        inner_validation["sample_weight"].to_numpy(),
                    )["weighted_rmse"]
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_alpha = alpha
                if best_alpha is None:
                    raise RuntimeError(
                        "Ridge alpha selection did not produce a candidate."
                    )
                selected_alphas[(test_end, feature_set, target_name)] = best_alpha
                model = _fit(outer_train, columns, target_column, best_alpha)
                prediction = model.predict(test.loc[:, columns])
                fold_predictions[f"prediction_{feature_set}_{target_name}"] = prediction
                metric_rows.append(
                    {
                        "test_window_end": test_end,
                        "feature_set": feature_set,
                        "target": target_name,
                        "alpha": best_alpha,
                        "train_max_window_end": train_max,
                        "test_players": int(len(test)),
                        **_metrics(
                            test[target_column].to_numpy(),
                            prediction,
                            test["sample_weight"].to_numpy(),
                        ),
                    }
                )

            offense_prediction = fold_predictions[f"prediction_{feature_set}_offense"]
            defense_prediction = fold_predictions[f"prediction_{feature_set}_defense"]
            fold_predictions[f"prediction_{feature_set}_net"] = (
                offense_prediction + defense_prediction
            )
            metric_rows.append(
                {
                    "test_window_end": test_end,
                    "feature_set": feature_set,
                    "target": "net",
                    "alpha": np.nan,
                    "train_max_window_end": train_max,
                    "test_players": int(len(test)),
                    **_metrics(
                        test["target_net"].to_numpy(),
                        fold_predictions[f"prediction_{feature_set}_net"].to_numpy(),
                        test["sample_weight"].to_numpy(),
                    ),
                }
            )
        prediction_rows.append(fold_predictions)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    summary = (
        fold_metrics.groupby(["feature_set", "target"], as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_window_end", "nunique"),
        )
        .sort_values(["target", "mean_weighted_rmse"], kind="stable")
    )

    run_id = f"statistical_impact_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_impact" / run_id
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "fold_predictions.parquet", index=False)
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)

    final_models = {}
    for feature_set, columns in feature_sets.items():
        for target_name, target_column in target_columns.items():
            alpha = selected_alphas[(max(test_window_ends), feature_set, target_name)]
            model = _fit(panel, columns, target_column, alpha)
            path = output / f"model_{feature_set}_{target_name}.joblib"
            joblib.dump(model, path)
            final_models[f"{feature_set}_{target_name}"] = {
                "path": str(path.resolve()),
                "alpha": alpha,
            }

    run = {
        "run_id": run_id,
        "model_family": "statistical_impact_ridge",
        "estimand": "three_season_normal_rapm_offense_and_defense",
        "status": "research_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_window_seasons": target_window_seasons,
            "input_window_seasons": 3,
            "first_complete_tracking_window": first_complete_tracking_window,
            "test_window_ends": list(test_window_ends),
            "purge_windows_between_train_and_test": target_window_seasons - 1,
            "alpha_grid": list(alpha_grid),
            "feature_sets": {
                name: list(values) for name, values in feature_sets.items()
            },
            "forbidden_primary_features": sorted(FORBIDDEN_PRIMARY_FEATURES),
            "sample_weight": "sqrt(min(Poss_Off, Poss_Def)); not an input feature",
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "joined_rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "summary": summary.to_dict(orient="records"),
        },
        "models": final_models,
        "caveats": [
            "The historical feature panel ends in 2024 and is not a current-season inference table.",
            "The existing panel minute-weights percentage and average features across seasons; rebuild those features with natural denominators before production.",
            "OnOffRtg and OnDefRtg appear only in the explicitly non-independent challenger.",
            "No uncertainty estimates are produced in this version.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
