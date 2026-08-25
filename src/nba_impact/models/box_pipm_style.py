"""Transparent BoxPIPM-style annual RAPM baseline.

This is deliberately *not* named PIPM.  The published PIPM combined a box
prior with luck-adjusted on/off information and non-public historical details.
This module isolates the reproducible part: a linear, box-score-only prior
trained against the same annual zero-prior RAPM labels as SPM.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import BOX_FEATURES, _metrics


BOX_PIPM_STYLE_FEATURES = BOX_FEATURES


def _model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _fit(
    frame: pd.DataFrame, features: tuple[str, ...], target: str, alpha: float
) -> Pipeline:
    model = _model(alpha)
    model.fit(
        frame.loc[:, features],
        frame[target],
        ridge__sample_weight=frame["sample_weight"],
    )
    return model


def _select_alpha(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha_grid: tuple[float, ...],
) -> float:
    """Select ridge strength only inside the held-out-season training panel."""
    seasons = tuple(sorted(int(value) for value in train["Season"].unique()))
    if len(seasons) < 3:
        raise ValueError(
            "BoxPIPM-style alpha selection requires at least three seasons."
        )
    scores: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        fold_scores = []
        for validation_season in seasons:
            inner_train = train.loc[train["Season"].ne(validation_season)]
            validation = train.loc[train["Season"].eq(validation_season)]
            prediction = _fit(inner_train, features, target, alpha).predict(
                validation.loc[:, features]
            )
            fold_scores.append(
                _metrics(
                    validation[target].to_numpy(),
                    prediction,
                    validation["sample_weight"].to_numpy(),
                )["weighted_rmse"]
            )
        scores.append((float(np.mean(fold_scores)), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def build_box_pipm_style_baseline(
    features_path: str | Path,
    targets_path: str | Path,
    *,
    artifact_root: str | Path,
    output_seasons: tuple[int, ...] = tuple(range(2017, 2025)),
    alpha_grid: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
) -> dict:
    """Build a leave-one-season-out, box-score-only RAPM baseline.

    The fixed feature list contains only 15 traditional per-100 box rates. It
    excludes position, age, minutes, games, on/off, team ratings, tracking, and
    external metrics.  Reliability enters only through sample weights.
    """
    if not output_seasons:
        raise ValueError("At least one output season is required.")
    if not alpha_grid or any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("BoxPIPM-style ridge alphas must be positive.")
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    missing = sorted(set(BOX_PIPM_STYLE_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"BoxPIPM-style features are missing: {missing}.")
    panel = features.merge(
        targets, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("BoxPIPM-style panel has duplicate player-season keys.")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for season in output_seasons:
        train = panel.loc[panel["Season"].ne(season)].copy()
        test = panel.loc[panel["Season"].eq(season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"BoxPIPM-style fold {season} has an empty partition.")
        fold = test[
            [
                "PLAYER_ID",
                "Season",
                "target_offense",
                "target_defense",
                "target_net",
                "Poss_Off",
                "Poss_Def",
                "sample_weight",
            ]
        ].copy()
        for side in ("offense", "defense"):
            target = f"target_{side}"
            alpha = _select_alpha(train, BOX_PIPM_STYLE_FEATURES, target, alpha_grid)
            prediction = _fit(train, BOX_PIPM_STYLE_FEATURES, target, alpha).predict(
                test.loc[:, BOX_PIPM_STYLE_FEATURES]
            )
            fold[f"box_pipm_style_{side}"] = prediction
            metric_rows.append(
                {
                    "test_season": int(season),
                    "component": side,
                    "selected_alpha": alpha,
                    "train_seasons": int(train["Season"].nunique()),
                    "test_rows": len(test),
                    **_metrics(
                        test[target].to_numpy(),
                        prediction,
                        test["sample_weight"].to_numpy(),
                    ),
                }
            )
        fold["box_pipm_style_net"] = (
            fold["box_pipm_style_offense"] + fold["box_pipm_style_defense"]
        )
        metric_rows.append(
            {
                "test_season": int(season),
                "component": "net",
                "selected_alpha": None,
                "train_seasons": int(train["Season"].nunique()),
                "test_rows": len(test),
                **_metrics(
                    test["target_net"].to_numpy(),
                    fold["box_pipm_style_net"].to_numpy(),
                    test["sample_weight"].to_numpy(),
                ),
            }
        )
        rows.append(fold)
    predictions = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("component", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values("component")
    )
    config = {
        "output_seasons": list(output_seasons),
        "features": list(BOX_PIPM_STYLE_FEATURES),
        "alpha_grid": list(alpha_grid),
        "evaluation": "leave-one-season-out; alpha selected only within the training seasons",
        "features_sha256": sha256_file(features_path),
        "targets_sha256": sha256_file(targets_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(config, sort_keys=True)).hex[
        :10
    ]
    output = (
        Path(artifact_root)
        / "models"
        / "box_pipm_style"
        / f"box_pipm_style_v1_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    run = {
        "run_id": output.name,
        "model_family": "box_pipm_style_linear_ridge",
        "status": "research_baseline",
        "estimand": "annual zero-prior RAPM via a transparent box-score-only prior",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "oof_rows": len(predictions),
            "duplicate_oof_keys": int(
                predictions.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
            "nonfinite_predictions": int(
                (
                    ~np.isfinite(
                        predictions[
                            [
                                "box_pipm_style_offense",
                                "box_pipm_style_defense",
                                "box_pipm_style_net",
                            ]
                        ]
                    )
                )
                .sum()
                .sum()
            ),
        },
        "metrics": summary.to_dict(orient="records"),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "This is not a replication of full PIPM.",
            "Full PIPM used luck-adjusted on/off information and non-public historical details.",
            "This baseline intentionally excludes on/off, team ratings, position, minutes, games, age, and tracking data.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
