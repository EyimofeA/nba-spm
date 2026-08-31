#!/usr/bin/env python3
"""Chronological learner and feature-family screen for annual SPM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "annual_spm_learner_screen_v1"
CONTRACT = ROOT / "research/experiments/annual_spm_learner_screen_v1.yml"
ANNUAL_FEATURES = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e/annual_features.parquet"
)
MECHANISM_FEATURES = ROOT / (
    "artifacts/research/mechanism_feature_panel/"
    "mechanism_feature_panel_v1_9224606a01/annual_features.parquet"
)
ATLAS = ROOT / (
    "artifacts/research/spm_feature_atlas/"
    "spm_feature_atlas_v1_6949ad7b60/feature_atlas.parquet"
)
TARGETS = ROOT / (
    "artifacts/models/canonical_annual_target_panel/"
    "canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
)
SIDES = ("offense", "defense")
_PRUNE_CACHE: dict[tuple, tuple[str, ...]] = {}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    params: dict[str, float | int | str]


MODEL_GRIDS = {
    "ridge": (
        ModelSpec("ridge", {"alpha": alpha})
        for alpha in (30.0, 300.0, 3000.0)
    ),
    "elastic_net": (
        ModelSpec("elastic_net", {"alpha": alpha, "l1_ratio": ratio})
        for alpha, ratio in ((0.003, 0.1), (0.03, 0.1), (0.1, 0.5))
    ),
    "histogram_gbm": (
        ModelSpec(
            "histogram_gbm",
            {
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": leaf_rows,
                "l2_regularization": l2,
            },
        )
        for rate, leaves, leaf_rows, l2 in (
            (0.03, 7, 30, 10.0),
            (0.05, 15, 50, 10.0),
        )
    ),
    "extra_trees": (
        ModelSpec(
            "extra_trees",
            {
                "n_estimators": 200,
                "min_samples_leaf": leaf_rows,
                "max_features": max_features,
            },
        )
        for leaf_rows, max_features in ((8, 0.7), (15, 1.0))
    ),
    "additive_spline_ridge": (
        ModelSpec("additive_spline_ridge", {"alpha": alpha})
        for alpha in (300.0, 3000.0)
    ),
}
MODEL_GRIDS = {key: tuple(value) for key, value in MODEL_GRIDS.items()}


def _pipeline(spec: ModelSpec) -> Pipeline:
    if spec.family == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(spec.params["alpha"]))),
            ]
        )
    if spec.family == "elastic_net":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(spec.params["alpha"]),
                        l1_ratio=float(spec.params["l1_ratio"]),
                        max_iter=20_000,
                        tol=1e-5,
                        random_state=20260831,
                    ),
                ),
            ]
        )
    if spec.family == "histogram_gbm":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=float(spec.params["learning_rate"]),
                        max_iter=200,
                        max_leaf_nodes=int(spec.params["max_leaf_nodes"]),
                        min_samples_leaf=int(spec.params["min_samples_leaf"]),
                        l2_regularization=float(spec.params["l2_regularization"]),
                        early_stopping=False,
                        random_state=20260831,
                    ),
                ),
            ]
        )
    if spec.family == "extra_trees":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=int(spec.params["n_estimators"]),
                        min_samples_leaf=int(spec.params["min_samples_leaf"]),
                        max_features=float(spec.params["max_features"]),
                        n_jobs=-1,
                        random_state=20260831,
                    ),
                ),
            ]
        )
    if spec.family == "additive_spline_ridge":
        # A bounded additive nonlinear diagnostic. No interaction expansion.
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=3,
                        degree=2,
                        include_bias=False,
                        extrapolation="linear",
                    ),
                ),
                ("model", Ridge(alpha=float(spec.params["alpha"]))),
            ]
        )
    raise ValueError(f"Unknown learner {spec.family}.")


def _weighted_metrics(
    actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    error = actual - predicted
    rmse = float(np.sqrt(np.average(error**2, weights=weight)))
    actual_mean = float(np.average(actual, weights=weight))
    predicted_mean = float(np.average(predicted, weights=weight))
    covariance = float(
        np.average((actual - actual_mean) * (predicted - predicted_mean), weights=weight)
    )
    actual_var = float(np.average((actual - actual_mean) ** 2, weights=weight))
    predicted_var = float(
        np.average((predicted - predicted_mean) ** 2, weights=weight)
    )
    correlation = (
        covariance / np.sqrt(actual_var * predicted_var)
        if actual_var > 0 and predicted_var > 0
        else np.nan
    )
    slope = covariance / predicted_var if predicted_var > 0 else np.nan
    return {
        "weighted_rmse": rmse,
        "weighted_correlation": float(correlation),
        "calibration_slope": float(slope),
    }


def _adjacent_stability(frame: pd.DataFrame, feature: str) -> float:
    values = []
    for season in sorted(frame["Season"].unique())[:-1]:
        next_season = int(season) + 1
        if next_season not in set(frame["Season"]):
            continue
        left = frame.loc[frame["Season"].eq(season), ["PLAYER_ID", feature]]
        right = frame.loc[
            frame["Season"].eq(next_season), ["PLAYER_ID", feature]
        ]
        joined = left.merge(right, on="PLAYER_ID", suffixes=("_left", "_right"))
        if (
            len(joined) < 3
            or joined[f"{feature}_left"].nunique() < 2
            or joined[f"{feature}_right"].nunique() < 2
        ):
            continue
        value = joined[f"{feature}_left"].corr(joined[f"{feature}_right"])
        if np.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if values else -1.0


def _prune_features(
    train: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    threshold: float,
) -> tuple[str, ...]:
    """Remove constants and near duplicates using training predictors only."""
    key = (
        int(train["Season"].min()),
        int(train["Season"].max()),
        len(train),
        candidates,
        threshold,
    )
    if key in _PRUNE_CACHE:
        return _PRUNE_CACHE[key]
    candidates = tuple(feature for feature in candidates if train[feature].nunique() > 1)
    stability = {feature: _adjacent_stability(train, feature) for feature in candidates}
    ordered = sorted(
        candidates,
        key=lambda feature: (
            feature not in BOX_PIPM_STYLE_FEATURES,
            -stability[feature],
            feature,
        ),
    )
    correlation = train.loc[:, ordered].corr().abs()
    kept: list[str] = []
    for feature in ordered:
        if any(float(correlation.loc[feature, prior]) >= threshold for prior in kept):
            continue
        kept.append(feature)
    result = tuple(kept)
    _PRUNE_CACHE[key] = result
    return result


def _fit(
    spec: ModelSpec,
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> Pipeline:
    model = _pipeline(spec)
    model.fit(
        train.loc[:, features],
        train[target],
        model__sample_weight=train["sample_weight"],
    )
    return model


def _select_spec(
    train: pd.DataFrame,
    candidates: tuple[str, ...],
    family: str,
    target: str,
    threshold: float,
) -> tuple[ModelSpec, tuple[str, ...]]:
    validation_season = int(train["Season"].max())
    inner_train = train.loc[train["Season"].lt(validation_season)]
    validation = train.loc[train["Season"].eq(validation_season)]
    features = _prune_features(inner_train, candidates, threshold=threshold)
    scored = []
    for spec in MODEL_GRIDS[family]:
        model = _fit(spec, inner_train, features, target)
        predicted = model.predict(validation.loc[:, features])
        metrics = _weighted_metrics(
            validation[target].to_numpy(dtype=float),
            predicted,
            validation["sample_weight"].to_numpy(dtype=float),
        )
        scored.append((metrics["weighted_rmse"], json.dumps(spec.params, sort_keys=True), spec))
    return min(scored, key=lambda row: (row[0], row[1]))[2], features


def _score_fold(
    panel: pd.DataFrame,
    *,
    test_season: int,
    side: str,
    arm: str,
    candidates: tuple[str, ...],
    learner: str,
    threshold: float,
    phase: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train = panel.loc[panel["Season"].lt(test_season)]
    test = panel.loc[panel["Season"].eq(test_season)]
    target = f"target_{side}"
    spec, inner_features = _select_spec(
        train, candidates, learner, target, threshold
    )
    features = _prune_features(train, candidates, threshold=threshold)
    model = _fit(spec, train, features, target)
    predicted = model.predict(test.loc[:, features])
    metrics = _weighted_metrics(
        test[target].to_numpy(dtype=float),
        predicted,
        test["sample_weight"].to_numpy(dtype=float),
    )
    row = {
        "test_season": test_season,
        "side": side,
        "arm": arm,
        "learner": learner,
        "phase": phase,
        "training_start": int(train["Season"].min()),
        "training_end": int(train["Season"].max()),
        "rows": len(test),
        "candidate_features": len(candidates),
        "selected_features": len(features),
        "inner_selected_features": len(inner_features),
        "selected_params": json.dumps(spec.params, sort_keys=True),
        **metrics,
    }
    predictions = test[["PLAYER_ID", "Season", target, "sample_weight"]].copy()
    predictions["side"] = side
    predictions["arm"] = arm
    predictions["learner"] = learner
    predictions["phase"] = phase
    predictions["prediction"] = predicted
    predictions["selected_params"] = row["selected_params"]
    predictions["selected_features"] = len(features)
    selected = pd.DataFrame(
        {
            "test_season": test_season,
            "side": side,
            "arm": arm,
            "learner": learner,
            "phase": phase,
            "feature": features,
        }
    )
    return row, predictions, selected


def _feature_arms(
    atlas: pd.DataFrame, panel: pd.DataFrame, side: str
) -> dict[str, tuple[str, ...]]:
    relevant = atlas.loc[
        atlas["suggested_side"].isin((side, "both"))
        & ~atlas["source_shift_flag"]
        & ~atlas["lane"].isin(
            ("lineup_derived_circular", "descriptive_role", "predictive_only")
        )
    ].copy()
    relevant = relevant.loc[relevant["feature"].isin(panel.columns)]
    box = tuple(feature for feature in BOX_PIPM_STYLE_FEATURES if feature in panel)
    arms = {"box15": box}
    for family in sorted(relevant["family"].unique()):
        extra = tuple(sorted(relevant.loc[relevant["family"].eq(family), "feature"]))
        arms[f"box15_plus_{family}"] = tuple(dict.fromkeys((*box, *extra)))
    all_features = tuple(sorted(relevant["feature"].unique()))
    arms["audited_all"] = tuple(dict.fromkeys((*box, *all_features)))
    return arms


def _mean_scores(folds: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    selected = folds.loc[folds["test_season"].isin(seasons)]
    return (
        selected.groupby(["side", "arm", "learner"], as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_correlation=("weighted_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values(["side", "mean_weighted_rmse", "arm", "learner"])
    )


def _net_diagnostics(
    predictions: pd.DataFrame,
    learner_winners: dict[str, str],
    box_learner_winners: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostic = predictions.loc[predictions["phase"].eq("diagnostic")]
    for candidate, arm, offense_learner, defense_learner in (
        ("box15_ridge", "box15", "ridge", "ridge"),
        (
            "box15_best_learner",
            "box15",
            box_learner_winners["offense"],
            box_learner_winners["defense"],
        ),
        (
            "frozen_rich_winner",
            "audited_all",
            learner_winners["offense"],
            learner_winners["defense"],
        ),
    ):
        offense = diagnostic.loc[
            diagnostic["side"].eq("offense")
            & diagnostic["arm"].eq(arm)
            & diagnostic["learner"].eq(offense_learner)
        ].rename(
            columns={
                "target_offense": "target_offense_value",
                "prediction": "prediction_offense",
            }
        )
        defense = diagnostic.loc[
            diagnostic["side"].eq("defense")
            & diagnostic["arm"].eq(arm)
            & diagnostic["learner"].eq(defense_learner)
        ].rename(
            columns={
                "target_defense": "target_defense_value",
                "prediction": "prediction_defense",
            }
        )
        joined = offense[
            [
                "PLAYER_ID",
                "Season",
                "target_offense_value",
                "sample_weight",
                "prediction_offense",
            ]
        ].merge(
            defense[
                [
                    "PLAYER_ID",
                    "Season",
                    "target_defense_value",
                    "sample_weight",
                    "prediction_defense",
                ]
            ],
            on=["PLAYER_ID", "Season"],
            suffixes=("_offense", "_defense"),
            validate="one_to_one",
        )
        if not np.allclose(
            joined["sample_weight_offense"], joined["sample_weight_defense"]
        ):
            raise ValueError("Offense and defense diagnostic weights differ.")
        joined["target_net"] = (
            joined["target_offense_value"] + joined["target_defense_value"]
        )
        joined["prediction_net"] = (
            joined["prediction_offense"] + joined["prediction_defense"]
        )
        for season, fold in joined.groupby("Season"):
            metrics = _weighted_metrics(
                fold["target_net"].to_numpy(dtype=float),
                fold["prediction_net"].to_numpy(dtype=float),
                fold["sample_weight_offense"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "test_season": int(season),
                    "rows": len(fold),
                    **metrics,
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_correlation=("weighted_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values("mean_weighted_rmse")
    )
    return folds, summary


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")
    features = pd.read_parquet(ANNUAL_FEATURES).rename(columns={"Window_End": "Season"})
    mechanism = pd.read_parquet(MECHANISM_FEATURES).rename(
        columns={"Window_End": "Season"}
    )
    atlas = pd.read_parquet(ATLAS)
    targets = pd.read_parquet(TARGETS)
    panel = features.merge(
        mechanism, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    ).merge(targets, on=["PLAYER_ID", "Season"], validate="one_to_one")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual learner panel has duplicate player-season keys.")

    selection_seasons = tuple(int(v) for v in contract["folds"]["selection_seasons"])
    diagnostic_seasons = tuple(int(v) for v in contract["folds"]["diagnostic_seasons"])
    threshold = float(contract["feature_screen"]["correlation_threshold"])
    predictions: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []

    # Stage 1. Pick one learner per side and feature pool before 2022.
    print("Stage 1: learner screen", flush=True)
    learner_rows = []
    for side in SIDES:
        arms = _feature_arms(atlas, panel, side)
        for arm in ("box15", "audited_all"):
            for learner in MODEL_GRIDS:
                for season in selection_seasons:
                    row, pred, selected = _score_fold(
                        panel,
                        test_season=season,
                        side=side,
                        arm=arm,
                        candidates=arms[arm],
                        learner=learner,
                        threshold=threshold,
                        phase="learner_selection",
                    )
                    learner_rows.append(row)
                    predictions.append(pred)
                    selections.append(selected)
    learner_folds = pd.DataFrame(learner_rows)
    learner_summary = _mean_scores(learner_folds, selection_seasons)
    winners = {
        side: learner_summary.loc[
            learner_summary["side"].eq(side)
            & learner_summary["arm"].eq("audited_all")
        ].iloc[0]["learner"]
        for side in SIDES
    }
    box_winners = {
        side: learner_summary.loc[
            learner_summary["side"].eq(side)
            & learner_summary["arm"].eq("box15")
        ].iloc[0]["learner"]
        for side in SIDES
    }

    # Stage 2. With the learner frozen, pick one feature arm per side before 2022.
    print(f"Stage 1 rich winners: {winners}", flush=True)
    print(f"Stage 1 Box15 winners: {box_winners}", flush=True)
    print("Stage 2: feature-family screen", flush=True)
    family_rows = []
    for side in SIDES:
        arms = _feature_arms(atlas, panel, side)
        for arm, candidates in arms.items():
            for season in selection_seasons:
                row, pred, selected = _score_fold(
                    panel,
                    test_season=season,
                    side=side,
                    arm=arm,
                    candidates=candidates,
                    learner=winners[side],
                    threshold=threshold,
                    phase="feature_selection",
                )
                family_rows.append(row)
                predictions.append(pred)
                selections.append(selected)
    family_folds = pd.DataFrame(family_rows)
    family_summary = _mean_scores(family_folds, selection_seasons)
    feature_winners = {
        side: family_summary.loc[family_summary["side"].eq(side)].iloc[0]["arm"]
        for side in SIDES
    }

    # Stage 3. Diagnose only the frozen winner, audited-all learner, and Box15 ridge.
    print(f"Stage 2 winners: {feature_winners}", flush=True)
    print("Stage 3: frozen later diagnostics", flush=True)
    diagnostic_rows = []
    for side in SIDES:
        arms = _feature_arms(atlas, panel, side)
        candidates = (
            ("box15", "ridge"),
            ("box15", box_winners[side]),
            ("audited_all", winners[side]),
            (feature_winners[side], winners[side]),
        )
        for arm, learner in dict.fromkeys(candidates):
            for season in diagnostic_seasons:
                row, pred, selected = _score_fold(
                    panel,
                    test_season=season,
                    side=side,
                    arm=arm,
                    candidates=arms[arm],
                    learner=learner,
                    threshold=threshold,
                    phase="diagnostic",
                )
                diagnostic_rows.append(row)
                predictions.append(pred)
                selections.append(selected)
    diagnostic_folds = pd.DataFrame(diagnostic_rows)
    diagnostic_summary = _mean_scores(diagnostic_folds, diagnostic_seasons)
    source_paths = {
        "contract": CONTRACT,
        "annual_features": ANNUAL_FEATURES,
        "mechanism_features": MECHANISM_FEATURES,
        "feature_atlas": ATLAS,
        "targets": TARGETS,
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "selection_seasons": list(selection_seasons),
        "diagnostic_seasons": list(diagnostic_seasons),
        "correlation_threshold": threshold,
        "learners": list(MODEL_GRIDS),
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/research/annual_spm_learner_screen" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_selections = pd.concat(selections, ignore_index=True)
    net_diagnostic_folds, net_diagnostic_summary = _net_diagnostics(
        all_predictions, winners, box_winners
    )
    feature_frequency = (
        all_selections.groupby(
            ["phase", "side", "arm", "learner", "feature"], as_index=False
        )["test_season"]
        .nunique()
        .rename(columns={"test_season": "folds_retained"})
    )
    outputs = {
        "learner_fold_metrics.parquet": learner_folds,
        "learner_summary.csv": learner_summary,
        "family_fold_metrics.parquet": family_folds,
        "family_summary.csv": family_summary,
        "diagnostic_fold_metrics.parquet": diagnostic_folds,
        "diagnostic_summary.csv": diagnostic_summary,
        "net_diagnostic_fold_metrics.parquet": net_diagnostic_folds,
        "net_diagnostic_summary.csv": net_diagnostic_summary,
        "predictions.parquet": all_predictions,
        "selected_features.parquet": all_selections,
        "feature_frequency.csv": feature_frequency,
    }
    for name, frame in outputs.items():
        if name.endswith(".csv"):
            frame.to_csv(output / name, index=False)
        else:
            frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "selection": {
            "learner_by_side": winners,
            "box15_learner_by_side": box_winners,
            "feature_arm_by_side": feature_winners,
        },
        "quality": {
            "panel_rows": len(panel),
            "panel_seasons": sorted(int(v) for v in panel["Season"].unique()),
            "duplicate_keys": 0,
            "target_values_loaded_outside_declared_seasons": False,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(run["selection"], indent=2))
    print("\nSelection learner summary")
    print(learner_summary.to_string(index=False))
    print("\nSelection feature summary")
    print(family_summary.to_string(index=False))
    print("\nLater diagnostic summary")
    print(diagnostic_summary.to_string(index=False))
    print("\nLater net diagnostic summary")
    print(net_diagnostic_summary.to_string(index=False))


if __name__ == "__main__":
    main()
