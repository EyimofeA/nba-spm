#!/usr/bin/env python3
"""Run a leak-safe Box15 factor-residual AIO tournament."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.historical_factor_features import REBOUND_FEATURES, SHOT_FEATURES
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit as _fit_box, _select_alpha
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _prior_frame

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
BASE_FEATURES = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_8be676bd0f/five_year_features.parquet"
)
SPECIALIST_ROOT = ROOT / "artifacts/research/historical_specialist_features"
FACTOR_TARGETS = (
    ROOT
    / "artifacts/research/historical_factor_targets"
    / "historical_factor_targets_v1_f4894bf588/five_year_factor_targets.parquet"
)
NORMAL_TARGETS = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
OUTPUT_ROOT = ROOT / "artifacts/research/historical_factor_residual_tournament"
PLAYER_SHEETS = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
ROLE_ROOT = ROOT / "artifacts/features/side_roles/side_roles_v1_2c228f4b9e"
RATING_SEASONS = tuple(range(2021, 2027))
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
CANDIDATES = ("box15", "box15_ts", "box15_oreb", "box15_ts_oreb")
MODEL_ORDER = (
    "box15",
    "box15_ts",
    "box15_oreb",
    "box15_ts_oreb",
    "zero_prior_rapm",
    "box15_aio",
    "box15_ts_aio",
    "box15_oreb_aio",
    "box15_ts_oreb_aio",
)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260829


@dataclass(frozen=True)
class ModelSpec:
    family: str
    alpha: float
    l1_ratio: float = 0.0
    max_leaf_nodes: int = 0


FACTOR_SPECS = (
    ModelSpec("ridge", 100.0),
    ModelSpec("elastic_net", 0.01, 0.1),
    ModelSpec("boosted_tree", 10.0, max_leaf_nodes=15),
)
BOX15_ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
RESIDUAL_SPEC = ModelSpec("ridge", 10.0)


def _latest_specialist_run() -> Path:
    runs = [path for path in SPECIALIST_ROOT.glob("historical_specialist_features_v1_*") if (path / "run.json").exists()]
    if not runs:
        raise ValueError("No complete historical specialist feature run exists.")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _pipeline(spec: ModelSpec) -> Pipeline:
    if spec.family == "ridge":
        model = Ridge(alpha=spec.alpha)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    if spec.family == "elastic_net":
        model = ElasticNet(alpha=spec.alpha, l1_ratio=spec.l1_ratio, max_iter=20_000, random_state=20260829)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=spec.max_leaf_nodes,
        l2_regularization=spec.alpha,
        random_state=20260829,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def _fit(spec: ModelSpec, frame: pd.DataFrame, features: tuple[str, ...], target: str, weight: str) -> Pipeline:
    frame = frame.loc[frame[target].notna() & frame[weight].notna() & frame[weight].gt(0)]
    if frame.empty:
        raise ValueError(f"No observed rows remain for {target}.")
    model = _pipeline(spec)
    model.fit(frame.loc[:, features], frame[target], model__sample_weight=frame[weight])
    return model


def _chronological_folds(frame: pd.DataFrame, minimum_history: int = 2) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    seasons = sorted(frame["Window_End"].unique())
    folds = []
    for index, season in enumerate(seasons):
        if index < minimum_history:
            continue
        folds.append((frame.loc[frame["Window_End"].lt(season)], frame.loc[frame["Window_End"].eq(season)]))
    return folds


def _select_spec(frame: pd.DataFrame, features: tuple[str, ...], target: str, weight: str, specs: tuple[ModelSpec, ...], *, minimum_history: int = 2) -> tuple[ModelSpec, pd.DataFrame]:
    folds = _chronological_folds(frame, minimum_history=minimum_history)
    if not folds:
        raise ValueError("Chronological model selection has no valid folds.")
    rows = []
    for spec in specs:
        fold_mse = []
        for inner, validation in folds:
            prediction = _fit(spec, inner, features, target, weight).predict(validation.loc[:, features])
            observed = validation[target].notna() & validation[weight].notna() & validation[weight].gt(0)
            fold_mse.append(float(np.average(np.square(validation.loc[observed, target] - prediction[observed]), weights=validation.loc[observed, weight])))
        rows.append({**spec.__dict__, "equal_window_mse": float(np.mean(fold_mse)), "folds": len(fold_mse)})
    grid = pd.DataFrame(rows).sort_values(["equal_window_mse", "family", "alpha", "l1_ratio", "max_leaf_nodes"], kind="stable")
    best = grid.iloc[0]
    selected = ModelSpec(str(best.family), float(best.alpha), float(best.l1_ratio), int(best.max_leaf_nodes))
    grid["selected"] = False
    grid.loc[grid.index[0], "selected"] = True
    return selected, grid


def _loso_oof(frame: pd.DataFrame, features: tuple[str, ...], target: str, weight: str, spec: ModelSpec) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for season in sorted(frame["Window_End"].unique()):
        inner = frame.loc[frame["Window_End"].ne(season)]
        validation = frame.loc[frame["Window_End"].eq(season)]
        result.loc[validation.index] = _fit(spec, inner, features, target, weight).predict(validation.loc[:, features])
    return result


def _box15_fit(
    frame: pd.DataFrame,
    target: str,
    weight: str = "normal_weight",
) -> tuple[Pipeline, float]:
    train = frame.rename(columns={"Window_End": "Season"}).copy()
    train = train.loc[train[target].notna() & train[weight].notna() & train[weight].gt(0)]
    train["sample_weight"] = train[weight]
    alpha = _select_alpha(train, BOX_PIPM_STYLE_FEATURES, target, BOX15_ALPHA_GRID)
    return _fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha), alpha


def _box15_oof(frame: pd.DataFrame, target: str, alpha: float) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for season in sorted(frame["Window_End"].unique()):
        inner = frame.loc[frame["Window_End"].ne(season)].rename(columns={"Window_End": "Season"}).copy()
        inner["sample_weight"] = inner["normal_weight"]
        validation = frame.loc[frame["Window_End"].eq(season)]
        model = _fit_box(inner, BOX_PIPM_STYLE_FEATURES, target, alpha)
        result.loc[validation.index] = model.predict(validation.loc[:, BOX_PIPM_STYLE_FEATURES])
    return result


def _feature_contract(panel: pd.DataFrame) -> dict[tuple[str, str], tuple[str, ...]]:
    shot_offense = tuple(feature for feature in SHOT_FEATURES if not feature.startswith("defender_") and feature != "rim_deterrence_vs_scorer_p100_eb" and feature != "has_assigned_shot_defense")
    shot_defense = tuple(feature for feature in SHOT_FEATURES if feature.startswith("defender_") or feature in {"rim_deterrence_vs_scorer_p100_eb", "has_assigned_shot_defense"})
    oreb_offense = tuple(feature for feature in REBOUND_FEATURES if "oreb" in feature and "dreb" not in feature) + ("OREB_p100",)
    oreb_defense = tuple(feature for feature in REBOUND_FEATURES if "dreb" in feature or "boxout" in feature or feature in {"player_height_inches", "rebound_conversion_above_expected_eb", "has_boxout_tracking"}) + ("DREB_p100", "dreb_contests_p100", "dreb_chances_p100")
    contract = {
        ("shooting_ts", "offense"): tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *shot_offense))),
        ("shooting_ts", "defense"): tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *shot_defense))),
        ("opponent_oreb_prevention", "offense"): tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *oreb_offense))),
        ("opponent_oreb_prevention", "defense"): tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *oreb_defense))),
    }
    missing = sorted({feature for features in contract.values() for feature in features} - set(panel.columns))
    if missing:
        raise ValueError(f"Specialist feature contract is missing {missing}.")
    return contract


def _load_panel() -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[str, ...]], Path]:
    specialist_run = _latest_specialist_run()
    base_features = pd.read_parquet(BASE_FEATURES)
    specialist = pd.read_parquet(specialist_run / "five_year_features.parquet")
    normal = pd.read_parquet(NORMAL_TARGETS)
    factors = pd.read_parquet(FACTOR_TARGETS).rename(columns={"player_id": "PLAYER_ID"})
    normal_fields = ["PLAYER_ID", "Window_End", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    panel = (
        base_features.merge(specialist, on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one")
        .merge(normal[normal_fields], on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one")
        .merge(factors.drop(columns=["PLAYER_NAME", "Window_Start"]), on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one")
    )
    panel = panel.loc[panel["Window_End"].between(2018, 2026)].copy()
    panel["normal_weight"] = np.sqrt(np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1))
    panel["shooting_ts_offense_weight"] = np.sqrt(panel["shooting_ts_off_exposure"].clip(lower=1))
    panel["shooting_ts_defense_weight"] = np.sqrt(panel["shooting_ts_def_exposure"].clip(lower=1))
    panel["opponent_oreb_prevention_offense_weight"] = np.sqrt(panel["opponent_oreb_prevention_off_exposure"].clip(lower=1))
    panel["opponent_oreb_prevention_defense_weight"] = np.sqrt(panel["opponent_oreb_prevention_def_exposure"].clip(lower=1))
    if panel["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered the factor tournament.")
    return panel, _feature_contract(panel), specialist_run


def _fit_priors(
    panel: pd.DataFrame,
    contract: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_rows = []
    selection_rows = []
    factor_metric_rows = []
    factor_prediction_rows = []
    for rating_season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
        test = panel.loc[panel["Window_End"].eq(rating_season)].copy()
        base_test: dict[str, np.ndarray] = {}
        base_oof: dict[str, pd.Series] = {}
        factor_test: dict[tuple[str, str], np.ndarray] = {}
        factor_oof: dict[tuple[str, str], pd.Series] = {}
        for side in ("offense", "defense"):
            target = f"target_{side}"
            model, alpha = _box15_fit(train, target)
            base_test[side] = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
            base_oof[side] = _box15_oof(train, target, alpha)
            selection_rows.append(pd.DataFrame([{
                "family": "ridge", "alpha": alpha, "l1_ratio": 0.0,
                "max_leaf_nodes": 0, "equal_window_mse": np.nan, "folds": np.nan,
                "selected": True, "rating_season": rating_season,
                "stage": "box15_normal", "component": side,
            }]))
        for factor in ("shooting_ts", "opponent_oreb_prevention"):
            for side in ("offense", "defense"):
                target = f"{factor}_{side}"
                weight = f"{factor}_{side}_weight"
                features = contract[(factor, side)]
                spec, grid = _select_spec(train, features, target, weight, FACTOR_SPECS)
                model = _fit(spec, train, features, target, weight)
                prediction = model.predict(test.loc[:, features])
                box_model, box_alpha = _box15_fit(train, target, weight)
                box_prediction = box_model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
                factor_test[(factor, side)] = prediction
                factor_oof[(factor, side)] = _loso_oof(train, features, target, weight, spec)
                observed = test[target].notna() & test[weight].notna() & test[weight].gt(0)
                for candidate, candidate_prediction, description in (
                    ("box15_factor", box_prediction, ModelSpec("ridge", box_alpha)),
                    ("specialist_factor", prediction, spec),
                ):
                    metric = _metrics(test.loc[observed, target].to_numpy(), candidate_prediction[observed], test.loc[observed, weight].to_numpy())
                    factor_metric_rows.append({"rating_season": rating_season, "factor": factor, "component": side, "candidate": candidate, **description.__dict__, "r2": float(r2_score(test.loc[observed, target], candidate_prediction[observed], sample_weight=test.loc[observed, weight])), **metric})
                    predicted = test.loc[observed, ["PLAYER_ID", "Window_End", target, weight]].copy()
                    predicted["factor"] = factor
                    predicted["component"] = side
                    predicted["candidate"] = candidate
                    predicted["prediction"] = candidate_prediction[observed]
                    predicted["residual"] = predicted[target] - predicted["prediction"]
                    predicted["squared_error"] = predicted["residual"].pow(2)
                    factor_prediction_rows.append(predicted.rename(columns={target: "target", weight: "weight"}))
                grid["rating_season"] = rating_season; grid["stage"] = factor; grid["component"] = side
                selection_rows.append(grid)

        outputs = {candidate: test[["PLAYER_ID", "Window_End"]].copy() for candidate in CANDIDATES}
        for side in ("offense", "defense"):
            outputs["box15"][side] = base_test[side]
            for candidate, factors in {
                "box15_ts": ("shooting_ts",),
                "box15_oreb": ("opponent_oreb_prevention",),
                "box15_ts_oreb": ("shooting_ts", "opponent_oreb_prevention"),
            }.items():
                signal_names = [f"signal_{factor}" for factor in factors]
                residual = train[["Window_End", f"target_{side}", "normal_weight"]].copy()
                residual["base"] = base_oof[side]
                for name, factor in zip(signal_names, factors, strict=True):
                    residual[name] = factor_oof[(factor, side)]
                residual = residual.dropna()
                residual["residual_target"] = residual[f"target_{side}"] - residual["base"]
                residual_model = _fit(RESIDUAL_SPEC, residual, tuple(signal_names), "residual_target", "normal_weight")
                test_signals = pd.DataFrame({name: factor_test[(factor, side)] for name, factor in zip(signal_names, factors, strict=True)})
                outputs[candidate][side] = base_test[side] + residual_model.predict(test_signals)
                selection_rows.append(pd.DataFrame([{
                    **RESIDUAL_SPEC.__dict__, "equal_window_mse": np.nan,
                    "folds": int(residual["Window_End"].nunique()), "selected": True,
                    "rating_season": rating_season, "stage": candidate,
                    "component": side,
                }]))
        for candidate, prior in outputs.items():
            prior["net"] = prior["offense"] + prior["defense"]
            prior_rows.append(_prior_frame(prior, candidate))
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.DataFrame(factor_metric_rows),
        pd.concat(selection_rows, ignore_index=True),
        pd.concat(factor_prediction_rows, ignore_index=True),
    )


def _adversarial_source_drift(panel: pd.DataFrame, features: tuple[str, ...]) -> dict[str, float]:
    frame = panel.loc[panel["Window_End"].isin((2018, 2019, 2020, 2024, 2025, 2026))].copy()
    target = frame["Window_End"].ge(2024).astype(int)
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    for held_out in sorted(frame["Window_End"].unique()):
        train = frame["Window_End"].ne(held_out)
        test = ~train
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=10.0, random_state=20260829))])
        model.fit(frame.loc[train, features], target.loc[train])
        prediction.loc[test] = model.predict_proba(frame.loc[test, features])[:, 1]
    return {"early_vs_late_auc": float(roc_auc_score(target, prediction)), "rows": int(len(frame))}


def _team_change_annotations(seasons: tuple[int, ...]) -> pd.DataFrame:
    by_season: dict[int, dict[int, set[int]]] = {}
    for season in range(min(seasons) - 1, max(seasons) + 1):
        frame = pd.read_parquet(PLAYER_SHEETS / f"{season}.parquet", columns=["PLAYER_ID", "TEAM_ID"])
        by_season[season] = frame.groupby("PLAYER_ID")["TEAM_ID"].agg(lambda values: set(map(int, values))).to_dict()
    rows = []
    for season in seasons:
        current = by_season[season]
        previous = by_season[season - 1]
        for player_id, teams in current.items():
            rows.append({
                "PLAYER_ID": player_id,
                "Window_End": season,
                "team_change": "changed" if player_id in previous and teams.isdisjoint(previous[player_id]) else "same_or_new",
            })
    return pd.DataFrame(rows)


def _subgroup_metrics(factor_predictions: pd.DataFrame, specialist_run: Path) -> pd.DataFrame:
    source = pd.read_parquet(
        specialist_run / "five_year_features.parquet",
        columns=["PLAYER_ID", "Window_End", "shot_assignment_exposure_fraction", "boxout_source_exposure_fraction"],
    )
    offense_roles = pd.read_parquet(ROLE_ROOT / "offense_assignments.parquet", columns=["PLAYER_ID", "Season", "off_role_cluster"]).rename(columns={"Season": "Window_End"})
    defense_roles = pd.read_parquet(ROLE_ROOT / "defense_assignments.parquet", columns=["PLAYER_ID", "Season", "def_role_cluster"]).rename(columns={"Season": "Window_End"})
    frame = (
        factor_predictions.merge(source, on=["PLAYER_ID", "Window_End"], how="left", validate="many_to_one")
        .merge(offense_roles, on=["PLAYER_ID", "Window_End"], how="left", validate="many_to_one")
        .merge(defense_roles, on=["PLAYER_ID", "Window_End"], how="left", validate="many_to_one")
        .merge(_team_change_annotations(RATING_SEASONS), on=["PLAYER_ID", "Window_End"], how="left", validate="many_to_one")
    )
    frame["role"] = np.where(frame["component"].eq("offense"), frame["off_role_cluster"], frame["def_role_cluster"])
    frame["role"] = frame["role"].fillna("unavailable").astype(str)
    frame["team_change"] = frame["team_change"].fillna("inactive_current_season")
    frame["exposure"] = pd.cut(frame["weight"].pow(2), [0, 1000, 5000, np.inf], labels=["under_1000", "1000_to_4999", "5000_plus"], include_lowest=True).astype(str)
    source_fraction = np.where(frame["factor"].eq("shooting_ts"), frame["shot_assignment_exposure_fraction"], frame["boxout_source_exposure_fraction"])
    frame["source_coverage"] = pd.cut(source_fraction, [-np.inf, 0, 0.99, np.inf], labels=["unavailable", "partial", "near_complete"]).astype(str)
    rows = []
    for dimension in ("role", "exposure", "team_change", "source_coverage"):
        for keys, group in frame.groupby(["factor", "component", "candidate", dimension], dropna=False):
            if len(group) < 10:
                continue
            rows.append({
                "factor": keys[0], "component": keys[1], "candidate": keys[2],
                "dimension": dimension, "group": str(keys[3]), "rows": len(group),
                "weighted_mse": float(np.average(group["squared_error"], weights=group["weight"])),
                "mean_residual": float(np.average(group["residual"], weights=group["weight"])),
            })
    output = pd.DataFrame(rows)
    baseline = output.loc[output["candidate"].eq("box15_factor"), ["factor", "component", "dimension", "group", "weighted_mse"]].rename(columns={"weighted_mse": "box15_weighted_mse"})
    output = output.merge(baseline, on=["factor", "component", "dimension", "group"], how="left", validate="many_to_one")
    output["mse_delta_vs_box15"] = output["weighted_mse"] - output["box15_weighted_mse"]
    return output.sort_values(["factor", "component", "dimension", "group", "candidate"], kind="stable")


def main() -> None:
    panel, contract, specialist_run = _load_panel()
    priors, factor_metrics, selection, factor_predictions = _fit_priors(panel, contract)
    base.RATING_SEASONS = RATING_SEASONS
    base.EVALUATED_RATING_SEASONS = EVALUATED_RATING_SEASONS
    base.PRIOR_CANDIDATES = CANDIDATES
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {frozenset(("box15_aio", f"{candidate}_aio")) for candidate in CANDIDATES[1:]}
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    fold_metrics, summary = base._game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = base.paired_game_bootstrap(games, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    drift_features = tuple(dict.fromkeys((*SHOT_FEATURES, *REBOUND_FEATURES)))
    drift = _adversarial_source_drift(panel, drift_features)
    subgroup_metrics = _subgroup_metrics(factor_predictions, specialist_run)
    sources = {
        "base_features": BASE_FEATURES,
        "specialist_features": specialist_run / "five_year_features.parquet",
        "specialist_manifest": specialist_run / "run.json",
        "factor_targets": FACTOR_TARGETS,
        "normal_targets": NORMAL_TARGETS,
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": "historical_factor_residual_tournament_v1",
        "rating_seasons": list(RATING_SEASONS),
        "evaluated_rating_seasons": list(EVALUATED_RATING_SEASONS),
        "candidates": list(CANDIDATES),
        "factor_model_families": sorted({spec.family for spec in FACTOR_SPECS}),
        "feature_contract": {f"{factor}_{side}": list(features) for (factor, side), features in contract.items()},
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "whole game within outcome season"},
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "season_2027": "forbidden",
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"historical_factor_residual_tournament_v1_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "factor_metrics.parquet": factor_metrics,
        "factor_predictions.parquet": factor_predictions,
        "subgroup_metrics.parquet": subgroup_metrics,
        "selection.parquet": selection,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": fold_metrics,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    reference = bootstrap_models.set_index("candidate").loc["box15_aio"]
    decisions = []
    for candidate in CANDIDATES[1:]:
        challenger = bootstrap_models.set_index("candidate").loc[f"{candidate}_aio"]
        pair = bootstrap_pairs.loc[bootstrap_pairs["candidate"].eq("box15_aio") & bootstrap_pairs["reference"].eq(f"{candidate}_aio")].iloc[0]
        rmse_improvement = float(reference["equal_season_rmse"] - challenger["equal_season_rmse"])
        correlation = summary.set_index("candidate")
        decisions.append({
            "candidate": f"{candidate}_aio",
            "rmse_improvement_points_per_game": rmse_improvement,
            "paired_mse_interval_favors_candidate": bool(pair["bootstrap_95_low"] > 0),
            "margin_correlation_delta": float(correlation.loc[f"{candidate}_aio", "mean_margin_correlation"] - correlation.loc["box15_aio", "mean_margin_correlation"]),
            "passes_primary_gate": bool(rmse_improvement >= 0.05 and pair["bootstrap_95_low"] > 0 and correlation.loc[f"{candidate}_aio", "mean_margin_correlation"] - correlation.loc["box15_aio", "mean_margin_correlation"] >= -0.01),
        })
    run = {
        "run_id": output.name,
        "status": "research_challenger" if any(item["passes_primary_gate"] for item in decisions) else "research_null",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {"panel_rows": int(len(panel)), "identical_games_within_fold": True, "adversarial_source_drift": drift, "season_2027_loaded": False},
        "decisions": decisions,
        "files": {},
        "forbidden_interpretation": "Factor fit alone does not establish a better all-in-one. Season 2027 was not loaded and remains the untouched confirmation season.",
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "rows": int(len(frame)), "sha256": sha256_file(output / name)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
