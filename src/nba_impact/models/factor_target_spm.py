"""Sparse factor-target SPM and teammate-context ablation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.data.statistical_features_v2 import _engineer_window
from nba_impact.models.factor_reconstruction import (
    fit_weighted_ridge,
    predict_weighted_ridge,
    weighted_metrics,
)
from nba_impact.models.hand_selected_sparse_spm import build_annual_auxiliary_features


EXPERIMENT_ID = "factor_target_sparse_spm_v1"
FACTORS = ("shooting_ts", "turnover", "offensive_rebound")
SIDES = ("offense", "defense")

INDIVIDUAL_FEATURES = {
    ("shooting_ts", "offense"): (
        "zts_pct_points",
        "playtype_difficulty_pct_points",
        "shooting_proficiency_2017_eb",
        "crafted_spacing_stable_v1",
        "at_rim_fga_p100",
    ),
    ("shooting_ts", "defense"): (
        "dfg_two_point_equivalent_saved_p100",
        "rim_points_saved_p100",
        "contested_2pt_p100",
        "contested_3pt_p100",
    ),
    ("turnover", "offense"): (
        "turnover_to_load_2017_eb",
        "live_ball_turnovers_p100",
        "bad_pass_turnovers_p100",
        "lost_ball_turnovers_p100",
        "offensive_fouls_p100",
    ),
    ("turnover", "defense"): (
        "STL_p100_relative",
        "deflections_p100",
        "charges_drawn_p100",
        "event_stops_p100",
    ),
    ("offensive_rebound", "offense"): (
        "OREB_p100",
        "self_oreb_p100",
        "oreb_contests_p100",
        "oreb_chances_p100",
        "offensive_boxouts_p100",
    ),
    ("offensive_rebound", "defense"): (
        "DREB_p100_relative",
        "dreb_contests_p100",
        "dreb_chances_p100",
        "dreb_contested_share",
        "defensive_boxouts_p100",
    ),
}

CONTEXT_SOURCES = {
    "teammate_spacing": ("crafted_spacing_stable_v1", "OffPoss"),
    "teammate_creation": ("box_creation_2017_eb_p100", "OffPoss"),
    "teammate_rim_pressure": ("at_rim_fga_p100", "OffPoss"),
    "teammate_turnover_to_load": ("turnover_to_load_2017_eb", "OffPoss"),
    "teammate_offensive_load": ("offensive_load_2017_eb_p100", "OffPoss"),
    "teammate_oreb": ("OREB_p100", "OffPoss"),
    "teammate_dreb": ("DREB_p100_relative", "DefPoss"),
    "teammate_dreb_contests": ("dreb_contests_p100", "DefPoss"),
    "teammate_event_stops": ("event_stops_p100", "DefPoss"),
    "teammate_deflections": ("deflections_p100", "DefPoss"),
    "teammate_rim_points_saved": ("rim_points_saved_p100", "DefPoss"),
    "teammate_contested_shots": ("contested_shots_p100", "DefPoss"),
}

CONTEXT_FEATURES = {
    ("shooting_ts", "offense"): (
        "teammate_spacing",
        "teammate_creation",
        "teammate_rim_pressure",
    ),
    ("shooting_ts", "defense"): (
        "teammate_rim_points_saved",
        "teammate_contested_shots",
        "teammate_event_stops",
    ),
    ("turnover", "offense"): (
        "teammate_spacing",
        "teammate_creation",
        "teammate_turnover_to_load",
    ),
    ("turnover", "defense"): (
        "teammate_event_stops",
        "teammate_deflections",
    ),
    ("offensive_rebound", "offense"): (
        "teammate_oreb",
        "teammate_spacing",
        "teammate_offensive_load",
    ),
    ("offensive_rebound", "defense"): (
        "teammate_dreb",
        "teammate_dreb_contests",
        "teammate_rim_points_saved",
    ),
}


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return 100.0 * pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    ).where(lambda values: values.gt(0))


def build_related_feature_panel(
    player_sheet_dir: str | Path,
    *,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    seasons: tuple[int, ...] = (2024, 2025, 2026),
) -> tuple[pd.DataFrame, dict]:
    """Build same-season individual and leave-one-out teammate features."""
    root = Path(player_sheet_dir)
    requested = range(min(seasons) - 2, max(seasons) + 1)
    available = [season for season in requested if (root / f"{season}.parquet").exists()]
    if any(season not in available for season in seasons):
        raise ValueError("A requested player-sheet season is unavailable.")
    loaded = {
        season: _load_source(root / f"{season}.parquet", season)[0]
        for season in available
    }
    first_available = min(loaded)
    outputs = []
    for season in seasons:
        raw = loaded[season]
        base = _aggregate_window([raw], season)
        seasonal = [
            _aggregate_window([loaded[value]], value)
            for value in range(max(first_available, season - 2), season + 1)
        ]
        seasonal = [seasonal[0]] * (3 - len(seasonal)) + seasonal
        engineered = _engineer_window(base, [raw], seasonal)
        identity = raw[
            ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
        ].copy()
        raw_index = raw.set_index("PLAYER_ID")
        engineered["oreb_contests_p100"] = _safe_rate(
            engineered["PLAYER_ID"].map(raw_index.get("OREB_CONTEST")),
            engineered["OffPoss"],
        )
        engineered["oreb_chances_p100"] = _safe_rate(
            engineered["PLAYER_ID"].map(raw_index.get("OREB_CHANCES")),
            engineered["OffPoss"],
        )
        engineered["offensive_boxouts_p100"] = _safe_rate(
            engineered["PLAYER_ID"].map(raw_index.get("hustle_OFF_BOXOUTS")),
            engineered["OffPoss"],
        )
        engineered["defensive_boxouts_p100"] = _safe_rate(
            engineered["PLAYER_ID"].map(raw_index.get("hustle_DEF_BOXOUTS")),
            engineered["DefPoss"],
        )
        engineered["Season"] = season
        outputs.append(
            engineered.merge(identity, on="PLAYER_ID", how="left", validate="one_to_one")
        )
    features = pd.concat(outputs, ignore_index=True)
    auxiliary, auxiliary_quality = build_annual_auxiliary_features(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
        seasons=tuple(available),
    )
    auxiliary_columns = [
        column
        for column in auxiliary.columns
        if column not in {"PLAYER_ID", "Season", "DefPoss"}
    ]
    features = features.merge(
        auxiliary[["PLAYER_ID", "Season", *auxiliary_columns]],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    )
    features["contested_shots_p100"] = (
        features["contested_2pt_p100"] + features["contested_3pt_p100"]
    )
    features = add_leave_one_out_teammate_context(features)

    required = {
        feature
        for values in (*INDIVIDUAL_FEATURES.values(), *CONTEXT_FEATURES.values())
        for feature in values
    }
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Related feature panel is missing {missing}.")
    quality = {
        "rows": int(len(features)),
        "duplicate_player_seasons": int(
            features.duplicated(["PLAYER_ID", "Season"]).sum()
        ),
        "team_id_coverage": float(features["TEAM_ID"].notna().mean()),
        "feature_missing_fraction": {
            feature: float(features[feature].isna().mean())
            for feature in sorted(required)
        },
        "auxiliary": auxiliary_quality,
        "teammate_context_definition": (
            "Possession-weighted mean of recorded same-team players excluding the focal player."
        ),
    }
    if quality["duplicate_player_seasons"]:
        raise ValueError("Related feature panel has duplicate player-season keys.")
    return features, quality


def add_leave_one_out_teammate_context(features: pd.DataFrame) -> pd.DataFrame:
    """Add team-season aggregates that remove the focal player's contribution."""
    output = features.copy()
    group_keys = ["Season", "TEAM_ID"]
    for destination, (source, weight_column) in CONTEXT_SOURCES.items():
        value = pd.to_numeric(output[source], errors="coerce")
        weight = pd.to_numeric(output[weight_column], errors="coerce")
        valid = value.notna() & weight.gt(0) & output["TEAM_ID"].notna()
        contribution = (value * weight).where(valid, 0.0)
        valid_weight = weight.where(valid, 0.0)
        team_contribution = contribution.groupby(
            [output[key] for key in group_keys], dropna=False
        ).transform("sum")
        team_weight = valid_weight.groupby(
            [output[key] for key in group_keys], dropna=False
        ).transform("sum")
        teammate_weight = team_weight - valid_weight
        output[destination] = (team_contribution - contribution) / teammate_weight.where(
            teammate_weight.gt(0)
        )
    return output


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    weight: str,
    alpha: float,
) -> tuple[np.ndarray, Pipeline]:
    model = _pipeline(alpha)
    model.fit(
        train.loc[:, features],
        train[target],
        model__sample_weight=train[weight],
    )
    return model.predict(test.loc[:, features]), model


def _metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    stage: str,
    candidate: str,
    target: str,
    side: str,
    alpha: float | None,
    weight: str,
) -> dict:
    return {
        "stage": stage,
        "candidate": candidate,
        "target": target,
        "side": side,
        "alpha": alpha,
        "players": int(len(frame)),
        **weighted_metrics(
            frame[f"target_{target}_{side}"].to_numpy(dtype=float),
            prediction,
            frame[weight].to_numpy(dtype=float),
        ),
    }


def _factor_weight(factor: str, side: str) -> str:
    return f"weight_{factor}_{side}"


def _prepare_panel(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    panel = targets.merge(
        features,
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    for factor in FACTORS:
        for side in SIDES:
            exposure = f"{factor}_{'off' if side == 'offense' else 'def'}_exposure"
            panel[_factor_weight(factor, side)] = np.sqrt(
                pd.to_numeric(panel[exposure], errors="coerce").clip(lower=1)
            )
            panel = panel.rename(
                columns={f"{factor}_{side}": f"target_{factor}_{side}"}
            )
    panel["weight_normal"] = np.sqrt(
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1)
    )
    return panel


def _fit_factor_variant(
    panel: pd.DataFrame,
    *,
    candidate: str,
    include_context: bool,
    feature_map: dict[tuple[str, str], tuple[str, ...]] | None = None,
    context_map: dict[tuple[str, str], tuple[str, ...]] | None = None,
    alphas: tuple[float, ...],
    development_season: int,
    selection_season: int,
    diagnostic_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], list[dict]]:
    feature_map = INDIVIDUAL_FEATURES if feature_map is None else feature_map
    context_map = CONTEXT_FEATURES if context_map is None else context_map
    development = panel.loc[panel["Season"].eq(development_season)]
    selection = panel.loc[panel["Season"].eq(selection_season)]
    train = panel.loc[panel["Season"].isin((development_season, selection_season))]
    diagnostic = panel.loc[panel["Season"].eq(diagnostic_season)]
    selection_metrics = []
    selected = []
    predictions = diagnostic[["PLAYER_ID", "Season"]].copy()
    models = {}
    diagnostic_metrics = []
    for factor in FACTORS:
        for side in SIDES:
            feature_names = feature_map[(factor, side)]
            if include_context:
                feature_names = feature_names + context_map[(factor, side)]
            target = f"target_{factor}_{side}"
            weight = _factor_weight(factor, side)
            candidates = []
            for alpha in alphas:
                prediction, _ = _fit_predict(
                    development,
                    selection,
                    features=feature_names,
                    target=target,
                    weight=weight,
                    alpha=alpha,
                )
                row = _metric_row(
                    selection,
                    prediction,
                    stage="selection",
                    candidate=candidate,
                    target=factor,
                    side=side,
                    alpha=alpha,
                    weight=weight,
                )
                candidates.append(row)
                selection_metrics.append(row)
            winner = min(candidates, key=lambda row: (row["weighted_rmse"], row["alpha"]))
            selected.append(
                {
                    "candidate": candidate,
                    "target": factor,
                    "side": side,
                    "alpha": float(winner["alpha"]),
                    "features": list(feature_names),
                }
            )
            prediction, model = _fit_predict(
                train,
                diagnostic,
                features=feature_names,
                target=target,
                weight=weight,
                alpha=float(winner["alpha"]),
            )
            predictions[f"predicted_{factor}_{side}"] = prediction
            models[f"{factor}_{side}"] = model
            diagnostic_metrics.append(
                _metric_row(
                    diagnostic,
                    prediction,
                    stage="diagnostic",
                    candidate=candidate,
                    target=factor,
                    side=side,
                    alpha=float(winner["alpha"]),
                    weight=weight,
                )
            )
    return (
        predictions,
        pd.DataFrame([*selection_metrics, *diagnostic_metrics]),
        models,
        selected,
    )


def _direct_feature_names(side: str, include_context: bool) -> tuple[str, ...]:
    values = []
    for factor in FACTORS:
        values.extend(INDIVIDUAL_FEATURES[(factor, side)])
        if include_context:
            values.extend(CONTEXT_FEATURES[(factor, side)])
    return tuple(dict.fromkeys(values))


def _fit_direct_variant(
    panel: pd.DataFrame,
    *,
    candidate: str,
    include_context: bool,
    feature_map: dict[str, tuple[str, ...]] | None = None,
    context_map: dict[str, tuple[str, ...]] | None = None,
    alphas: tuple[float, ...],
    development_season: int,
    selection_season: int,
    diagnostic_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict[str, Pipeline]]:
    if feature_map is None:
        feature_map = {
            side: _direct_feature_names(side, False) for side in SIDES
        }
    if context_map is None:
        context_map = {
            side: tuple(
                dict.fromkeys(
                    feature
                    for factor in FACTORS
                    for feature in CONTEXT_FEATURES[(factor, side)]
                )
            )
            for side in SIDES
        }
    development = panel.loc[panel["Season"].eq(development_season)]
    selection = panel.loc[panel["Season"].eq(selection_season)]
    train = panel.loc[panel["Season"].isin((development_season, selection_season))]
    diagnostic = panel.loc[panel["Season"].eq(diagnostic_season)]
    predictions = diagnostic[["PLAYER_ID", "Season"]].copy()
    metrics = []
    selected = []
    models = {}
    for side in SIDES:
        feature_names = feature_map[side]
        if include_context:
            feature_names = tuple(dict.fromkeys((*feature_names, *context_map[side])))
        target = f"target_normal_{side}"
        candidates = []
        for alpha in alphas:
            prediction, _ = _fit_predict(
                development,
                selection,
                features=feature_names,
                target=target,
                weight="weight_normal",
                alpha=alpha,
            )
            row = _metric_row(
                selection,
                prediction,
                stage="selection",
                candidate=candidate,
                target="normal",
                side=side,
                alpha=alpha,
                weight="weight_normal",
            )
            candidates.append(row)
            metrics.append(row)
        winner = min(candidates, key=lambda row: (row["weighted_rmse"], row["alpha"]))
        selected.append(
            {
                "candidate": candidate,
                "target": "normal",
                "side": side,
                "alpha": float(winner["alpha"]),
                "features": list(feature_names),
            }
        )
        prediction, model = _fit_predict(
            train,
            diagnostic,
            features=feature_names,
            target=target,
            weight="weight_normal",
            alpha=float(winner["alpha"]),
        )
        predictions[f"predicted_normal_{side}"] = prediction
        models[side] = model
        metrics.append(
            _metric_row(
                diagnostic,
                prediction,
                stage="diagnostic",
                candidate=candidate,
                target="normal",
                side=side,
                alpha=float(winner["alpha"]),
                weight="weight_normal",
            )
        )
    predictions["predicted_normal_net"] = (
        predictions["predicted_normal_offense"]
        + predictions["predicted_normal_defense"]
    )
    return predictions, pd.DataFrame(metrics), selected, models


def _coefficient_rows(
    models: dict[str, Pipeline],
    selected: list[dict],
) -> list[dict]:
    rows = []
    by_key = {(row["target"], row["side"]): row for row in selected}
    for key, model in models.items():
        if "_" in key and key.rsplit("_", 1)[1] in SIDES:
            target, side = key.rsplit("_", 1)
        else:
            target, side = "normal", key
        contract = by_key[(target, side)]
        inputs = np.asarray(contract["features"], dtype=object)
        terms = model.named_steps["impute"].get_feature_names_out(inputs)
        coefficients = model.named_steps["model"].coef_
        rows.extend(
            {
                "candidate": contract["candidate"],
                "target": target,
                "side": side,
                "alpha": contract["alpha"],
                "term": str(term),
                "standardized_coefficient": float(coefficient),
            }
            for term, coefficient in zip(terms, coefficients, strict=True)
        )
    return rows


def _reconstruction_models(
    train: pd.DataFrame, *, alpha: float
) -> dict[str, object]:
    models = {}
    for side in SIDES:
        columns = [f"target_{factor}_{side}" for factor in FACTORS]
        models[side] = fit_weighted_ridge(
            train[columns].to_numpy(dtype=float),
            train[f"target_normal_{side}"].to_numpy(dtype=float),
            train["weight_normal"].to_numpy(dtype=float),
            alpha=alpha,
        )
    return models


def _apply_reconstruction(
    models: dict[str, object],
    frame: pd.DataFrame,
    *,
    predicted: bool,
) -> dict[str, np.ndarray]:
    output = {}
    for side in SIDES:
        prefix = "predicted" if predicted else "target"
        columns = [f"{prefix}_{factor}_{side}" for factor in FACTORS]
        output[side] = predict_weighted_ridge(
            models[side], frame[columns].to_numpy(dtype=float)
        )
    output["net"] = output["offense"] + output["defense"]
    return output


def _score_normal(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    stage: str,
    candidate: str,
) -> list[dict]:
    rows = []
    for side in (*SIDES, "net"):
        rows.append(
            {
                "stage": stage,
                "candidate": candidate,
                "side": side,
                "players": int(len(frame)),
                **weighted_metrics(
                    frame[f"target_normal_{side}"].to_numpy(dtype=float),
                    predictions[side],
                    frame["weight_normal"].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def run_factor_target_spm(
    *,
    player_sheet_dir: str | Path,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    factor_panel_path: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    contract = json.loads(Path(contract_path).read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Factor-target SPM requires a frozen contract.")
    if contract["untouched_confirmation_season"] in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    source_paths = {
        "playtype": Path(playtype_path),
        "dfg": Path(dfg_path),
        "rim_dfg": Path(rim_dfg_path),
        "hustle": Path(hustle_path),
        "factor_panel": Path(factor_panel_path),
        "contract": Path(contract_path),
        "source_code": Path(__file__),
        "statistical_features_builder": Path(_aggregate_window.__code__.co_filename),
        "statistical_features_v2_builder": Path(_engineer_window.__code__.co_filename),
        "auxiliary_feature_builder": Path(build_annual_auxiliary_features.__code__.co_filename),
        "factor_reconstruction": Path(fit_weighted_ridge.__code__.co_filename),
    }
    for season in range(min(contract["seasons"]) - 2, max(contract["seasons"]) + 1):
        source_paths[f"player_sheet_{season}"] = Path(player_sheet_dir) / f"{season}.parquet"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    config = {
        "contract": contract,
        "individual_features": {
            f"{factor}_{side}": list(values)
            for (factor, side), values in INDIVIDUAL_FEATURES.items()
        },
        "context_features": {
            f"{factor}_{side}": list(values)
            for (factor, side), values in CONTEXT_FEATURES.items()
        },
        "context_sources": {
            key: list(value) for key, value in CONTEXT_SOURCES.items()
        },
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "research" / "factor_target_spm" / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())

    features, feature_quality = build_related_feature_panel(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
        seasons=tuple(contract["seasons"]),
    )
    target = pd.read_parquet(factor_panel_path).rename(
        columns={
            "target_offense": "target_normal_offense",
            "target_defense": "target_normal_defense",
            "target_net": "target_normal_net",
        }
    )
    panel = _prepare_panel(features, target)
    minimum = float(contract["minimum_normal_possessions_per_side"])
    panel = panel.loc[
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).ge(minimum)
    ].copy()
    if panel["Season"].value_counts().sort_index().to_dict() != {2024: 365, 2025: 381, 2026: 387}:
        raise ValueError("Factor-target panel does not reproduce the pinned eligible rows.")

    alphas = tuple(float(value) for value in contract["ridge_alphas"])
    development = int(contract["development_season"])
    selection = int(contract["selection_season"])
    diagnostic = int(contract["reused_diagnostic_season"])
    individual_predictions, individual_metrics, individual_models, individual_selected = _fit_factor_variant(
        panel,
        candidate="factor_individual",
        include_context=False,
        alphas=alphas,
        development_season=development,
        selection_season=selection,
        diagnostic_season=diagnostic,
    )
    context_predictions, context_metrics, context_models, context_selected = _fit_factor_variant(
        panel,
        candidate="factor_teammate_context",
        include_context=True,
        alphas=alphas,
        development_season=development,
        selection_season=selection,
        diagnostic_season=diagnostic,
    )
    direct_individual, direct_individual_metrics, direct_individual_selected, direct_individual_models = _fit_direct_variant(
        panel,
        candidate="direct_individual",
        include_context=False,
        alphas=alphas,
        development_season=development,
        selection_season=selection,
        diagnostic_season=diagnostic,
    )
    direct_context, direct_context_metrics, direct_context_selected, direct_context_models = _fit_direct_variant(
        panel,
        candidate="direct_teammate_context",
        include_context=True,
        alphas=alphas,
        development_season=development,
        selection_season=selection,
        diagnostic_season=diagnostic,
    )

    diagnostic_rows = panel.loc[panel["Season"].eq(diagnostic)].copy()
    train_rows = panel.loc[panel["Season"].isin((development, selection))]
    reconstruction = _reconstruction_models(
        train_rows, alpha=float(contract["factor_to_points_alpha"])
    )
    oracle = _apply_reconstruction(reconstruction, diagnostic_rows, predicted=False)
    individual_joined = diagnostic_rows.merge(
        individual_predictions,
        on=["PLAYER_ID", "Season"],
        validate="one_to_one",
    )
    context_joined = diagnostic_rows.merge(
        context_predictions,
        on=["PLAYER_ID", "Season"],
        validate="one_to_one",
    )
    factor_individual = _apply_reconstruction(
        reconstruction, individual_joined, predicted=True
    )
    factor_context = _apply_reconstruction(
        reconstruction, context_joined, predicted=True
    )
    direct_individual_dict = {
        side: direct_individual[f"predicted_normal_{side}"].to_numpy(dtype=float)
        for side in (*SIDES, "net")
    }
    direct_context_dict = {
        side: direct_context[f"predicted_normal_{side}"].to_numpy(dtype=float)
        for side in (*SIDES, "net")
    }
    mean = {
        side: np.full(
            len(diagnostic_rows),
            np.average(
                train_rows[f"target_normal_{side}"],
                weights=train_rows["weight_normal"],
            ),
        )
        for side in SIDES
    }
    mean["net"] = mean["offense"] + mean["defense"]
    normal_metrics = pd.DataFrame(
        [
            *_score_normal(diagnostic_rows, mean, stage="diagnostic", candidate="mean"),
            *_score_normal(diagnostic_rows, oracle, stage="diagnostic", candidate="oracle_factor_reconstruction"),
            *_score_normal(diagnostic_rows, factor_individual, stage="diagnostic", candidate="factor_individual"),
            *_score_normal(diagnostic_rows, factor_context, stage="diagnostic", candidate="factor_teammate_context"),
            *_score_normal(diagnostic_rows, direct_individual_dict, stage="diagnostic", candidate="direct_individual"),
            *_score_normal(diagnostic_rows, direct_context_dict, stage="diagnostic", candidate="direct_teammate_context"),
        ]
    )
    factor_metrics = pd.concat(
        [individual_metrics, context_metrics], ignore_index=True
    )
    direct_metrics = pd.concat(
        [direct_individual_metrics, direct_context_metrics], ignore_index=True
    )
    selected_models = pd.DataFrame(
        [
            *individual_selected,
            *context_selected,
            *direct_individual_selected,
            *direct_context_selected,
        ]
    )
    coefficients = pd.DataFrame(
        [
            *_coefficient_rows(individual_models, individual_selected),
            *_coefficient_rows(context_models, context_selected),
            *_coefficient_rows(direct_individual_models, direct_individual_selected),
            *_coefficient_rows(direct_context_models, direct_context_selected),
        ]
    )
    predictions = diagnostic_rows[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "Season",
            "target_normal_offense",
            "target_normal_defense",
            "target_normal_net",
            "weight_normal",
            *[f"target_{factor}_{side}" for factor in FACTORS for side in SIDES],
            *[_factor_weight(factor, side) for factor in FACTORS for side in SIDES],
        ]
    ].copy()
    for candidate, values in (
        ("oracle_factor_reconstruction", oracle),
        ("factor_individual", factor_individual),
        ("factor_teammate_context", factor_context),
        ("direct_individual", direct_individual_dict),
        ("direct_teammate_context", direct_context_dict),
    ):
        for side in (*SIDES, "net"):
            predictions[f"{candidate}_{side}"] = values[side]
    for candidate, frame in (
        ("factor_individual", individual_predictions),
        ("factor_teammate_context", context_predictions),
    ):
        for factor in FACTORS:
            for side in SIDES:
                predictions[f"{candidate}_{factor}_{side}"] = frame[
                    f"predicted_{factor}_{side}"
                ].to_numpy(dtype=float)

    output.mkdir(parents=True, exist_ok=False)
    factor_metrics.to_parquet(output / "factor_target_metrics.parquet", index=False)
    direct_metrics.to_parquet(output / "direct_target_metrics.parquet", index=False)
    normal_metrics.to_parquet(output / "normal_rapm_metrics.parquet", index=False)
    selected_models.to_parquet(output / "selected_models.parquet", index=False)
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    predictions.to_parquet(output / "predictions_2026.parquet", index=False)
    context_columns = ["PLAYER_ID", "Season", "TEAM_ID", *CONTEXT_SOURCES]
    features[context_columns].to_parquet(output / "teammate_context.parquet", index=False)

    normal_net = normal_metrics.loc[normal_metrics["side"].eq("net")].sort_values(
        "weighted_rmse", kind="stable"
    )
    factor_diagnostic = factor_metrics.loc[factor_metrics["stage"].eq("diagnostic")]
    context_wins = 0
    for factor in FACTORS:
        for side in SIDES:
            rows = factor_diagnostic.loc[
                factor_diagnostic["target"].eq(factor)
                & factor_diagnostic["side"].eq(side)
            ].set_index("candidate")
            context_wins += int(
                rows.loc["factor_teammate_context", "weighted_rmse"]
                < rows.loc["factor_individual", "weighted_rmse"]
            )
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            **feature_quality,
            "eligible_rows_by_season": {
                str(key): int(value)
                for key, value in panel["Season"].value_counts().sort_index().items()
            },
            "season_2027_loaded": False,
            "net_identity_max_error": float(
                max(
                    np.max(np.abs(values["offense"] + values["defense"] - values["net"]))
                    for values in (
                        oracle,
                        factor_individual,
                        factor_context,
                        direct_individual_dict,
                        direct_context_dict,
                    )
                )
            ),
        },
        "normal_rapm_net_comparison": normal_net.to_dict("records"),
        "teammate_context_factor_target_wins": {
            "wins": context_wins,
            "comparisons": len(FACTORS) * len(SIDES),
        },
        "decision": (
            "Retain teammate context for research follow-up."
            if context_wins >= 4
            else "Do not retain teammate context; it improves fewer than four of six factor targets."
        ),
        "caveats": [
            "Factor RAPM and normal RAPM share the same season and lineup design; reconstruction is not independent validation.",
            "Team context uses the one recorded annual TEAM_ID, so traded-player context is approximate.",
            "Same-season teammate features describe context and may absorb team or scheme strength; they are not player skill.",
            "Defended-rim source data end in 2025, so 2026 rim fields use the builder's neutral same-season fallback.",
            "Season 2026 is reused diagnostic evidence and Season 2027 remains untouched.",
        ],
        "paths": {
            "factor_target_metrics": "factor_target_metrics.parquet",
            "direct_target_metrics": "direct_target_metrics.parquet",
            "normal_rapm_metrics": "normal_rapm_metrics.parquet",
            "selected_models": "selected_models.parquet",
            "coefficients": "coefficients.parquet",
            "predictions_2026": "predictions_2026.parquet",
            "teammate_context": "teammate_context.parquet",
        },
        "forbidden_interpretation": (
            "Causal factor credit, independent validation, future forecast, production rating, or teammate skill attribution."
        ),
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest
