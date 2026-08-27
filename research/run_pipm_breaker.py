#!/usr/bin/env python3
"""Separate the target, feature, learner and context reasons BoxPIPM can win.

This is a reused historical experiment. It does not modify the public SPM or
the production site. Every candidate becomes a prior for the same one-season
RAPM update and scores the same future games.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _load_source
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.external_impact_benchmark import normalize_player_name
from nba_impact.data.quality import audit_possession_frame
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.models.spm_role_team_win_benchmark import _TotalsParser
from run_aio_prior_bakeoff import (
    _annual_onoff,
    _game_metrics,
    _paired_bootstrap,
    _prior_frame,
    _rolling_onoff,
    _team_context,
)
from run_aio_prior_canonical_followup import (
    _annual_from_frame,
    _center,
    _remap_annual,
    _solve,
)
from run_ryan_target_spm import _panel


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pipm_breaker_v1"
PRIOR_CHECKPOINT_VERSION = "pipm_breaker_priors_v3"
RATING_SEASONS = (2021, 2022, 2023)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CORRELATION_REPORT_THRESHOLD = 0.95
CORRELATION_PRUNE_THRESHOLD = 0.98
TEAM_ALIASES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}

# Explicit source aliases, verified against season, team and minutes. These are
# not fuzzy matches and they do not change the canonical identity dimension.
BBREF_NAME_ALIASES = {
    "Jeff Taylor": "Jeffery Taylor",
    "Vítor Luiz Faverani": "Vitor Faverani",
    "Hamady N'Diaye": "Hamady Ndiaye",
    "Tibor Pleiß": "Tibor Pleiss",
    "Vince Hunter": "Vincent Hunter",
    "Cameron Reynolds": "Cam Reynolds",
    "Mitch Creek": "Mitchell Creek",
    "Vince Edwards": "Vincent Edwards",
    "RJ Nembhard Jr.": "Ruben Nembhard Jr.",
    "Tre Scott": "Trevon Scott",
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    config: dict[str, float | int]


def _pipeline(spec: ModelSpec) -> Pipeline:
    if spec.family == "ridge":
        estimator = Ridge(alpha=float(spec.config["alpha"]))
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    if spec.family == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec.config["alpha"]),
            l1_ratio=float(spec.config["l1_ratio"]),
            max_iter=20_000,
            tol=1e-5,
            selection="cyclic",
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    if spec.family == "histogram_gbm":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(spec.config["learning_rate"]),
            max_iter=250,
            max_leaf_nodes=int(spec.config["max_leaf_nodes"]),
            min_samples_leaf=int(spec.config["min_samples_leaf"]),
            l2_regularization=float(spec.config["l2_regularization"]),
            early_stopping=False,
            random_state=20260827,
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unknown model family {spec.family}.")


def _model_grid(*, ridge_only: bool = False) -> tuple[ModelSpec, ...]:
    ridge = tuple(ModelSpec("ridge", {"alpha": alpha}) for alpha in ALPHA_GRID)
    if ridge_only:
        return ridge
    elastic = tuple(
        ModelSpec("elastic_net", {"alpha": alpha, "l1_ratio": ratio})
        for alpha, ratio in (
            (0.001, 0.1),
            (0.01, 0.1),
            (0.03, 0.1),
            (0.1, 0.1),
            (0.1, 0.5),
        )
    )
    histogram = tuple(
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
            (0.03, 7, 30, 1.0),
            (0.03, 7, 50, 10.0),
            (0.03, 15, 30, 10.0),
            (0.07, 7, 30, 10.0),
            (0.07, 15, 50, 10.0),
        )
    )
    return (*ridge, *elastic, *histogram)


def _fit_model(
    spec: ModelSpec,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> Pipeline:
    model = _pipeline(spec)
    model.fit(
        frame.loc[:, features],
        frame[target],
        model__sample_weight=frame["sample_weight"],
    )
    return model


def _weighted_rmse(actual: pd.Series, predicted: np.ndarray, weights: pd.Series) -> float:
    error = actual.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.average(error**2, weights=weights.to_numpy(dtype=float))))


def _select_model(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    *,
    ridge_only: bool,
) -> tuple[ModelSpec, pd.DataFrame]:
    validation_end = int(train["Window_End"].max())
    inner_train = train.loc[train["Window_End"].lt(validation_end)]
    validation = train.loc[train["Window_End"].eq(validation_end)]
    if inner_train["Window_End"].nunique() < 1 or validation.empty:
        raise ValueError("Model selection requires training history and one validation window.")
    rows = []
    for spec in _model_grid(ridge_only=ridge_only):
        fitted = _fit_model(spec, inner_train, features, target)
        prediction = fitted.predict(validation.loc[:, features])
        rows.append(
            {
                "family": spec.family,
                "config": json.dumps(spec.config, sort_keys=True),
                "validation_window_end": validation_end,
                "weighted_rmse": _weighted_rmse(
                    validation[target], prediction, validation["sample_weight"]
                ),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["weighted_rmse", "family", "config"], kind="stable"
    )
    winner = scores.iloc[0]
    return ModelSpec(winner["family"], json.loads(winner["config"])), scores


def _bbref_starts_minutes(
    html_root: Path,
    player_sheet_root: Path,
    seasons: range,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage = []
    for season in seasons:
        parser = _TotalsParser()
        path = html_root / f"nba_{season}_totals.html"
        parser.feed(path.read_text(encoding="utf-8"))
        source = pd.DataFrame(parser.rows)
        required = {
            "name_display",
            "team_name_abbr",
            "games",
            "games_started",
            "mp",
        }
        if missing := sorted(required - set(source.columns)):
            raise ValueError(f"Basketball-Reference season {season} lacks {missing}.")
        source = source.loc[
            source["name_display"].ne("Player")
            & source["name_display"].ne("League Average")
            & ~source["team_name_abbr"].astype(str).str.endswith("TM")
        ].copy()
        source["canonical_source_name"] = source["name_display"].replace(
            BBREF_NAME_ALIASES
        )
        source["name_key"] = source["canonical_source_name"].map(
            normalize_player_name
        )
        source["team"] = source["team_name_abbr"].replace(TEAM_ALIASES)
        for column in ("games", "games_started", "mp"):
            source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0.0)

        identity = pd.read_parquet(
            player_sheet_root / f"{season}.parquet",
            columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"],
        ).drop_duplicates()
        identity["name_key"] = identity["PLAYER_NAME"].map(normalize_player_name)
        identity["team"] = identity["TEAM_ABBREVIATION"].replace(TEAM_ALIASES)
        identity = identity.drop_duplicates(["PLAYER_ID", "name_key", "team"])
        name_counts = identity.groupby("name_key")["PLAYER_ID"].nunique()
        unique_names = set(name_counts.loc[name_counts.eq(1)].index)
        unique_identity = (
            identity.loc[identity["name_key"].isin(unique_names), ["PLAYER_ID", "name_key"]]
            .drop_duplicates("name_key")
        )
        ambiguous_identity = identity.loc[
            ~identity["name_key"].isin(unique_names),
            ["PLAYER_ID", "name_key", "team"],
        ].drop_duplicates()
        if ambiguous_identity.duplicated(["name_key", "team"]).any():
            ambiguous = ambiguous_identity.loc[
                ambiguous_identity.duplicated(["name_key", "team"], keep=False),
                ["name_key", "team"],
            ].drop_duplicates()
            raise ValueError(
                f"Season {season} has ambiguous player name/team keys: "
                f"{ambiguous.to_dict(orient='records')[:5]}"
            )
        matched = source.merge(
            unique_identity,
            on="name_key",
            how="left",
            validate="many_to_one",
        )
        unresolved = matched["PLAYER_ID"].isna()
        if unresolved.any() and not ambiguous_identity.empty:
            resolved = matched.loc[unresolved, ["name_key", "team"]].merge(
                ambiguous_identity,
                on=["name_key", "team"],
                how="left",
                validate="many_to_one",
            )
            matched.loc[unresolved, "PLAYER_ID"] = resolved["PLAYER_ID"].to_numpy()
        source_minutes = float(matched["mp"].sum())
        matched_minutes = float(matched.loc[matched["PLAYER_ID"].notna(), "mp"].sum())
        coverage.append(
            {
                "Season": season,
                "source_rows": len(source),
                "matched_rows": int(matched["PLAYER_ID"].notna().sum()),
                "source_minutes": source_minutes,
                "matched_minutes": matched_minutes,
                "minute_match_rate": matched_minutes / source_minutes,
                "unmatched_players": "|".join(
                    matched.loc[matched["PLAYER_ID"].isna(), "name_display"].astype(str)
                ),
            }
        )
        matched = matched.dropna(subset=["PLAYER_ID"]).copy()
        matched["PLAYER_ID"] = matched["PLAYER_ID"].astype(int)
        annual = matched.groupby("PLAYER_ID", as_index=False).agg(
            games=("games", "sum"),
            games_started=("games_started", "sum"),
            bbref_minutes=("mp", "sum"),
        )
        annual["Season"] = season
        rows.append(annual)
    coverage_frame = pd.DataFrame(coverage)
    if coverage_frame["minute_match_rate"].min() < 0.995:
        raise ValueError("Basketball-Reference starts/minutes coverage is below 99.5%.")
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Basketball-Reference annual player keys are duplicated.")
    return output, coverage_frame


def _position_group(value: object) -> str:
    position = str(value).upper()
    if position in {"PG", "SG"}:
        return "guard"
    if position in {"SF", "PF"}:
        return "forward"
    return "center"


def _annual_context(
    player_sheet_root: Path,
    starts: pd.DataFrame,
    seasons: range,
) -> pd.DataFrame:
    rows = []
    for season in seasons:
        frame, _ = _load_source(player_sheet_root / f"{season}.parquet", season)
        required = {
            "PLAYER_ID",
            "GP",
            "MIN",
            "Pos",
            "OREB_PCT",
            "OffPoss",
        }
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Player sheet season {season} lacks {missing}.")
        annual = frame.loc[:, sorted(required)].copy()
        annual["Season"] = season
        for column in ("PLAYER_ID", "GP", "MIN", "OREB_PCT", "OffPoss"):
            annual[column] = pd.to_numeric(annual[column], errors="coerce")
        annual = annual.dropna(subset=["PLAYER_ID"]).copy()
        annual["PLAYER_ID"] = annual["PLAYER_ID"].astype(int)
        annual["position_group"] = annual["Pos"].map(_position_group)
        valid = annual["OREB_PCT"].notna() & annual["OffPoss"].gt(0)
        league_center = float(
            np.average(
                annual.loc[valid, "OREB_PCT"],
                weights=annual.loc[valid, "OffPoss"],
            )
        )
        position_centers = (
            annual.loc[valid]
            .groupby("position_group", sort=False)
            .apply(
                lambda group: float(
                    np.average(group["OREB_PCT"], weights=group["OffPoss"])
                ),
                include_groups=False,
            )
            .to_dict()
        )
        shrink = 500.0
        annual["oreb_pct_stable"] = (
            annual["OREB_PCT"].fillna(league_center) * annual["OffPoss"].fillna(0.0)
            + league_center * shrink
        ) / (annual["OffPoss"].fillna(0.0) + shrink)
        annual["position_oreb_center"] = annual["position_group"].map(
            position_centers
        ).fillna(league_center)
        annual["position_adjusted_oreb"] = (
            annual["oreb_pct_stable"] - annual["position_oreb_center"]
        )
        rows.append(
            annual.merge(
                starts.loc[starts["Season"].eq(season)],
                on=["PLAYER_ID", "Season"],
                how="left",
                validate="one_to_one",
            )
        )
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual context contains duplicate player-season keys.")
    return output


def _rolling_context(annual: pd.DataFrame, window_ends: range) -> pd.DataFrame:
    rows = []
    for end in window_ends:
        window = annual.loc[annual["Season"].between(end - 4, end)]
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            minutes = float(group["MIN"].fillna(0.0).sum())
            games = float(group["games"].fillna(group["GP"]).sum())
            starts = float(group["games_started"].fillna(0.0).sum())
            weights = group["OffPoss"].fillna(0.0).clip(lower=0.0)
            position_oreb = (
                float(np.average(group["position_adjusted_oreb"], weights=weights))
                if weights.sum() > 0
                else np.nan
            )
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "Window_End": int(end),
                    "minutes_5y": minutes,
                    "log_minutes_5y": float(np.log1p(minutes)),
                    "games_5y": games,
                    "games_started_5y": starts,
                    "starter_share_squared_5y": (starts / games) ** 2 if games > 0 else 0.0,
                    "position_adjusted_oreb_5y": position_oreb,
                }
            )
    output = pd.DataFrame(rows)
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Rolling context contains duplicate player-window keys.")
    return output


def _add_spacing(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    required = {"FG2A_p100", "FG2M_p100", "FG3A_p100", "FG3M_p100", "fg3_pct_eb"}
    if missing := sorted(required - set(output.columns)):
        raise ValueError(f"Spacing candidate lacks {missing}.")
    output["league_efg"] = np.nan
    for end, group in output.groupby("Window_End"):
        weights = group["courtsignal_exposure"].clip(lower=0.0)
        attempts = group["FG2A_p100"] + group["FG3A_p100"]
        made_value = group["FG2M_p100"] + 1.5 * group["FG3M_p100"]
        valid = attempts.gt(0) & weights.gt(0)
        denominator = float(np.sum(attempts[valid] * weights[valid]))
        league_efg = (
            float(np.sum(made_value[valid] * weights[valid]) / denominator)
            if denominator > 0
            else np.nan
        )
        output.loc[group.index, "league_efg"] = league_efg
    output["spacing_value_above_average_p100"] = output["FG3A_p100"] * (
        1.5 * output["fg3_pct_eb"] - output["league_efg"]
    )
    return output


def _correlation_audit(
    panel: pd.DataFrame,
    features: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    rows = []
    for side, columns in features.items():
        correlation = panel.loc[:, columns].corr(min_periods=100)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                value = correlation.at[left, right]
                if pd.notna(value) and abs(value) >= CORRELATION_REPORT_THRESHOLD:
                    rows.append(
                        {
                            "side": side,
                            "feature_a": left,
                            "feature_b": right,
                            "correlation": float(value),
                            "absolute_correlation": float(abs(value)),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["side", "absolute_correlation", "feature_a", "feature_b"],
        ascending=[True, False, True, True],
        kind="stable",
    )


def _pruned_features(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> tuple[str, ...]:
    numeric = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    weighted_target = train[target]
    scores = []
    for feature in features:
        valid = numeric[feature].notna() & weighted_target.notna()
        score = (
            abs(float(numeric.loc[valid, feature].corr(weighted_target.loc[valid])))
            if valid.sum() >= 50
            else 0.0
        )
        scores.append((feature, score, float(numeric[feature].isna().mean())))
    ordered = [
        feature
        for feature, _, _ in sorted(scores, key=lambda row: (-row[1], row[2], row[0]))
    ]
    correlation = numeric.corr(min_periods=100).abs()
    kept = []
    for feature in ordered:
        if not kept or all(
            pd.isna(correlation.at[feature, other])
            or correlation.at[feature, other] < CORRELATION_PRUNE_THRESHOLD
            for other in kept
        ):
            kept.append(feature)
    return tuple(kept)


def _target_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    side: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    weight = train["courtsignal_exposure"].to_numpy(dtype=float).copy()
    weight /= weight.sum()
    court = train[f"courtsignal_{side}"].to_numpy(dtype=float)
    ryan = train[f"ryan_{side}"].to_numpy(dtype=float)
    court_mean = float(np.sum(weight * court))
    ryan_mean = float(np.sum(weight * ryan))
    court_sd = float(np.sqrt(np.sum(weight * (court - court_mean) ** 2)))
    ryan_sd = float(np.sqrt(np.sum(weight * (ryan - ryan_mean) ** 2)))
    if ryan_sd <= 0:
        raise ValueError(f"Ryan {side} target has zero spread.")
    scale = court_sd / ryan_sd
    train[f"ryan_rescaled_{side}"] = (train[f"ryan_{side}"] - ryan_mean) * scale + court_mean
    test[f"ryan_rescaled_{side}"] = (test[f"ryan_{side}"] - ryan_mean) * scale + court_mean
    return train, test


def _fit_direct_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    *,
    ridge_only: bool,
) -> tuple[np.ndarray, ModelSpec, pd.DataFrame]:
    spec, scores = _select_model(
        train, features, target, ridge_only=ridge_only
    )
    fitted = _fit_model(spec, train, features, target)
    return fitted.predict(test.loc[:, features]), spec, scores


def _forward_predictions(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    spec: ModelSpec,
) -> np.ndarray:
    prediction = pd.Series(index=frame.index, dtype=float)
    for validation_end in sorted(frame["Window_End"].unique()):
        validation = frame.loc[frame["Window_End"].eq(validation_end)]
        fold_train = frame.loc[frame["Window_End"].lt(validation_end)]
        if fold_train["Window_End"].nunique() < 2:
            continue
        model = _fit_model(spec, fold_train, features, target)
        prediction.loc[validation.index] = model.predict(validation.loc[:, features])
    return prediction.loc[frame.index].to_numpy(dtype=float)


def _residual_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_features: tuple[str, ...],
    residual_features: tuple[str, ...],
    target: str,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    base_spec, base_scores = _select_model(
        train, base_features, target, ridge_only=True
    )
    base_oof = _forward_predictions(
        train, base_features, target, base_spec
    )
    eligible = np.isfinite(base_oof)
    residual_train = train.loc[eligible].copy()
    residual_train["base_oof"] = base_oof[eligible]
    residual_train["residual_target"] = (
        residual_train[target].to_numpy(dtype=float) - residual_train["base_oof"]
    )
    base_model = _fit_model(base_spec, train, base_features, target)
    base_test = base_model.predict(test.loc[:, base_features])
    if residual_train["Window_End"].nunique() < 2:
        selection = base_scores.assign(stage="base")
        return base_test, {
            "base_family": base_spec.family,
            "base_config": base_spec.config,
            "residual_family": None,
            "residual_config": None,
            "gamma": 0.0,
            "reason": "insufficient_forward_history",
        }, selection
    residual_spec, residual_scores = _select_model(
        residual_train,
        residual_features,
        "residual_target",
        ridge_only=False,
    )
    gamma_end = int(residual_train["Window_End"].max())
    gamma_train = residual_train.loc[residual_train["Window_End"].lt(gamma_end)]
    gamma_validation = residual_train.loc[residual_train["Window_End"].eq(gamma_end)]
    gamma_model = _fit_model(
        residual_spec, gamma_train, residual_features, "residual_target"
    )
    residual_oof = gamma_model.predict(gamma_validation.loc[:, residual_features])
    gamma_rows = []
    for gamma in GAMMA_GRID:
        prediction = gamma_validation["base_oof"].to_numpy(dtype=float) + gamma * residual_oof
        gamma_rows.append(
            {
                "gamma": gamma,
                "weighted_rmse": _weighted_rmse(
                    gamma_validation[target], prediction, gamma_validation["sample_weight"]
                ),
            }
        )
    gamma_scores = pd.DataFrame(gamma_rows).sort_values(
        ["weighted_rmse", "gamma"], kind="stable"
    )
    gamma = float(gamma_scores.iloc[0]["gamma"])
    residual_model = _fit_model(
        residual_spec, residual_train, residual_features, "residual_target"
    )
    prediction = base_test + gamma * residual_model.predict(
        test.loc[:, residual_features]
    )
    selection = pd.concat(
        [
            base_scores.assign(stage="base"),
            residual_scores.assign(stage="residual"),
            gamma_scores.assign(
                family="gamma",
                config=lambda frame: frame["gamma"].map(lambda value: json.dumps({"gamma": value})),
                validation_window_end=int(train["Window_End"].max()),
                stage="gamma",
            )[["family", "config", "validation_window_end", "weighted_rmse", "stage"]],
        ],
        ignore_index=True,
    )
    return prediction, {
        "base_family": base_spec.family,
        "base_config": base_spec.config,
        "residual_family": residual_spec.family,
        "residual_config": residual_spec.config,
        "gamma": gamma,
    }, selection


def _fit_priors(
    panel: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    pruned_rows = []
    target_metrics = []
    full_union = {
        side: tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *selected[side])))
        for side in ("offense", "defense")
    }
    context = {
        "offense": (
            "minutes_5y",
            "log_minutes_5y",
            "starter_share_squared_5y",
            "position_adjusted_oreb_5y",
            "spacing_value_above_average_p100",
        ),
        "defense": (
            "minutes_5y",
            "log_minutes_5y",
            "starter_share_squared_5y",
        ),
    }
    onoff = {
        "offense": ("raw_onoff_offense_5y",),
        "defense": ("raw_onoff_defense_5y",),
    }
    candidate_predictions: dict[tuple[int, str], pd.DataFrame] = {}

    for season in RATING_SEASONS:
        train_base = panel.loc[panel["Window_End"].lt(season)].copy()
        test_base = panel.loc[panel["Window_End"].eq(season)].copy()
        if train_base["Window_End"].nunique() < 3 or test_base.empty:
            raise ValueError(f"PIPM breaker fold {season} lacks chronological history.")
        train_base["sample_weight"] = train_base["courtsignal_exposure"]
        test_base["sample_weight"] = test_base["courtsignal_exposure"]
        for side in ("offense", "defense"):
            train, test = _target_columns(train_base, test_base, side)
            pruned = _pruned_features(
                train, full_union[side], f"ryan_rescaled_{side}"
            )
            pruned_rows.extend(
                {
                    "rating_season": season,
                    "side": side,
                    "feature": feature,
                    "kept": feature in pruned,
                }
                for feature in full_union[side]
            )
            definitions = {
                "box_courtsignal_ridge": (
                    BOX_PIPM_STYLE_FEATURES,
                    f"courtsignal_{side}",
                    True,
                ),
                "box_courtsignal_tuned": (
                    BOX_PIPM_STYLE_FEATURES,
                    f"courtsignal_{side}",
                    False,
                ),
                "box_ryan_ridge": (
                    BOX_PIPM_STYLE_FEATURES,
                    f"ryan_rescaled_{side}",
                    True,
                ),
                "box_ryan_tuned": (
                    BOX_PIPM_STYLE_FEATURES,
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "full_courtsignal_ridge": (
                    full_union[side],
                    f"courtsignal_{side}",
                    True,
                ),
                "full_courtsignal_tuned": (
                    full_union[side],
                    f"courtsignal_{side}",
                    False,
                ),
                "full_ryan_ridge": (
                    full_union[side],
                    f"ryan_rescaled_{side}",
                    True,
                ),
                "full_ryan_tuned": (
                    full_union[side],
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "pruned_full_ryan_tuned": (
                    pruned,
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "box_ryan_minutes_starts": (
                    tuple(
                        dict.fromkeys(
                            (
                                *BOX_PIPM_STYLE_FEATURES,
                                "log_minutes_5y",
                                "starter_share_squared_5y",
                            )
                        )
                    ),
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "box_ryan_raw_minutes": (
                    tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, "minutes_5y"))),
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "box_ryan_rebound_spacing": (
                    tuple(
                        dict.fromkeys(
                            (
                                *BOX_PIPM_STYLE_FEATURES,
                                *(
                                    (
                                        "position_adjusted_oreb_5y",
                                        "spacing_value_above_average_p100",
                                    )
                                    if side == "offense"
                                    else ()
                                ),
                            )
                        )
                    ),
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "box_ryan_onoff": (
                    tuple(dict.fromkeys((*BOX_PIPM_STYLE_FEATURES, *onoff[side]))),
                    f"ryan_rescaled_{side}",
                    False,
                ),
                "box_ryan_all_context": (
                    tuple(
                        dict.fromkeys(
                            (*BOX_PIPM_STYLE_FEATURES, *context[side], *onoff[side])
                        )
                    ),
                    f"ryan_rescaled_{side}",
                    False,
                ),
            }
            for candidate, (features, target, ridge_only) in definitions.items():
                prediction, spec, scores = _fit_direct_prior(
                    train,
                    test,
                    features,
                    target,
                    ridge_only=ridge_only,
                )
                key = (season, candidate)
                if key not in candidate_predictions:
                    candidate_predictions[key] = test[["PLAYER_ID", "Window_End"]].copy()
                candidate_predictions[key][side] = prediction
                selections.append(
                    scores.assign(
                        rating_season=season,
                        candidate=candidate,
                        side=side,
                        selected=lambda frame, spec=spec: (
                            frame["family"].eq(spec.family)
                            & frame["config"].eq(json.dumps(spec.config, sort_keys=True))
                        ),
                        feature_count=len(features),
                    )
                )
                error = test[target].to_numpy(dtype=float) - prediction
                target_metrics.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "target": target,
                        "feature_count": len(features),
                        "target_rmse": float(np.sqrt(np.mean(error**2))),
                        "target_correlation": float(
                            np.corrcoef(test[target].to_numpy(dtype=float), prediction)[0, 1]
                        ),
                    }
                )

            residual_features = tuple(
                feature
                for feature in dict.fromkeys(
                    (*full_union[side], *context[side])
                )
                if feature not in BOX_PIPM_STYLE_FEATURES
            )
            residual_prediction, residual_spec, residual_scores = _residual_prediction(
                train,
                test,
                BOX_PIPM_STYLE_FEATURES,
                residual_features,
                f"ryan_rescaled_{side}",
            )
            key = (season, "box_ryan_residual")
            if key not in candidate_predictions:
                candidate_predictions[key] = test[["PLAYER_ID", "Window_End"]].copy()
            candidate_predictions[key][side] = residual_prediction
            selections.append(
                residual_scores.assign(
                    rating_season=season,
                    candidate="box_ryan_residual",
                    side=side,
                    selected=False,
                    feature_count=len(residual_features),
                )
            )
            target_metrics.append(
                {
                    "rating_season": season,
                    "candidate": "box_ryan_residual",
                    "side": side,
                    "target": f"ryan_rescaled_{side}",
                    "feature_count": len(residual_features),
                    "target_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    test[f"ryan_rescaled_{side}"].to_numpy(dtype=float)
                                    - residual_prediction
                                )
                                ** 2
                            )
                        )
                    ),
                    "target_correlation": float(
                        np.corrcoef(
                            test[f"ryan_rescaled_{side}"].to_numpy(dtype=float),
                            residual_prediction,
                        )[0, 1]
                    ),
                    "residual_spec": json.dumps(residual_spec, sort_keys=True),
                }
            )

    for (_, candidate), frame in sorted(candidate_predictions.items()):
        if not {"offense", "defense"}.issubset(frame.columns):
            raise ValueError(f"Candidate {candidate} is missing one side.")
        frame["net"] = frame["offense"] + frame["defense"]
        rows.append(_prior_frame(frame, candidate))
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(selections, ignore_index=True),
        pd.DataFrame(pruned_rows),
        pd.DataFrame(target_metrics),
    )


def _lineup(value: object) -> tuple[int, ...]:
    players = tuple(
        sorted(
            int(player)
            for player in str(value).split("|")
            if player and player.lower() != "nan"
        )
    )
    return players


def _free_throws_made(events: object) -> int:
    return sum(
        "free throw" in line.lower() and "miss" not in line.lower()
        for line in str(events).splitlines()
    )


def _final_scores(schedule_root: Path, season: int) -> pd.DataFrame:
    with gzip.open(schedule_root / f"leaguegamelog_{season}.json.gz", "rt") as handle:
        payload = json.load(handle)
    result = payload["resultSets"][0]
    games = pd.DataFrame(result["rowSet"], columns=result["headers"])
    games["game_id"] = games["GAME_ID"].astype(str).str[-8:].astype(int)
    schedule = pd.read_parquet(
        schedule_root / f"schedule_{season}.parquet",
        columns=["game_id", "home_team_id", "away_team_id"],
    )
    schedule["game_id"] = schedule["game_id"].astype(str).str[-8:].astype(int)
    games = games.merge(schedule, on="game_id", validate="many_to_one")
    games["expected_home"] = np.where(
        games["TEAM_ID"].eq(games["home_team_id"]), games["PTS"], np.nan
    )
    games["expected_away"] = np.where(
        games["TEAM_ID"].eq(games["away_team_id"]), games["PTS"], np.nan
    )
    return games.groupby("game_id", as_index=False).agg(
        expected_home=("expected_home", "max"),
        expected_away=("expected_away", "max"),
    )


def _build_poss_data_cache(
    source_root: Path,
    schedule_root: Path,
    season: int,
) -> tuple[pd.DataFrame, dict]:
    derived_root = source_root / "derived"
    cache_path = derived_root / f"matchups_{season}.parquet"
    manifest_path = cache_path.with_suffix(".manifest.json")
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("output_sha256") == sha256_file(cache_path):
            return pd.read_parquet(cache_path), manifest["quality"]

    files = sorted(source_root.glob(f"{season}/???_{season}_clips_with_players.csv"))
    files = [
        path
        for path in files
        if re.fullmatch(
            rf"[A-Z]{{3}}_{season}_clips_with_players\.csv", path.name
        )
    ]
    if len(files) != 30:
        raise ValueError(f"Expected 30 possession files for {season}, found {len(files)}.")
    required = [
        "index",
        "EVENTS",
        "FG2M",
        "FG3M",
        "GAMEDATE",
        "GAMEID",
        "PERIOD",
        "STARTSCOREDIFFERENTIAL",
        "STARTTIME",
        "ENDTIME",
        "STARTTYPE",
        "TEAM_ID",
        "players_on",
        "opp_players_on",
    ]
    source = pd.concat(
        [pd.read_csv(path, usecols=required, low_memory=False) for path in files],
        ignore_index=True,
    )
    key = [
        "GAMEID",
        "TEAM_ID",
        "PERIOD",
        "STARTTIME",
        "ENDTIME",
        "STARTTYPE",
        "STARTSCOREDIFFERENTIAL",
        "EVENTS",
    ]
    source["index"] = pd.to_numeric(source["index"], errors="coerce")
    source["offense_lineup"] = source["players_on"].map(_lineup)
    source["defense_lineup"] = source["opp_players_on"].map(_lineup)
    source["valid_lineup"] = (
        source["offense_lineup"].map(len).eq(5)
        & source["defense_lineup"].map(len).eq(5)
        & source["offense_lineup"].map(lambda values: len(set(values))).eq(5)
        & source["defense_lineup"].map(lambda values: len(set(values))).eq(5)
        & ~pd.Series(
            [
                bool(set(offense) & set(defense))
                for offense, defense in zip(
                    source["offense_lineup"], source["defense_lineup"]
                )
            ],
            index=source.index,
        )
    )
    source = source.sort_values([*key, "index"], kind="stable")
    valid_terminal = source.loc[source["valid_lineup"]].drop_duplicates(
        key, keep="last"
    )
    fallback_terminal = source.drop_duplicates(key, keep="last")
    possessions = pd.concat(
        [
            valid_terminal,
            fallback_terminal.loc[
                ~pd.MultiIndex.from_frame(fallback_terminal[key]).isin(
                    pd.MultiIndex.from_frame(valid_terminal[key])
                )
            ],
        ],
        ignore_index=True,
    )
    valid_lineup = possessions["valid_lineup"]
    invalid_lineup_rows = int((~valid_lineup).sum())
    possessions["pts"] = (
        2.0 * pd.to_numeric(possessions["FG2M"], errors="coerce").fillna(0.0)
        + 3.0 * pd.to_numeric(possessions["FG3M"], errors="coerce").fillna(0.0)
        + possessions["EVENTS"].map(_free_throws_made)
    )
    schedule = pd.read_parquet(
        schedule_root / f"schedule_{season}.parquet",
        columns=["game_id", "game_date", "home_team_id", "away_team_id"],
    )
    schedule["GAMEID"] = schedule["game_id"].astype(str).str[-8:].astype(int)
    possessions["GAMEID"] = pd.to_numeric(possessions["GAMEID"], errors="raise").astype(int)
    possessions["TEAM_ID"] = pd.to_numeric(possessions["TEAM_ID"], errors="raise").astype(int)
    possessions = possessions.merge(
        schedule[["GAMEID", "game_date", "home_team_id", "away_team_id"]],
        on="GAMEID",
        how="inner",
        validate="many_to_one",
    )
    possessions["home_poss"] = possessions["TEAM_ID"].eq(
        possessions["home_team_id"]
    ).astype(int)
    wrong_team = ~(
        possessions["TEAM_ID"].eq(possessions["home_team_id"])
        | possessions["TEAM_ID"].eq(possessions["away_team_id"])
    )
    if wrong_team.any():
        raise ValueError(f"Possession source has {int(wrong_team.sum())} invalid team-game rows.")
    possessions["start_seconds"] = (
        (pd.to_numeric(possessions["PERIOD"], errors="raise") - 1) * 720
        + possessions["STARTTIME"].astype(str).str.split(":").map(
            lambda values: 720 - (int(values[0]) * 60 + int(values[1]))
        )
    )
    possessions = possessions.sort_values(
        ["GAMEID", "start_seconds", "index"], kind="stable"
    ).reset_index(drop=True)
    repaired_rows = 0
    game_positions = {
        game_id: list(indices)
        for game_id, indices in possessions.groupby("GAMEID", sort=False).groups.items()
    }
    for game_id, indices in game_positions.items():
        local_position = {index: position for position, index in enumerate(indices)}
        for index in indices:
            if bool(possessions.at[index, "valid_lineup"]):
                continue
            team_id = int(possessions.at[index, "TEAM_ID"])
            position = local_position[index]
            candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            for distance in range(1, 5):
                for candidate_position in (position + distance, position - distance):
                    if not 0 <= candidate_position < len(indices):
                        continue
                    candidate_index = indices[candidate_position]
                    if (
                        bool(possessions.at[candidate_index, "valid_lineup"])
                        and int(possessions.at[candidate_index, "TEAM_ID"]) != team_id
                    ):
                        candidates.append(
                            (
                                possessions.at[candidate_index, "defense_lineup"],
                                possessions.at[candidate_index, "offense_lineup"],
                            )
                        )
                if candidates:
                    break
            if candidates:
                offense, defense = candidates[0]
                possessions.at[index, "offense_lineup"] = offense
                possessions.at[index, "defense_lineup"] = defense
                possessions.at[index, "valid_lineup"] = True
                repaired_rows += 1
    unresolved_lineup_rows = int((~possessions["valid_lineup"]).sum())
    possessions = possessions.loc[possessions["valid_lineup"]].copy()
    possessions["num"] = possessions.groupby("GAMEID").cumcount().add(1)

    rows = []
    for row in possessions.itertuples(index=False):
        offense = row.offense_lineup
        defense = row.defense_lineup
        home = offense if row.home_poss else defense
        away = defense if row.home_poss else offense
        rows.append(
            {
                "home_poss": int(row.home_poss),
                "pts": float(row.pts),
                **{f"a{index}": int(player) for index, player in enumerate(away, 1)},
                **{f"h{index}": int(player) for index, player in enumerate(home, 1)},
                "season": season,
                "date": row.game_date,
                "period": int(row.PERIOD),
                "num": int(row.num),
                "gameid": str(int(row.GAMEID)).zfill(10),
            }
        )
    frame = pd.DataFrame(rows)
    expected = _final_scores(schedule_root, season)
    scored = frame.assign(
        game_id=frame["gameid"].astype(str).str[-8:].astype(int),
        home_points=np.where(frame["home_poss"].eq(1), frame["pts"], 0.0),
        away_points=np.where(frame["home_poss"].eq(0), frame["pts"], 0.0),
    ).groupby("game_id", as_index=False).agg(
        observed_home=("home_points", "sum"),
        observed_away=("away_points", "sum"),
        possessions=("pts", "size"),
    )
    score_check = scored.merge(expected, on="game_id", how="left", validate="one_to_one")
    score_check["side_score_matches"] = (
        score_check["observed_home"].eq(score_check["expected_home"])
        & score_check["observed_away"].eq(score_check["expected_away"])
    )
    score_check["total_score_matches"] = (
        score_check["observed_home"] + score_check["observed_away"]
    ).eq(score_check["expected_home"] + score_check["expected_away"])
    failed_games = set(
        score_check.loc[~score_check["total_score_matches"], "game_id"]
    )
    if failed_games:
        frame = frame.loc[
            ~frame["gameid"].astype(str).str[-8:].astype(int).isin(failed_games)
        ].copy()
        frame["num"] = frame.groupby("gameid").cumcount().add(1)
    report = audit_possession_frame(frame, expected_season=season)
    if not report.passed:
        failures = "; ".join(
            f"{issue.code}={issue.count}" for issue in report.issues
        )
        raise ValueError(f"Derived possession cache failed QA for {season}: {failures}")
    quality = {
        "season": season,
        "source_files": len(files),
        "raw_event_rows": int(len(source)),
        "candidate_possessions": int(len(possessions)),
        "invalid_lineup_rows": invalid_lineup_rows,
        "lineup_rows_repaired": repaired_rows,
        "unresolved_lineup_rows": unresolved_lineup_rows,
        "score_mismatch_games_quarantined": len(failed_games),
        "retained_games": int(frame["gameid"].nunique()),
        "retained_possessions": int(len(frame)),
        "side_score_match_rate_before_quarantine": float(
            score_check["side_score_matches"].mean()
        ),
        "total_score_match_rate_before_quarantine": float(
            score_check["total_score_matches"].mean()
        ),
    }
    derived_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    write_json_atomic(
        {
            "dataset": "gabriel_poss_data_terminal_lineups",
            "season": season,
            "source_repository": "https://github.com/gabriel1200/poss_data",
            "source_files": [
                {"path": str(path.relative_to(source_root)), "sha256": sha256_file(path)}
                for path in files
            ],
            "output_sha256": sha256_file(cache_path),
            "quality": quality,
        },
        manifest_path,
    )
    return frame, quality


def _fit_aio(
    priors: pd.DataFrame,
    *,
    possession_source: Path,
    schedule_root: Path,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = []
    games = []
    coverage = []
    possession_quality = []
    candidates = tuple(sorted(priors["candidate"].unique()))
    for season in RATING_SEASONS:
        possession_frame, quality = _build_poss_data_cache(
            possession_source,
            schedule_root,
            season,
        )
        possession_quality.append(quality)
        direct = _annual_from_frame(possession_frame, season)
        matrix_dir = matrix_root / f"5y_end_{season}"
        annual = _remap_annual(direct, np.load(matrix_dir / "player_ids.npy"))
        for candidate in candidates:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(season)
            ]
            center, report = _center(prior, annual)
            beta, intercept = _solve(annual, center, scale=1.0)
            game = stored_evaluation_predictions(matrix_dir, beta, intercept)
            game["candidate"] = candidate
            game["rating_season"] = season
            game["test_season"] = season + 1
            game["squared_error"] = (
                game["actual_margin"] - game["predicted_margin"]
            ) ** 2
            games.append(game)
            n = len(annual.players)
            rating = pd.DataFrame(
                {
                    "PLAYER_ID": annual.players,
                    "offense": 100.0 * beta[:n],
                    "defense": -100.0 * beta[n : 2 * n],
                    "Poss_Off": annual.off_possessions,
                    "Poss_Def": annual.def_possessions,
                }
            )
            rating["net"] = rating["offense"] + rating["defense"]
            rating["candidate"] = candidate
            rating["rating_season"] = season
            ratings.append(rating)
            coverage.append(
                {"candidate": candidate, "rating_season": season, **report}
            )
        print(f"PIPM breaker AIO fold {season}->{season + 1}: complete", flush=True)
    return (
        pd.concat(ratings, ignore_index=True),
        pd.concat(games, ignore_index=True),
        pd.DataFrame(coverage),
        pd.DataFrame(possession_quality),
    )


def _game_summary(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = games.groupby(["test_season", "candidate"])["game_id"].nunique().unstack()
    if counts.isna().any().any() or not counts.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates do not score identical games.")
    actual_hashes = (
        games.assign(
            actual_key=lambda frame: (
                frame["game_id"].astype(str)
                + ":"
                + frame["actual_margin"].astype(str)
            )
        )
        .groupby(["test_season", "candidate"])["actual_key"]
        .apply(
            lambda values: hashlib.sha256(
                "|".join(sorted(values)).encode()
            ).hexdigest()
        )
        .unstack()
    )
    if not actual_hashes.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates do not score identical outcomes.")
    rows = []
    for (candidate, rating_season, test_season), frame in games.groupby(
        ["candidate", "rating_season", "test_season"]
    ):
        rows.append(
            {
                "candidate": candidate,
                "rating_season": int(rating_season),
                "test_season": int(test_season),
                **_game_metrics(frame),
            }
        )
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    return metrics, summary


def _load_attached_pipm(
    path: Path,
    player_sheet_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(path)
    source["Season"] = pd.to_numeric(
        source["Season"].astype(str).str[-2:], errors="raise"
    ).astype(int)
    source["Season"] = np.where(source["Season"] >= 70, 1900 + source["Season"], 2000 + source["Season"])
    source["name_key"] = source["Player"].map(normalize_player_name)
    for column in ("MP", "O-PIPM", "D-PIPM", "PIPM", "Wins Added"):
        source[column] = pd.to_numeric(
            source[column].astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False),
            errors="coerce",
        )
    source = source.loc[source["Season"].between(2014, 2021)].copy()
    identities = []
    for season in range(2014, 2022):
        frame, _ = _load_source(player_sheet_root / f"{season}.parquet", season)
        keep = frame[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates().copy()
        keep["Season"] = season
        keep["name_key"] = keep["PLAYER_NAME"].map(normalize_player_name)
        identities.append(keep)
    identity = pd.concat(identities, ignore_index=True)
    ambiguous = identity.groupby(["Season", "name_key"])["PLAYER_ID"].nunique()
    ambiguous_keys = ambiguous.loc[ambiguous.gt(1)].index
    ambiguous_frame = pd.MultiIndex.from_frame(identity[["Season", "name_key"]]).isin(ambiguous_keys)
    identity = identity.loc[~ambiguous_frame].drop_duplicates(["Season", "name_key"])
    matched = source.merge(
        identity[["PLAYER_ID", "PLAYER_NAME", "Season", "name_key"]],
        on=["Season", "name_key"],
        how="left",
        validate="many_to_one",
    )
    coverage = pd.DataFrame(
        [
            {
                "rows": len(source),
                "matched_rows": int(matched["PLAYER_ID"].notna().sum()),
                "source_minutes": float(source["MP"].sum()),
                "matched_minutes": float(matched.loc[matched["PLAYER_ID"].notna(), "MP"].sum()),
                "minute_match_rate": float(
                    matched.loc[matched["PLAYER_ID"].notna(), "MP"].sum() / source["MP"].sum()
                ),
                "duplicate_player_season_names": int(source.duplicated(["Player", "Season"]).sum()),
                "scope": "regular season plus playoffs",
            }
        ]
    )
    return matched.dropna(subset=["PLAYER_ID"]).assign(PLAYER_ID=lambda frame: frame["PLAYER_ID"].astype(int)), coverage


def _pipm_agreement(
    ratings: pd.DataFrame,
    pipm: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate, candidate_ratings in ratings.loc[
        ratings["rating_season"].eq(2021)
    ].groupby("candidate"):
        matched = candidate_ratings.merge(
            pipm.loc[pipm["Season"].eq(2021), ["PLAYER_ID", "O-PIPM", "D-PIPM", "PIPM", "MP"]],
            on="PLAYER_ID",
            how="inner",
            validate="one_to_one",
        )
        matched = matched.loc[matched["MP"].ge(250)].copy()
        for component, external in (
            ("offense", "O-PIPM"),
            ("defense", "D-PIPM"),
            ("net", "PIPM"),
        ):
            rows.append(
                {
                    "candidate": candidate,
                    "component": component,
                    "rows": len(matched),
                    "pearson": float(matched[component].corr(matched[external])),
                    "spearman": float(matched[component].corr(matched[external], method="spearman")),
                    "rmse": float(np.sqrt(np.mean((matched[component] - matched[external]) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/research/spm_target_horizon_full/spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument(
        "--ryan-ratings",
        type=Path,
        default=ROOT / "research/rapm_lab/data/external/user_downloads/ryan_davis_multi_rapm.csv",
    )
    parser.add_argument(
        "--player-sheet-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
    )
    parser.add_argument(
        "--bbref-root",
        type=Path,
        default=ROOT / "data/lake/bronze/basketball_reference/player_totals",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=ROOT / "data/lake/bronze/official_game_schedule_1997_2026",
    )
    parser.add_argument(
        "--possession-source",
        type=Path,
        default=ROOT / "research/rapm_lab/external/external/poss_data",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument(
        "--pipm-csv",
        type=Path,
        default=Path("/Users/eadebayo/Downloads/PIPM Player Finder through 2021 - Database.csv"),
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "research/rapm_lab/external/external/pipm_breaker_checkpoints",
    )
    args = parser.parse_args()

    if 2027 in RATING_SEASONS or 2027 in tuple(season + 1 for season in RATING_SEASONS):
        raise ValueError("Season 2027 is reserved and must not be loaded.")
    panel, selected = _panel(args.features, args.targets, args.ryan_ratings)
    starts, starts_coverage = _bbref_starts_minutes(
        args.bbref_root, args.player_sheet_root, range(2014, 2024)
    )
    annual = _annual_context(args.player_sheet_root, starts, range(2014, 2024))
    rolling = _rolling_context(annual, range(2018, 2024))
    team = _team_context(args.schedule_root, range(2014, 2024))
    annual_onoff = _annual_onoff(args.player_sheet_root, team, range(2014, 2024))
    rolling_onoff = _rolling_onoff(annual_onoff, tuple(range(2018, 2024)))
    panel = panel.merge(
        rolling, on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one"
    ).merge(
        rolling_onoff,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    panel = _add_spacing(panel)
    auxiliary = (
        "minutes_5y",
        "log_minutes_5y",
        "starter_share_squared_5y",
        "position_adjusted_oreb_5y",
        "spacing_value_above_average_p100",
        "raw_onoff_offense_5y",
        "raw_onoff_defense_5y",
    )
    if panel.loc[:, auxiliary].isna().all().any():
        raise ValueError("At least one auxiliary feature is entirely missing.")

    audit_features = {
        "offense": tuple(
            dict.fromkeys(
                (
                    *BOX_PIPM_STYLE_FEATURES,
                    *selected["offense"],
                    *auxiliary,
                )
            )
        ),
        "defense": tuple(
            dict.fromkeys(
                (
                    *BOX_PIPM_STYLE_FEATURES,
                    *selected["defense"],
                    *auxiliary,
                )
            )
        ),
    }
    correlation = _correlation_audit(panel, audit_features)
    checkpoint_identity = hashlib.sha256(
        json.dumps(
            {
                "version": PRIOR_CHECKPOINT_VERSION,
                "features": sha256_file(args.features),
                "targets": sha256_file(args.targets),
                "ryan": sha256_file(args.ryan_ratings),
                "selected": {side: list(values) for side, values in selected.items()},
                "auxiliary": list(auxiliary),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    checkpoint = args.checkpoint_root / checkpoint_identity
    checkpoint_paths = {
        "priors": checkpoint / "priors.parquet",
        "selections": checkpoint / "selections.parquet",
        "pruned": checkpoint / "pruned.parquet",
        "target_metrics": checkpoint / "target_metrics.parquet",
    }
    if all(path.exists() for path in checkpoint_paths.values()):
        priors = pd.read_parquet(checkpoint_paths["priors"])
        selections = pd.read_parquet(checkpoint_paths["selections"])
        pruned = pd.read_parquet(checkpoint_paths["pruned"])
        target_metrics = pd.read_parquet(checkpoint_paths["target_metrics"])
        print(f"Loaded prior checkpoint {checkpoint.name}", flush=True)
    else:
        priors, selections, pruned, target_metrics = _fit_priors(panel, selected)
        checkpoint.mkdir(parents=True, exist_ok=True)
        priors.to_parquet(checkpoint_paths["priors"], index=False)
        selections.to_parquet(checkpoint_paths["selections"], index=False)
        pruned.to_parquet(checkpoint_paths["pruned"], index=False)
        target_metrics.to_parquet(checkpoint_paths["target_metrics"], index=False)
        write_json_atomic(
            {
                "version": PRIOR_CHECKPOINT_VERSION,
                "identity": checkpoint_identity,
                "paths": {name: path.name for name, path in checkpoint_paths.items()},
            },
            checkpoint / "manifest.json",
        )
        print(f"Saved prior checkpoint {checkpoint.name}", flush=True)
    ratings, games, prior_coverage, possession_quality = _fit_aio(
        priors,
        possession_source=args.possession_source,
        schedule_root=args.schedule_root,
        matrix_root=args.matrix_root,
    )
    game_metrics, summary = _game_summary(games)
    bootstrap = _paired_bootstrap(
        games,
        baseline="box_courtsignal_ridge",
        draws=5_000,
        seed=20260827,
    )
    pipm, pipm_coverage = _load_attached_pipm(args.pipm_csv, args.player_sheet_root)
    pipm_agreement = _pipm_agreement(ratings, pipm)

    source_paths = {
        "contract": ROOT / "research/experiments/pipm_breaker_v1.yml",
        "features": args.features,
        "targets": args.targets,
        "ryan_ratings": args.ryan_ratings,
        "attached_pipm": args.pipm_csv,
        "runner": Path(__file__),
        **{
            f"bbref_totals_{season}": args.bbref_root / f"nba_{season}_totals.html"
            for season in range(2014, 2024)
        },
        **{
            f"player_sheet_{season}": args.player_sheet_root / f"{season}.parquet"
            for season in range(2014, 2024)
        },
        **{
            f"possession_cache_{season}": args.possession_source / "derived" / f"matchups_{season}.parquet"
            for season in RATING_SEASONS
        },
        **{
            f"possession_manifest_{season}": args.possession_source / "derived" / f"matchups_{season}.manifest.json"
            for season in RATING_SEASONS
        },
    }
    config = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "box_features": list(BOX_PIPM_STYLE_FEATURES),
        "selected_features": {side: list(values) for side, values in selected.items()},
        "auxiliary_features": list(auxiliary),
        "alpha_grid": list(ALPHA_GRID),
        "gamma_grid": list(GAMMA_GRID),
        "correlation_report_threshold": CORRELATION_REPORT_THRESHOLD,
        "correlation_prune_threshold": CORRELATION_PRUNE_THRESHOLD,
        "aio": {
            "lambda_offense": 3000.0,
            "lambda_defense": 3000.0,
            "lambda_home": 300.0,
            "center_scale": 1.0,
            "likelihood_seasons": 1,
        },
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.artifact_root / "research/pipm_breaker" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "summary": summary,
        "game_metrics": game_metrics,
        "game_predictions": games,
        "prior_ratings": priors,
        "posterior_ratings": ratings,
        "prior_coverage": prior_coverage,
        "possession_source_quality": possession_quality,
        "model_selection": selections,
        "target_metrics": target_metrics,
        "feature_correlations": correlation,
        "pruned_feature_decisions": pruned,
        "starts_source_coverage": starts_coverage,
        "pipm_source_coverage": pipm_coverage,
        "pipm_agreement": pipm_agreement,
        "paired_bootstrap": bootstrap,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    component_error = float(
        (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
    )
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "estimand_id": "prior_validation_suite_v1",
        "status": "reused_historical_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "panel_windows": sorted(panel["Window_End"].unique().tolist()),
            "minimum_bbref_minute_match_rate": float(starts_coverage["minute_match_rate"].min()),
            "pipm_minute_match_rate": float(pipm_coverage.iloc[0]["minute_match_rate"]),
            "identical_game_rows": True,
            "component_identity_max_error": component_error,
            "season_2027_rows": 0,
        },
        "paths": {name: f"{name}.parquet" for name in outputs},
        "caveats": [
            "All scored seasons are reused development evidence.",
            "The attached PIPM file combines regular season and playoffs and is used only for agreement.",
            "Minutes and starts encode availability and role, not a basketball skill.",
            "Ordinary on/off reuses lineup-outcome information also present in the AIO likelihood.",
            "Correlation pruning is a separately scored candidate and does not alter the full feature bank.",
            "Season 2027 remains untouched.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
