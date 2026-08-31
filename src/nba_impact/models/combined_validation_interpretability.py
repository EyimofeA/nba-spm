"""Exact row-aligned validation and additive Box15 AIO accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from nba_impact.models.impact_validation_suite import COMPONENTS, weighted_correlation


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{label} is missing {missing}.")


def reject_season_2027(frame: pd.DataFrame, season_columns: Sequence[str]) -> None:
    """Fail closed if any supplied panel contains the untouched 2027 season."""
    for column in season_columns:
        if column in frame and frame[column].dropna().astype(int).ge(2027).any():
            raise ValueError(f"{column} contains forbidden Season 2027 or later.")


def align_prior_predictions(
    priors: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    candidates: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """Intersect player-window rows before scoring statistical priors."""
    candidates = tuple(candidates)
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("Prior validation requires at least two unique candidates.")
    _require_columns(
        priors,
        {
            "PLAYER_ID",
            "Window_End",
            "candidate",
            "prior_offense_per_100",
            "prior_defense_per_100",
            "prior_net_per_100",
        },
        "Prior panel",
    )
    _require_columns(
        targets,
        {
            "PLAYER_ID",
            "Window_End",
            "target_offense",
            "target_defense",
            "target_net",
        },
        "Target panel",
    )
    targets = targets.copy()
    if "sample_weight" not in targets:
        _require_columns(targets, {"Poss_Off", "Poss_Def"}, "Target panel")
        targets["sample_weight"] = np.sqrt(
            np.minimum(targets["Poss_Off"], targets["Poss_Def"]).clip(lower=1)
        )
    reject_season_2027(priors, ("Window_End",))
    reject_season_2027(targets, ("Window_End",))
    selected = priors.loc[priors["candidate"].isin(candidates)].copy()
    key = ["PLAYER_ID", "Window_End"]
    if selected.duplicated(["candidate", *key]).any():
        raise ValueError("Prior candidate-player-window keys must be unique.")
    if targets.duplicated(key).any():
        raise ValueError("Target player-window keys must be unique.")
    found = set(selected["candidate"].unique())
    if missing := sorted(set(candidates) - found):
        raise ValueError(f"Prior panel lacks candidates {missing}.")
    common = (
        selected.groupby(key, as_index=False)["candidate"]
        .nunique()
        .loc[lambda frame: frame["candidate"].eq(len(candidates)), key]
    )
    if common.empty:
        raise ValueError("Prior candidates have no common player-window rows.")
    aligned = selected.merge(common, on=key, how="inner", validate="many_to_one")
    aligned = aligned.merge(targets, on=key, how="inner", validate="many_to_one")
    expected = len(common) * len(candidates)
    if len(aligned) != expected:
        raise ValueError("One or more common prior rows lack a target.")
    counts = aligned.groupby("candidate").size()
    if not counts.eq(len(common)).all():
        raise AssertionError("Prior candidates were not scored on identical rows.")
    audit = {
        "candidate_count": len(candidates),
        "common_player_windows": int(len(common)),
        "rows_per_candidate": int(len(common)),
        "seasons": int(common["Window_End"].nunique()),
        "identical_row_set": True,
        "season_2027_rows": 0,
    }
    return aligned.sort_values(["Window_End", "candidate", "PLAYER_ID"]), audit


def score_prior_predictions(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute component metrics by season and equal-season summaries."""
    rows: list[dict] = []
    for (season, candidate), fold in aligned.groupby(
        ["Window_End", "candidate"], sort=True
    ):
        weight = fold["sample_weight"].to_numpy(dtype=float)
        for component in COMPONENTS:
            actual = fold[f"target_{component}"].to_numpy(dtype=float)
            predicted = fold[f"prior_{component}_per_100"].to_numpy(dtype=float)
            error = actual - predicted
            rows.append(
                {
                    "rating_season": int(season),
                    "candidate": candidate,
                    "component": component,
                    "players": int(len(fold)),
                    "weighted_mse": float(np.average(error**2, weights=weight)),
                    "weighted_rmse": float(
                        np.sqrt(np.average(error**2, weights=weight))
                    ),
                    "weighted_correlation": weighted_correlation(
                        actual, predicted, weight
                    ),
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["candidate", "component"], as_index=False)
        .agg(
            folds=("rating_season", "nunique"),
            mean_weighted_mse=("weighted_mse", "mean"),
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_correlation=("weighted_correlation", "mean"),
        )
        .sort_values(["component", "mean_weighted_rmse", "candidate"])
        .reset_index(drop=True)
    )
    return folds, summary


def align_game_predictions(
    predictions: pd.DataFrame,
    *,
    candidates: Sequence[str],
    key_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """Require identical complete games and outcomes for every candidate."""
    candidates = tuple(candidates)
    key_columns = tuple(key_columns)
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("Game validation requires at least two unique candidates.")
    _require_columns(
        predictions,
        {"candidate", "actual_margin", "predicted_margin", *key_columns},
        "Game prediction panel",
    )
    reject_season_2027(
        predictions,
        tuple(column for column in key_columns if "season" in column.lower()),
    )
    selected = predictions.loc[predictions["candidate"].isin(candidates)].copy()
    if selected.duplicated(["candidate", *key_columns]).any():
        raise ValueError("Game candidate keys must be unique.")
    found = set(selected["candidate"].unique())
    if missing := sorted(set(candidates) - found):
        raise ValueError(f"Game panel lacks candidates {missing}.")
    common = (
        selected.groupby(list(key_columns), as_index=False)["candidate"]
        .nunique()
        .loc[lambda frame: frame["candidate"].eq(len(candidates)), list(key_columns)]
    )
    if common.empty:
        raise ValueError("Game candidates have no common evaluation rows.")
    aligned = selected.merge(
        common, on=list(key_columns), how="inner", validate="many_to_one"
    )
    outcome_counts = aligned.groupby(list(key_columns))["actual_margin"].nunique()
    if not outcome_counts.eq(1).all():
        raise ValueError("Candidates do not share identical actual game margins.")
    counts = aligned.groupby("candidate").size()
    if not counts.eq(len(common)).all():
        raise AssertionError("Game candidates were not scored on identical rows.")
    audit = {
        "candidate_count": len(candidates),
        "common_games": int(len(common)),
        "rows_per_candidate": int(len(common)),
        "identical_row_set": True,
        "identical_actual_margins": True,
        "season_2027_rows": 0,
    }
    return aligned.sort_values([*key_columns, "candidate"]), audit


def game_metrics(frame: pd.DataFrame) -> dict:
    """Score final game margins and calibration."""
    actual = frame["actual_margin"].to_numpy(dtype=float)
    predicted = frame["predicted_margin"].to_numpy(dtype=float)
    error = actual - predicted
    predicted_variance = float(np.var(predicted, ddof=0))
    slope = (
        float(np.cov(actual, predicted, ddof=0)[0, 1] / predicted_variance)
        if predicted_variance > 0
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return {
        "games": int(len(frame)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": correlation,
        "calibration_intercept": float(np.mean(actual) - slope * np.mean(predicted)),
        "calibration_slope": slope,
        "actual_margin_sd": float(np.std(actual, ddof=0)),
        "predicted_margin_sd": float(np.std(predicted, ddof=0)),
    }


def score_game_predictions(
    aligned: pd.DataFrame,
    *,
    fold_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fold metrics and equal-fold summaries without pooling seasons."""
    fold_columns = tuple(fold_columns)
    rows = []
    for keys, frame in aligned.groupby([*fold_columns, "candidate"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip([*fold_columns, "candidate"], keys, strict=True)),
                **game_metrics(frame),
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            folds=(fold_columns[0], "size"),
            mean_mse=("mse", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_correlation=("correlation", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values(["mean_mse", "candidate"])
        .reset_index(drop=True)
    )
    summary["equal_fold_rmse_from_mse"] = np.sqrt(summary["mean_mse"])
    return folds, summary


def paired_game_bootstrap(
    aligned: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    season_column: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired whole games within season, then weight seasons equally."""
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required.")
    selected = aligned.loc[aligned["candidate"].isin((candidate, reference))]
    seasons: list[tuple[np.ndarray, np.ndarray]] = []
    point_deltas = []
    for _, fold in selected.groupby(season_column, sort=True):
        wide = fold.pivot(index="game_id", columns="candidate")
        actual = wide["actual_margin"]
        if actual.isna().any().any() or not actual.nunique(axis=1).eq(1).all():
            raise ValueError("Bootstrap candidates must share complete outcomes.")
        candidate_error = (
            actual[candidate].to_numpy(dtype=float)
            - wide["predicted_margin"][candidate].to_numpy(dtype=float)
        ) ** 2
        reference_error = (
            actual[reference].to_numpy(dtype=float)
            - wide["predicted_margin"][reference].to_numpy(dtype=float)
        ) ** 2
        seasons.append((candidate_error, reference_error))
        point_deltas.append(float(candidate_error.mean() - reference_error.mean()))
    if not seasons:
        raise ValueError("Bootstrap received no common seasons.")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=float)
    for draw in range(draws):
        deltas = []
        for candidate_error, reference_error in seasons:
            index = rng.integers(0, len(candidate_error), len(candidate_error))
            deltas.append(
                float(candidate_error[index].mean() - reference_error[index].mean())
            )
        sampled[draw] = float(np.mean(deltas))
    low, high = np.quantile(sampled, (0.025, 0.975))
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "reference": reference,
                "seasons": len(seasons),
                "mean_mse_delta_candidate_minus_reference": float(
                    np.mean(point_deltas)
                ),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(sampled < 0)),
                "bootstrap_draws": draws,
                "seed": seed,
            }
        ]
    )


def linear_group_contributions(
    model,
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Decompose a fitted linear pipeline exactly into intercept and groups."""
    feature_names = tuple(feature_names)
    assigned = [feature for values in groups.values() for feature in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(feature_names):
        raise ValueError("Interpretation groups must partition the model features.")
    transformed = np.asarray(model[:-1].transform(features.loc[:, feature_names]))
    transformed_names = tuple(model[:-1].get_feature_names_out(feature_names))
    if transformed_names != feature_names:
        raise ValueError("Exact accounting does not allow added imputation indicators.")
    estimator = model.steps[-1][1]
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) != len(feature_names):
        raise ValueError("Exact accounting requires a one-output linear estimator.")
    raw = transformed * coefficients
    output = pd.DataFrame(index=features.index)
    output["prior_intercept"] = float(estimator.intercept_)
    for group, names in groups.items():
        indexes = [feature_names.index(name) for name in names]
        output[group] = raw[:, indexes].sum(axis=1)
    output["raw_prediction"] = model.predict(features.loc[:, feature_names])
    reconstructed = output[["prior_intercept", *groups]].sum(axis=1)
    output["identity_error"] = output["raw_prediction"] - reconstructed
    if output["identity_error"].abs().max() > 1e-10:
        raise AssertionError("Linear group contributions do not reconstruct predictions.")
    return output


def build_aio_component_ledger(
    *,
    feature_panel: pd.DataFrame,
    raw_priors: pd.DataFrame,
    active_leaderboard: pd.DataFrame,
    models: Mapping[str, object],
    feature_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    rating_season: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the exact centered-prior plus RAPM-update identity for active players."""
    if rating_season >= 2027:
        raise ValueError("Season 2027 is forbidden.")
    key = "PLAYER_ID"
    current_features = feature_panel.loc[
        feature_panel["Window_End"].eq(rating_season), [key, *feature_names]
    ].copy()
    if current_features.duplicated(key).any():
        raise ValueError("Current feature rows must be unique by player.")
    current_raw = raw_priors.loc[
        raw_priors["Window_End"].eq(rating_season)
        & raw_priors["candidate"].eq("box_15")
    ].copy()
    if current_raw.duplicated(key).any():
        raise ValueError("Current raw Box15 priors must be unique by player.")
    identity_columns = [key, "PLAYER_NAME", "TEAM_ABBREVIATION"]
    prior = active_leaderboard.loc[
        active_leaderboard["candidate"].eq("box_15"),
        [*identity_columns, "offense", "defense", "net", "Poss_Off", "Poss_Def"],
    ].rename(
        columns={component: f"centered_prior_{component}" for component in COMPONENTS}
    )
    aio = active_leaderboard.loc[
        active_leaderboard["candidate"].eq("box_15_aio"),
        [key, "offense", "defense", "net", "rank"],
    ].rename(columns={component: f"aio_{component}" for component in COMPONENTS})
    current = prior.merge(aio, on=key, validate="one_to_one")
    current = current.merge(current_features, on=key, validate="one_to_one")
    current = current.merge(
        current_raw[
            [
                key,
                "prior_offense_per_100",
                "prior_defense_per_100",
                "prior_net_per_100",
            ]
        ],
        on=key,
        validate="one_to_one",
    )
    ledger_rows: list[dict] = []
    wide = current[
        [
            *identity_columns,
            "Poss_Off",
            "Poss_Def",
            "rank",
            *(f"aio_{component}" for component in COMPONENTS),
        ]
    ].copy()
    side_values: dict[str, pd.DataFrame] = {}
    offsets: dict[str, float] = {}
    for side in ("offense", "defense"):
        contributions = linear_group_contributions(
            models[side],
            current,
            feature_names=feature_names,
            groups=groups,
        )
        raw_column = f"prior_{side}_per_100"
        raw_error = contributions["raw_prediction"] - current[raw_column]
        if raw_error.abs().max() > 1e-8:
            raise AssertionError(f"Saved {side} prior differs from model prediction.")
        centering = current[f"centered_prior_{side}"] - contributions["raw_prediction"]
        if float(centering.max() - centering.min()) > 1e-8:
            raise AssertionError(f"The {side} prior centering offset is not constant.")
        offset = float(centering.mean())
        offsets[side] = offset
        values = pd.DataFrame(index=current.index)
        values["prior_baseline"] = contributions["prior_intercept"] + offset
        for group in groups:
            values[group] = contributions[group]
        values["rapm_update"] = (
            current[f"aio_{side}"] - current[f"centered_prior_{side}"]
        )
        values["aio"] = current[f"aio_{side}"]
        values["identity_error"] = values["aio"] - values[
            ["prior_baseline", *groups, "rapm_update"]
        ].sum(axis=1)
        side_values[side] = values
    side_values["net"] = side_values["offense"] + side_values["defense"]
    side_values["net"]["aio"] = current["aio_net"]
    side_values["net"]["identity_error"] = side_values["net"]["aio"] - side_values[
        "net"
    ][["prior_baseline", *groups, "rapm_update"]].sum(axis=1)
    for side, values in side_values.items():
        for component in ("prior_baseline", *groups, "rapm_update"):
            ledger_rows.extend(
                {
                    **current.loc[index, identity_columns].to_dict(),
                    "rating_season": rating_season,
                    "side": side,
                    "component": component,
                    "value_points_per_100": float(values.loc[index, component]),
                    "aio_points_per_100": float(values.loc[index, "aio"]),
                    "additive_to_aio": True,
                }
                for index in values.index
            )
        wide[f"identity_error_{side}"] = values["identity_error"]
        for component in ("prior_baseline", *groups, "rapm_update"):
            wide[f"{side}_{component}"] = values[component]
    ledger = pd.DataFrame(ledger_rows)
    maximum_error = float(
        wide[[f"identity_error_{side}" for side in COMPONENTS]].abs().max().max()
    )
    if maximum_error > 1e-8:
        raise AssertionError("AIO component ledger failed its exact identity.")
    quality = {
        "active_players": int(len(current)),
        "offense_centering_offset": offsets["offense"],
        "defense_centering_offset": offsets["defense"],
        "maximum_identity_error": maximum_error,
        "season_2027_rows": 0,
    }
    return ledger, wide.sort_values("rank"), quality


def build_factor_skill_panel(
    factor_predictions: pd.DataFrame,
    active_players: pd.DataFrame,
    *,
    rating_season: int = 2026,
) -> pd.DataFrame:
    """Return non-additive TS, turnover, and rebound skill estimates."""
    if rating_season >= 2027:
        raise ValueError("Season 2027 is forbidden.")
    wanted = {
        "shooting_ts": "true_shooting_skill",
        "turnover_avoidance": "turnover_avoidance_skill",
        "opponent_oreb_prevention": "rebounding_skill",
    }
    _require_columns(
        factor_predictions,
        {
            "PLAYER_ID",
            "Window_End",
            "factor",
            "component",
            "candidate",
            "prediction",
        },
        "Factor prediction panel",
    )
    selected = factor_predictions.loc[
        factor_predictions["Window_End"].eq(rating_season)
        & factor_predictions["candidate"].eq("specialist_factor")
        & factor_predictions["factor"].isin(wanted)
    ].copy()
    if selected.duplicated(["PLAYER_ID", "factor", "component"]).any():
        raise ValueError("Factor skill keys must be unique.")
    identities = active_players[
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
    ].drop_duplicates()
    if identities["PLAYER_ID"].duplicated().any():
        raise ValueError("Active player identities must be unique.")
    selected = selected.merge(
        identities, on="PLAYER_ID", how="inner", validate="many_to_one"
    )
    selected["skill"] = selected["factor"].map(wanted)
    selected["score"] = selected["prediction"]
    selected["units"] = "factor_target_units"
    selected["additive_to_aio"] = False
    return selected[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "Window_End",
            "skill",
            "component",
            "score",
            "units",
            "additive_to_aio",
        ]
    ].sort_values(["skill", "component", "score"], ascending=[True, True, False])
