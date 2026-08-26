"""Chronological validation suite for frozen player-impact priors."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_impact.models.aging_balanced_validation import _age_basis
from nba_impact.models.rapm import (
    RapmConfig,
    RapmDesign,
    build_design,
    fit_coefficients_with_center,
)


COMPONENTS = ("offense", "defense", "net")
DEFAULT_TEST_WEIGHTS = {
    "next_season_game_margin": 0.50,
    "midseason_adaptation": 0.25,
    "forward_annual_impact": 0.15,
    "reverse_annual_impact": 0.05,
    "same_season_rapm_fit": 0.05,
}


def weighted_correlation(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Return a finite-weight Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    x, y, weight = x[keep], y[keep], weight[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    weight = weight / weight.sum()
    x_centered = x - np.sum(weight * x)
    y_centered = y - np.sum(weight * y)
    denominator = math.sqrt(
        float(np.sum(weight * x_centered**2) * np.sum(weight * y_centered**2))
    )
    return float(np.sum(weight * x_centered * y_centered) / denominator)


def chronological_game_halves(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split a season at the midpoint of whole games, ordered without outcomes."""
    required = {"gameid", "date"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Possession frame is missing {missing}.")
    games = frame[["gameid", "date"]].copy()
    games["gameid"] = games["gameid"].astype(str)
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    if games.groupby("gameid")["date"].nunique().gt(1).any():
        raise ValueError("A game cannot have multiple dates.")
    games = games.drop_duplicates("gameid").sort_values(
        ["date", "gameid"], kind="stable"
    ).reset_index(drop=True)
    if len(games) < 2:
        raise ValueError("At least two games are required for a chronological split.")
    split = len(games) // 2
    games["half"] = np.where(np.arange(len(games)) < split, "first", "second")
    half_by_game = games.set_index("gameid")["half"]
    labels = frame["gameid"].astype(str).map(half_by_game)
    first = labels.eq("first").to_numpy()
    second = labels.eq("second").to_numpy()
    if np.any(first & second) or not np.all(first | second):
        raise AssertionError("Every possession must belong to exactly one half.")
    return first, second, games


def prior_center(
    design: RapmDesign,
    prior: pd.DataFrame,
    *,
    row_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map positive-good offense and defense ratings into RAPM coefficients."""
    required = {"PLAYER_ID", "offense", "defense"}
    if missing := sorted(required - set(prior.columns)):
        raise ValueError(f"Prior is missing {missing}.")
    if prior["PLAYER_ID"].duplicated().any():
        raise ValueError("Prior player IDs must be unique.")
    lookup = prior.set_index("PLAYER_ID")[["offense", "defense"]]
    aligned = lookup.reindex(design.players)
    known = aligned[["offense", "defense"]].notna().all(axis=1).to_numpy()
    offense = aligned["offense"].fillna(0.0).to_numpy(dtype=float) / 100.0
    points_allowed = -aligned["defense"].fillna(0.0).to_numpy(dtype=float) / 100.0
    n_players = len(design.players)
    train = design.X[row_mask]
    off_exposure = np.asarray(train[:, :n_players].sum(axis=0)).ravel()
    def_exposure = np.asarray(train[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    if off_exposure.sum() <= 0 or def_exposure.sum() <= 0:
        raise ValueError("Training rows must contain offensive and defensive exposure.")
    offense -= float(np.average(offense, weights=off_exposure))
    points_allowed -= float(np.average(points_allowed, weights=def_exposure))
    center = np.concatenate([offense, points_allowed, np.asarray([0.0])])
    quality = {
        "players_in_design": int(n_players),
        "players_with_prior": int(known.sum()),
        "train_off_possession_coverage": float(off_exposure[known].sum() / off_exposure.sum()),
        "train_def_possession_coverage": float(def_exposure[known].sum() / def_exposure.sum()),
    }
    return center, quality


def game_margin_predictions(
    design: RapmDesign,
    beta: np.ndarray,
    intercept: float,
    row_mask: np.ndarray,
) -> pd.DataFrame:
    """Aggregate possession predictions into home-team game margins."""
    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != (design.X.shape[0],):
        raise ValueError("Row mask length must match the RAPM design.")
    rows = np.flatnonzero(mask)
    game_ids = design.game_ids[rows].astype(str)
    signs = np.where(design.home_offense[rows], 1.0, -1.0)
    predicted_points = np.asarray(design.X[rows] @ beta).ravel() + intercept
    actual_points = design.y[rows]
    possessions = pd.DataFrame(
        {
            "game_id": game_ids,
            "actual": signs * actual_points,
            "predicted": signs * predicted_points,
        }
    )
    return possessions.groupby("game_id", as_index=False).agg(
        actual_margin=("actual", "sum"),
        predicted_margin=("predicted", "sum"),
        possession_rows=("actual", "size"),
    )


def game_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Score an identical set of whole-game margins."""
    required = {"actual_margin", "predicted_margin"}
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Game predictions are missing {missing}.")
    error = predictions["actual_margin"] - predictions["predicted_margin"]
    return {
        "games": int(len(predictions)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": float(
            predictions["actual_margin"].corr(predictions["predicted_margin"])
        ),
    }


def paired_game_mse_intervals(
    predictions: pd.DataFrame,
    *,
    season_column: str = "season",
    draws: int = 5_000,
    seed: int = 20260827,
) -> pd.DataFrame:
    """Bootstrap paired whole games and average season MSE deltas equally."""
    required = {
        "candidate",
        season_column,
        "game_id",
        "actual_margin",
        "predicted_margin",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Paired game predictions are missing {missing}.")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required.")
    frame = predictions.copy()
    frame["squared_error"] = (
        frame["actual_margin"] - frame["predicted_margin"]
    ) ** 2
    candidates = sorted(frame["candidate"].unique())
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            observed_deltas: list[float] = []
            bootstrap_by_season: list[np.ndarray] = []
            for season, season_frame in frame.groupby(season_column, sort=True):
                wide = season_frame.pivot(
                    index="game_id", columns="candidate", values="squared_error"
                )
                if left not in wide or right not in wide:
                    continue
                pair = wide[[left, right]].dropna()
                if pair.empty:
                    continue
                delta = (pair[left] - pair[right]).to_numpy(dtype=float)
                observed_deltas.append(float(delta.mean()))
                indices = rng.integers(0, len(delta), size=(draws, len(delta)))
                bootstrap_by_season.append(delta[indices].mean(axis=1))
            if not observed_deltas:
                continue
            bootstrap = np.mean(np.column_stack(bootstrap_by_season), axis=1)
            rows.append(
                {
                    "left_candidate": left,
                    "right_candidate": right,
                    "seasons": len(observed_deltas),
                    "mean_mse_delta_left_minus_right": float(np.mean(observed_deltas)),
                    "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                    "probability_left_better": float(np.mean(bootstrap < 0)),
                    "bootstrap_draws": draws,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def evaluate_midseason_adaptation(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    season: int,
    candidates: tuple[str, ...],
    config: RapmConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Update prior-season ratings on the first half and score the second half."""
    season_frame = possessions.loc[
        pd.to_numeric(possessions["season"], errors="raise").eq(season)
    ].copy()
    if season_frame.empty:
        raise ValueError(f"No possessions are available for season {season}.")
    first, second, games = chronological_game_halves(season_frame)
    design = build_design(season_frame, include_home=True)
    config = config or RapmConfig(seasons=(season,))
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    expected_game_ids: set[str] | None = None
    for candidate in candidates:
        if candidate == "zero_prior_rapm":
            center = np.zeros(design.X.shape[1], dtype=float)
            quality = {
                "players_in_design": len(design.players),
                "players_with_prior": 0,
                "train_off_possession_coverage": 0.0,
                "train_def_possession_coverage": 0.0,
            }
        else:
            candidate_prior = priors.loc[
                priors["candidate"].eq(candidate)
                & pd.to_numeric(priors["Season"], errors="raise").eq(season - 1)
            ]
            if candidate_prior.empty:
                raise ValueError(f"{candidate} has no prior for season {season - 1}.")
            center, quality = prior_center(design, candidate_prior, row_mask=first)
        beta, intercept = fit_coefficients_with_center(
            design,
            config,
            center,
            center_scale=1.0,
            row_mask=first,
        )
        predictions = game_margin_predictions(design, beta, intercept, second)
        game_ids = set(predictions["game_id"])
        expected_game_ids = game_ids if expected_game_ids is None else expected_game_ids
        if game_ids != expected_game_ids:
            raise AssertionError("Every candidate must score identical second-half games.")
        predictions["test_id"] = "midseason_adaptation"
        predictions["candidate"] = candidate
        predictions["season"] = season
        prediction_rows.append(predictions)
        metric_rows.append(
            {
                "test_id": "midseason_adaptation",
                "candidate": candidate,
                "season": season,
                **game_metrics(predictions),
            }
        )
        coverage_rows.append({"candidate": candidate, "season": season, **quality})
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(coverage_rows).merge(
            games.groupby("half", as_index=False).size().rename(columns={"size": "games"}),
            how="cross",
        ),
    )


def build_adjacent_annual_metrics(
    ratings: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    candidates: tuple[str, ...],
    ages: pd.DataFrame | None = None,
    minimum_aging_training_seasons: int = 3,
    aging_ridge_alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score same-season, forward, and reverse annual RAPM relationships."""
    rating_required = {"PLAYER_ID", "Season", "candidate", *COMPONENTS}
    target_required = {
        "PLAYER_ID",
        "Season",
        "sample_weight",
        *(f"target_{side}" for side in COMPONENTS),
    }
    if missing := sorted(rating_required - set(ratings.columns)):
        raise ValueError(f"Rating panel is missing {missing}.")
    if missing := sorted(target_required - set(targets.columns)):
        raise ValueError(f"Target panel is missing {missing}.")
    if ratings.duplicated(["PLAYER_ID", "Season", "candidate"]).any():
        raise ValueError("Rating panel keys must be unique.")
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Target panel keys must be unique.")
    if ages is not None:
        age_required = {"PLAYER_ID", "Season", "AGE"}
        if missing := sorted(age_required - set(ages.columns)):
            raise ValueError(f"Age panel is missing {missing}.")
        if ages.duplicated(["PLAYER_ID", "Season"]).any():
            raise ValueError("Age panel keys must be unique.")
    rows: list[dict] = []
    matched_rows: list[pd.DataFrame] = []
    test_specs = (
        ("same_season_rapm_fit", 0),
        ("forward_annual_impact", 1),
        ("reverse_annual_impact", -1),
    )
    for test_id, step in test_specs:
        shifted = targets.copy()
        shifted["Season"] = shifted["Season"] - step
        shifted = shifted.rename(
            columns={
                "sample_weight": "evaluation_weight",
                **{f"target_{side}": f"evaluation_{side}" for side in COMPONENTS},
            }
        )
        age_adjustments: list[pd.DataFrame] = []
        if ages is not None and step != 0:
            transitions = targets.merge(
                shifted,
                on=["PLAYER_ID", "Season"],
                how="inner",
                validate="one_to_one",
            ).merge(
                ages[["PLAYER_ID", "Season", "AGE"]],
                on=["PLAYER_ID", "Season"],
                how="inner",
                validate="one_to_one",
            )
            for season in sorted(transitions["Season"].unique().astype(int)):
                train = transitions.loc[transitions["Season"].lt(season)]
                test = transitions.loc[transitions["Season"].eq(season)]
                if train["Season"].nunique() < minimum_aging_training_seasons:
                    continue
                adjustment = test[["PLAYER_ID", "Season"]].copy()
                train_weight = np.minimum(
                    train["sample_weight"], train["evaluation_weight"]
                ).to_numpy(dtype=float)
                for component in COMPONENTS:
                    change = (
                        train[f"evaluation_{component}"]
                        - train[f"target_{component}"]
                    )
                    model = Ridge(alpha=aging_ridge_alpha, fit_intercept=False)
                    model.fit(
                        _age_basis(train["AGE"]),
                        change,
                        sample_weight=train_weight,
                    )
                    adjustment[f"expected_age_change_{component}"] = model.predict(
                        _age_basis(test["AGE"])
                    )
                age_adjustments.append(adjustment)
        adjustment_panel = (
            pd.concat(age_adjustments, ignore_index=True)
            if age_adjustments
            else pd.DataFrame(columns=["PLAYER_ID", "Season"])
        )
        origin_weights = targets[["PLAYER_ID", "Season", "sample_weight"]].rename(
            columns={"sample_weight": "origin_weight"}
        )
        for candidate in candidates:
            candidate_ratings = ratings.loc[ratings["candidate"].eq(candidate)].merge(
                origin_weights,
                on=["PLAYER_ID", "Season"],
                how="inner",
                validate="one_to_one",
            )
            matched = candidate_ratings.merge(
                shifted,
                on=["PLAYER_ID", "Season"],
                how="inner",
                validate="one_to_one",
            )
            matched["evaluation_weight"] = np.minimum(
                matched["origin_weight"], matched["evaluation_weight"]
            )
            if not adjustment_panel.empty:
                matched = matched.merge(
                    adjustment_panel,
                    on=["PLAYER_ID", "Season"],
                    how="left",
                    validate="one_to_one",
                )
            for season, fold in matched.groupby("Season", sort=True):
                for component in COMPONENTS:
                    prediction = fold[component].to_numpy(dtype=float)
                    weight = fold["evaluation_weight"].to_numpy(dtype=float)
                    variants = {
                        "raw": (
                            fold[f"evaluation_{component}"].to_numpy(dtype=float),
                            prediction,
                            weight,
                        )
                    }
                    adjustment_column = f"expected_age_change_{component}"
                    if adjustment_column in fold:
                        age_known = fold[adjustment_column].notna().to_numpy()
                        if age_known.sum() >= 3:
                            variants["aging_adjusted"] = (
                                (
                                    fold.loc[age_known, f"evaluation_{component}"]
                                    - fold.loc[age_known, adjustment_column]
                                ).to_numpy(dtype=float),
                                prediction[age_known],
                                weight[age_known],
                            )
                    for variant, (target, variant_prediction, variant_weight) in variants.items():
                        error = target - variant_prediction
                        rows.append(
                            {
                                "test_id": test_id,
                                "variant": variant,
                                "candidate": candidate,
                                "season": int(season),
                                "component": component,
                                "rows": int(len(target)),
                                "mse": float(np.average(error**2, weights=variant_weight)),
                                "rmse": float(np.sqrt(np.average(error**2, weights=variant_weight))),
                                "correlation": weighted_correlation(
                                    target, variant_prediction, variant_weight
                                ),
                            }
                        )
            matched["test_id"] = test_id
            matched["candidate"] = candidate
            matched_rows.append(matched)
    return pd.DataFrame(rows), pd.concat(matched_rows, ignore_index=True)


def composite_ranking(
    test_scores: pd.DataFrame,
    *,
    weights: Mapping[str, float] = DEFAULT_TEST_WEIGHTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine unlike tests through within-fold percentile ranks only."""
    required = {"test_id", "candidate", "fold", "value", "higher_is_better"}
    if missing := sorted(required - set(test_scores.columns)):
        raise ValueError(f"Composite scores are missing {missing}.")
    if set(test_scores["test_id"]) - set(weights):
        raise ValueError("Every test must have a declared composite weight.")
    if any(value < 0 for value in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("Composite weights must be non-negative and sum to one.")
    ranked_parts: list[pd.DataFrame] = []
    for (_, _), fold in test_scores.groupby(["test_id", "fold"], sort=False):
        if fold["higher_is_better"].nunique() != 1:
            raise ValueError("A test fold cannot mix metric directions.")
        ascending = not bool(fold["higher_is_better"].iloc[0])
        part = fold.copy()
        part["rank"] = part["value"].rank(method="average", ascending=ascending)
        n = len(part)
        part["percentile_score"] = 1.0 if n == 1 else 1.0 - (part["rank"] - 1.0) / (n - 1.0)
        ranked_parts.append(part)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    per_test = ranked.groupby(["candidate", "test_id"], as_index=False).agg(
        folds=("fold", "nunique"),
        mean_percentile_score=("percentile_score", "mean"),
        mean_value=("value", "mean"),
    )
    per_test["declared_weight"] = per_test["test_id"].map(weights)
    rows: list[dict] = []
    for candidate, frame in per_test.groupby("candidate", sort=False):
        available_weight = float(frame["declared_weight"].sum())
        normalized = frame["declared_weight"] / available_weight
        rows.append(
            {
                "candidate": candidate,
                "tests_available": int(len(frame)),
                "declared_weight_coverage": available_weight,
                "weighted_percentile_score": float(
                    np.sum(normalized * frame["mean_percentile_score"])
                ),
                "equal_weight_percentile_score": float(frame["mean_percentile_score"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["weighted_rank"] = summary["weighted_percentile_score"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["equal_weight_rank"] = summary["equal_weight_percentile_score"].rank(
        method="min", ascending=False
    ).astype(int)
    return summary.sort_values("weighted_rank", kind="stable"), ranked
