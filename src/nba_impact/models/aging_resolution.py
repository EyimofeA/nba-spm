"""Walk-forward aging-curve resolution and trailing-window diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPONENTS = ("offense", "defense", "net")


@dataclass(frozen=True)
class AgingResolutionResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    curves: pd.DataFrame
    quality: dict[str, object]


def load_age_panel(source_dir, seasons: tuple[int, ...]) -> pd.DataFrame:
    """Load one unique season-level age for each player."""
    rows: list[pd.DataFrame] = []
    for season in seasons:
        path = source_dir / f"{season}.csv"
        frame = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce")
        frame["AGE"] = pd.to_numeric(frame["AGE"], errors="coerce")
        frame = frame.dropna(subset=["PLAYER_ID", "AGE"])
        if frame.groupby("PLAYER_ID")["AGE"].nunique().gt(1).any():
            raise ValueError(f"Conflicting ages in {path}.")
        frame = frame.drop_duplicates("PLAYER_ID")
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frame["Season"] = int(season)
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Age panel keys must be unique.")
    return panel


def build_annual_transitions(targets: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    """Join adjacent annual RAPM observations without future-season leakage."""
    required = {
        "PLAYER_ID",
        "Season",
        "Poss_Off",
        "Poss_Def",
        *(f"target_{component}" for component in COMPONENTS),
    }
    if missing := sorted(required - set(targets.columns)):
        raise ValueError(f"Targets are missing {missing}.")
    annual = targets[list(required)].merge(
        ages, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    annual["weight"] = np.sqrt(
        annual[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1.0)
    )
    following = annual[
        ["PLAYER_ID", "Season", "weight", *(f"target_{c}" for c in COMPONENTS)]
    ].copy()
    following["Season"] -= 1
    following = following.rename(
        columns={
            "weight": "next_weight",
            **{f"target_{c}": f"next_{c}" for c in COMPONENTS},
        }
    )
    transitions = annual.merge(
        following, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    transitions["evaluation_weight"] = np.minimum(
        transitions["weight"], transitions["next_weight"]
    )
    for component in COMPONENTS:
        transitions[f"delta_{component}"] = (
            transitions[f"next_{component}"] - transitions[f"target_{component}"]
        )
    return transitions


def kernel_age_change(
    train_age: np.ndarray,
    train_change: np.ndarray,
    train_weight: np.ndarray,
    test_age: np.ndarray,
    *,
    bandwidth: float,
    minimum_effective_weight: float = 1e-12,
) -> np.ndarray:
    """Gaussian-kernel estimate of expected annual rating change by age."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    train_age = np.asarray(train_age, dtype=float)
    train_change = np.asarray(train_change, dtype=float)
    train_weight = np.asarray(train_weight, dtype=float)
    test_age = np.asarray(test_age, dtype=float)
    if not (
        np.isfinite(train_age).all()
        and np.isfinite(train_change).all()
        and np.isfinite(train_weight).all()
        and np.isfinite(test_age).all()
    ):
        raise ValueError("Aging inputs must be finite.")
    if (train_weight < 0).any() or train_weight.sum() <= 0:
        raise ValueError("Training weights must be nonnegative with positive total.")
    distance = (test_age[:, None] - train_age[None, :]) / bandwidth
    weights = np.exp(-0.5 * distance**2) * train_weight[None, :]
    denominator = weights.sum(axis=1)
    global_change = float(np.average(train_change, weights=train_weight))
    numerator = weights @ train_change
    return np.where(
        denominator > minimum_effective_weight,
        numerator / np.maximum(denominator, minimum_effective_weight),
        global_change,
    )


def _trailing_rating(
    annual: pd.DataFrame,
    origin_season: int,
    window: int,
) -> pd.DataFrame:
    history = annual.loc[
        annual["Season"].between(origin_season - window + 1, origin_season)
    ].copy()
    expected = set(range(origin_season - window + 1, origin_season + 1))
    complete = history.groupby("PLAYER_ID")["Season"].agg(lambda x: set(x) == expected)
    history = history.loc[history["PLAYER_ID"].isin(complete.index[complete])]
    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    for player_id, group in history.groupby("PLAYER_ID", sort=False):
        latest = group.loc[group["Season"].eq(origin_season)]
        if latest.empty:
            continue
        row: dict[str, float | int] = {
            "PLAYER_ID": int(player_id),
            "Season": int(origin_season),
            "AGE": float(latest["AGE"].iloc[0]),
        }
        for component in COMPONENTS:
            row[f"base_{component}"] = float(
                np.average(group[f"target_{component}"], weights=group["weight"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_metrics(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict:
    error = np.asarray(actual) - np.asarray(predicted)
    weight = np.asarray(weight)
    rmse = math.sqrt(float(np.average(error**2, weights=weight)))
    mean_actual = float(np.average(actual, weights=weight))
    mean_predicted = float(np.average(predicted, weights=weight))
    covariance = float(np.average((actual - mean_actual) * (predicted - mean_predicted), weights=weight))
    variance_actual = float(np.average((actual - mean_actual) ** 2, weights=weight))
    variance_predicted = float(np.average((predicted - mean_predicted) ** 2, weights=weight))
    correlation = (
        covariance / math.sqrt(variance_actual * variance_predicted)
        if variance_actual > 0 and variance_predicted > 0
        else float("nan")
    )
    return {"weighted_rmse": rmse, "correlation": correlation}


def evaluate_aging_resolutions(
    targets: pd.DataFrame,
    ages: pd.DataFrame,
    *,
    bandwidths: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    trailing_windows: tuple[int, ...] = (1, 3, 5),
    minimum_training_origins: int = 3,
) -> AgingResolutionResult:
    """Walk forward over annual ratings and compare age smoothing resolutions."""
    if not bandwidths or len(set(bandwidths)) != len(bandwidths):
        raise ValueError("bandwidths must be nonempty and unique.")
    if not trailing_windows or any(window < 1 for window in trailing_windows):
        raise ValueError("trailing_windows must contain positive integers.")
    transitions = build_annual_transitions(targets, ages)
    annual = targets.merge(
        ages, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    ).copy()
    annual["weight"] = np.sqrt(
        annual[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1.0)
    )
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int]] = []
    for origin in sorted(transitions["Season"].astype(int).unique()):
        train = transitions.loc[transitions["Season"].lt(origin)].copy()
        test = transitions.loc[transitions["Season"].eq(origin)].copy()
        if train["Season"].nunique() < minimum_training_origins or test.empty:
            continue
        changes: dict[tuple[str, float], np.ndarray] = {}
        for component in COMPONENTS:
            for bandwidth in bandwidths:
                changes[(component, bandwidth)] = kernel_age_change(
                    train["AGE"].to_numpy(),
                    train[f"delta_{component}"].to_numpy(),
                    train["evaluation_weight"].to_numpy(),
                    test["AGE"].to_numpy(),
                    bandwidth=bandwidth,
                )
                for age, value in zip(
                    np.arange(19.0, 44.0 + 0.1, 0.1),
                    kernel_age_change(
                        train["AGE"].to_numpy(),
                        train[f"delta_{component}"].to_numpy(),
                        train["evaluation_weight"].to_numpy(),
                        np.arange(19.0, 44.0 + 0.1, 0.1),
                        bandwidth=bandwidth,
                    ),
                ):
                    curve_rows.append(
                        {
                            "origin_season": origin,
                            "component": component,
                            "bandwidth_years": bandwidth,
                            "age": round(float(age), 1),
                            "expected_annual_change": float(value),
                        }
                    )
        for window in trailing_windows:
            base = _trailing_rating(annual, origin, window)
            if base.empty:
                continue
            scored = test[
                [
                    "PLAYER_ID",
                    "Season",
                    "AGE",
                    "evaluation_weight",
                    *(f"next_{component}" for component in COMPONENTS),
                ]
            ].merge(base, on=["PLAYER_ID", "Season", "AGE"], validate="one_to_one")
            if scored.empty:
                continue
            position = test.reset_index(drop=True).reset_index().set_index("PLAYER_ID")["index"]
            indices = scored["PLAYER_ID"].map(position).to_numpy(dtype=int)
            for component in COMPONENTS:
                variants: list[tuple[str, float | None, np.ndarray]] = [
                    ("no_age", None, scored[f"base_{component}"].to_numpy())
                ]
                for bandwidth in bandwidths:
                    variants.append(
                        (
                            "age_adjusted",
                            bandwidth,
                            scored[f"base_{component}"].to_numpy()
                            + changes[(component, bandwidth)][indices],
                        )
                    )
                for variant, bandwidth, predicted in variants:
                    actual = scored[f"next_{component}"].to_numpy()
                    weight = scored["evaluation_weight"].to_numpy()
                    metrics = _weighted_metrics(actual, predicted, weight)
                    metric_rows.append(
                        {
                            "origin_season": origin,
                            "target_season": origin + 1,
                            "trailing_window_seasons": window,
                            "component": component,
                            "variant": variant,
                            "bandwidth_years": bandwidth,
                            "rows": len(scored),
                            "training_origins": int(train["Season"].nunique()),
                            **metrics,
                        }
                    )
                    frame = scored[["PLAYER_ID", "Season", "AGE", "evaluation_weight"]].copy()
                    frame["target_season"] = origin + 1
                    frame["trailing_window_seasons"] = window
                    frame["component"] = component
                    frame["variant"] = variant
                    frame["bandwidth_years"] = bandwidth
                    frame["actual"] = actual
                    frame["predicted"] = predicted
                    prediction_rows.append(frame)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    curves = pd.DataFrame(curve_rows)
    fractional_ages = np.modf(ages["AGE"].to_numpy(dtype=float))[0]
    quality = {
        "transition_rows": int(len(transitions)),
        "players": int(transitions["PLAYER_ID"].nunique()),
        "origin_seasons": sorted(transitions["Season"].astype(int).unique().tolist()),
        "age_fractional_values": sorted(np.unique(np.round(fractional_ages, 6)).tolist()),
        "subannual_age_resolution_supported": bool(np.any(np.abs(fractional_ages) > 1e-9)),
    }
    return AgingResolutionResult(metrics, predictions, curves, quality)
