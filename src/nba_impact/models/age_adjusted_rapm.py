"""Age-standardized long-window RAPM with explicit lineup-age controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.models.rapm import RapmConfig, RapmDesign


@dataclass(frozen=True)
class AgeDesign:
    X: csr_matrix
    ages: np.ndarray
    reference_age: int
    known_slots: int
    total_slots: int


@dataclass(frozen=True)
class AgeAdjustedRapmFit:
    player_coefficients: np.ndarray
    age_coefficients: np.ndarray
    intercept: float
    ages: np.ndarray
    reference_age: int
    rows: int


def season_decay_weights(
    seasons: np.ndarray,
    *,
    window_end: int,
    half_life_years: float | None,
) -> np.ndarray:
    """Return season-constant exponential weights with newest season at one."""
    values = np.asarray(seasons, dtype=int)
    if half_life_years is None:
        return np.ones(len(values), dtype=float)
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    return np.power(2.0, (values - int(window_end)) / float(half_life_years))


def build_age_design(
    frame: pd.DataFrame,
    ages: pd.DataFrame,
    *,
    minimum_age: int = 19,
    maximum_age: int = 43,
    reference_age: int = 27,
) -> AgeDesign:
    """Count offensive and defensive players at each non-reference age."""
    if not minimum_age <= reference_age <= maximum_age:
        raise ValueError("reference_age must lie inside the modeled age range")
    required = {"PLAYER_ID", "Season", "AGE"}
    if missing := sorted(required - set(ages.columns)):
        raise ValueError(f"Age panel is missing columns: {missing}")
    panel = ages[list(required)].copy()
    panel["PLAYER_ID"] = pd.to_numeric(panel["PLAYER_ID"], errors="raise").astype(int)
    panel["Season"] = pd.to_numeric(panel["Season"], errors="raise").astype(int)
    panel["AGE"] = pd.to_numeric(panel["AGE"], errors="raise")
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Age panel must be unique by player-season.")
    lookup = panel.set_index(["Season", "PLAYER_ID"])["AGE"]

    away = frame.loc[:, AWAY_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home = frame.loc[:, HOME_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home_offense = frame["home_poss"].to_numpy(dtype=bool, copy=False)
    offense = np.where(home_offense[:, None], home, away)
    defense = np.where(home_offense[:, None], away, home)
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=int)

    def resolve(players: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = pd.MultiIndex.from_arrays(
            [np.repeat(seasons, 5), players.ravel()],
            names=["Season", "PLAYER_ID"],
        )
        values = lookup.reindex(keys).to_numpy(dtype=float)
        known = np.isfinite(values)
        values = np.where(known, values, float(reference_age))
        return np.clip(np.rint(values).astype(int), minimum_age, maximum_age), known

    offense_age, offense_known = resolve(offense)
    defense_age, defense_known = resolve(defense)
    modeled_ages = np.asarray(
        [age for age in range(minimum_age, maximum_age + 1) if age != reference_age],
        dtype=int,
    )
    age_to_column = {age: index for index, age in enumerate(modeled_ages)}
    row_base = np.repeat(np.arange(len(frame), dtype=np.int64), 5)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for side, values in enumerate((offense_age, defense_age)):
        keep = values != reference_age
        row_parts.append(row_base[keep])
        column_parts.append(
            np.asarray(
                [age_to_column[int(age)] + side * len(modeled_ages) for age in values[keep]],
                dtype=np.int64,
            )
        )
    matrix = csr_matrix(
        (
            np.ones(sum(len(rows) for rows in row_parts), dtype=float),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(len(frame), 2 * len(modeled_ages)),
    )
    return AgeDesign(
        X=matrix,
        ages=modeled_ages,
        reference_age=reference_age,
        known_slots=int(offense_known.sum() + defense_known.sum()),
        total_slots=int(offense_known.size + defense_known.size),
    )


def fit_age_adjusted_rapm(
    design: RapmDesign,
    age_design: AgeDesign,
    config: RapmConfig,
    *,
    age_penalty: float,
    row_mask: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
) -> AgeAdjustedRapmFit:
    """Fit player and lineup-age indicators in one ridge regression."""
    if age_penalty < 0:
        raise ValueError("age_penalty must be nonnegative")
    if age_design.X.shape[0] != design.X.shape[0]:
        raise ValueError("Player and age designs must have identical rows.")
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    if mask.shape != (design.X.shape[0],) or not mask.any():
        raise ValueError("row_mask must select at least one design row")
    weights = (
        np.ones(design.X.shape[0], dtype=float)
        if row_weights is None
        else np.asarray(row_weights, dtype=float)
    )
    if (
        weights.shape != (design.X.shape[0],)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not weights[mask].sum() > 0
    ):
        raise ValueError("row_weights must be finite, nonnegative, and have positive selected mass")
    player_x = design.X[mask]
    age_x = age_design.X[mask]
    matrix = hstack([player_x, age_x], format="csr")
    y = design.y[mask]
    selected_weights = weights[mask]
    root_weight = np.sqrt(selected_weights)
    weighted_matrix = diags(root_weight, format="csr") @ matrix
    intercept = float(np.average(y, weights=selected_weights))
    n_players = len(design.players)
    penalties = np.concatenate(
        [
            np.full(n_players, config.lambda_off, dtype=float),
            np.full(n_players, config.lambda_def, dtype=float),
            np.asarray([config.lambda_home], dtype=float)
            if config.include_home
            else np.empty(0, dtype=float),
            np.full(age_x.shape[1], age_penalty, dtype=float),
        ]
    )
    lhs = (weighted_matrix.T @ weighted_matrix).tocsr() + diags(
        penalties, format="csr"
    )
    rhs = np.asarray(weighted_matrix.T @ ((y - intercept) * root_weight)).ravel()
    try:
        coefficients, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        coefficients, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        coefficients = spsolve(lhs.tocsc(), rhs)
    coefficients = np.asarray(coefficients, dtype=float)
    player = coefficients[: design.X.shape[1]].copy()
    age = coefficients[design.X.shape[1] :].copy()

    off_counts = np.asarray(
        player_x[:, :n_players].T @ selected_weights
    ).ravel()
    def_counts = np.asarray(
        player_x[:, n_players : 2 * n_players].T @ selected_weights
    ).ravel()
    off_mean = float(np.average(player[:n_players], weights=off_counts))
    def_mean = float(
        np.average(player[n_players : 2 * n_players], weights=def_counts)
    )
    player[:n_players] -= off_mean
    player[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)
    return AgeAdjustedRapmFit(
        player_coefficients=player,
        age_coefficients=age,
        intercept=intercept,
        ages=age_design.ages,
        reference_age=age_design.reference_age,
        rows=int(mask.sum()),
    )


def predict_age_adjusted_rapm(
    fit: AgeAdjustedRapmFit,
    design: RapmDesign,
    age_design: AgeDesign,
    *,
    row_mask: np.ndarray | None = None,
    include_age: bool = True,
) -> np.ndarray:
    mask = (
        np.ones(design.X.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    prediction = np.asarray(design.X[mask] @ fit.player_coefficients).ravel()
    prediction += fit.intercept
    if include_age:
        prediction += np.asarray(age_design.X[mask] @ fit.age_coefficients).ravel()
    return prediction


def age_curve(fit: AgeAdjustedRapmFit) -> pd.DataFrame:
    """Return the per-player age effect relative to the reference age."""
    count = len(fit.ages)
    offense = dict(zip(fit.ages, 100.0 * fit.age_coefficients[:count]))
    defense = dict(zip(fit.ages, -100.0 * fit.age_coefficients[count:]))
    rows = []
    for age in range(int(fit.ages.min()), int(fit.ages.max()) + 1):
        off = 0.0 if age == fit.reference_age else float(offense[age])
        deff = 0.0 if age == fit.reference_age else float(defense[age])
        rows.append({"age": age, "offense": off, "defense": deff, "net": off + deff})
    return pd.DataFrame(rows)


def current_age_player_coefficients(
    fit: AgeAdjustedRapmFit,
    design: RapmDesign,
    ages: pd.DataFrame,
    *,
    season: int,
) -> tuple[np.ndarray, float]:
    """Add each player's observed-season age effect to their base coefficient.

    This produces current-age ratings. It is distinct from publishing every
    player at the omitted reference age.
    """
    required = {"PLAYER_ID", "Season", "AGE"}
    if missing := sorted(required - set(ages.columns)):
        raise ValueError(f"Age panel is missing columns: {missing}")
    selected = ages.loc[pd.to_numeric(ages["Season"]).eq(int(season))].copy()
    selected["PLAYER_ID"] = pd.to_numeric(selected["PLAYER_ID"], errors="raise").astype(int)
    selected["AGE"] = pd.to_numeric(selected["AGE"], errors="coerce")
    if selected["PLAYER_ID"].duplicated().any():
        raise ValueError("Current-age panel must be unique by player.")
    age_lookup = selected.set_index("PLAYER_ID")["AGE"]
    player_ages = age_lookup.reindex(design.players).to_numpy(dtype=float)
    known = np.isfinite(player_ages)
    player_ages = np.where(known, np.rint(player_ages), fit.reference_age).astype(int)
    player_ages = np.clip(player_ages, int(fit.ages.min()), int(fit.ages.max()))
    age_to_column = {int(age): index for index, age in enumerate(fit.ages)}
    count = len(fit.ages)
    coefficients = fit.player_coefficients.copy()
    n_players = len(design.players)
    for index, age in enumerate(player_ages):
        if int(age) == fit.reference_age:
            continue
        age_column = age_to_column[int(age)]
        coefficients[index] += fit.age_coefficients[age_column]
        coefficients[n_players + index] += fit.age_coefficients[count + age_column]
    return coefficients, float(known.mean())
