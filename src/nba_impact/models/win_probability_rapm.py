"""Conserved possession-credit RAPM from player-neutral win probability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, ratings_table


@dataclass(frozen=True)
class WinProbabilityRapmResult:
    ratings: pd.DataFrame
    game_conservation: pd.DataFrame
    quality: dict[str, float | int]


def build_conserved_wp_target(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability_context",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Credit each possession with the change to the next state or final result."""
    required = {
        "possession_id",
        "gameid",
        "num",
        "home_poss",
        "home_win",
        probability_column,
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Win-probability frame is missing {missing}.")
    # Legacy possession numbers restart each quarter. Prefer the game-global
    # state index, or include period when only source possession numbers exist.
    sequence = (
        ["possession_index_before"] if "possession_index_before" in frame
        else ["period", "num"] if "period" in frame else ["num"]
    )
    order = ["gameid", *sequence]
    if frame[order].isna().any().any() or frame.duplicated(order).any():
        raise ValueError("Possession chronology must be complete and unambiguous.")
    ordered = frame.sort_values(order, kind="stable").copy()
    if ordered.duplicated("possession_id").any():
        raise ValueError("Win-probability possession IDs must be unique.")
    probability = pd.to_numeric(ordered[probability_column], errors="raise").astype(float)
    if not np.isfinite(probability).all() or probability.lt(0).any() or probability.gt(1).any():
        raise ValueError("Win probabilities must be finite and in [0, 1].")
    for column in ("home_win", "home_poss"):
        if not ordered[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain binary values.")
    if ordered.groupby("gameid")["home_win"].nunique().ne(1).any():
        raise ValueError("Each game must have one consistent final result.")
    next_probability = probability.groupby(ordered["gameid"]).shift(-1)
    terminal = ordered["home_win"].astype(float)
    probability_after = next_probability.fillna(terminal)
    ordered["home_wp_before"] = probability
    ordered["home_wp_after"] = probability_after
    ordered["home_wp_change"] = probability_after - probability
    ordered["offense_wp_change"] = ordered["home_wp_change"] * np.where(
        ordered["home_poss"].astype(bool), 1.0, -1.0
    )
    conservation = ordered.groupby("gameid", as_index=False).agg(
        first_home_wp=("home_wp_before", "first"),
        final_home_result=("home_win", "first"),
        summed_home_wp_change=("home_wp_change", "sum"),
        possessions=("possession_id", "size"),
    )
    conservation["required_change"] = (
        conservation["final_home_result"].astype(float) - conservation["first_home_wp"]
    )
    conservation["conservation_error"] = (
        conservation["summed_home_wp_change"] - conservation["required_change"]
    )
    return ordered, conservation


def build_log_odds_wp_target(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability_context",
    epsilon: float = 0.025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Credit possessions with clipped home-win log-odds changes."""
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half.")
    ordered, _ = build_conserved_wp_target(
        frame, probability_column=probability_column
    )
    before = ordered["home_wp_before"].clip(epsilon, 1.0 - epsilon)
    after = ordered["home_wp_after"].clip(epsilon, 1.0 - epsilon)
    ordered["home_log_odds_before"] = np.log(before / (1.0 - before))
    ordered["home_log_odds_after"] = np.log(after / (1.0 - after))
    ordered["home_log_odds_change"] = (
        ordered["home_log_odds_after"] - ordered["home_log_odds_before"]
    )
    ordered["offense_log_odds_change"] = ordered["home_log_odds_change"] * np.where(
        ordered["home_poss"].astype(bool), 1.0, -1.0
    )
    conservation = ordered.groupby("gameid", as_index=False).agg(
        first_home_log_odds=("home_log_odds_before", "first"),
        final_home_log_odds=("home_log_odds_after", "last"),
        summed_home_log_odds_change=("home_log_odds_change", "sum"),
        possessions=("possession_id", "size"),
    )
    conservation["required_log_odds_change"] = (
        conservation["final_home_log_odds"]
        - conservation["first_home_log_odds"]
    )
    conservation["conservation_error"] = (
        conservation["summed_home_log_odds_change"]
        - conservation["required_log_odds_change"]
    )
    return ordered, conservation


def fit_win_probability_rapm(
    frame: pd.DataFrame,
    config: RapmConfig,
    *,
    probability_column: str = "probability_context",
) -> WinProbabilityRapmResult:
    """Fit offense, defense, and net WPA credit with positive-good signs."""
    target, conservation = build_conserved_wp_target(
        frame, probability_column=probability_column
    )
    model_frame = target.copy()
    model_frame["pts"] = model_frame["offense_wp_change"]
    design = build_design(model_frame, include_home=config.include_home)
    beta, intercept = fit_coefficients(design, config)
    ratings = ratings_table(design, beta).rename(
        columns={
            "offense_per_100": "offense_wp_units_per_100",
            "defense_per_100": "defense_wp_units_per_100",
            "net_per_100": "net_wp_units_per_100",
        }
    )
    for component in ("offense", "defense", "net"):
        ratings[f"{component}_wp_percentage_points_per_100"] = (
            100.0 * ratings[f"{component}_wp_units_per_100"]
        )
    quality = {
        "possessions": int(len(target)),
        "games": int(target["gameid"].nunique()),
        "players": int(len(ratings)),
        "maximum_game_conservation_error": float(
            conservation["conservation_error"].abs().max()
        ),
        "mean_absolute_possession_wp_change": float(
            target["home_wp_change"].abs().mean()
        ),
        "terminal_jump_share_of_absolute_credit": float(
            target.groupby("gameid").tail(1)["home_wp_change"].abs().sum()
            / target["home_wp_change"].abs().sum()
        ),
        "intercept_wp_units_per_possession": float(intercept),
    }
    return WinProbabilityRapmResult(ratings, conservation, quality)
