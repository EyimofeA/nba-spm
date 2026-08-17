"""Build season-relative player skill profiles for the public site.

These are descriptive percentiles of observed/engineered inputs. They are not
causal attribution and they are deliberately separate from RAPM/AIO ratings.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PROFILE_AXES: dict[str, tuple[tuple[str, int], ...]] = {
    "shooting": (
        ("true_shooting_pct_relative", 1),
        ("shooting_proficiency_2017_eb", 1),
        ("zts_pct_points", 1),
    ),
    "spacing": (("crafted_spacing_stable_v1", 1),),
    "creation": (
        ("box_creation_2017_eb_p100", 1),
        ("behavioral_passer_score_v1", 1),
        ("assist_to_load_2017_eb", 1),
        ("creation_to_load_2017_eb", 1),
        ("potential_assists_p100_relative", 1),
    ),
    "security": (
        ("turnover_to_load_2017_eb", -1),
        ("TOV_p100_relative", -1),
        ("live_ball_turnovers_p100", -1),
        ("bad_pass_turnovers_p100", -1),
    ),
    "rim_pressure": (
        ("FTA_p100_relative", 1),
        ("at_rim_frequency_relative", 1),
        ("shooting_fouls_drawn_p100", 1),
        ("PFD_p100", 1),
    ),
    "rebounding": (
        ("OREB_p100", 1),
        ("DREB_p100", 1),
        ("rebound_contests_p100", 1),
        ("recovered_blocks_p100", 1),
    ),
    "shot_defense": (
        ("rim_points_saved_p100", 1),
        ("dfg_diff_pct_eb", -1),
        ("rim_diff_pct_eb", -1),
        ("matchup_opponent_adjusted_points_saved_p100_eb", 1),
        ("matchup_shotmaking_points_saved_vs_scorer_p100_eb", 1),
    ),
    "disruption": (
        ("STL_p100", 1),
        ("BLK_p100", 1),
        ("deflections_p100", 1),
        ("charges_drawn_p100", 1),
        ("matchup_turnovers_forced_vs_scorer_p100_eb", 1),
    ),
    "suppression": (
        ("matchup_fga_suppressed_vs_scorer_p100_eb", 1),
        ("matchup_three_pa_suppressed_vs_scorer_p100_eb", 1),
        ("matchup_assists_suppressed_vs_scorer_p100_eb", 1),
        ("matchup_shooting_fouls_prevented_vs_scorer_p100_eb", 1),
    ),
}

PROFILE_SIDES: dict[str, tuple[str, ...]] = {
    "offense": ("shooting", "spacing", "creation", "security", "rim_pressure", "rebounding"),
    "defense": ("shot_defense", "disruption", "suppression", "rebounding"),
}


def _percentile(values: pd.Series, direction: int) -> pd.Series:
    """Return season-relative 0-100 percentiles where higher is always better."""
    numeric = pd.to_numeric(values, errors="coerce") * direction
    return numeric.rank(method="average", pct=True) * 100.0


def build_player_skill_profiles(
    features: pd.DataFrame,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Create compact skill percentiles for each available player-season."""
    season_values = {int(value) for value in seasons}
    season_column = "Season" if "Season" in features else "Window_End"
    required = {"PLAYER_ID", season_column}
    if missing := required - set(features.columns):
        raise ValueError(f"Profile input is missing columns: {sorted(missing)}")

    frame = features.loc[features[season_column].isin(season_values)].copy()
    frame = frame.rename(columns={season_column: "Season"})
    result = frame[["PLAYER_ID", "Season"]].copy()
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)
    result["Season"] = result["Season"].astype(int)

    for axis, metrics in PROFILE_AXES.items():
        percentiles = []
        for column, direction in metrics:
            if column not in frame:
                continue
            percentiles.append(
                frame.groupby("Season", sort=False)[column]
                .transform(lambda values, sign=direction: _percentile(values, sign))
                .rename(column)
            )
        if percentiles:
            result[axis] = pd.concat(percentiles, axis=1).mean(axis=1, skipna=True)
        else:
            result[axis] = np.nan

    if result.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Profile input contains duplicate player-seasons.")
    bounded = result[list(PROFILE_AXES)].stack().dropna()
    if not bounded.between(0.0, 100.0).all():
        raise ValueError("Profile percentiles must stay in [0, 100].")
    return result.sort_values(["PLAYER_ID", "Season"], kind="stable").reset_index(drop=True)
