"""Cross-fitted shooting-luck targets and teammate-shot lineup transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS


def leave_game_out_empirical_bayes_rate(
    attempts: pd.DataFrame,
    *,
    prior_attempts: float,
) -> np.ndarray:
    """Estimate player-season make rates excluding the current game."""
    if prior_attempts <= 0:
        raise ValueError("prior_attempts must be positive.")
    required = {"season", "game_id", "player_id", "made"}
    if missing := sorted(required - set(attempts.columns)):
        raise ValueError(f"Attempt rows are missing {missing}.")
    frame = attempts.reset_index(drop=True).copy()
    player = frame.groupby(["season", "player_id"])["made"].agg(["sum", "count"])
    game = frame.groupby(["season", "player_id", "game_id"])["made"].agg(["sum", "count"])
    league = frame.groupby("season")["made"].agg(["sum", "count"])
    player_key = pd.MultiIndex.from_frame(frame[["season", "player_id"]])
    game_key = pd.MultiIndex.from_frame(frame[["season", "player_id", "game_id"]])
    season = frame["season"].to_numpy()
    makes = player["sum"].reindex(player_key).to_numpy() - game["sum"].reindex(game_key).to_numpy()
    count = player["count"].reindex(player_key).to_numpy() - game["count"].reindex(game_key).to_numpy()
    league_rate = (league["sum"] / league["count"]).reindex(season).to_numpy()
    return (makes + prior_attempts * league_rate) / (count + prior_attempts)


def replace_shooter_with_dummy(
    shots: pd.DataFrame,
    *,
    dummy_player_id: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Remove the shooter from the offensive five while retaining teammates."""
    output = shots.copy()
    missing = 0
    for index, row in output.iterrows():
        columns = HOME_PLAYER_COLUMNS if bool(row["home_poss"]) else AWAY_PLAYER_COLUMNS
        matches = [column for column in columns if int(row[column]) == int(row["shooter_id"])]
        if len(matches) != 1:
            missing += 1
            continue
        output.at[index, matches[0]] = dummy_player_id
    return output, missing
