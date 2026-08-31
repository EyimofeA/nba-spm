"""Small bivariate offense-defense Kalman filter for annual RAPM research."""

from __future__ import annotations

import numpy as np
import pandas as pd


COMPONENTS = ("offense", "defense")


def filter_bivariate_annual_rapm(
    observations: pd.DataFrame,
    *,
    phi: float,
    process_sd: float,
    process_correlation: float,
    use_observation_covariance: bool,
) -> pd.DataFrame:
    """Filter annual offense and defense observations without future rows."""
    required = {
        "PLAYER_ID", "Season", "target_offense", "target_defense",
        "observation_variance_offense", "observation_variance_defense",
        "observation_covariance_offense_defense", "Poss_Off", "Poss_Def",
    }
    if missing := sorted(required - set(observations.columns)):
        raise ValueError(f"Bivariate state-space input is missing {missing}.")
    if not 0 < phi < 1 or process_sd <= 0 or abs(process_correlation) >= 1:
        raise ValueError("Invalid state-space process parameters.")
    values = observations.sort_values(["PLAYER_ID", "Season"], kind="stable").copy()
    numeric = sorted(required - {"PLAYER_ID", "Season"})
    if not np.isfinite(values[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Bivariate state-space input must be finite.")
    process_covariance = process_sd**2 * np.asarray(
        [[1.0, process_correlation], [process_correlation, 1.0]], dtype=float
    )
    stationary_covariance = process_covariance / (1.0 - phi**2)
    rows: list[dict] = []
    for player_id, player in values.groupby("PLAYER_ID", sort=False):
        state = np.zeros(2, dtype=float)
        covariance = stationary_covariance.copy()
        previous_season: int | None = None
        for row in player.itertuples(index=False):
            season = int(row.Season)
            gap = 1 if previous_season is None else season - previous_season
            if gap <= 0:
                raise ValueError("Player seasons must be strictly increasing.")
            transition = phi**gap
            gap_process_covariance = process_covariance * (
                (1.0 - phi ** (2 * gap)) / (1.0 - phi**2)
            )
            predicted_state = transition * state
            predicted_covariance = transition**2 * covariance + gap_process_covariance
            observed = np.asarray([row.target_offense, row.target_defense], dtype=float)
            observed_cross = (
                float(row.observation_covariance_offense_defense)
                if use_observation_covariance
                else 0.0
            )
            observation_covariance = np.asarray(
                [
                    [row.observation_variance_offense, observed_cross],
                    [observed_cross, row.observation_variance_defense],
                ],
                dtype=float,
            )
            eigenvalues = np.linalg.eigvalsh(observation_covariance)
            if eigenvalues.min() <= 0:
                observation_covariance += np.eye(2) * (abs(eigenvalues.min()) + 1e-9)
            gain = predicted_covariance @ np.linalg.inv(
                predicted_covariance + observation_covariance
            )
            innovation = observed - predicted_state
            state = predicted_state + gain @ innovation
            identity = np.eye(2)
            covariance = (
                (identity - gain) @ predicted_covariance @ (identity - gain).T
                + gain @ observation_covariance @ gain.T
            )
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "Season": season,
                    "filtered_offense": float(state[0]),
                    "filtered_defense": float(state[1]),
                    "filtered_net": float(state.sum()),
                    "forecast_next_offense": float(phi * state[0]),
                    "forecast_next_defense": float(phi * state[1]),
                    "forecast_next_net": float(phi * state.sum()),
                    "filtered_variance_offense": float(covariance[0, 0]),
                    "filtered_variance_defense": float(covariance[1, 1]),
                    "filtered_covariance_offense_defense": float(covariance[0, 1]),
                    "kalman_gain_offense_from_offense": float(gain[0, 0]),
                    "kalman_gain_offense_from_defense": float(gain[0, 1]),
                    "kalman_gain_defense_from_offense": float(gain[1, 0]),
                    "kalman_gain_defense_from_defense": float(gain[1, 1]),
                    "Poss_Off": float(row.Poss_Off),
                    "Poss_Def": float(row.Poss_Def),
                    "season_gap": gap,
                }
            )
            previous_season = season
    output = pd.DataFrame(rows)
    error = (output["filtered_offense"] + output["filtered_defense"] - output["filtered_net"]).abs().max()
    if float(error) > 1e-12:
        raise AssertionError("Filtered offense plus defense must equal net.")
    return output
