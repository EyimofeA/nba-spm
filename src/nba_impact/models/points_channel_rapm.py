"""Additive scoring-channel decomposition of possession RAPM.

This is not a multinomial probability model. It uses the linearity of ridge to
split possession points into conserved channels, fit every channel against the
same lineup design, and recover ordinary points RAPM by summing the results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from nba_impact.models.rapm import (
    RapmConfig,
    _penalty,
    build_design,
    fit_coefficients,
    ratings_table,
)


POINT_CHANNELS = ("one_point", "two_point", "three_plus")


@dataclass(frozen=True)
class PointsChannelRapmResult:
    ratings: pd.DataFrame
    channel_intercepts: dict[str, float]
    channel_home_effects_per_100: dict[str, float]
    quality: dict[str, float | int]


def build_points_channel_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return additive point targets whose row sum equals possession points."""
    points = pd.to_numeric(frame["pts"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(points).all() or (points < 0).any():
        raise ValueError("Possession points must be finite and nonnegative.")
    rounded = np.rint(points)
    if not np.allclose(points, rounded, atol=1e-12):
        raise ValueError("Points-channel RAPM requires integer possession points.")
    targets = pd.DataFrame(
        {
            "one_point": np.where(rounded == 1.0, 1.0, 0.0),
            "two_point": np.where(rounded == 2.0, 2.0, 0.0),
            "three_plus": np.where(rounded >= 3.0, rounded, 0.0),
        },
        index=frame.index,
    )
    if not np.allclose(targets.sum(axis=1), points, atol=1e-12):
        raise AssertionError("Scoring channels do not conserve possession points.")
    return targets


def _season_adjust_targets(
    targets: pd.DataFrame,
    seasons: pd.Series,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Remove each channel's season mean without breaking additivity."""
    season_values = pd.to_numeric(seasons, errors="raise").astype(int)
    adjusted = targets.copy()
    metadata: dict[str, dict[str, float]] = {}
    for channel in POINT_CHANNELS:
        season_means = targets[channel].groupby(season_values).mean().sort_index()
        overall_mean = float(targets[channel].mean())
        adjusted[channel] = (
            targets[channel] - season_values.map(season_means) + overall_mean
        )
        metadata[channel] = {
            "overall": overall_mean,
            **{
                str(int(season)): float(value)
                for season, value in season_means.items()
            },
        }
    return adjusted.loc[:, POINT_CHANNELS].to_numpy(dtype=np.float64), metadata


def fit_points_channel_rapm(
    frame: pd.DataFrame,
    config: RapmConfig,
    *,
    names: pd.DataFrame | None = None,
) -> PointsChannelRapmResult:
    """Fit conserved 1-point, 2-point, and 3-plus-point RAPM channels."""
    targets = build_points_channel_targets(frame)
    adjusted_targets, _ = _season_adjust_targets(targets, frame["season"])

    adjusted_frame = frame.copy()
    season_points = frame.groupby("season")["pts"].transform("mean")
    adjusted_frame["pts"] = (
        pd.to_numeric(frame["pts"], errors="raise")
        - season_points
        + float(pd.to_numeric(frame["pts"], errors="raise").mean())
    )
    design = build_design(adjusted_frame, include_home=config.include_home)
    X = design.X.tocsr()
    channel_intercepts_array = adjusted_targets.mean(axis=0)
    centered = adjusted_targets - channel_intercepts_array
    penalty = _penalty(config, len(design.players))
    lhs = (X.T @ X).tocsc() + diags(penalty, format="csc")
    rhs = np.asarray(X.T @ centered)
    factor = splu(lhs)
    beta = np.column_stack(
        [factor.solve(rhs[:, index]) for index in range(len(POINT_CHANNELS))]
    )

    n_players = len(design.players)
    off_counts = np.asarray(X[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    intercepts = channel_intercepts_array.copy()
    for index in range(len(POINT_CHANNELS)):
        off_mean = float(np.average(beta[:n_players, index], weights=off_counts))
        def_mean = float(
            np.average(
                beta[n_players : 2 * n_players, index], weights=def_counts
            )
        )
        beta[:n_players, index] -= off_mean
        beta[n_players : 2 * n_players, index] -= def_mean
        intercepts[index] += 5.0 * (off_mean + def_mean)

    ratings = pd.DataFrame({"PLAYER_ID": design.players})
    if names is not None and {"PLAYER_ID", "PLAYER_NAME"}.issubset(names.columns):
        ratings = ratings.merge(
            names[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID"),
            on="PLAYER_ID",
            how="left",
            validate="one_to_one",
        )
    ratings["Poss_Off"] = design.off_possessions
    ratings["Poss_Def"] = design.def_possessions
    for index, channel in enumerate(POINT_CHANNELS):
        ratings[f"{channel}_offense"] = 100.0 * beta[:n_players, index]
        ratings[f"{channel}_defense"] = -100.0 * beta[
            n_players : 2 * n_players, index
        ]
        ratings[f"{channel}_net"] = (
            ratings[f"{channel}_offense"] + ratings[f"{channel}_defense"]
        )
    for component in ("offense", "defense", "net"):
        ratings[component] = ratings[
            [f"{channel}_{component}" for channel in POINT_CHANNELS]
        ].sum(axis=1)

    canonical_beta, canonical_intercept = fit_coefficients(design, config)
    canonical = ratings_table(design, canonical_beta).rename(
        columns={
            "player_id": "PLAYER_ID",
            "offense_per_100": "canonical_offense",
            "defense_per_100": "canonical_defense",
            "net_per_100": "canonical_net",
        }
    )
    checked = ratings.merge(
        canonical[["PLAYER_ID", "canonical_offense", "canonical_defense", "canonical_net"]],
        on="PLAYER_ID",
        validate="one_to_one",
    )
    component_errors = [
        np.abs(checked[component] - checked[f"canonical_{component}"]).max()
        for component in ("offense", "defense", "net")
    ]
    identity_errors = [
        np.abs(
            ratings[component]
            - ratings[[f"{channel}_{component}" for channel in POINT_CHANNELS]].sum(axis=1)
        ).max()
        for component in ("offense", "defense", "net")
    ]
    channel_identity_error = max(
        float(
            np.abs(
                ratings[f"{channel}_net"]
                - ratings[f"{channel}_offense"]
                - ratings[f"{channel}_defense"]
            ).max()
        )
        for channel in POINT_CHANNELS
    )
    channel_intercepts = {
        channel: float(intercepts[index])
        for index, channel in enumerate(POINT_CHANNELS)
    }
    channel_home = {
        channel: (
            float(100.0 * beta[2 * n_players, index])
            if config.include_home
            else 0.0
        )
        for index, channel in enumerate(POINT_CHANNELS)
    }
    quality = {
        "possession_rows": int(len(frame)),
        "games": int(frame["gameid"].nunique()),
        "players": int(n_players),
        "maximum_target_recomposition_error": float(
            np.abs(
                targets.sum(axis=1).to_numpy()
                - pd.to_numeric(frame["pts"], errors="raise").to_numpy(dtype=np.float64)
            ).max()
        ),
        "maximum_rating_recomposition_error": float(max(identity_errors)),
        "maximum_channel_net_identity_error": channel_identity_error,
        "maximum_canonical_rapm_error": float(max(component_errors)),
        "intercept_recomposition_error": float(
            abs(sum(channel_intercepts.values()) - canonical_intercept)
        ),
    }
    return PointsChannelRapmResult(
        ratings=ratings.sort_values("net", ascending=False).reset_index(drop=True),
        channel_intercepts=channel_intercepts,
        channel_home_effects_per_100=channel_home,
        quality=quality,
    )
