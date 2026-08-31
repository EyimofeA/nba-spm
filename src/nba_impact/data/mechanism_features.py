"""Same-season mechanism features for SPM research.

These features compress observed player-season inputs into basketball concepts.
They never use RAPM, future seasons, external ratings, or player identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


OFFENSE_MECHANISM_FEATURES = (
    "pass_value_per_potential_assist_eb",
    "load_adjusted_shot_quality_residual",
    "load_adjusted_creation_residual",
    "spacing_creation_interaction",
)

DEFENSE_MECHANISM_FEATURES = (
    "dreb_conversion_above_expected_eb",
    "foul_adjusted_activity_residual",
    "workload_adjusted_shot_suppression_residual",
    "rim_protection_workload_value",
)

MECHANISM_FEATURES = (*OFFENSE_MECHANISM_FEATURES, *DEFENSE_MECHANISM_FEATURES)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Mechanism feature input lacks {column}.")
    return pd.to_numeric(frame[column], errors="coerce")


def _season_eb_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    exposure: pd.Series,
    season: pd.Series,
    *,
    prior_opportunities: float,
) -> pd.Series:
    """Shrink a rate toward its same-season opportunity-weighted center."""
    opportunities = (denominator * exposure / 100.0).clip(lower=0.0)
    value = numerator / denominator.where(denominator.gt(0))
    table = pd.DataFrame(
        {"value": value, "opportunities": opportunities, "season": season}
    )
    centers = {}
    for label, group in table.groupby("season", sort=False):
        valid = group["value"].notna() & group["opportunities"].gt(0)
        centers[label] = (
            float(
                np.average(
                    group.loc[valid, "value"],
                    weights=group.loc[valid, "opportunities"],
                )
            )
            if valid.any()
            else 0.0
        )
    center = table["season"].map(centers).astype(float)
    reliability = opportunities / (opportunities + prior_opportunities)
    return reliability * value.fillna(center) + (1.0 - reliability) * center


def _leave_one_out_weighted_residual(
    frame: pd.DataFrame,
    response: str,
    predictors: tuple[str, ...],
    weight: str,
) -> pd.Series:
    """Return analytic leave-one-out residuals from a weighted linear projection."""
    columns = (response, *predictors, weight)
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & numeric[weight].gt(0)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() <= len(predictors) + 2:
        return result.fillna(0.0)

    data = numeric.loc[valid]
    design = np.column_stack(
        [np.ones(len(data)), *(data[column].to_numpy() for column in predictors)]
    )
    weights = data[weight].to_numpy(dtype=float).clip(min=1.0)
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_response = data[response].to_numpy(dtype=float) * root_weight
    inverse = np.linalg.pinv(weighted_design.T @ weighted_design)
    coefficients = inverse @ weighted_design.T @ weighted_response
    fitted = design @ coefficients
    leverage = np.einsum(
        "ij,jk,ik->i", weighted_design, inverse, weighted_design
    ).clip(0.0, 0.999999)
    residual = (data[response].to_numpy(dtype=float) - fitted) / (1.0 - leverage)
    result.loc[valid] = residual
    return result.fillna(0.0)


def compute_mechanism_features(annual: pd.DataFrame) -> pd.DataFrame:
    """Build eight same-season, target-free SPM research features."""
    required_keys = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss"}
    if missing := sorted(required_keys - set(annual.columns)):
        raise ValueError(f"Mechanism feature input lacks keys {missing}.")
    if annual.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Mechanism feature input has duplicate player-seasons.")

    frame = annual.copy()
    season = pd.to_numeric(frame["Window_End"], errors="raise").astype(int)
    output = frame[["PLAYER_ID", "Window_End"]].copy()

    output["pass_value_per_potential_assist_eb"] = _season_eb_rate(
        _numeric(frame, "assist_points_created_p100"),
        _numeric(frame, "potential_assists_p100"),
        _numeric(frame, "OffPoss"),
        season,
        prior_opportunities=100.0,
    )

    shot_parts = []
    creation_parts = []
    activity_parts = []
    suppression_parts = []
    for _, group in frame.assign(_season=season).groupby("_season", sort=False):
        shot_parts.append(
            _leave_one_out_weighted_residual(
                group,
                "shot_quality_average_relative",
                (
                    "offensive_load_2017_eb_p100",
                    "at_rim_frequency_eb",
                    "arc3_frequency_eb",
                ),
                "OffPoss",
            )
        )
        creation_parts.append(
            _leave_one_out_weighted_residual(
                group,
                "box_creation_2017_eb_p100",
                (
                    "offensive_load_2017_eb_p100",
                    "touches_p100",
                    "potential_assists_p100",
                ),
                "OffPoss",
            )
        )
        activity_group = group.copy()
        activity_group["_defensive_activity"] = (
            _numeric(activity_group, "contested_2pt_p100")
            + _numeric(activity_group, "contested_3pt_p100")
            + _numeric(activity_group, "deflections_p100")
            + _numeric(activity_group, "recovered_blocks_p100")
        )
        activity_group["_mechanism_weight"] = _numeric(
            activity_group, "DefPoss"
        ) * _numeric(activity_group, "has_hustle_tracking")
        activity_parts.append(
            _leave_one_out_weighted_residual(
                activity_group,
                "_defensive_activity",
                ("PF_p100",),
                "_mechanism_weight",
            )
        )
        suppression_group = group.copy()
        suppression_group["_mechanism_weight"] = _numeric(
            suppression_group, "DefPoss"
        ) * _numeric(suppression_group, "has_matchup_tracking")
        suppression_parts.append(
            _leave_one_out_weighted_residual(
                suppression_group,
                "matchup_opponent_adjusted_points_saved_p100_eb",
                (
                    "dfg_attempts_p100",
                    "rim_dfga_p100",
                ),
                "_mechanism_weight",
            )
        )

    offense_reliability = _numeric(frame, "OffPoss").clip(lower=0.0) / (
        _numeric(frame, "OffPoss").clip(lower=0.0) + 500.0
    )
    defense_reliability = _numeric(frame, "DefPoss").clip(lower=0.0) / (
        _numeric(frame, "DefPoss").clip(lower=0.0) + 500.0
    )
    output["load_adjusted_shot_quality_residual"] = (
        offense_reliability * pd.concat(shot_parts).sort_index()
    )
    output["load_adjusted_creation_residual"] = (
        offense_reliability * pd.concat(creation_parts).sort_index()
    )
    output["foul_adjusted_activity_residual"] = (
        defense_reliability * pd.concat(activity_parts).sort_index()
    )
    output["workload_adjusted_shot_suppression_residual"] = (
        defense_reliability * pd.concat(suppression_parts).sort_index()
    )

    spacing = _numeric(frame, "crafted_spacing_stable_v1")
    creation = _numeric(frame, "box_creation_2017_eb_p100")
    output["spacing_creation_interaction"] = spacing * creation

    rebound_conversion = _numeric(frame, "DREB_p100") / _numeric(
        frame, "dreb_chances_p100"
    ).where(_numeric(frame, "dreb_chances_p100").gt(0))
    rebound_frame = frame.copy()
    rebound_frame["_rebound_conversion"] = rebound_conversion
    rebound_frame["_contest_share"] = _numeric(
        frame, "dreb_contests_p100"
    ) / _numeric(frame, "dreb_chances_p100").where(
        _numeric(frame, "dreb_chances_p100").gt(0)
    )
    rebound_residual_parts = []
    for _, group in rebound_frame.assign(_season=season).groupby(
        "_season", sort=False
    ):
        rebound_residual_parts.append(
            _leave_one_out_weighted_residual(
                group,
                "_rebound_conversion",
                ("_contest_share", "dreb_chances_p100"),
                "DefPoss",
            )
        )
    rebound_residual = pd.concat(rebound_residual_parts).sort_index()
    rebound_opportunities = (
        _numeric(frame, "dreb_chances_p100")
        * _numeric(frame, "DefPoss")
        / 100.0
    ).clip(lower=0.0)
    rebound_reliability = rebound_opportunities / (rebound_opportunities + 100.0)
    output["dreb_conversion_above_expected_eb"] = (
        rebound_reliability * rebound_residual
    )

    rim_workload = _numeric(frame, "rim_dfga_p100").clip(lower=0.0)
    output["rim_protection_workload_value"] = _numeric(
        frame, "rim_points_saved_p100"
    ) * np.sqrt(rim_workload)

    output[list(MECHANISM_FEATURES)] = output[list(MECHANISM_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return output[["PLAYER_ID", "Window_End", *MECHANISM_FEATURES]]
