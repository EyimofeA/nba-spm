"""Coverage audit for frozen SPM feature contracts.

Coverage means an observed upstream value. It does not count a median or zero
fill as observed. The audit separates source absence from a valid undefined
rate caused by zero opportunities.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .statistical_features import (
    CORE_RATE_SPECS,
    NATURAL_WEIGHTED_AVERAGES,
    RATIO_SPECS,
    TRACKING_RATE_SPECS,
)


DFG_FEATURES = {
    "dfg_attempts_p100",
    "dfg_diff_pct_eb",
    "dfg_two_point_equivalent_saved_p100",
}
RIM_DFG_FEATURES = {
    "rim_dfga_p100",
    "rim_diff_pct_eb",
    "rim_points_saved_p100_raw",
    "rim_points_saved_p100",
    "rim_matchup_attempt_share",
}
HUSTLE_FEATURES = {
    "deflections_p100",
    "charges_drawn_p100",
    "contested_2pt_p100",
    "contested_3pt_p100",
    "contested_3pt_share",
    "def_loose_balls_recovered_p100",
}
PLAYTYPE_FEATURES = {
    "player_ts_pct",
    "league_ts_pct",
    "relative_ts_pct_points",
    "playtype_expected_ts_pct",
    "playtype_difficulty_pct_points",
    "zts_pct_points",
    "playtype_poe_per_75",
    "transition_share",
    "transition_poe_per_75",
}


def feature_source_family(feature: str) -> str:
    """Return the upstream family that determines observed coverage."""
    if feature in PLAYTYPE_FEATURES:
        return "playtype"
    if feature in DFG_FEATURES:
        return "dfg"
    if feature in RIM_DFG_FEATURES:
        return "rim_dfg"
    if feature in HUSTLE_FEATURES:
        return "hustle"
    if feature.startswith("matchup_"):
        return "matchup_defense"
    return "player_sheet"


def _feature_reason(feature: str, family: str) -> tuple[str, str]:
    if family == "playtype":
        return (
            "source_eligibility",
            "The playtype build requires at least 250 minutes and 50 qualifying "
            "playtype possessions. Players below either threshold have no observed row.",
        )
    if family == "hustle":
        return (
            "source_history_and_row_absence",
            "NBA hustle data starts in 2018 in the pinned source. A small number of "
            "later player-seasons also lack an observed hustle row.",
        )
    if family == "matchup_defense":
        return (
            "source_history_and_matchup_exposure",
            "The pinned scorer-defender matchup source starts in 2018. Later omissions "
            "are players without a positive observed matchup assignment.",
        )
    if family == "dfg":
        return (
            "no_defended_shot_row",
            "The closest-defender dashboard omits player-seasons without an observed "
            "defended-shot row. The pipeline does not treat that absence as zero defense.",
        )
    if family == "rim_dfg":
        return (
            "no_rim_defended_shot_row",
            "The rim-defense dashboard omits player-seasons without an observed rim "
            "defended-shot row. The pipeline does not treat that absence as zero defense.",
        )

    base = feature.removesuffix("_relative")
    if base in RATIO_SPECS or base in {
        "true_shooting_pct",
        "self_oreb_adjusted_true_shooting_pct",
    }:
        return (
            "zero_opportunity_or_missing_input",
            "The rate is undefined when its player-season denominator is zero. It is "
            "also missing when the player sheet lacks the required tracking field or "
            "lineup-derived possession exposure.",
        )
    if base in CORE_RATE_SPECS or base in TRACKING_RATE_SPECS:
        numerator, denominator = (CORE_RATE_SPECS | TRACKING_RATE_SPECS)[base]
        return (
            "missing_player_sheet_input",
            f"The player sheet lacks {numerator} or {denominator} for some rows. Most "
            "broad per-100 gaps come from missing lineup-derived OffPoss/DefPoss on "
            "low-exposure player-seasons; event-specific tracking fields add more gaps.",
        )
    if base in NATURAL_WEIGHTED_AVERAGES:
        source, weight = NATURAL_WEIGHTED_AVERAGES[base]
        return (
            "missing_weighted_average_input",
            f"The player sheet lacks {source}, or has no positive {weight} weight, for "
            "some player-seasons.",
        )
    return (
        "derived_input_missing",
        "At least one upstream player-sheet input or lineup-derived possession exposure "
        "needed by this derived feature is unavailable.",
    )


def normalize_source_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a source frame to unique integer PLAYER_ID/Season keys."""
    season_column = next(
        (column for column in ("Season", "Window_End", "year") if column in frame),
        None,
    )
    if season_column is None or "PLAYER_ID" not in frame:
        raise ValueError("Coverage source needs PLAYER_ID and Season/Window_End/year.")
    keys = frame[["PLAYER_ID", season_column]].copy()
    keys.columns = ["PLAYER_ID", "Season"]
    keys["PLAYER_ID"] = pd.to_numeric(keys["PLAYER_ID"], errors="coerce")
    keys["Season"] = pd.to_numeric(keys["Season"], errors="coerce")
    return keys.dropna().drop_duplicates().astype(int)


def _source_membership(
    panel: pd.DataFrame,
    source_keys: pd.DataFrame,
    *,
    panel_kind: str,
) -> pd.Series:
    observed = set(map(tuple, source_keys[["PLAYER_ID", "Season"]].to_numpy()))
    if panel_kind == "annual":
        return pd.Series(
            [
                (int(player), int(season)) in observed
                for player, season in panel[["PLAYER_ID", "Window_End"]].itertuples(index=False)
            ],
            index=panel.index,
        )
    if panel_kind != "five_year":
        raise ValueError(f"Unknown panel kind {panel_kind}.")
    return pd.Series(
        [
            any((int(player), season) in observed for season in range(int(end) - 4, int(end) + 1))
            for player, end in panel[["PLAYER_ID", "Window_End"]].itertuples(index=False)
        ],
        index=panel.index,
    )


def audit_feature_coverage(
    annual: pd.DataFrame,
    five_year: pd.DataFrame,
    selected: Mapping[str, tuple[str, ...]],
    source_keys: Mapping[str, pd.DataFrame],
    *,
    threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-feature and per-season observed-coverage ledgers."""
    selected_union = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    normalized = {family: normalize_source_keys(frame) for family, frame in source_keys.items()}
    feature_rows: list[dict] = []
    season_rows: list[dict] = []
    for panel_kind, panel in (("annual", annual), ("five_year", five_year)):
        for feature in selected_union:
            family = feature_source_family(feature)
            if family == "player_sheet":
                observed = panel[feature].notna()
            else:
                if family not in normalized:
                    raise ValueError(f"Missing source keys for {family}.")
                # A source row is necessary but not sufficient. The feature can
                # remain unavailable when its exposure denominator is missing.
                observed = panel[feature].notna() & _source_membership(
                    panel, normalized[family], panel_kind=panel_kind
                )
            fraction = float(observed.mean())
            reason_code, reason = _feature_reason(feature, family)
            sides = "+".join(
                side for side in ("offense", "defense") if feature in selected[side]
            )
            feature_rows.append(
                {
                    "panel": panel_kind,
                    "feature": feature,
                    "side": sides,
                    "source_family": family,
                    "rows": int(len(panel)),
                    "observed_rows": int(observed.sum()),
                    "coverage_fraction": fraction,
                    "below_threshold": bool(fraction < threshold),
                    "reason_code": reason_code if fraction < threshold else "complete",
                    "reason": reason if fraction < threshold else "Observed coverage meets the gate.",
                }
            )
            for season, index in panel.groupby("Window_End", sort=True).groups.items():
                season_observed = observed.loc[index]
                season_rows.append(
                    {
                        "panel": panel_kind,
                        "feature": feature,
                        "Window_End": int(season),
                        "rows": int(len(index)),
                        "observed_rows": int(season_observed.sum()),
                        "coverage_fraction": float(season_observed.mean()),
                    }
                )
    summary = pd.DataFrame(feature_rows).sort_values(
        ["panel", "coverage_fraction", "feature"]
    ).reset_index(drop=True)
    by_season = pd.DataFrame(season_rows).sort_values(
        ["panel", "feature", "Window_End"]
    ).reset_index(drop=True)
    unexplained = summary.loc[
        summary["below_threshold"] & summary["reason_code"].eq("")
    ]
    if not unexplained.empty:
        raise ValueError(f"Unexplained feature coverage: {unexplained['feature'].tolist()}")
    return summary, by_season
