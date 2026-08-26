"""Basketball-domain feature engineering layered on the validated v1 table."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.player_skill_features import PLAYER_SKILL_MODEL_FEATURES
from nba_impact.data.behavior_roles import ROLE_MODEL_FEATURES
from nba_impact.data.statistical_features import RATIO_SPECS, _aggregate_window, _load_source

TEMPORAL_FEATURES = (
    "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
    "DREB_p100", "PF_p100", "FTA_p100", "FG3A_p100", "drives_p100",
    "touches_p100", "potential_assists_p100", "true_shooting_pct",
    "at_rim_frequency", "arc3_frequency", "shot_quality_average",
    "rebound_contests_p100", "rebound_chances_p100", "recovered_blocks_p100",
)

RELATIVE_FEATURES = (
    "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
    "DREB_p100", "PF_p100", "FTA_p100", "FG3A_p100", "drives_p100",
    "potential_assists_p100", "true_shooting_pct", "at_rim_frequency",
    "arc3_frequency", "shot_quality_average",
    "self_oreb_adjusted_true_shooting_pct",
)

EXTRA_COUNTS = (
    "PtsAssisted2s", "PtsAssisted3s", "PtsUnassisted2s", "PtsUnassisted3s",
    "PULL_UP_FGA", "CATCH_SHOOT_FGA", "DRIVE_PASSES", "DRIVES",
    "AST", "POTENTIAL_AST", "AST_PTS_CREATED", "TOUCHES", "PTS",
    "LiveBallTurnovers", "PASSES_MADE", "PFD", "FGA", "FG2A", "FG2M",
    "FG3A", "FG3M",
    "REB_CONTEST", "REB_CHANCES", "DREB_CONTEST", "DREB_UNCONTEST",
    "RecoveredBlocks", "Charge_Fouls_Drawn", "Offensive_Fouls_Drawn",
    "BLK", "STL", "PF", "DefPoss", "OffPoss",
    "PAINT_TOUCHES", "POST_TOUCHES", "ELBOW_TOUCHES",
)

BOUNDED_FEATURES = (
    "self_created_point_share", "assisted_three_share", "pull_up_attempt_share",
    "potential_assist_conversion", "drive_pass_rate",
    "rebound_contest_share", "dreb_contested_share",
    "rim_and_three_frequency", "midrange_frequency", "effective_fg_pct",
    "three_point_attempt_rate",
)

PUBLIC_BENCHMARK_FEATURES = (
    "shooting_proficiency_2017",
    "box_creation_2017_p100",
    "offensive_load_2017_p100",
    "assist_to_load_2017",
    "turnover_to_load_2017",
    "creation_to_load_2017",
    "behavioral_passer_score_v1",
    "crafted_spacing_proxy_v1",
)

PRIMARY_PUBLIC_INSPIRED_FEATURES = (
    "shooting_proficiency_2017_eb",
    "box_creation_2017_eb_p100",
    "offensive_load_2017_eb_p100",
    "assist_to_load_2017_eb",
    "turnover_to_load_2017_eb",
    "creation_to_load_2017_eb",
    "behavioral_passer_score_v1",
    "crafted_spacing_stable_v1",
)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def _bounded_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = _ratio(numerator, denominator)
    return ratio.where(numerator <= denominator)


def _weighted_zscore(values: pd.Series, weights: pd.Series) -> pd.Series:
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    center = float(np.average(values.loc[valid], weights=weights.loc[valid]))
    variance = float(
        np.average((values.loc[valid] - center) ** 2, weights=weights.loc[valid])
    )
    if variance <= 0:
        return pd.Series(0.0, index=values.index, dtype=float).where(values.notna())
    return (values - center) / np.sqrt(variance)


def _shrink_rate(
    values: pd.Series,
    exposure: pd.Series,
    *,
    strength: float = 500.0,
) -> pd.Series:
    valid = values.notna() & exposure.gt(0)
    if not valid.any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    center = float(np.average(values.loc[valid], weights=exposure.loc[valid]))
    reliability = exposure.clip(lower=0) / (exposure.clip(lower=0) + strength)
    return reliability * values + (1.0 - reliability) * center


def _pooled_counts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    for column in EXTRA_COUNTS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby("PLAYER_ID", as_index=False)[list(EXTRA_COUNTS)].sum(min_count=1)


def _prior_strength(denominator: str) -> float:
    if denominator == "TOUCHES":
        return 500.0
    if denominator in {"DRIVES", "TOV"}:
        return 150.0
    if "TOUCHES" in denominator:
        return 100.0
    return 100.0


def _stabilized_ratios(frames: list[pd.DataFrame], player_ids: pd.Series) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    needed = {column for values in RATIO_SPECS.values() for column in values}
    for column in needed:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    sums = frame.groupby("PLAYER_ID", as_index=True)[sorted(needed)].sum(min_count=1)
    output = pd.DataFrame({"PLAYER_ID": player_ids})
    for name, (numerator, denominator) in RATIO_SPECS.items():
        league_denominator = frame[denominator].sum()
        league_rate = (
            frame[numerator].sum() / league_denominator
            if league_denominator > 0
            else np.nan
        )
        strength = _prior_strength(denominator)
        stabilized = (sums[numerator].fillna(0) + strength * league_rate) / (
            sums[denominator].fillna(0) + strength
        )
        output[f"{name}_eb"] = output["PLAYER_ID"].map(stabilized)
    return output


def _entropy(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    shares = frame[columns].fillna(0).clip(lower=0)
    total = shares.sum(axis=1)
    shares = shares.div(total.replace(0, np.nan), axis=0)
    entropy = -(shares.where(shares > 0) * np.log(shares.where(shares > 0))).sum(axis=1)
    return entropy.where(total > 0)


def _engineer_window(
    base: pd.DataFrame,
    frames: list[pd.DataFrame],
    seasonal: list[pd.DataFrame],
) -> pd.DataFrame:
    output = base.copy()
    stabilized = _stabilized_ratios(frames, output["PLAYER_ID"])
    output = output.merge(stabilized, on="PLAYER_ID", validate="one_to_one")
    counts = _pooled_counts(frames).set_index("PLAYER_ID")
    mapped = {column: output["PLAYER_ID"].map(counts[column]) for column in EXTRA_COUNTS}

    engineered: dict[str, pd.Series] = {}

    assisted = mapped["PtsAssisted2s"] + mapped["PtsAssisted3s"]
    unassisted = mapped["PtsUnassisted2s"] + mapped["PtsUnassisted3s"]
    engineered["self_created_point_share"] = _ratio(unassisted, assisted + unassisted)
    engineered["assisted_three_share"] = _ratio(
        mapped["PtsAssisted3s"], mapped["PtsAssisted3s"] + mapped["PtsUnassisted3s"]
    )
    engineered["pull_up_attempt_share"] = _ratio(
        mapped["PULL_UP_FGA"], mapped["PULL_UP_FGA"] + mapped["CATCH_SHOOT_FGA"]
    )
    engineered["potential_assist_conversion"] = _bounded_ratio(
        mapped["AST"], mapped["POTENTIAL_AST"]
    )
    engineered["assist_points_per_touch"] = _ratio(mapped["AST_PTS_CREATED"], mapped["TOUCHES"])
    engineered["total_points_created_per_touch"] = _ratio(
        mapped["PTS"] + mapped["AST_PTS_CREATED"], mapped["TOUCHES"]
    )
    engineered["drive_pass_rate"] = _ratio(mapped["DRIVE_PASSES"], mapped["DRIVES"])
    creation_events = mapped["DRIVES"] + mapped["POTENTIAL_AST"] + mapped["PULL_UP_FGA"]
    engineered["creation_load_p100"] = 100 * _ratio(creation_events, mapped["OffPoss"])
    engineered["live_ball_turnovers_per_creation"] = _ratio(
        mapped["LiveBallTurnovers"], creation_events
    )
    engineered["foul_pressure_per_fga"] = _ratio(mapped["PFD"], mapped["FGA"])
    engineered["stocks_p100_def"] = 100 * _ratio(mapped["STL"] + mapped["BLK"], mapped["DefPoss"])
    engineered["defensive_activity_p100"] = 100 * _ratio(
        mapped["STL"] + mapped["BLK"] + mapped["REB_CONTEST"], mapped["DefPoss"]
    )
    # Gabriel's public ``Stops`` field is this event count. It is not Dean
    # Oliver's team-allocated Stop%; retain the narrower name and components.
    observed_stops = (
        mapped["STL"].fillna(0.0)
        + mapped["RecoveredBlocks"].fillna(0.0)
        + mapped["Charge_Fouls_Drawn"].fillna(0.0)
        + mapped["Offensive_Fouls_Drawn"].fillna(0.0)
    )
    engineered["event_stops_p100"] = 100 * _ratio(observed_stops, mapped["DefPoss"])
    engineered["rebound_contest_share"] = _ratio(mapped["REB_CONTEST"], mapped["REB_CHANCES"])
    engineered["dreb_contested_share"] = _ratio(
        mapped["DREB_CONTEST"], mapped["DREB_CONTEST"] + mapped["DREB_UNCONTEST"]
    )
    engineered["block_recovery_rate"] = _ratio(mapped["RecoveredBlocks"], mapped["BLK"])
    engineered["stocks_per_foul"] = _ratio(mapped["STL"] + mapped["BLK"], mapped["PF"])
    engineered["interior_role_load"] = 100 * _ratio(
        mapped["PAINT_TOUCHES"] + mapped["POST_TOUCHES"] + mapped["ELBOW_TOUCHES"],
        mapped["OffPoss"],
    )

    zone_frequencies = [
        "at_rim_frequency", "short_mid_frequency", "long_mid_frequency",
        "corner3_frequency", "arc3_frequency",
    ]
    engineered["shot_profile_entropy"] = _entropy(output, zone_frequencies)
    engineered["effective_shot_zones"] = np.exp(engineered["shot_profile_entropy"])
    engineered["rim_and_three_frequency"] = (
        output["at_rim_frequency"] + output["corner3_frequency"] + output["arc3_frequency"]
    )
    engineered["midrange_frequency"] = output["short_mid_frequency"] + output["long_mid_frequency"]
    engineered["expected_zone_points"] = (
        2 * output["at_rim_frequency"] * output["at_rim_accuracy_eb"]
        + 2 * output["short_mid_frequency"] * output["short_mid_accuracy_eb"]
        + 2 * output["long_mid_frequency"] * output["long_mid_accuracy_eb"]
        + 3 * output["corner3_frequency"] * output["corner3_accuracy_eb"]
        + 3 * output["arc3_frequency"] * output["arc3_accuracy_eb"]
    )

    fga_p100 = output["FG2A_p100"] + output["FG3A_p100"]
    fgm_p100 = output["FG2M_p100"] + output["FG3M_p100"]
    engineered["effective_fg_pct"] = _ratio(
        fgm_p100 + 0.5 * output["FG3M_p100"], fga_p100
    )
    engineered["three_point_attempt_rate"] = _ratio(output["FG3A_p100"], fga_p100)
    engineered["free_throw_rate"] = _ratio(output["FTA_p100"], fga_p100)

    shooting_proficiency = (
        2.0 / (1.0 + np.exp(-output["FG3A_p100"].clip(lower=0))) - 1.0
    ) * output["fg3_pct"]
    engineered["shooting_proficiency_2017"] = shooting_proficiency
    box_creation = (
        0.1843 * output["AST_p100"]
        + 0.0969 * (output["PTS_p100"] + output["TOV_p100"])
        - 2.3021 * shooting_proficiency
        + 0.0582
        * output["AST_p100"]
        * (output["PTS_p100"] + output["TOV_p100"])
        * shooting_proficiency
        - 1.1942
    )
    engineered["box_creation_2017_p100"] = box_creation
    offensive_load = (
        0.75 * (output["AST_p100"] - 0.38 * box_creation)
        + fga_p100
        + 0.44 * output["FTA_p100"]
        + box_creation
        + output["TOV_p100"]
    )
    engineered["offensive_load_2017_p100"] = offensive_load
    engineered["assist_to_load_2017"] = _ratio(output["AST_p100"], offensive_load)
    engineered["turnover_to_load_2017"] = _ratio(output["TOV_p100"], offensive_load)
    engineered["creation_to_load_2017"] = _ratio(box_creation, offensive_load)

    weights = output["OffPoss"]
    pts_eb = _shrink_rate(output["PTS_p100"], weights)
    ast_eb = _shrink_rate(output["AST_p100"], weights)
    tov_eb = _shrink_rate(output["TOV_p100"], weights)
    fga_eb = _shrink_rate(fga_p100, weights)
    fta_eb = _shrink_rate(output["FTA_p100"], weights)
    fg3a_eb = _shrink_rate(output["FG3A_p100"], weights)
    shooting_proficiency_eb = (
        2.0 / (1.0 + np.exp(-fg3a_eb.clip(lower=0))) - 1.0
    ) * output["fg3_pct_eb"]
    engineered["shooting_proficiency_2017_eb"] = shooting_proficiency_eb
    box_creation_eb = (
        0.1843 * ast_eb
        + 0.0969 * (pts_eb + tov_eb)
        - 2.3021 * shooting_proficiency_eb
        + 0.0582 * ast_eb * (pts_eb + tov_eb) * shooting_proficiency_eb
        - 1.1942
    )
    engineered["box_creation_2017_eb_p100"] = box_creation_eb
    offensive_load_eb = (
        0.75 * (ast_eb - 0.38 * box_creation_eb)
        + fga_eb
        + 0.44 * fta_eb
        + box_creation_eb
        + tov_eb
    )
    engineered["offensive_load_2017_eb_p100"] = offensive_load_eb
    load_exposure = weights * offensive_load_eb.clip(lower=0) / 100.0
    assist_to_load_eb = _shrink_rate(
        _ratio(ast_eb, offensive_load_eb), load_exposure, strength=150.0
    )
    turnover_to_load_eb = _shrink_rate(
        _ratio(tov_eb, offensive_load_eb), load_exposure, strength=150.0
    )
    creation_to_load_eb = _shrink_rate(
        _ratio(box_creation_eb, offensive_load_eb), load_exposure, strength=150.0
    )
    engineered["assist_to_load_2017_eb"] = assist_to_load_eb
    engineered["turnover_to_load_2017_eb"] = turnover_to_load_eb
    engineered["creation_to_load_2017_eb"] = creation_to_load_eb
    engineered["behavioral_passer_score_v1"] = (
        _weighted_zscore(offensive_load_eb, weights).clip(-4.0, 4.0)
        + 3.0 * _weighted_zscore(assist_to_load_eb, weights).clip(-4.0, 4.0)
        - 2.0 * _weighted_zscore(turnover_to_load_eb, weights).clip(-4.0, 4.0)
        + 0.5 * _weighted_zscore(creation_to_load_eb, weights).clip(-4.0, 4.0)
    )
    league_fga = float(mapped["FGA"].sum(skipna=True))
    league_efg = (
        float((mapped["FG2M"] + 1.5 * mapped["FG3M"]).sum(skipna=True)) / league_fga
        if league_fga > 0
        else np.nan
    )
    engineered["crafted_spacing_proxy_v1"] = (
        output["FG3A_p100"] * (1.5 * output["fg3_pct"]) - league_efg
    )
    engineered["crafted_spacing_stable_v1"] = (
        fg3a_eb * (1.5 * output["fg3_pct_eb"]) - league_efg
    )

    for feature in RELATIVE_FEATURES:
        defensive_rates = {"STL_p100", "BLK_p100", "DREB_p100", "PF_p100"}
        weights = output["DefPoss"] if feature in defensive_rates else output["OffPoss"]
        valid = output[feature].notna() & weights.gt(0)
        center = (
            np.average(output.loc[valid, feature], weights=weights[valid])
            if valid.any()
            else np.nan
        )
        engineered[f"{feature}_relative"] = output[feature] - center

    seasonal_indexed = [frame.set_index("PLAYER_ID") for frame in seasonal]
    for feature in TEMPORAL_FEATURES:
        values = pd.concat([frame[feature] for frame in seasonal_indexed], axis=1)
        values.columns = ["old", "middle", "latest"]
        latest = output["PLAYER_ID"].map(values["latest"])
        trend = output["PLAYER_ID"].map((values["latest"] - values["old"]) / 2)
        volatility = output["PLAYER_ID"].map(values.std(axis=1, ddof=0))
        engineered[f"{feature}_latest"] = latest.fillna(output[feature])
        engineered[f"{feature}_trend"] = trend.fillna(0.0)
        engineered[f"{feature}_volatility"] = volatility.fillna(0.0)
    return pd.concat([output, pd.DataFrame(engineered, index=output.index)], axis=1)


def build_statistical_features_v2(
    source_dir: str | Path,
    base_features_path: str | Path,
    *,
    artifact_root: str | Path,
    window_ends: tuple[int, ...] = tuple(range(2016, 2025)),
    pooled_window_seasons: int = 3,
    playtype_features_path: str | Path | None = None,
    defensive_tracking_features_path: str | Path | None = None,
    assist_quality_features_path: str | Path | None = None,
    matchup_defense_features_path: str | Path | None = None,
    player_skill_features_path: str | Path | None = None,
    behavior_roles_path: str | Path | None = None,
    offense_roles_path: str | Path | None = None,
    defense_roles_path: str | Path | None = None,
    source_overrides: Mapping[int, str | Path] | None = None,
) -> dict:
    if pooled_window_seasons < 1:
        raise ValueError("pooled_window_seasons must be positive.")
    source = Path(source_dir)
    base = pd.read_parquet(base_features_path)
    loaded = {}
    source_hashes = {}
    source_overrides = dict(source_overrides or {})
    history_seasons = max(3, pooled_window_seasons)
    for season in range(min(window_ends) - history_seasons + 1, max(window_ends) + 1):
        path = Path(source_overrides.get(season, source / f"{season}.csv"))
        loaded[season] = _load_source(path, season)[0]
        source_hashes[str(season)] = sha256_file(path)
    outputs = []
    for end in window_ends:
        frames = [
            loaded[season]
            for season in range(end - pooled_window_seasons + 1, end + 1)
        ]
        temporal_frames = [loaded[season] for season in range(end - 2, end + 1)]
        seasonal = [
            _aggregate_window([frame], season)
            for frame, season in zip(
                temporal_frames, range(end - 2, end + 1), strict=True
            )
        ]
        base_window = base.loc[base["Window_End"].eq(end)].copy()
        outputs.append(_engineer_window(base_window, frames, seasonal))
    features = pd.concat(outputs, ignore_index=True)
    playtype_feature_names: list[str] = []
    if playtype_features_path is not None:
        playtype = pd.read_parquet(playtype_features_path).rename(
            columns={"Season": "Window_End"}
        )
        if playtype.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Playtype feature keys are not unique.")
        playtype_feature_names = [
            column for column in playtype.columns
            if column not in {"PLAYER_ID", "Window_End", "synergy_possessions"}
        ]
        features = features.merge(
            playtype[["PLAYER_ID", "Window_End", *playtype_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    defensive_tracking_feature_names: list[str] = []
    if defensive_tracking_features_path is not None:
        defensive = pd.read_parquet(defensive_tracking_features_path).rename(
            columns={"Season": "Window_End"}
        )
        if defensive.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Defensive tracking feature keys are not unique.")
        defensive_tracking_feature_names = [
            column for column in defensive.columns
            if column not in {"PLAYER_ID", "Window_End"}
        ]
        features = features.merge(
            defensive[["PLAYER_ID", "Window_End", *defensive_tracking_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    assist_quality_feature_names: list[str] = []
    if assist_quality_features_path is not None:
        assist = pd.read_parquet(assist_quality_features_path).rename(
            columns={"Season": "Window_End"}
        )
        if assist.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Assist-quality feature keys are not unique.")
        assist_quality_feature_names = [
            column for column in assist.columns
            if column not in {"PLAYER_ID", "Window_End"}
        ]
        features = features.merge(
            assist[["PLAYER_ID", "Window_End", *assist_quality_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    matchup_defense_feature_names: list[str] = []
    if matchup_defense_features_path is not None:
        matchup = pd.read_parquet(matchup_defense_features_path).rename(
            columns={"Season": "Window_End"}
        )
        if matchup.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Matchup-defense feature keys are not unique.")
        matchup_defense_feature_names = [
            column
            for column in matchup.columns
            if column not in {"PLAYER_ID", "Window_End", "matchup_possessions"}
        ]
        features = features.merge(
            matchup[["PLAYER_ID", "Window_End", *matchup_defense_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    player_skill_feature_names: list[str] = []
    if player_skill_features_path is not None:
        skill = pd.read_parquet(player_skill_features_path).rename(
            columns={"Season": "Window_End"}
        )
        if skill.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Player-skill feature keys are not unique.")
        player_skill_feature_names = [
            column for column in PLAYER_SKILL_MODEL_FEATURES if column in skill.columns
        ]
        features = features.merge(
            skill[["PLAYER_ID", "Window_End", *player_skill_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    behavior_role_feature_names: list[str] = []
    if behavior_roles_path is not None:
        roles = pd.read_parquet(behavior_roles_path).rename(
            columns={"Season": "Window_End"}
        )
        if roles.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError("Behavior-role feature keys are not unique.")
        behavior_role_feature_names = [
            column for column in ROLE_MODEL_FEATURES if column in roles.columns
        ]
        if behavior_role_feature_names != list(ROLE_MODEL_FEATURES):
            raise ValueError("Behavior-role artifact is missing frozen model candidates.")
        features = features.merge(
            roles[["PLAYER_ID", "Window_End", *behavior_role_feature_names]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    side_role_feature_names: dict[str, list[str]] = {"offense": [], "defense": []}
    for side, path, prefix in (
        ("offense", offense_roles_path, "off_role"),
        ("defense", defense_roles_path, "def_role"),
    ):
        if path is None:
            continue
        roles = pd.read_parquet(path).rename(columns={"Season": "Window_End"})
        if roles.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError(f"{side.title()} role feature keys are not unique.")
        axes = sorted(
            (column for column in roles if column.startswith(f"{prefix}_axis_")),
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        affinities = sorted(
            (column for column in roles if column.startswith(f"{prefix}_affinity_")),
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        candidates = [*axes, *affinities[:-1]]
        if not axes or len(affinities) < 2:
            raise ValueError(f"{side.title()} role artifact has no usable role map.")
        side_role_feature_names[side] = candidates
        features = features.merge(
            roles[["PLAYER_ID", "Window_End", *candidates]],
            on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one",
        )
    defense_role_interaction_feature_names: list[str] = []
    defensive_skill_roots = (
        "dfg_diff_pct_eb", "rim_points_saved_p100", "deflections_p100",
        "charges_drawn_p100", "matchup_opponent_adjusted_points_saved_p100_eb",
        "matchup_shotmaking_points_saved_vs_scorer_p100_eb",
    )
    defense_axes = [
        feature for feature in side_role_feature_names["defense"]
        if feature.startswith("def_role_axis_")
    ]
    for skill in defensive_skill_roots:
        if skill not in features:
            continue
        for axis in defense_axes:
            interaction = f"{skill}_x_{axis}"
            features[interaction] = features[skill] * features[axis]
            defense_role_interaction_feature_names.append(interaction)
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("V2 feature keys are not unique.")
    new_features = [column for column in features if column not in base.columns]
    raw_new_features = features[new_features].copy()
    missing_by_feature = raw_new_features.isna().mean().sort_values(ascending=False)
    for feature in new_features:
        window_median = features.groupby("Window_End")[feature].transform("median")
        features[feature] = features[feature].fillna(window_median).fillna(0.0)
    bound_violations = {
        column: int(((features[column] < 0) | (features[column] > 1)).sum())
        for column in BOUNDED_FEATURES
    }
    audit = pd.DataFrame(
        {
            "feature": new_features,
            "missing_fraction_before_neutral_fill": [
                float(raw_new_features[column].isna().mean()) for column in new_features
            ],
            "minimum": [float(features[column].min()) for column in new_features],
            "median": [float(features[column].median()) for column in new_features],
            "maximum": [float(features[column].max()) for column in new_features],
        }
    )
    config = {
        "base_features_sha256": sha256_file(base_features_path),
        "source_hashes": source_hashes,
        "builder_sha256": sha256_file(Path(__file__)),
        "window_ends": list(window_ends),
        "pooled_window_seasons": pooled_window_seasons,
        "playtype_features_sha256": (
            sha256_file(playtype_features_path) if playtype_features_path else None
        ),
        "defensive_tracking_features_sha256": (
            sha256_file(defensive_tracking_features_path)
            if defensive_tracking_features_path else None
        ),
        "assist_quality_features_sha256": (
            sha256_file(assist_quality_features_path)
            if assist_quality_features_path else None
        ),
        "matchup_defense_features_sha256": (
            sha256_file(matchup_defense_features_path)
            if matchup_defense_features_path else None
        ),
        "player_skill_features_sha256": (
            sha256_file(player_skill_features_path)
            if player_skill_features_path else None
        ),
        "behavior_roles_sha256": (
            sha256_file(behavior_roles_path) if behavior_roles_path else None
        ),
        "offense_roles_sha256": (
            sha256_file(offense_roles_path) if offense_roles_path else None
        ),
        "defense_roles_sha256": (
            sha256_file(defense_roles_path) if defense_roles_path else None
        ),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"statistical_features_v2_{identity}"
    output = Path(artifact_root) / "features" / "statistical_impact" / run_id
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.parquet"
    audit_path = output / "audit.parquet"
    features.to_parquet(features_path, index=False)
    audit.to_parquet(audit_path, index=False)
    run = {
        "run_id": run_id,
        "dataset": "statistical_impact_features_v2",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": len(features),
            "players": features["PLAYER_ID"].nunique(),
            "total_features": len(features.columns) - 4,
            "new_features": len(new_features),
            "duplicate_keys": 0,
            "infinite_values": int(np.isinf(features.select_dtypes(include="number")).sum().sum()),
            "bounded_feature_violations": int(sum(bound_violations.values())),
            "max_new_feature_missing_fraction_before_neutral_fill": float(missing_by_feature.max()),
            "new_feature_missing_values_after_neutral_fill": int(
                features[new_features].isna().sum().sum()
            ),
        },
        "bounded_feature_violations": bound_violations,
        "public_benchmark_features": list(PUBLIC_BENCHMARK_FEATURES),
        "primary_public_inspired_features": list(PRIMARY_PUBLIC_INSPIRED_FEATURES),
        "playtype_feature_names": playtype_feature_names,
        "defensive_tracking_feature_names": defensive_tracking_feature_names,
        "assist_quality_feature_names": assist_quality_feature_names,
        "matchup_defense_feature_names": matchup_defense_feature_names,
        "player_skill_feature_names": player_skill_feature_names,
        "behavior_role_feature_names": behavior_role_feature_names,
        "side_role_feature_names": side_role_feature_names,
        "defense_role_interaction_feature_names": defense_role_interaction_feature_names,
        "public_benchmark_provenance": {
            "source": "https://craftednba.com/glossary",
            "box_creation_and_offensive_load": "Ben Taylor public formulas as reproduced by CraftedNBA",
            "primary_variants": "possession-shrunk within each player window; derived z-score components clipped to [-4, 4]",
            "behavioral_passer_score_v1": "CraftedNBA-inspired; excludes height and positional standardization; uses shrunk rates",
            "crafted_spacing_proxy_v1": "CraftedNBA formula with FG3A per 100 and pooled window league eFG; unit choice made explicit by this project",
        },
        "new_feature_names": new_features,
        "features_path": str(features_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "artifact_path": str(output.resolve()),
    }
    if run["quality"]["infinite_values"]:
        raise ValueError("V2 features contain infinite values.")
    if run["quality"]["bounded_feature_violations"]:
        raise ValueError("V2 features contain bounded-feature violations.")
    write_json_atomic(run, output / "run.json")
    return run
