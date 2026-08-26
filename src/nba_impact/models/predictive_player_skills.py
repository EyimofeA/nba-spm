"""Chronologically selected empirical-Bayes estimates of current player skills.

The module keeps skill and impact estimands separate.  It selects stabilization
parameters with future-season observations through 2024, freezes them, and then
updates the displayed 2026 estimates with observations through 2026.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


Family = Literal["binomial", "rate", "continuous"]


@dataclass(frozen=True)
class SkillSpec:
    key: str
    label: str
    group: str
    family: Family
    unit: str
    scale: float = 1.0
    numerator: str | None = None
    denominator: str | None = None
    value_column: str | None = None
    exposure_column: str | None = None
    higher_is_better: bool = True
    source_family: str = "gabriel_player_sheets"
    definition: str = ""


SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec("free_throw_pct", "Free throws", "shooting", "binomial", "percent", 100, "FTM", "FTA", definition="Free throws made divided by attempts."),
    SkillSpec("rim_pct", "Rim finishing", "shooting", "binomial", "percent", 100, "AtRimFGM", "AtRimFGA", definition="Made field goals at the rim divided by attempts at the rim."),
    SkillSpec("short_mid_pct", "Short midrange", "shooting", "binomial", "percent", 100, "ShortMidRangeFGM", "ShortMidRangeFGA", definition="Short-midrange makes divided by attempts."),
    SkillSpec("long_mid_pct", "Long midrange", "shooting", "binomial", "percent", 100, "LongMidRangeFGM", "LongMidRangeFGA", definition="Long-midrange makes divided by attempts."),
    SkillSpec("corner_three_pct", "Corner three", "shooting", "binomial", "percent", 100, "Corner3FGM", "Corner3FGA", definition="Corner-three makes divided by attempts."),
    SkillSpec("above_break_three_pct", "Above-break three", "shooting", "binomial", "percent", 100, "Arc3FGM", "Arc3FGA", definition="Above-break three-point makes divided by attempts."),
    SkillSpec("three_point_pct", "Three-point shooting", "shooting", "binomial", "percent", 100, "FG3M", "FG3A", definition="All three-point makes divided by attempts."),
    SkillSpec("catch_shoot_three_pct", "Catch-and-shoot three", "shooting", "binomial", "percent", 100, "CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A", definition="Catch-and-shoot three-point makes divided by attempts."),
    SkillSpec("pull_up_three_pct", "Pull-up three", "shooting", "binomial", "percent", 100, "PULL_UP_FG3M", "PULL_UP_FG3A", definition="Pull-up three-point makes divided by attempts."),
    SkillSpec("shot_quality", "Shot quality", "shooting", "continuous", "points_per_shot", value_column="shot_difficulty_expected_points_per_attempt", exposure_column="FieldGoalAttempts", source_family="player_skill_tracking", definition="Expected points per attempt from defender-distance and two-versus-three-point shot mix."),
    SkillSpec("shotmaking_above_expected", "Shotmaking", "shooting", "continuous", "points_per_100", value_column="shot_making_points_above_expected_p100_eb", exposure_column="OffPoss", source_family="player_skill_tracking", definition="Realized points above the player-neutral shot expectation per 100 offensive possessions."),
    SkillSpec("zts", "zTS", "shooting", "continuous", "percentage_points", value_column="zts_pct_points", exposure_column="synergy_possessions", source_family="playtype_impact", definition="True-shooting percentage minus the expectation from the player's playtype mix."),
    SkillSpec("assist_creation", "Assist creation", "creation", "rate", "per_100", 100, "AST_PTS_CREATED", "OffPoss", definition="Points created by assists per 100 offensive possessions."),
    SkillSpec("potential_assists", "Potential assists", "creation", "rate", "per_100", 100, "POTENTIAL_AST", "OffPoss", definition="Potential assists per 100 offensive possessions."),
    SkillSpec("rim_assists", "Rim assists", "creation", "rate", "per_100", 100, "AtRimAssists", "OffPoss", definition="Assists on rim attempts per 100 offensive possessions."),
    SkillSpec("three_point_assists", "Three-point assists", "creation", "rate", "per_100", 100, "ThreePointAssists", "OffPoss", definition="Assists on corner and above-break threes per 100 offensive possessions."),
    SkillSpec("passing_efficiency", "Passing efficiency", "creation", "rate", "points_per_opportunity", 1, "AST_PTS_CREATED", "POTENTIAL_AST", definition="Assist points created per potential assist."),
    SkillSpec("turnover_rate", "Ball security", "creation", "rate", "percent", 100, "TOV", "UsageEvents", higher_is_better=False, definition="Turnovers divided by shooting, free-throw and turnover usage events."),
    SkillSpec("live_ball_turnover_rate", "Live-ball security", "creation", "rate", "percent", 100, "LiveBallTurnovers", "UsageEvents", higher_is_better=False, definition="Live-ball turnovers divided by shooting, free-throw and turnover usage events."),
    SkillSpec("drive_creation", "Drive creation", "creation", "rate", "per_100_drives", 100, "DRIVE_AST", "DRIVES", definition="Assists created per 100 drives."),
    SkillSpec("rim_pressure", "Rim pressure", "creation", "rate", "per_100", 100, "AtRimFGA", "OffPoss", definition="Rim attempts per 100 offensive possessions."),
    SkillSpec("free_throw_pressure", "Free-throw pressure", "creation", "rate", "per_100", 100, "FTA", "OffPoss", definition="Free-throw attempts per 100 offensive possessions."),
    SkillSpec("offensive_load", "Offensive load", "creation", "continuous", "per_100", value_column="offensive_load_2017_p100", exposure_column="OffPoss", source_family="statistical_features", definition="The public Offensive Load composite in the statistical feature panel."),
    SkillSpec("offensive_rebound_rate", "Offensive rebounding", "rebounding", "rate", "percent", 100, "OREB", "OREB_CHANCES", definition="Offensive rebounds per recorded offensive rebound chance; scored as a rate because source counts are not bounded binomial trials."),
    SkillSpec("defensive_rebound_rate", "Defensive rebounding", "rebounding", "rate", "percent", 100, "DREB", "DREB_CHANCES", definition="Defensive rebounds per recorded defensive rebound chance; scored as a rate because source counts are not bounded binomial trials."),
    SkillSpec("contested_rebound_conversion", "Contested rebounding", "rebounding", "rate", "percent", 100, "REB_CONTEST", "REB_CHANCES", definition="Contested rebounds secured per recorded rebound chance; an available-data proxy for contested conversion."),
    SkillSpec("rim_deterrence", "Rim deterrence", "defense", "continuous", "attempts_suppressed_per_100", value_column="rim_deterrence", exposure_column="DefPoss", source_family="defensive_tracking", definition="Season-average rim attempts faced per 100 minus the player's observed rate; positive means fewer rim attempts faced."),
    SkillSpec("rim_points_saved", "Rim points saved", "defense", "continuous", "points_per_100", value_column="rim_points_saved_p100", exposure_column="DefPoss", source_family="defensive_tracking", definition="Estimated rim points saved per 100 defensive possessions from observed rim shot defense."),
    SkillSpec("non_rim_shot_suppression", "Non-rim suppression", "defense", "continuous", "attempts_suppressed_per_100", value_column="non_rim_shot_suppression", exposure_column="DefPoss", source_family="defensive_tracking", definition="Season-average non-rim attempts faced per 100 minus the player's observed rate; an observational suppression proxy."),
    SkillSpec("matchup_adjusted_points_saved", "Matchup points saved", "defense", "continuous", "points_per_100_matchups", value_column="matchup_opponent_adjusted_points_saved_p100_eb", exposure_column="matchup_possessions", source_family="matchup_defense", definition="Opponent-adjusted points saved per 100 assigned matchup possessions, already source-level EB shrunk."),
    SkillSpec("turnovers_forced", "Turnovers forced", "defense", "continuous", "per_100_matchups", value_column="matchup_turnovers_forced_vs_scorer_p100_eb", exposure_column="matchup_possessions", source_family="matchup_defense", definition="Turnovers forced versus scorer expectation per 100 assigned matchup possessions."),
    SkillSpec("foul_discipline", "Foul discipline", "defense", "continuous", "per_100_matchups", value_column="matchup_shooting_fouls_prevented_vs_scorer_p100_eb", exposure_column="matchup_possessions", source_family="matchup_defense", definition="Shooting fouls prevented versus scorer expectation per 100 assigned matchup possessions."),
    SkillSpec("deflections", "Deflections", "defense", "continuous", "per_100", value_column="deflections_p100", exposure_column="DefPoss", source_family="defensive_tracking", definition="Deflections per 100 defensive possessions."),
    SkillSpec("recovered_blocks", "Recovered blocks", "defense", "rate", "per_100", 100, "RecoveredBlocks", "DefPoss", definition="Blocks recovered by the defense per 100 defensive possessions."),
)


def skill_definitions() -> pd.DataFrame:
    """Return the machine-readable skill dictionary."""
    return pd.DataFrame(asdict(spec) for spec in SKILL_SPECS)


def load_player_skill_panel(
    player_sheet_dir: str | Path,
    *,
    seasons: range,
    player_skill_path: str | Path,
    playtype_path: str | Path,
    statistical_path: str | Path,
    tracking_path: str | Path,
    tracking_dfg_observations_path: str | Path,
    tracking_rim_observations_path: str | Path,
    tracking_hustle_observations_path: str | Path,
    matchup_path: str | Path,
) -> pd.DataFrame:
    """Load the annual numerators, denominators, identity, and derived sources."""
    required = {
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "OffPoss", "DefPoss",
        "FTM", "FTA", "AtRimFGM", "AtRimFGA", "ShortMidRangeFGM", "ShortMidRangeFGA",
        "LongMidRangeFGM", "LongMidRangeFGA", "Corner3FGM", "Corner3FGA", "Arc3FGM",
        "Arc3FGA", "FG2A", "FG3M", "FG3A", "CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A",
        "PULL_UP_FG3M", "PULL_UP_FG3A", "AST_PTS_CREATED", "POTENTIAL_AST",
        "AtRimAssists", "Corner3Assists", "Arc3Assists", "TOV", "LiveBallTurnovers",
        "DRIVES", "DRIVE_AST", "OREB", "DREB", "OREB_CHANCES", "DREB_CHANCES",
        "REB_CONTEST", "REB_CHANCES", "RecoveredBlocks",
    }
    rows: list[pd.DataFrame] = []
    for season in seasons:
        path = Path(player_sheet_dir) / f"{season}.csv"
        header = set(pd.read_csv(path, nrows=0).columns)
        if missing := required - header:
            raise ValueError(f"Player sheet {season} is missing {sorted(missing)}")
        frame = pd.read_csv(path, usecols=sorted(required), low_memory=False).drop_duplicates()
        frame["Season"] = int(season)
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["PLAYER_ID"] = pd.to_numeric(panel["PLAYER_ID"], errors="raise").astype(int)
    panel["Season"] = panel["Season"].astype(int)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Player sheets contain duplicate player-season rows.")
    numeric = required - {"PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"}
    panel[list(numeric)] = panel[list(numeric)].apply(pd.to_numeric, errors="coerce")
    panel["ThreePointAssists"] = panel["Corner3Assists"].fillna(0) + panel["Arc3Assists"].fillna(0)
    panel["FieldGoalAttempts"] = panel["FG2A"].fillna(0) + panel["FG3A"].fillna(0)
    panel["UsageEvents"] = (
        panel["FG2A"].fillna(0) + panel["FG3A"].fillna(0)
        + 0.44 * panel["FTA"].fillna(0) + panel["TOV"].fillna(0)
    )

    merge_sources = [
        (player_skill_path, ["shot_difficulty_expected_points_per_attempt", "shot_making_points_above_expected_p100_eb"]),
        (playtype_path, ["zts_pct_points", "synergy_possessions"]),
        (statistical_path, ["offensive_load_2017_p100"]),
        (tracking_path, ["dfg_attempts_p100", "rim_dfga_p100", "rim_points_saved_p100", "deflections_p100"]),
        (matchup_path, ["matchup_possessions", "matchup_opponent_adjusted_points_saved_p100_eb", "matchup_turnovers_forced_vs_scorer_p100_eb", "matchup_shooting_fouls_prevented_vs_scorer_p100_eb"]),
    ]
    for path, columns in merge_sources:
        available = pd.read_parquet(path, columns=None)
        season_column = "Season" if "Season" in available else "Window_End"
        source = available[["PLAYER_ID", season_column, *columns]].rename(columns={season_column: "Season"})
        source[["PLAYER_ID", "Season"]] = source[["PLAYER_ID", "Season"]].astype(int)
        if source.duplicated(["PLAYER_ID", "Season"]).any():
            raise ValueError(f"Derived skill source has duplicate keys: {path}")
        panel = panel.merge(source, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one")

    observation_sources = {
        "has_dfg_observation": tracking_dfg_observations_path,
        "has_rim_observation": tracking_rim_observations_path,
        "has_hustle_observation": tracking_hustle_observations_path,
    }
    for indicator, path in observation_sources.items():
        observed = pd.read_csv(path, usecols=["PLAYER_ID", "year"], low_memory=False).rename(columns={"year": "Season"})
        observed[["PLAYER_ID", "Season"]] = observed[["PLAYER_ID", "Season"]].apply(pd.to_numeric, errors="coerce")
        observed = observed.dropna().astype({"PLAYER_ID": int, "Season": int}).drop_duplicates()
        observed[indicator] = True
        panel = panel.merge(observed, on=["PLAYER_ID", "Season"], how="left", validate="many_to_one")
        panel[indicator] = panel[indicator].fillna(False).astype(bool)

    # Convert observed attempt volume into season-relative suppression proxies.
    defense_weight = panel["DefPoss"].where(panel["DefPoss"].gt(0))

    def weighted_season_center(values: pd.Series) -> pd.Series:
        valid = values.notna() & defense_weight.notna()
        weighted = (values * defense_weight).where(valid)
        weight = defense_weight.where(valid)
        totals = pd.DataFrame(
            {"Season": panel["Season"], "weighted": weighted, "weight": weight}
        ).groupby("Season", sort=False).sum(min_count=1)
        return totals["weighted"] / totals["weight"].where(totals["weight"].gt(0))

    rim_center = weighted_season_center(panel["rim_dfga_p100"])
    panel["rim_deterrence"] = panel["Season"].map(rim_center) - panel["rim_dfga_p100"]
    panel["non_rim_attempts_p100"] = panel["dfg_attempts_p100"] - panel["rim_dfga_p100"]
    non_rim_center = weighted_season_center(panel["non_rim_attempts_p100"])
    panel["non_rim_shot_suppression"] = panel["Season"].map(non_rim_center) - panel["non_rim_attempts_p100"]
    return panel.sort_values(["Season", "PLAYER_ID"], kind="stable").reset_index(drop=True)


def _skill_frame(panel: pd.DataFrame, spec: SkillSpec) -> pd.DataFrame:
    columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season", "AGE"]
    frame = panel[columns].copy()
    if spec.family in {"binomial", "rate"}:
        assert spec.numerator and spec.denominator
        frame["numerator"] = pd.to_numeric(panel[spec.numerator], errors="coerce")
        frame["opportunities"] = pd.to_numeric(panel[spec.denominator], errors="coerce")
        frame["raw_value"] = spec.scale * frame["numerator"] / frame["opportunities"].where(frame["opportunities"].gt(0))
    else:
        assert spec.value_column and spec.exposure_column
        frame["raw_value"] = pd.to_numeric(panel[spec.value_column], errors="coerce")
        frame["opportunities"] = pd.to_numeric(panel[spec.exposure_column], errors="coerce")
        frame["numerator"] = frame["raw_value"] * frame["opportunities"] / spec.scale
    observed = pd.Series(True, index=frame.index)
    if spec.key in {"rim_deterrence", "rim_points_saved"}:
        observed = panel["has_rim_observation"]
    elif spec.key == "non_rim_shot_suppression":
        observed = panel["has_dfg_observation"] & panel["has_rim_observation"]
    elif spec.key == "deflections":
        observed = panel["has_hustle_observation"]
    frame.loc[~observed, ["numerator", "opportunities", "raw_value"]] = np.nan
    frame.loc[~frame["opportunities"].gt(0), ["numerator", "opportunities", "raw_value"]] = np.nan
    return frame


def _history_prediction(
    frame: pd.DataFrame,
    *,
    target_season: int,
    prior_strength: float,
    half_life: float | None,
    minimum_exposure: float,
    family: Family,
    scale: float,
    include_target: bool = False,
) -> tuple[pd.DataFrame, float]:
    cutoff = target_season if include_target else target_season - 1
    history = frame.loc[frame["Season"].le(cutoff)].dropna(subset=["raw_value", "opportunities"]).copy()
    if history.empty:
        raise ValueError(f"No skill history exists through {cutoff}.")
    if half_life is None:
        history["decay_weight"] = 1.0
    else:
        history["decay_weight"] = np.power(2.0, (history["Season"] - cutoff) / half_life)
    history["weighted_opportunities"] = history["opportunities"] * history["decay_weight"]
    if family in {"binomial", "rate"}:
        history["weighted_numerator"] = history["numerator"] * history["decay_weight"]
        total_opportunities = float(history["weighted_opportunities"].sum())
        center = float(history["weighted_numerator"].sum() / total_opportunities)
        grouped = history.groupby("PLAYER_ID", as_index=False).agg(
            weighted_numerator=("weighted_numerator", "sum"),
            effective_exposure=("weighted_opportunities", "sum"),
        )
        grouped["estimate"] = scale * (
            grouped["weighted_numerator"] + prior_strength * center
        ) / (grouped["effective_exposure"] + prior_strength)
    else:
        center = float(np.average(history["raw_value"], weights=history["weighted_opportunities"]))
        history["weighted_value"] = history["raw_value"] * history["weighted_opportunities"]
        grouped = history.groupby("PLAYER_ID", as_index=False).agg(
            weighted_value=("weighted_value", "sum"),
            effective_exposure=("weighted_opportunities", "sum"),
        )
        grouped["estimate"] = (
            grouped["weighted_value"] + prior_strength * center
        ) / (grouped["effective_exposure"] + prior_strength)
    grouped.loc[grouped["effective_exposure"].lt(minimum_exposure), "estimate"] = scale * center if family != "continuous" else center
    return grouped, scale * center if family != "continuous" else center


def _raw_previous_prediction(frame: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, float]:
    history = frame.loc[frame["Season"].lt(target_season)].dropna(subset=["raw_value", "opportunities"])
    previous = history.loc[history["Season"].eq(target_season - 1), ["PLAYER_ID", "raw_value", "opportunities"]].rename(
        columns={"raw_value": "estimate", "opportunities": "effective_exposure"}
    )
    same = history.loc[history["Season"].eq(target_season - 1)]
    center = float(np.average(same["raw_value"], weights=same["opportunities"])) if not same.empty else float(np.average(history["raw_value"], weights=history["opportunities"]))
    return previous, center


def _score_prediction(target: pd.DataFrame, prediction: np.ndarray, spec: SkillSpec) -> dict[str, float]:
    valid = target["opportunities"].gt(0) & target["raw_value"].notna() & np.isfinite(prediction)
    selected = target.loc[valid]
    pred = np.asarray(prediction)[valid.to_numpy()]
    weight = selected["opportunities"].to_numpy(dtype=float)
    actual = selected["raw_value"].to_numpy(dtype=float)
    if not len(selected):
        return {"rows": 0, "opportunities": 0.0, "primary": np.nan, "secondary": np.nan}
    if spec.family == "binomial":
        p = np.clip(pred / spec.scale, 1e-6, 1 - 1e-6)
        makes = selected["numerator"].to_numpy(dtype=float)
        attempts = weight
        misses = attempts - makes
        log_loss = -float((makes * np.log(p) + misses * np.log1p(-p)).sum() / attempts.sum())
        brier = float((makes * (1 - p) ** 2 + misses * p**2).sum() / attempts.sum())
        return {"rows": len(selected), "opportunities": float(weight.sum()), "primary": log_loss, "secondary": brier}
    rmse = float(np.sqrt(np.average((actual - pred) ** 2, weights=weight)))
    mae = float(np.average(np.abs(actual - pred), weights=weight))
    return {"rows": len(selected), "opportunities": float(weight.sum()), "primary": rmse, "secondary": mae}


def _age_features(age: pd.Series) -> np.ndarray:
    centered = pd.to_numeric(age, errors="coerce").fillna(27.0).to_numpy(dtype=float) - 27.0
    return np.column_stack([centered, centered**2])


def _age_adjusted_prediction(
    frame: pd.DataFrame,
    spec: SkillSpec,
    *,
    test_season: int,
    prior_strength: float,
    half_life: float | None,
    minimum_exposure: float,
    alpha: float,
) -> tuple[pd.DataFrame, float]:
    training_pairs: list[pd.DataFrame] = []
    first = max(int(frame["Season"].min()) + 2, 2016)
    for season in range(first, test_season):
        try:
            base, center = _history_prediction(
                frame, target_season=season, prior_strength=prior_strength,
                half_life=half_life, minimum_exposure=minimum_exposure,
                family=spec.family, scale=spec.scale,
            )
        except ValueError:
            continue
        target = frame.loc[frame["Season"].eq(season)].merge(base[["PLAYER_ID", "estimate"]], on="PLAYER_ID", how="left")
        target["estimate"] = target["estimate"].fillna(center)
        target = target.dropna(subset=["raw_value", "opportunities", "AGE"])
        if spec.family == "binomial":
            observed = np.clip(target["raw_value"].to_numpy() / spec.scale, 1e-5, 1 - 1e-5)
            base_p = np.clip(target["estimate"].to_numpy() / spec.scale, 1e-5, 1 - 1e-5)
            target["residual"] = np.log(observed / (1 - observed)) - np.log(base_p / (1 - base_p))
        else:
            target["residual"] = target["raw_value"] - target["estimate"]
        training_pairs.append(target)
    if not training_pairs:
        return _history_prediction(
            frame, target_season=test_season, prior_strength=prior_strength,
            half_life=half_life, minimum_exposure=minimum_exposure,
            family=spec.family, scale=spec.scale,
        )
    training = pd.concat(training_pairs, ignore_index=True)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(_age_features(training["AGE"]), training["residual"], sample_weight=np.sqrt(training["opportunities"].clip(lower=1)))
    base, center = _history_prediction(
        frame, target_season=test_season, prior_strength=prior_strength,
        half_life=half_life, minimum_exposure=minimum_exposure,
        family=spec.family, scale=spec.scale,
    )
    target_ages = frame.loc[frame["Season"].eq(test_season), ["PLAYER_ID", "AGE"]].drop_duplicates("PLAYER_ID")
    output = target_ages.merge(base, on="PLAYER_ID", how="left")
    output["estimate"] = output["estimate"].fillna(center)
    adjustment = model.predict(_age_features(output["AGE"]))
    if spec.family == "binomial":
        p = np.clip(output["estimate"].to_numpy() / spec.scale, 1e-5, 1 - 1e-5)
        logit = np.log(p / (1 - p)) + adjustment
        output["estimate"] = spec.scale / (1 + np.exp(-logit))
    else:
        output["estimate"] = output["estimate"] + adjustment
    return output.drop(columns="AGE"), center


def tune_skill(
    frame: pd.DataFrame,
    spec: SkillSpec,
    *,
    selection_seasons: tuple[int, ...],
    prior_grid: tuple[float, ...],
    half_life_grid: tuple[float, ...],
    minimum_exposure_grid: tuple[float, ...],
    age_alpha_grid: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw, career EB, decayed EB, age, and the role support gate."""
    rows: list[dict] = []

    def evaluate(arm: str, prior: float, half_life: float | None, minimum: float, age_alpha: float | None = None) -> None:
        for test_season in selection_seasons:
            target = frame.loc[frame["Season"].eq(test_season)].copy()
            if arm == "raw_previous_season":
                predictions, center = _raw_previous_prediction(frame, test_season)
            elif arm == "time_decayed_eb_plus_age":
                assert age_alpha is not None
                predictions, center = _age_adjusted_prediction(
                    frame, spec, test_season=test_season, prior_strength=prior,
                    half_life=half_life, minimum_exposure=minimum, alpha=age_alpha,
                )
            else:
                predictions, center = _history_prediction(
                    frame, target_season=test_season, prior_strength=prior,
                    half_life=half_life, minimum_exposure=minimum,
                    family=spec.family, scale=spec.scale,
                )
            scored = target.merge(predictions[["PLAYER_ID", "estimate"]], on="PLAYER_ID", how="left")
            scored["estimate"] = scored["estimate"].fillna(center)
            metric = _score_prediction(scored, scored["estimate"].to_numpy(), spec)
            rows.append({
                "skill": spec.key, "arm": arm, "prior_strength": float(prior),
                "half_life_years": "none" if half_life is None else f"{half_life:g}",
                "minimum_exposure": float(minimum), "age_alpha": age_alpha,
                "test_season": int(test_season), **metric,
            })

    evaluate("raw_previous_season", 0.0, None, 0.0)
    for prior in prior_grid:
        for minimum in minimum_exposure_grid:
            evaluate("career_eb", prior, None, minimum)
            for half_life in half_life_grid:
                evaluate("time_decayed_eb", prior, half_life, minimum)

    folds = pd.DataFrame(rows)
    summary_columns = ["skill", "arm", "prior_strength", "half_life_years", "minimum_exposure", "age_alpha"]
    summary = folds.groupby(summary_columns, as_index=False, dropna=False).agg(
        primary=("primary", "mean"), secondary=("secondary", "mean"),
        folds=("test_season", "nunique"), rows=("rows", "sum"), opportunities=("opportunities", "sum"),
    )
    base = summary.loc[summary["arm"].eq("time_decayed_eb")].sort_values(
        ["primary", "secondary", "prior_strength", "minimum_exposure", "half_life_years"], kind="stable"
    ).iloc[0]
    selected_half = float(base["half_life_years"])
    for alpha in age_alpha_grid:
        evaluate(
            "time_decayed_eb_plus_age", float(base["prior_strength"]), selected_half,
            float(base["minimum_exposure"]), age_alpha=float(alpha),
        )
    folds = pd.DataFrame(rows)
    summary = folds.groupby(summary_columns, as_index=False, dropna=False).agg(
        primary=("primary", "mean"), secondary=("secondary", "mean"),
        folds=("test_season", "nunique"), rows=("rows", "sum"), opportunities=("opportunities", "sum"),
    )
    summary = summary.sort_values(
        ["primary", "secondary", "prior_strength", "minimum_exposure", "half_life_years"], kind="stable"
    ).reset_index(drop=True)
    summary["selected"] = False
    selected_index = int(summary.index[0])
    summary["age_fold_wins"] = np.nan
    if summary.loc[selected_index, "arm"] == "time_decayed_eb_plus_age":
        age_row = summary.loc[selected_index]
        base_match = summary.loc[
            summary["arm"].eq("time_decayed_eb")
            & summary["prior_strength"].eq(age_row["prior_strength"])
            & summary["half_life_years"].eq(age_row["half_life_years"])
            & summary["minimum_exposure"].eq(age_row["minimum_exposure"])
        ]
        age_folds = folds.loc[
            folds["arm"].eq("time_decayed_eb_plus_age")
            & folds["prior_strength"].eq(age_row["prior_strength"])
            & folds["half_life_years"].eq(age_row["half_life_years"])
            & folds["minimum_exposure"].eq(age_row["minimum_exposure"])
            & folds["age_alpha"].eq(age_row["age_alpha"]),
            ["test_season", "primary"],
        ]
        base_folds = folds.loc[
            folds["arm"].eq("time_decayed_eb")
            & folds["prior_strength"].eq(age_row["prior_strength"])
            & folds["half_life_years"].eq(age_row["half_life_years"])
            & folds["minimum_exposure"].eq(age_row["minimum_exposure"]),
            ["test_season", "primary"],
        ]
        paired = age_folds.merge(base_folds, on="test_season", suffixes=("_age", "_base"))
        wins = int((paired["primary_age"] < paired["primary_base"]).sum())
        summary.loc[selected_index, "age_fold_wins"] = wins
        # An age correction must help in at least four of six future-season
        # folds.  This prevents a tiny pooled win from selecting an unstable
        # age curve after testing three penalties.
        if wins < 4 and not base_match.empty:
            selected_index = int(base_match.index[0])
    summary.loc[selected_index, "selected"] = True
    # The current descriptive role maps do not provide a pre-season role label
    # across every development fold, so a role-conditioned arm would leak or
    # change sample composition.  Record the support decision instead of fitting it.
    skipped = pd.DataFrame([{
        "skill": spec.key, "arm": "role_conditional", "prior_strength": np.nan,
        "half_life_years": "n/a", "minimum_exposure": np.nan, "age_alpha": np.nan,
        "primary": np.nan, "secondary": np.nan, "folds": 0, "rows": 0,
        "opportunities": 0.0, "selected": False, "age_fold_wins": np.nan, "status": "skipped",
        "reason": "No frozen pre-season role label with consistent coverage across all 2019-24 folds.",
    }])
    summary["status"] = "scored"
    summary["reason"] = None
    return folds, pd.concat([summary, skipped], ignore_index=True)


def _posterior_for_season(
    frame: pd.DataFrame,
    spec: SkillSpec,
    season: int,
    selected: pd.Series,
) -> tuple[pd.DataFrame, float]:
    """Apply the selected preseason estimator, then update with named-season data."""
    arm = str(selected["arm"])
    if arm == "raw_previous_season":
        current = frame.loc[
            frame["Season"].eq(season),
            ["PLAYER_ID", "raw_value", "opportunities"],
        ].rename(
            columns={"raw_value": "estimate", "opportunities": "effective_exposure"}
        )
        previous, _ = _raw_previous_prediction(frame, season)
        previous = previous.rename(
            columns={
                "estimate": "preseason_estimate",
                "effective_exposure": "preseason_effective_exposure",
            }
        )
        current = current.merge(previous, on="PLAYER_ID", how="left")
        current["preseason_precision"] = current[
            "preseason_effective_exposure"
        ].fillna(0.0)
        center_frame = frame.loc[frame["Season"].eq(season)].dropna(subset=["raw_value", "opportunities"])
        positive = center_frame["opportunities"].gt(0)
        center = (
            float(np.average(center_frame.loc[positive, "raw_value"], weights=center_frame.loc[positive, "opportunities"]))
            if positive.any()
            else np.nan
        )
        return current, center
    half_label = str(selected["half_life_years"])
    half = None if half_label == "none" else float(half_label)
    prior_strength = float(selected["prior_strength"])
    minimum_exposure = float(selected["minimum_exposure"])
    try:
        if arm == "time_decayed_eb_plus_age":
            preseason, center = _age_adjusted_prediction(
                frame,
                spec,
                test_season=season,
                prior_strength=prior_strength,
                half_life=half,
                minimum_exposure=minimum_exposure,
                alpha=float(selected["age_alpha"]),
            )
        else:
            preseason, center = _history_prediction(
                frame,
                target_season=season,
                prior_strength=prior_strength,
                half_life=half,
                minimum_exposure=minimum_exposure,
                family=spec.family,
                scale=spec.scale,
            )
    except ValueError:
        preseason = pd.DataFrame(
            columns=["PLAYER_ID", "estimate", "effective_exposure"]
        )
        center_frame = frame.loc[frame["Season"].eq(season)].dropna(
            subset=["raw_value", "opportunities"]
        )
        positive = center_frame["opportunities"].gt(0)
        center = (
            float(
                np.average(
                    center_frame.loc[positive, "raw_value"],
                    weights=center_frame.loc[positive, "opportunities"],
                )
            )
            if positive.any()
            else np.nan
        )

    target = frame.loc[
        frame["Season"].eq(season),
        ["PLAYER_ID", "raw_value", "opportunities"],
    ].copy()
    preseason = preseason.rename(
        columns={
            "estimate": "preseason_estimate",
            "effective_exposure": "preseason_effective_exposure",
        }
    )
    output = target.merge(preseason, on="PLAYER_ID", how="left")
    has_history = output["preseason_estimate"].notna()
    has_current = output["raw_value"].notna() & output["opportunities"].gt(0)
    output.loc[has_current, "preseason_estimate"] = output.loc[
        has_current, "preseason_estimate"
    ].fillna(center)
    history_exposure = output["preseason_effective_exposure"].fillna(0.0)
    history_exposure = history_exposure.where(
        history_exposure.ge(minimum_exposure), 0.0
    )
    output["preseason_effective_exposure"] = history_exposure
    output["preseason_precision"] = prior_strength + history_exposure
    current_exposure = output["opportunities"].where(has_current, 0.0).fillna(0.0)
    numerator = (
        output["preseason_estimate"] * output["preseason_precision"]
        + output["raw_value"].fillna(0.0) * current_exposure
    )
    denominator = output["preseason_precision"] + current_exposure
    output["estimate"] = numerator / denominator.where(denominator.gt(0))
    output.loc[~(has_history | has_current), "estimate"] = np.nan
    output["effective_exposure"] = history_exposure + current_exposure
    return output[
        [
            "PLAYER_ID",
            "estimate",
            "effective_exposure",
            "preseason_estimate",
            "preseason_effective_exposure",
            "preseason_precision",
        ]
    ], center


def build_skill_estimates(
    panel: pd.DataFrame,
    *,
    selection_seasons: tuple[int, ...],
    prior_grid: tuple[float, ...],
    half_life_grid: tuple[float, ...],
    minimum_exposure_grid: tuple[float, ...],
    age_alpha_grid: tuple[float, ...],
    output_seasons: tuple[int, ...],
    last_update: pd.DataFrame | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune every skill and return estimates, fold metrics, and decisions."""
    estimate_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    checkpoints = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoints is not None:
        checkpoints.mkdir(parents=True, exist_ok=True)
    for spec in SKILL_SPECS:
        frame = _skill_frame(panel, spec)
        fold_checkpoint = checkpoints / f"{spec.key}.folds.parquet" if checkpoints else None
        decision_checkpoint = checkpoints / f"{spec.key}.selection.parquet" if checkpoints else None
        estimate_checkpoint = checkpoints / f"{spec.key}.estimates.parquet" if checkpoints else None
        if fold_checkpoint and fold_checkpoint.exists() and decision_checkpoint and decision_checkpoint.exists():
            folds = pd.read_parquet(fold_checkpoint)
            decisions = pd.read_parquet(decision_checkpoint)
        else:
            folds, decisions = tune_skill(
                frame, spec, selection_seasons=selection_seasons,
                prior_grid=prior_grid, half_life_grid=half_life_grid,
                minimum_exposure_grid=minimum_exposure_grid, age_alpha_grid=age_alpha_grid,
            )
            if fold_checkpoint and decision_checkpoint:
                folds.to_parquet(fold_checkpoint, index=False)
                decisions.to_parquet(decision_checkpoint, index=False)
        selected = decisions.loc[decisions["selected"]].iloc[0]
        fold_rows.append(folds)
        decision_rows.append(decisions)
        if estimate_checkpoint and estimate_checkpoint.exists():
            skill_estimates = pd.read_parquet(estimate_checkpoint)
        else:
            season_rows: list[pd.DataFrame] = []
            for season in output_seasons:
                posterior, center = _posterior_for_season(frame, spec, season, selected)
                observed = frame.loc[frame["Season"].eq(season)].copy()
                result = observed.merge(
                    posterior[
                        [
                            "PLAYER_ID",
                            "estimate",
                            "effective_exposure",
                            "preseason_estimate",
                            "preseason_effective_exposure",
                            "preseason_precision",
                        ]
                    ],
                    on="PLAYER_ID",
                    how="left",
                )
                result["effective_exposure"] = result["effective_exposure"].fillna(0.0)
                result["skill"] = spec.key
                result["label"] = spec.label
                result["group"] = spec.group
                result["unit"] = spec.unit
                result["higher_is_better"] = spec.higher_is_better
                result["model_arm"] = selected["arm"]
                result["half_life_years"] = selected["half_life_years"]
                result["prior_strength"] = selected["prior_strength"]
                result["minimum_exposure"] = selected["minimum_exposure"]
                result["age_alpha"] = selected["age_alpha"]
                season_rows.append(result)
            skill_estimates = pd.concat(season_rows, ignore_index=True)
            if estimate_checkpoint:
                skill_estimates.to_parquet(estimate_checkpoint, index=False)
        estimate_rows.append(skill_estimates)
    estimates = pd.concat(estimate_rows, ignore_index=True)
    direction = np.where(estimates["higher_is_better"], 1.0, -1.0)
    estimates["percentile"] = estimates.assign(_rank=estimates["estimate"] * direction).groupby(
        ["Season", "skill"], sort=False
    )["_rank"].rank(method="average", pct=True) * 100
    estimates = estimates.sort_values(["PLAYER_ID", "skill", "Season"], kind="stable")
    estimates["year_over_year_change"] = estimates.groupby(["PLAYER_ID", "skill"], sort=False)["estimate"].diff()
    if last_update is not None:
        updates = last_update.copy()
        updates[["PLAYER_ID", "Season"]] = updates[["PLAYER_ID", "Season"]].astype(int)
        estimates = estimates.merge(updates, on=["PLAYER_ID", "Season"], how="left", validate="many_to_one")
    if "last_update_date" not in estimates:
        estimates["last_update_date"] = pd.NaT
    fallback = pd.to_datetime(estimates["Season"].astype(str) + "-06-30")
    estimates["last_update_date"] = pd.to_datetime(estimates["last_update_date"], errors="coerce").fillna(fallback).dt.date.astype(str)
    estimates["standard_error"] = np.nan
    for spec in SKILL_SPECS:
        mask = estimates["skill"].eq(spec.key)
        if spec.family == "binomial":
            p = np.clip(pd.to_numeric(estimates.loc[mask, "estimate"], errors="coerce").to_numpy(dtype=float) / spec.scale, 1e-6, 1 - 1e-6)
            n = (
                pd.to_numeric(estimates.loc[mask, "effective_exposure"], errors="coerce").fillna(0).to_numpy(dtype=float)
                + pd.to_numeric(estimates.loc[mask, "prior_strength"], errors="coerce").fillna(0).to_numpy(dtype=float)
            )
            estimates.loc[mask, "standard_error"] = spec.scale * np.sqrt(p * (1 - p) / np.clip(n, 1, None))
    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season", "skill", "label", "group", "unit",
        "estimate", "raw_value", "opportunities", "percentile", "year_over_year_change",
        "model_arm", "half_life_years", "prior_strength", "minimum_exposure", "age_alpha",
        "effective_exposure", "preseason_estimate", "preseason_effective_exposure",
        "preseason_precision", "standard_error", "last_update_date", "higher_is_better",
    ]
    return estimates[keep].reset_index(drop=True), pd.concat(fold_rows, ignore_index=True), pd.concat(decision_rows, ignore_index=True)
