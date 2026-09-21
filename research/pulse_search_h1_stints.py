#!/usr/bin/env python3
"""H1 stint conversion and scoring helpers for the PULSE search pilots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.possessions import LINEUP_COLUMNS
from nba_impact.models.canonical_pulse import predict_next_season_games, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design, fit_stint_center_path
from research.pulse_search_h1_scores import V3_NESTED, WEB

def technical_free_throws(v3: pd.DataFrame) -> pd.DataFrame:
    work = v3.copy()
    work["game_id"] = work["gameId"].map(canonical_game_id)
    action = work["actionType"].astype(str).str.strip().str.casefold()
    subtype = work["subType"].astype(str) if "subType" in work else pd.Series("", index=work.index)
    description = work["description"].astype(str)
    made = action.eq("free throw")
    technical = subtype.str.contains("technical", case=False, na=False)
    missed = description.str.contains(r"^\s*MISS\b", case=False, na=False)
    selected = work.loc[made & technical & ~missed, ["game_id", "actionId", "teamId"]].copy()
    selected["event_order"] = pd.to_numeric(selected["actionId"], errors="raise").astype("int64")
    selected["team_id"] = pd.to_numeric(selected["teamId"], errors="coerce")
    selected["points"] = 1
    return selected[["game_id", "event_order", "team_id", "points"]]


def apply_technical_points(possessions: pd.DataFrame, assigned: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    mapped = assigned.merge(tech, on=["game_id", "event_order"], how="inner")
    output = possessions.copy()
    if mapped.empty:
        output["technical_points"] = 0
        output["points_excl_tech"] = output["points"]
        return output
    by_possession = mapped.groupby("possession_id", as_index=False)["points"].sum().rename(
        columns={"points": "technical_points"}
    )
    output = output.merge(by_possession, on="possession_id", how="left")
    output["technical_points"] = output["technical_points"].fillna(0).astype(int)
    output["points_excl_tech"] = output["points"] - output["technical_points"]
    if (output["points_excl_tech"] < 0).any():
        raise ValueError("Technical-FT exclusion produced negative possession points.")
    return output


def tag_segment_technicals(segments: pd.DataFrame, assigned: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    output = segments.copy()
    mapped = assigned.merge(tech, on=["game_id", "event_order"], how="inner")
    if mapped.empty:
        output["technical_points"] = 0
        output["points_excl_tech"] = output["points"]
        return output
    joined = mapped.merge(
        output[["possession_id", "segment_number", "start_order_number", "end_order_number"]],
        on="possession_id",
        how="left",
    )
    hit = (
        joined["event_order"].ge(joined["start_order_number"])
        & joined["event_order"].le(joined["end_order_number"])
    )
    by_segment = (
        joined.loc[hit]
        .groupby(["possession_id", "segment_number"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "technical_points"})
    )
    output = output.merge(by_segment, on=["possession_id", "segment_number"], how="left")
    output["technical_points"] = output["technical_points"].fillna(0).astype(int)
    output["points_excl_tech"] = output["points"] - output["technical_points"]
    if (output["points_excl_tech"] < 0).any():
        raise ValueError("Technical-FT exclusion produced negative segment points.")
    return output


def _attach_possession_meta(possessions: pd.DataFrame, segments: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    meta_columns = ["possession_id", "offense_is_home", "season_end", "game_id", *extra]
    meta = possessions.loc[:, [column for column in meta_columns if column in possessions.columns]]
    overlap = [column for column in meta.columns if column != "possession_id" and column in segments.columns]
    return segments.drop(columns=overlap).merge(meta, on="possession_id", how="inner")


def segments_to_stints(possessions: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    frame = _attach_possession_meta(possessions, segments)
    first = frame.groupby("possession_id", as_index=False)["segment_number"].min().rename(
        columns={"segment_number": "first_segment"}
    )
    frame = frame.merge(first, on="possession_id", how="left")
    frame["is_first"] = frame["segment_number"].eq(frame["first_segment"])
    excl = frame["points_excl_tech"] if "points_excl_tech" in frame else frame["points"]
    tech = frame["technical_points"] if "technical_points" in frame else 0
    frame["home_possessions"] = np.where(frame["offense_is_home"] & frame["is_first"], 1, 0)
    frame["away_possessions"] = np.where((~frame["offense_is_home"]) & frame["is_first"], 1, 0)
    frame["home_points"] = np.where(frame["offense_is_home"], frame["points"], 0)
    frame["away_points"] = np.where(~frame["offense_is_home"], frame["points"], 0)
    frame["home_points_excl"] = np.where(frame["offense_is_home"], excl, 0)
    frame["away_points_excl"] = np.where(~frame["offense_is_home"], excl, 0)
    frame["home_technical_points_excluded"] = np.where(frame["offense_is_home"], tech, 0)
    frame["away_technical_points_excluded"] = np.where(~frame["offense_is_home"], tech, 0)
    grouped = frame.groupby(["game_id", "season_end", *LINEUP_COLUMNS], as_index=False, sort=False).agg(
        home_possessions=("home_possessions", "sum"),
        away_possessions=("away_possessions", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        home_points_excl=("home_points_excl", "sum"),
        away_points_excl=("away_points_excl", "sum"),
        home_technical_points_excluded=("home_technical_points_excluded", "sum"),
        away_technical_points_excluded=("away_technical_points_excluded", "sum"),
    )
    grouped["season"] = grouped["season_end"].astype(int)
    if (grouped[["home_points_excl", "away_points_excl"]] < 0).any().any():
        raise ValueError("Stint technical-FT exclusion produced negative points.")
    return grouped


def segments_to_terminal(possessions: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Assign each possession's points to its last lineup. Blocked H6 comparison."""
    extra = tuple(column for column in ("points", "points_excl_tech", "technical_points") if column in possessions.columns)
    frame = _attach_possession_meta(possessions, segments, extra=extra)
    last = frame.groupby("possession_id", as_index=False)["segment_number"].max().rename(
        columns={"segment_number": "last_segment"}
    )
    frame = frame.merge(last, on="possession_id", how="inner")
    frame = frame.loc[frame["segment_number"].eq(frame["last_segment"])].copy()
    points = frame["points"]
    excl = frame["points_excl_tech"] if "points_excl_tech" in frame else points
    tech = frame["technical_points"] if "technical_points" in frame else 0
    frame["home_possessions"] = np.where(frame["offense_is_home"], 1, 0)
    frame["away_possessions"] = np.where(~frame["offense_is_home"], 1, 0)
    frame["home_points"] = np.where(frame["offense_is_home"], points, 0)
    frame["away_points"] = np.where(~frame["offense_is_home"], points, 0)
    frame["home_points_excl"] = np.where(frame["offense_is_home"], excl, 0)
    frame["away_points_excl"] = np.where(~frame["offense_is_home"], excl, 0)
    frame["home_technical_points_excluded"] = np.where(frame["offense_is_home"], tech, 0)
    frame["away_technical_points_excluded"] = np.where(~frame["offense_is_home"], tech, 0)
    grouped = frame.groupby(["game_id", "season_end", *LINEUP_COLUMNS], as_index=False, sort=False).agg(
        home_possessions=("home_possessions", "sum"),
        away_possessions=("away_possessions", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        home_points_excl=("home_points_excl", "sum"),
        away_points_excl=("away_points_excl", "sum"),
        home_technical_points_excluded=("home_technical_points_excluded", "sum"),
        away_technical_points_excluded=("away_technical_points_excluded", "sum"),
    )
    grouped["season"] = grouped["season_end"].astype(int)
    return grouped


def public_priors(season: int, family: str = "pulse") -> pd.DataFrame:
    rows = json.loads((WEB / f"leaderboard-{season}.json").read_text())
    frame = pd.DataFrame(rows)
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    frame["Window_End"] = season
    if family == "pulse":
        return frame.rename(columns={
            "pulse_prior_offense": "prior_offense_per_100",
            "pulse_prior_defense": "prior_defense_per_100",
        })
    if family == "rich":
        return frame.rename(columns={
            "rich_spm_offense": "prior_offense_per_100",
            "rich_spm_defense": "prior_defense_per_100",
        })
    raise ValueError(family)


def official_margins(scores: pd.DataFrame, season: int) -> pd.DataFrame:
    games = scores.loc[scores["project_season"].eq(season) & scores["season_type"].eq("regular")].copy()
    games["game_id"] = games["game_id"].map(canonical_game_id)
    games["v3_margin"] = games["home_score"].astype(float) - games["away_score"].astype(float)
    if "box_home_score" in games and games["box_home_score"].notna().any():
        games["box_margin"] = games["box_home_score"].astype(float) - games["box_away_score"].astype(float)
        games["official_margin"] = games["box_margin"].where(games["box_margin"].notna(), games["v3_margin"])
    else:
        games["box_margin"] = games["v3_margin"]
        games["official_margin"] = games["v3_margin"]
    return games[[
        "game_id", "official_margin", "v3_margin", "box_margin", "home_score", "away_score"
    ]]


def score_games(source, target, beta, intercept, official: pd.DataFrame, name: str, rating_season: int) -> pd.DataFrame:
    games = predict_next_season_games(source, target, beta, intercept)
    games = games.merge(official, on="game_id", how="inner", validate="one_to_one")
    output = games.rename(columns={"actual_margin": "excl_margin"})
    output["actual_margin"] = output["official_margin"]
    output["candidate"] = name
    output["rating_season"] = rating_season
    output["outcome_season"] = rating_season + 1
    output["squared_error_official"] = (output["official_margin"] - output["predicted_margin"]) ** 2
    output["squared_error_v3"] = (output["v3_margin"] - output["predicted_margin"]) ** 2
    output["squared_error_excl"] = (output["excl_margin"] - output["predicted_margin"]) ** 2
    return output


def fit_stint_arm(
    stints: pd.DataFrame,
    point_columns: tuple[str, str],
    priors,
    season: int,
    scale: float,
    *,
    lambda_off: float = 3000.0,
    lambda_def: float = 4500.0,
):
    frame = stints.copy()
    frame["home_points"] = frame[point_columns[0]]
    frame["away_points"] = frame[point_columns[1]]
    design = build_stint_design(frame)
    zero = np.zeros(design.X.shape[1])
    config = RapmConfig(
        (season,),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=300.0,
        data_scope="pulse_search_h1_v3_candidate",
    )
    rapm_beta, rapm_intercept = fit_stint_center_path(design, config, zero, center_scales=(0.0,))[0.0]
    center, coverage = stint_prior_center(design, priors, season)
    pulse_beta, pulse_intercept = fit_stint_center_path(
        design, config, center, center_scales=(scale,)
    )[scale]
    return design, {
        "rapm": (rapm_beta, rapm_intercept),
        "pulse": (pulse_beta, pulse_intercept),
    }, coverage, center


def paired_delta(games: pd.DataFrame, left: str, right: str, value: str) -> dict:
    wide = games.loc[games["candidate"].isin([left, right])].pivot(
        index=["outcome_season", "game_id"], columns="candidate", values=value
    )
    if wide.isna().any().any():
        raise ValueError(f"{left} and {right} did not score identical games on {value}.")
    delta = wide[left] - wide[right]
    rng = np.random.default_rng(20260921)
    values = [group.to_numpy() for _, group in delta.groupby(level="outcome_season")]
    draws = np.array([
        np.mean([part[rng.integers(0, len(part), len(part))].mean() for part in values])
        for _ in range(2000)
    ])
    observed = float(np.mean([part.mean() for part in values]))
    return {
        "mean": observed,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "games": int(len(wide)),
    }

