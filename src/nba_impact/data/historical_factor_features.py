"""Mechanism-specific shooting and rebound features for factor SPM research."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SHOT_FEATURES = (
    "shot_quality_created_points_per_shot",
    "shotmaking_above_expected_p100_eb",
    "rim_attempt_share",
    "rim_shotmaking_above_expected_p100_eb",
    "midrange_attempt_share",
    "three_attempt_share",
    "assisted_attempt_share",
    "transition_attempt_share",
    "defender_expected_points_conceded_per_shot",
    "defender_shotmaking_points_saved_p100_eb",
    "defender_rim_expected_points_conceded_per_shot",
    "rim_deterrence_vs_scorer_p100_eb",
    "has_assigned_shot_defense",
)

REBOUND_FEATURES = (
    "self_oreb_adjusted_ts",
    "player_height_inches",
    "dreb_chances_p100_specialist",
    "dreb_contested_share_specialist",
    "dreb_defer_share_specialist",
    "average_dreb_distance",
    "rebound_conversion_above_expected_eb",
    "oreb_chances_p100_specialist",
    "oreb_contested_share_specialist",
    "oreb_defer_share_specialist",
    "average_oreb_distance",
    "oreb_conversion_above_expected_eb",
    "offensive_boxouts_p100_specialist",
    "defensive_boxouts_p100_specialist",
    "boxout_team_rebound_conversion_eb",
    "boxout_player_rebound_conversion_eb",
    "height_x_defensive_boxouts",
    "height_x_dreb_contested_share",
    "height_x_offensive_boxouts",
    "height_x_oreb_contested_share",
    "has_boxout_tracking",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _clock_seconds(value: object) -> float:
    text = str(value)
    if not text.startswith("PT"):
        return np.nan
    text = text[2:]
    minutes = 0.0
    if "M" in text:
        minute_text, text = text.split("M", 1)
        minutes = float(minute_text or 0)
    seconds = float(text.removesuffix("S") or 0)
    return 60.0 * minutes + seconds


def _qualifier_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value)).lower()
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.lower()
    return " ".join(map(str, parsed)).lower() if isinstance(parsed, list) else text.lower()


def _shot_rows(events: pd.DataFrame) -> pd.DataFrame:
    action = events["actionType"].fillna("").astype(str).str.lower()
    shots = events.loc[action.isin(("2pt", "3pt"))].copy()
    shots["game_int"] = pd.to_numeric(shots["game_id"], errors="raise").astype("int64")
    shots["event_int"] = pd.to_numeric(shots["actionNumber"], errors="raise").astype("int64")
    shots["scorer_id"] = pd.to_numeric(shots["person_id"], errors="coerce").astype("Int64")
    shots["made"] = shots["shotResult"].fillna("").astype(str).str.lower().eq("made")
    shots["shot_value"] = np.where(action.loc[shots.index].eq("3pt"), 3.0, 2.0)
    shots["actual_points"] = shots["made"].astype(float) * shots["shot_value"]
    x = pd.to_numeric(shots["xLegacy"], errors="coerce") / 10.0
    y = pd.to_numeric(shots["yLegacy"], errors="coerce") / 10.0
    fallback_distance = pd.Series(
        np.where(shots["shot_value"].eq(3), 24.0, 10.0), index=shots.index
    )
    shots["distance"] = np.sqrt(np.square(x) + np.square(y)).fillna(
        fallback_distance
    )
    shots["zone"] = np.select(
        [
            shots["distance"].le(4.0),
            shots["distance"].le(14.0),
            shots["shot_value"].eq(2.0),
            x.abs().ge(22.0) & y.le(14.0),
        ],
        ["rim", "short_midrange", "long_midrange", "corner_three"],
        default="above_break_three",
    )
    assisted = shots.get("assisted", pd.Series(False, index=shots.index))
    shots["assisted_flag"] = assisted.fillna(False).astype(bool)
    qualifier = shots.get("qualifier", pd.Series("", index=shots.index)).map(_qualifier_text)
    description = shots["description"].fillna("").astype(str).str.lower()
    previous = shots.get("previous_action", pd.Series("", index=shots.index)).fillna("").astype(str).str.lower()
    shots["transition_flag"] = (
        qualifier.str.contains("transition|fast break", regex=True)
        | description.str.contains("fast break", regex=False)
        | previous.str.contains("turnover|steal", regex=True)
    )
    clock = shots.get("clock", pd.Series(np.nan, index=shots.index)).map(_clock_seconds)
    shots["late_period_flag"] = clock.le(4.0).fillna(False)
    return shots


def _leave_game_out_expected_points(shots: pd.DataFrame, prior_attempts: float) -> pd.Series:
    keys = ["zone", "assisted_flag", "transition_flag", "late_period_flag"]
    game = (
        shots.groupby([*keys, "game_int"], as_index=False)
        .agg(game_points=("actual_points", "sum"), game_attempts=("actual_points", "size"))
    )
    totals = game.groupby(keys, as_index=False).agg(
        group_points=("game_points", "sum"), group_attempts=("game_attempts", "sum")
    )
    game = game.merge(totals, on=keys, validate="many_to_one")
    league = float(shots["actual_points"].mean())
    game["expected_points"] = (
        game["group_points"] - game["game_points"] + prior_attempts * league
    ) / (game["group_attempts"] - game["game_attempts"] + prior_attempts)
    return shots[[*keys, "game_int"]].merge(
        game[[*keys, "game_int", "expected_points"]],
        on=[*keys, "game_int"],
        how="left",
        validate="many_to_one",
    )["expected_points"]


def build_shot_context_features(
    events: pd.DataFrame,
    assignments: pd.DataFrame | None,
    matchup_pairs: pd.DataFrame | None,
    season: int,
    *,
    quality_prior_attempts: float = 100.0,
    shotmaking_prior_attempts: float = 500.0,
    deterrence_prior_possessions: float = 500.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build annual cross-fitted shot quality and assignment-aware defense features."""
    shots = _shot_rows(events)
    shots["expected_points"] = _leave_game_out_expected_points(
        shots, quality_prior_attempts
    ).to_numpy()
    shots["shotmaking_residual"] = shots["actual_points"] - shots["expected_points"]
    shots["is_rim"] = shots["zone"].eq("rim")
    shots["is_midrange"] = shots["zone"].isin(("short_midrange", "long_midrange"))
    shots["is_three"] = shots["shot_value"].eq(3.0)

    offense = shots.dropna(subset=["scorer_id"]).groupby("scorer_id", as_index=False).agg(
        shot_attempts=("actual_points", "size"),
        expected_points=("expected_points", "sum"),
        residual_points=("shotmaking_residual", "sum"),
        rim_attempts=("is_rim", "sum"),
        rim_residual_points=("shotmaking_residual", lambda value: value[shots.loc[value.index, "is_rim"]].sum()),
        midrange_attempts=("is_midrange", "sum"),
        three_attempts=("is_three", "sum"),
        assisted_attempts=("assisted_flag", "sum"),
        transition_attempts=("transition_flag", "sum"),
    ).rename(columns={"scorer_id": "PLAYER_ID"})
    offense["shot_quality_created_points_per_shot"] = offense["expected_points"] / offense["shot_attempts"]
    reliability = offense["shot_attempts"] / (
        offense["shot_attempts"] + shotmaking_prior_attempts
    )
    offense["shotmaking_above_expected_p100_eb"] = (
        100.0 * reliability * offense["residual_points"] / offense["shot_attempts"]
    )
    offense["rim_attempt_share"] = offense["rim_attempts"] / offense["shot_attempts"]
    offense["rim_shotmaking_above_expected_p100_eb"] = (
        100.0 * reliability * offense["rim_residual_points"] / offense["shot_attempts"]
    )
    offense["midrange_attempt_share"] = offense["midrange_attempts"] / offense["shot_attempts"]
    offense["three_attempt_share"] = offense["three_attempts"] / offense["shot_attempts"]
    offense["assisted_attempt_share"] = offense["assisted_attempts"] / offense["shot_attempts"]
    offense["transition_attempt_share"] = offense["transition_attempts"] / offense["shot_attempts"]

    defense = pd.DataFrame(columns=["PLAYER_ID"])
    assignment_quality: dict[str, object] = {
        "assignment_source_available": False,
        "assigned_shots": 0,
        "assigned_shot_fraction": 0.0,
    }
    if assignments is not None and len(assignments):
        tags = assignments[["gi", "ei", "def_id"]].copy()
        tags.columns = ["game_int", "event_int", "defender_id"]
        for column in tags:
            tags[column] = pd.to_numeric(tags[column], errors="coerce")
        tags = tags.dropna().astype("int64").drop_duplicates()
        tags["defender_tags"] = tags.groupby(["game_int", "event_int"])[
            "defender_id"
        ].transform("nunique")
        tags["assignment_weight"] = 1.0 / tags["defender_tags"]
        assigned = shots.merge(
            tags,
            on=["game_int", "event_int"],
            how="inner",
            validate="one_to_many",
        )
        assigned["weighted_expected"] = assigned["assignment_weight"] * assigned["expected_points"]
        assigned["weighted_saved"] = -assigned["assignment_weight"] * assigned["shotmaking_residual"]
        assigned["weighted_attempt"] = assigned["assignment_weight"]
        assigned["weighted_rim_attempt"] = assigned["assignment_weight"] * assigned["is_rim"]
        assigned["weighted_rim_expected"] = assigned["weighted_expected"] * assigned["is_rim"]
        defense = assigned.groupby("defender_id", as_index=False).agg(
            assigned_attempts=("weighted_attempt", "sum"),
            expected_points_conceded=("weighted_expected", "sum"),
            shotmaking_points_saved=("weighted_saved", "sum"),
            assigned_rim_attempts=("weighted_rim_attempt", "sum"),
            assigned_rim_expected=("weighted_rim_expected", "sum"),
        ).rename(columns={"defender_id": "PLAYER_ID"})
        defense_reliability = defense["assigned_attempts"] / (
            defense["assigned_attempts"] + shotmaking_prior_attempts
        )
        defense["defender_expected_points_conceded_per_shot"] = (
            defense["expected_points_conceded"] / defense["assigned_attempts"]
        )
        defense["defender_shotmaking_points_saved_p100_eb"] = (
            100.0
            * defense_reliability
            * defense["shotmaking_points_saved"]
            / defense["assigned_attempts"]
        )
        defense["defender_rim_expected_points_conceded_per_shot"] = (
            defense["assigned_rim_expected"]
            / defense["assigned_rim_attempts"].where(defense["assigned_rim_attempts"].gt(0))
        ).fillna(float(shots.loc[shots["is_rim"], "expected_points"].mean()))
        assignment_quality = {
            "assignment_source_available": True,
            "assignment_rows": int(len(tags)),
            "multi_tagged_shots": int(tags.loc[tags["defender_tags"].gt(1), ["game_int", "event_int"]].drop_duplicates().shape[0]),
            "assigned_shots": int(assigned[["game_int", "event_int"]].drop_duplicates().shape[0]),
            "assigned_shot_fraction": float(
                assigned[["game_int", "event_int"]].drop_duplicates().shape[0]
                / max(1, shots[["game_int", "event_int"]].drop_duplicates().shape[0])
            ),
        }

        if matchup_pairs is not None and len(matchup_pairs):
            pairs = matchup_pairs[
                ["person_id", "matchups_person_id", "partial_possessions"]
            ].copy()
            pairs.columns = ["scorer_id", "defender_id", "pair_possessions"]
            pairs = pairs.apply(pd.to_numeric, errors="coerce").dropna()
            pairs = pairs.groupby(["scorer_id", "defender_id"], as_index=False)[
                "pair_possessions"
            ].sum()
            rim = assigned.groupby(["scorer_id", "defender_id"], as_index=False).agg(
                rim_attempts=("weighted_rim_attempt", "sum")
            )
            pairs = pairs.merge(rim, on=["scorer_id", "defender_id"], how="left")
            pairs["rim_attempts"] = pairs["rim_attempts"].fillna(0.0)
            scorer_possessions = pairs.groupby("scorer_id")["pair_possessions"].transform("sum")
            scorer_rim = pairs.groupby("scorer_id")["rim_attempts"].transform("sum")
            other_possessions = scorer_possessions - pairs["pair_possessions"]
            league_rim_rate = float(pairs["rim_attempts"].sum() / pairs["pair_possessions"].sum())
            expected_rate = (
                (scorer_rim - pairs["rim_attempts"])
                / other_possessions.where(other_possessions.gt(0))
            ).fillna(league_rim_rate)
            pairs["rim_attempts_prevented"] = (
                expected_rate * pairs["pair_possessions"] - pairs["rim_attempts"]
            )
            deterrence = pairs.groupby("defender_id", as_index=False).agg(
                matchup_possessions=("pair_possessions", "sum"),
                rim_attempts_prevented=("rim_attempts_prevented", "sum"),
            ).rename(columns={"defender_id": "PLAYER_ID"})
            det_reliability = deterrence["matchup_possessions"] / (
                deterrence["matchup_possessions"] + deterrence_prior_possessions
            )
            deterrence["rim_deterrence_vs_scorer_p100_eb"] = (
                100.0
                * det_reliability
                * deterrence["rim_attempts_prevented"]
                / deterrence["matchup_possessions"]
            )
            defense = defense.merge(
                deterrence[["PLAYER_ID", "rim_deterrence_vs_scorer_p100_eb"]],
                on="PLAYER_ID",
                how="outer",
                validate="one_to_one",
            )

    output = offense.merge(defense, on="PLAYER_ID", how="outer", validate="one_to_one")
    output["Season"] = int(season)
    for exposure in ("shot_attempts", "assigned_attempts"):
        if exposure not in output:
            output[exposure] = 0.0
    output["shot_context_off_exposure"] = output["shot_attempts"].fillna(0.0)
    output["shot_context_def_exposure"] = output["assigned_attempts"].fillna(0.0)
    output["has_assigned_shot_defense"] = output["assigned_attempts"].fillna(0).gt(0).astype(float)
    for feature in SHOT_FEATURES:
        if feature not in output:
            output[feature] = 0.0
    output[list(SHOT_FEATURES)] = output[list(SHOT_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    quality = {
        "season": int(season),
        "shot_rows": int(len(shots)),
        "expected_quality_definition": (
            "Leave-one-game-out empirical expected points by location zone, assisted status, "
            "transition context, and final-four-seconds game-clock context."
        ),
        "shot_clock_available": False,
        "defender_distance_available": False,
        "defender_assignment_policy": "fractional_equal_weight_for_multi_tagged_shots",
        **assignment_quality,
    }
    return output[
        [
            "PLAYER_ID",
            "Season",
            "shot_context_off_exposure",
            "shot_context_def_exposure",
            *SHOT_FEATURES,
        ]
    ], quality


def build_rebound_responsibility_features(
    player_sheet: pd.DataFrame,
    season: int,
    *,
    conversion_prior_chances: float = 100.0,
    boxout_prior_events: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build chance-, box-out-, and height-conditioned rebound features."""
    source_rows = len(player_sheet)
    if player_sheet.duplicated("PLAYER_ID").any():
        player_sheet = player_sheet.groupby("PLAYER_ID", as_index=False).first()
    output = player_sheet[["PLAYER_ID"]].copy()
    output["Season"] = int(season)
    off_poss = _numeric(player_sheet, "OffPoss")
    def_poss = _numeric(player_sheet, "DefPoss")
    output["rebound_context_exposure"] = def_poss.fillna(0.0)
    attempts = _numeric(player_sheet, "FGA") + 0.44 * _numeric(player_sheet, "FTA")
    adjusted_attempts = attempts - _numeric(player_sheet, "SelfOReb").fillna(0.0)
    output["self_oreb_adjusted_ts"] = _numeric(player_sheet, "PTS") / (
        2.0 * adjusted_attempts.where(adjusted_attempts.gt(0))
    )
    output["player_height_inches"] = _numeric(player_sheet, "PLAYER_HEIGHT_INCHES")
    chances = _numeric(player_sheet, "DREB_CHANCES")
    rebounds = _numeric(player_sheet, "DREB")
    contests = _numeric(player_sheet, "DREB_CONTEST")
    defers = _numeric(player_sheet, "DREB_CHANCE_DEFER")
    distance = _numeric(player_sheet, "AVG_DREB_DIST")
    output["dreb_chances_p100_specialist"] = 100.0 * chances / def_poss.where(def_poss.gt(0))
    output["dreb_contested_share_specialist"] = contests / chances.where(chances.gt(0))
    output["dreb_defer_share_specialist"] = defers / chances.where(chances.gt(0))
    output["average_dreb_distance"] = distance

    predictor = pd.DataFrame(
        {
            "height": output["player_height_inches"],
            "contest_share": output["dreb_contested_share_specialist"],
            "defer_share": output["dreb_defer_share_specialist"],
            "distance": distance,
        }
    )
    observed = rebounds / chances.where(chances.gt(0))
    valid = predictor.notna().all(axis=1) & observed.notna() & chances.gt(0)
    expected = pd.Series(float(np.average(observed[valid], weights=chances[valid])), index=output.index)
    if valid.sum() >= 20:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(predictor.loc[valid], observed.loc[valid], ridge__sample_weight=chances.loc[valid])
        expected.loc[predictor.notna().all(axis=1)] = model.predict(
            predictor.loc[predictor.notna().all(axis=1)]
        )
    reliability = chances / (chances + conversion_prior_chances)
    output["rebound_conversion_above_expected_eb"] = reliability * (observed - expected)

    oreb_chances = _numeric(player_sheet, "OREB_CHANCES")
    offensive_rebounds = _numeric(player_sheet, "OREB")
    oreb_contests = _numeric(player_sheet, "OREB_CONTEST")
    oreb_defers = _numeric(player_sheet, "OREB_CHANCE_DEFER")
    oreb_distance = _numeric(player_sheet, "AVG_OREB_DIST")
    output["oreb_chances_p100_specialist"] = (
        100.0 * oreb_chances / off_poss.where(off_poss.gt(0))
    )
    output["oreb_contested_share_specialist"] = (
        oreb_contests / oreb_chances.where(oreb_chances.gt(0))
    )
    output["oreb_defer_share_specialist"] = (
        oreb_defers / oreb_chances.where(oreb_chances.gt(0))
    )
    output["average_oreb_distance"] = oreb_distance
    oreb_predictor = pd.DataFrame(
        {
            "height": output["player_height_inches"],
            "contest_share": output["oreb_contested_share_specialist"],
            "defer_share": output["oreb_defer_share_specialist"],
            "distance": oreb_distance,
        }
    )
    oreb_observed = offensive_rebounds / oreb_chances.where(oreb_chances.gt(0))
    oreb_valid = (
        oreb_predictor.notna().all(axis=1)
        & oreb_observed.notna()
        & oreb_chances.gt(0)
    )
    oreb_center = (
        float(np.average(oreb_observed[oreb_valid], weights=oreb_chances[oreb_valid]))
        if oreb_valid.any()
        else 0.0
    )
    oreb_expected = pd.Series(oreb_center, index=output.index)
    if oreb_valid.sum() >= 20:
        oreb_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        oreb_model.fit(
            oreb_predictor.loc[oreb_valid],
            oreb_observed.loc[oreb_valid],
            ridge__sample_weight=oreb_chances.loc[oreb_valid],
        )
        predictable = oreb_predictor.notna().all(axis=1)
        oreb_expected.loc[predictable] = oreb_model.predict(
            oreb_predictor.loc[predictable]
        )
    oreb_reliability = oreb_chances / (oreb_chances + conversion_prior_chances)
    output["oreb_conversion_above_expected_eb"] = (
        oreb_reliability * (oreb_observed - oreb_expected)
    )

    def_boxouts = _numeric(player_sheet, "hustle_DEF_BOXOUTS")
    off_boxouts = _numeric(player_sheet, "hustle_OFF_BOXOUTS")
    team_rebounds = _numeric(player_sheet, "hustle_BOX_OUT_PLAYER_TEAM_REBS")
    player_rebounds = _numeric(player_sheet, "hustle_BOX_OUT_PLAYER_REBS")
    output["defensive_boxouts_p100_specialist"] = 100.0 * def_boxouts / def_poss.where(def_poss.gt(0))
    output["offensive_boxouts_p100_specialist"] = (
        100.0 * off_boxouts / off_poss.where(off_poss.gt(0))
    )
    league_team_conversion = float(team_rebounds.sum() / def_boxouts.sum()) if def_boxouts.sum() > 0 else 0.0
    league_player_conversion = float(player_rebounds.sum() / def_boxouts.sum()) if def_boxouts.sum() > 0 else 0.0
    output["boxout_team_rebound_conversion_eb"] = (
        team_rebounds + boxout_prior_events * league_team_conversion
    ) / (def_boxouts + boxout_prior_events)
    output["boxout_player_rebound_conversion_eb"] = (
        player_rebounds + boxout_prior_events * league_player_conversion
    ) / (def_boxouts + boxout_prior_events)
    centered_height = output["player_height_inches"] - output["player_height_inches"].median()
    output["height_x_defensive_boxouts"] = centered_height * output["defensive_boxouts_p100_specialist"]
    output["height_x_dreb_contested_share"] = centered_height * output["dreb_contested_share_specialist"]
    output["height_x_offensive_boxouts"] = (
        centered_height * output["offensive_boxouts_p100_specialist"]
    )
    output["height_x_oreb_contested_share"] = (
        centered_height * output["oreb_contested_share_specialist"]
    )
    output["has_boxout_tracking"] = def_boxouts.notna().astype(float)

    for feature in REBOUND_FEATURES:
        output[feature] = output[feature].replace([np.inf, -np.inf], np.nan)
        median = output[feature].median()
        output[feature] = output[feature].fillna(0.0 if pd.isna(median) else median)
    quality = {
        "season": int(season),
        "source_rows": int(source_rows),
        "unique_players": int(len(player_sheet)),
        "rebound_chance_coverage": float(chances.notna().mean()),
        "boxout_coverage": float(def_boxouts.notna().mean()),
        "rebound_probability_definition": (
            "Empirical-Bayes residual of defensive-rebound conversion after height, "
            "contest share, defer share, and rebound distance."
        ),
        "unique_boxout_responsibility_available": bool(def_boxouts.notna().any()),
    }
    return output[
        ["PLAYER_ID", "Season", "rebound_context_exposure", *REBOUND_FEATURES]
    ], quality
