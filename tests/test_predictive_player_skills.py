from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.predictive_player_skills import (
    SkillSpec,
    _history_prediction,
    _posterior_for_season,
    _score_prediction,
    skill_definitions,
    tune_skill,
)


def _shooting_frame() -> pd.DataFrame:
    rows = []
    for season in range(2014, 2025):
        rows.extend(
            [
                {
                    "PLAYER_ID": 1,
                    "PLAYER_NAME": "A",
                    "TEAM_ABBREVIATION": "AAA",
                    "Season": season,
                    "AGE": 20 + season - 2014,
                    "numerator": 40 + season - 2014,
                    "opportunities": 100.0,
                    "raw_value": 40.0 + season - 2014,
                },
                {
                    "PLAYER_ID": 2,
                    "PLAYER_NAME": "B",
                    "TEAM_ABBREVIATION": "BBB",
                    "Season": season,
                    "AGE": 25 + season - 2014,
                    "numerator": 60 - (season - 2014),
                    "opportunities": 100.0,
                    "raw_value": 60.0 - (season - 2014),
                },
            ]
        )
    return pd.DataFrame(rows)


def test_history_prediction_excludes_target_season() -> None:
    frame = _shooting_frame()
    before, _ = _history_prediction(
        frame,
        target_season=2024,
        prior_strength=25.0,
        half_life=2.0,
        minimum_exposure=0.0,
        family="binomial",
        scale=100.0,
    )
    changed = frame.copy()
    changed.loc[changed["Season"].eq(2024), "numerator"] = 0.0
    changed.loc[changed["Season"].eq(2024), "raw_value"] = 0.0
    after, _ = _history_prediction(
        changed,
        target_season=2024,
        prior_strength=25.0,
        half_life=2.0,
        minimum_exposure=0.0,
        family="binomial",
        scale=100.0,
    )
    pd.testing.assert_frame_equal(before, after)


def test_current_posterior_includes_named_season() -> None:
    frame = _shooting_frame()
    before, _ = _history_prediction(
        frame,
        target_season=2024,
        prior_strength=25.0,
        half_life=2.0,
        minimum_exposure=0.0,
        family="binomial",
        scale=100.0,
        include_target=True,
    )
    changed = frame.copy()
    changed.loc[(changed["Season"].eq(2024)) & changed["PLAYER_ID"].eq(1), ["numerator", "raw_value"]] = [100.0, 100.0]
    after, _ = _history_prediction(
        changed,
        target_season=2024,
        prior_strength=25.0,
        half_life=2.0,
        minimum_exposure=0.0,
        family="binomial",
        scale=100.0,
        include_target=True,
    )
    assert after.set_index("PLAYER_ID").loc[1, "estimate"] > before.set_index("PLAYER_ID").loc[1, "estimate"]


def test_binomial_score_matches_grouped_events() -> None:
    spec = SkillSpec("test", "Test", "shooting", "binomial", "percent", 100)
    target = pd.DataFrame(
        {"numerator": [1.0, 0.0], "opportunities": [1.0, 1.0], "raw_value": [100.0, 0.0]}
    )
    scores = _score_prediction(target, np.array([75.0, 25.0]), spec)
    assert np.isclose(scores["primary"], -np.log(0.75))
    assert np.isclose(scores["secondary"], 0.0625)


def test_rebound_sources_are_scored_as_rates_not_impossible_binomials() -> None:
    definitions = skill_definitions().set_index("key")
    assert definitions.loc["offensive_rebound_rate", "family"] == "rate"
    assert definitions.loc["defensive_rebound_rate", "family"] == "rate"


def test_age_selected_posterior_applies_preseason_age_then_current_update() -> None:
    frame = _shooting_frame()
    selected_age = pd.Series(
        {
            "arm": "time_decayed_eb_plus_age",
            "prior_strength": 25.0,
            "half_life_years": "2",
            "minimum_exposure": 0.0,
            "age_alpha": 10.0,
        }
    )
    selected_base = selected_age.copy()
    selected_base["arm"] = "time_decayed_eb"
    spec = SkillSpec("test", "Test", "shooting", "binomial", "percent", 100)
    age, _ = _posterior_for_season(frame, spec, 2024, selected_age)
    base, _ = _posterior_for_season(frame, spec, 2024, selected_base)
    joined = age.merge(base, on="PLAYER_ID", suffixes=("_age", "_base"))
    assert not np.allclose(
        joined["preseason_estimate_age"], joined["preseason_estimate_base"]
    )
    player = age.set_index("PLAYER_ID").loc[1]
    observed = frame.loc[
        frame["PLAYER_ID"].eq(1) & frame["Season"].eq(2024)
    ].iloc[0]
    expected = (
        player["preseason_estimate"] * player["preseason_precision"]
        + observed["raw_value"] * observed["opportunities"]
    ) / (player["preseason_precision"] + observed["opportunities"])
    assert np.isclose(player["estimate"], expected)


def test_tuning_records_all_arms_and_role_support_skip() -> None:
    spec = SkillSpec("test", "Test", "shooting", "binomial", "percent", 100)
    folds, decisions = tune_skill(
        _shooting_frame(),
        spec,
        selection_seasons=(2019, 2020, 2021, 2022, 2023, 2024),
        prior_grid=(25.0,),
        half_life_grid=(1.0, 2.0),
        minimum_exposure_grid=(0.0,),
        age_alpha_grid=(10.0,),
    )
    assert folds["test_season"].max() == 2024
    assert {
        "raw_previous_season",
        "career_eb",
        "time_decayed_eb",
        "time_decayed_eb_plus_age",
    }.issubset(set(folds["arm"]))
    role = decisions.loc[decisions["arm"].eq("role_conditional")].iloc[0]
    assert role["status"] == "skipped"
    assert decisions["selected"].eq(True).sum() == 1  # noqa: E712
