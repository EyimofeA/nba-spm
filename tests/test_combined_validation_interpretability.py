from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.models.combined_validation_interpretability import (
    align_game_predictions,
    align_prior_predictions,
    build_aio_component_ledger,
    build_factor_skill_panel,
    linear_group_contributions,
    paired_game_bootstrap,
    score_game_predictions,
    score_prior_predictions,
)


def _prior_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for candidate, shift in (("simple", 0.0), ("rich", 0.5)):
        for player, target in ((1, 1.0), (2, -1.0)):
            rows.append(
                {
                    "PLAYER_ID": player,
                    "Window_End": 2025,
                    "candidate": candidate,
                    "prior_offense_per_100": target + shift,
                    "prior_defense_per_100": -target + shift,
                    "prior_net_per_100": 2 * shift,
                }
            )
    rows.append(
        {
            "PLAYER_ID": 3,
            "Window_End": 2025,
            "candidate": "rich",
            "prior_offense_per_100": 99.0,
            "prior_defense_per_100": 99.0,
            "prior_net_per_100": 198.0,
        }
    )
    targets = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Window_End": [2025, 2025, 2025],
            "target_offense": [1.0, -1.0, 0.0],
            "target_defense": [-1.0, 1.0, 0.0],
            "target_net": [0.0, 0.0, 0.0],
            "sample_weight": [1.0, 2.0, 100.0],
        }
    )
    return pd.DataFrame(rows), targets


def test_prior_metrics_use_candidate_intersection_before_scoring() -> None:
    priors, targets = _prior_rows()
    aligned, audit = align_prior_predictions(
        priors, targets, candidates=("simple", "rich")
    )
    folds, summary = score_prior_predictions(aligned)
    assert audit["common_player_windows"] == 2
    assert aligned.groupby("candidate").size().to_dict() == {"rich": 2, "simple": 2}
    simple = summary.loc[
        summary["candidate"].eq("simple") & summary["component"].eq("net")
    ].iloc[0]
    assert simple["mean_weighted_rmse"] == pytest.approx(0.0)
    assert set(folds["players"]) == {2}


def test_prior_validation_rejects_nonfinite_values() -> None:
    priors, targets = _prior_rows()
    targets.loc[0, "target_offense"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        align_prior_predictions(priors, targets, candidates=("simple", "rich"))


def _game_rows() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        for game_id, actual in ((f"{season}-a", 3.0), (f"{season}-b", -2.0)):
            for candidate, prediction in (("simple", actual), ("rich", 0.0)):
                rows.append(
                    {
                        "rating_season": season - 1,
                        "test_season": season,
                        "game_id": game_id,
                        "candidate": candidate,
                        "actual_margin": actual,
                        "predicted_margin": prediction,
                    }
                )
    return pd.DataFrame(rows)


def test_game_validation_enforces_identical_outcomes_and_is_deterministic() -> None:
    aligned, audit = align_game_predictions(
        _game_rows(),
        candidates=("simple", "rich"),
        key_columns=("rating_season", "test_season", "game_id"),
    )
    folds, summary = score_game_predictions(
        aligned, fold_columns=("rating_season", "test_season")
    )
    assert audit["common_games"] == 4
    assert summary.iloc[0]["candidate"] == "simple"
    assert summary.iloc[0]["mean_mse"] == pytest.approx(0.0)
    first = paired_game_bootstrap(
        aligned,
        candidate="simple",
        reference="rich",
        season_column="test_season",
        draws=200,
        seed=7,
    )
    second = paired_game_bootstrap(
        aligned,
        candidate="simple",
        reference="rich",
        season_column="test_season",
        draws=200,
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["bootstrap_95_high"] < 0
    assert len(folds) == 4


def test_game_validation_rejects_outcome_mismatch_and_forbidden_season() -> None:
    mismatched = _game_rows()
    mismatched.loc[
        mismatched["candidate"].eq("rich") & mismatched["game_id"].eq("2024-a"),
        "actual_margin",
    ] = 4.0
    with pytest.raises(ValueError, match="identical actual"):
        align_game_predictions(
            mismatched,
            candidates=("simple", "rich"),
            key_columns=("rating_season", "test_season", "game_id"),
        )
    future = _game_rows().assign(test_season=2027)
    with pytest.raises(ValueError, match="forbidden Season 2027"):
        align_game_predictions(
            future,
            candidates=("simple", "rich"),
            key_columns=("rating_season", "test_season", "game_id"),
        )
    nonfinite = _game_rows()
    nonfinite.loc[0, "predicted_margin"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        align_game_predictions(
            nonfinite,
            candidates=("simple", "rich"),
            key_columns=("rating_season", "test_season", "game_id"),
        )


def _linear_model(features: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=0.1)),
        ]
    )
    target = 2.0 * features["shot"] - features["turnover"] + 0.5 * features["rebound"]
    return model.fit(features, target)


def test_linear_groups_and_aio_ledger_reconstruct_exactly() -> None:
    training = pd.DataFrame(
        {
            "shot": [0.0, 1.0, 2.0, 3.0],
            "turnover": [3.0, 2.0, 1.0, 0.0],
            "rebound": [1.0, 2.0, 1.0, 2.0],
        }
    )
    model = _linear_model(training)
    groups = {
        "shooting_scoring": ("shot",),
        "turnover": ("turnover",),
        "rebounding": ("rebound",),
    }
    contributions = linear_group_contributions(
        model,
        training,
        feature_names=("shot", "turnover", "rebound"),
        groups=groups,
    )
    assert contributions["identity_error"].abs().max() < 1e-12

    current = training.iloc[:2].copy()
    current.insert(0, "Window_End", 2026)
    current.insert(0, "PLAYER_ID", [1, 2])
    raw = model.predict(current[["shot", "turnover", "rebound"]])
    priors = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Window_End": [2026, 2026],
            "candidate": ["box_15", "box_15"],
            "prior_offense_per_100": raw,
            "prior_defense_per_100": raw,
            "prior_net_per_100": 2 * raw,
        }
    )
    rows = []
    for candidate, update in (("box_15", 0.0), ("box_15_aio", 0.25)):
        for index, player in enumerate((1, 2)):
            offense = raw[index] + 0.1 + update
            defense = raw[index] - 0.2 + update
            rows.append(
                {
                    "PLAYER_ID": player,
                    "PLAYER_NAME": f"P{player}",
                    "TEAM_ABBREVIATION": "TST",
                    "candidate": candidate,
                    "offense": offense,
                    "defense": defense,
                    "net": offense + defense,
                    "Poss_Off": 10,
                    "Poss_Def": 10,
                    "rank": player,
                }
            )
    ledger, summary, quality = build_aio_component_ledger(
        feature_panel=current,
        raw_priors=priors,
        active_leaderboard=pd.DataFrame(rows),
        models={"offense": model, "defense": model},
        feature_names=("shot", "turnover", "rebound"),
        groups=groups,
    )
    assert quality["maximum_identity_error"] < 1e-10
    assert quality["offense_centering_offset"] == pytest.approx(0.1)
    assert quality["defense_centering_offset"] == pytest.approx(-0.2)
    assert set(ledger["side"]) == {"offense", "defense", "net"}
    assert summary.filter(like="identity_error").abs().max().max() < 1e-10


def test_factor_skills_are_explicitly_non_additive() -> None:
    factors = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1],
            "Window_End": [2026, 2026, 2026],
            "factor": ["shooting_ts", "turnover_avoidance", "opponent_oreb_prevention"],
            "component": ["offense", "offense", "defense"],
            "candidate": ["specialist_factor"] * 3,
            "prediction": [1.0, 2.0, 3.0],
        }
    )
    active = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "PLAYER_NAME": ["Player"],
            "TEAM_ABBREVIATION": ["TST"],
        }
    )
    output = build_factor_skill_panel(factors, active)
    assert not output["additive_to_aio"].any()
    assert set(output["skill"]) == {
        "true_shooting_skill",
        "turnover_avoidance_skill",
        "rebounding_skill",
    }
    assert output["units"].nunique() == 3
    assert output["sign_convention"].notna().all()
