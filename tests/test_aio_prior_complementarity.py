from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from research.run_aio_prior_complementarity import (
    AnnualMatrix,
    _cross_fitted_box_defense,
    _metric_row,
    _validate_contract,
    _variance_model_multipliers,
    assert_component_identity,
    assert_identical_games,
    canonical_frame_hash,
    checkpoint_frame,
    coefficient_center,
    fully_lagged_pairs,
    future_reference_seasons,
    lambda_components,
    outcome_censored_features,
    past_reference,
    select_from_history,
)


def _contract() -> dict:
    return {
        "experiment_id": "aio_prior_complementarity_v1",
        "seasons": {
            "rating": list(range(2016, 2026)),
            "outcome": list(range(2017, 2027)),
            "design_selection_outcomes": list(range(2017, 2022)),
            "later_diagnostic_outcomes": list(range(2022, 2027)),
        },
        "evaluation": {"bootstrap_draws": 5000, "bootstrap_seed": 7},
        "player_precision": {
            "alpha": 10,
            "variance_clip_quantiles": [0.1, 0.9],
            "precision_multiplier_clip": [0.5, 2.0],
        },
    }


def _matrix(players=(1, 2)) -> AnnualMatrix:
    n = len(players)
    off = np.asarray([float(player) for player in players])
    deff = off + 1.0
    return AnnualMatrix(
        season=2020,
        players=np.asarray(players),
        xtx=csr_matrix((2 * n + 1, 2 * n + 1)),
        xty=np.zeros(2 * n + 1),
        off_exposure=off,
        def_exposure=deff,
        base_intercept=1.0,
        game_design=csr_matrix((1, 2 * n + 1)),
        game_ids=np.asarray(["g"]),
        actual_margin=np.asarray([0.0]),
        intercept_multiplier=np.asarray([0.0]),
        unknown_slots=np.asarray([0.0]),
    )


def test_contract_keeps_the_ten_requested_folds() -> None:
    _validate_contract(_contract())


def test_past_reference_excludes_season_t() -> None:
    current = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "rating_season": [2019],
            "reference": ["nine_year_current"],
            "target_offense": [2.0],
        }
    )
    shifted = past_reference(current)
    assert shifted.loc[0, "rating_season"] == 2020
    assert shifted.loc[0, "target_offense"] == 2.0
    assert shifted.loc[0, "reference"] == "nine_year_past"


def test_future_reference_excludes_season_t() -> None:
    seasons = future_reference_seasons(2020)
    assert seasons == (2021, 2022, 2023)
    assert 2020 not in seasons


def test_fully_lagged_alignment_uses_previous_season_features() -> None:
    features = pd.DataFrame({"PLAYER_ID": [1], "Season": [2019], "x": [3.0]})
    target = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "rating_season": [2019],
            "target_offense": [1.0],
            "target_defense": [2.0],
            "target_net": [3.0],
            "Poss_Off": [10.0],
            "Poss_Def": [10.0],
        }
    )
    paired = fully_lagged_pairs(features, target)
    assert paired.loc[0, "application_rating_season"] == 2020
    assert paired.loc[0, "Season"] == 2019


def test_defense_center_converts_positive_good_to_negative_points_allowed() -> None:
    prior = pd.DataFrame(
        {"PLAYER_ID": [1, 2], "prior_offense": [0.0, 0.0], "prior_defense": [2.0, -1.0]}
    )
    center, present = coefficient_center(prior, _matrix())
    assert present.all()
    assert center[2] < center[3]


def test_lambda_identity_matches_total_and_center_scale() -> None:
    zero, prior = lambda_components(4000.0, 0.25)
    beta = np.asarray([0.2, -0.3])
    center = np.asarray([0.7, -0.1])
    left = zero * np.sum(beta**2) + prior * np.sum((beta - center) ** 2)
    right = 4000.0 * np.sum((beta - 0.25 * center) ** 2)
    constant = prior * np.sum(center**2) - 4000.0 * np.sum((0.25 * center) ** 2)
    assert left == pytest.approx(right + constant)


def test_fold_selection_uses_only_earlier_outcomes_and_then_freezes() -> None:
    rows = pd.DataFrame(
        {
            "outcome_season": [2017, 2017, 2018, 2018, 2022, 2022],
            "choice": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "mse": [2.0, 1.0, 2.0, 1.0, 0.0, 99.0],
        }
    )
    early = select_from_history(
        rows,
        ["choice"],
        {"choice": 0.0},
        current_outcome_season=2018,
        selection_outcomes=tuple(range(2017, 2022)),
        minimum_folds=2,
    )
    frozen = select_from_history(
        rows,
        ["choice"],
        {"choice": 0.0},
        current_outcome_season=2023,
        selection_outcomes=tuple(range(2017, 2022)),
        minimum_folds=2,
    )
    assert early["choice"] == 0.0
    assert frozen["choice"] == 1.0


def test_defense_residual_labels_use_cross_fitted_box_predictions(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Season": np.repeat([2018, 2019, 2020], 2),
            "target_defense": np.arange(6, dtype=float),
            "sample_weight": np.ones(6),
            **{feature: np.ones(6) for feature in (
                "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
                "OREB_p100", "DREB_p100", "PF_p100", "PFD_p100", "FTA_p100",
                "FTM_p100", "FG2A_p100", "FG2M_p100", "FG3A_p100", "FG3M_p100",
            )},
        }
    )
    seen = []

    class FoldModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), self.value)

    def fake_fit(train, *_args, **_kwargs):
        seasons = tuple(sorted(train["Season"].unique()))
        seen.append(seasons)
        return FoldModel(float(sum(seasons)))

    monkeypatch.setattr(
        "research.run_aio_prior_complementarity.fit_box",
        fake_fit,
    )
    contract = _contract()
    contract["spm"] = {"box15": {"defense_alpha": 1000}}
    prediction = _cross_fitted_box_defense(frame, contract)
    assert seen == [(2019, 2020), (2018, 2020), (2018, 2019)]
    assert np.array_equal(
        prediction,
        np.repeat([4039.0, 4038.0, 4037.0], 2),
    )


def test_outcome_censor_keeps_activity_and_removes_result_fields() -> None:
    kept = outcome_censored_features(
        ("dfg_attempts_p100", "dfg_diff_pct_eb", "matchup_shotmaking_points_saved_x"),
        {"dfg_diff_pct_eb"},
        ("matchup_shotmaking_points_saved",),
    )
    assert kept == ("dfg_attempts_p100",)


def test_precision_multiplier_is_positive_bounded_and_deterministic() -> None:
    history = pd.DataFrame(
        {
            "rating_season": np.repeat([2018, 2019], 4),
            "reference_exposure": np.arange(1, 9) * 100,
            "absolute_offense_disagreement": np.linspace(0.1, 0.8, 8),
            "unavailable_source_family_count": [0, 1, 2, 3, 0, 1, 2, 3],
            "prior_offense": np.linspace(-1, 1, 8),
            "target_offense": np.linspace(-0.8, 0.9, 8),
            "sample_weight": np.ones(8),
        }
    )
    current = history.iloc[:4].copy()
    first = _variance_model_multipliers(history, current, side="offense", contract=_contract())
    second = _variance_model_multipliers(history, current, side="offense", contract=_contract())
    assert np.array_equal(first, second)
    assert first.min() >= 0.5
    assert first.max() <= 2.0


def test_player_order_does_not_change_center_by_player() -> None:
    prior = pd.DataFrame(
        {"PLAYER_ID": [1, 2], "prior_offense": [1.0, -1.0], "prior_defense": [2.0, -2.0]}
    )
    first, _ = coefficient_center(prior, _matrix((1, 2)))
    second, _ = coefficient_center(prior, _matrix((2, 1)))
    assert first[0] == pytest.approx(second[1])
    assert first[2] == pytest.approx(second[3])


def test_game_order_does_not_change_metrics() -> None:
    actual = np.asarray([1.0, -2.0, 3.0])
    predicted = np.asarray([0.0, -1.0, 2.0])
    first = _metric_row(actual, predicted)
    second = _metric_row(actual[::-1], predicted[::-1])
    for key in first:
        assert first[key] == pytest.approx(second[key])


def test_checkpoint_resume_does_not_rebuild(tmp_path: Path) -> None:
    calls = []

    def builder() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"x": [1]})

    path = tmp_path / "stage.parquet"
    checkpoint_frame(path, builder)
    checkpoint_frame(path, builder)
    assert calls == [1]


def test_identical_game_check_rejects_missing_candidate_rows() -> None:
    games = pd.DataFrame(
        {
            "outcome_season": [2020, 2020, 2020],
            "game_id": ["a", "a", "b"],
            "candidate": ["x", "y", "x"],
            "actual_margin": [1.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        assert_identical_games(games)


def test_component_identity_and_deterministic_hash() -> None:
    frame = pd.DataFrame({"id": [2, 1], "offense": [1.0, 2.0], "defense": [3.0, 4.0]})
    frame["net"] = frame["offense"] + frame["defense"]
    assert_component_identity(frame)
    assert canonical_frame_hash(frame, ["id"]) == canonical_frame_hash(frame.iloc[::-1], ["id"])
