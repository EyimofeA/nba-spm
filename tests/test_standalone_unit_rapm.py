import numpy as np

from nba_impact.models.standalone_unit_rapm import (
    fit_unit_rapm,
    predict_unit_rapm,
    unit_slot_coverage,
)


def test_pair_rapm_contains_no_individual_player_columns() -> None:
    offense = np.array(
        [[1, 2, 3, 4, 5]] * 80 + [[1, 6, 7, 8, 9]] * 80,
        dtype=np.int64,
    )
    defense = np.array([[10, 11, 12, 13, 14]] * 160, dtype=np.int64)
    home = np.tile([True, False], 80)
    points = np.array([2.0] * 80 + [0.0] * 80)
    fit = fit_unit_rapm(
        offense,
        defense,
        home,
        points,
        order=2,
        unit_penalty=1.0,
        minimum_exposure=2,
    )
    predicted = predict_unit_rapm(fit, offense, defense, home)

    assert fit.combinations.shape[1] == 2
    assert fit.coefficients.shape == (2 * len(fit.combinations) + 1,)
    assert predicted[:80].mean() > predicted[80:].mean()


def test_unseen_lineup_units_receive_zero_update() -> None:
    offense = np.array([[1, 2, 3, 4, 5]] * 20, dtype=np.int64)
    defense = np.array([[6, 7, 8, 9, 10]] * 20, dtype=np.int64)
    home = np.ones(20, dtype=bool)
    points = np.ones(20)
    fit = fit_unit_rapm(
        offense,
        defense,
        home,
        points,
        order=5,
        unit_penalty=10.0,
        minimum_exposure=1,
    )
    unseen_offense = np.array([[11, 12, 13, 14, 15]], dtype=np.int64)
    unseen_defense = np.array([[16, 17, 18, 19, 20]], dtype=np.int64)

    home_prediction = predict_unit_rapm(
        fit, unseen_offense, unseen_defense, np.array([True])
    )[0]
    away_prediction = predict_unit_rapm(
        fit, unseen_offense, unseen_defense, np.array([False])
    )[0]
    assert np.isclose((home_prediction + away_prediction) / 2, fit.intercept)
    assert unit_slot_coverage(fit, unseen_offense, unseen_defense) == 0.0


def test_venue_is_not_collapsed_when_lineups_repeat() -> None:
    offense = np.array([[1, 2, 3, 4, 5]] * 80, dtype=np.int64)
    defense = np.array([[6, 7, 8, 9, 10]] * 80, dtype=np.int64)
    home = np.array([True] * 40 + [False] * 40)
    points = np.array([2.0] * 40 + [0.0] * 40)
    fit = fit_unit_rapm(
        offense,
        defense,
        home,
        points,
        order=5,
        unit_penalty=1000.0,
        home_penalty=1.0,
        minimum_exposure=1,
    )
    predicted = predict_unit_rapm(fit, offense, defense, home)
    assert predicted[:40].mean() > predicted[40:].mean()


def test_exposure_buckets_keep_low_sample_units_with_extra_shrinkage() -> None:
    offense = np.array(
        [[1, 2, 3, 4, 5]] * 20 + [[1, 6, 7, 8, 9]] * 5,
        dtype=np.int64,
    )
    defense = np.array([[10, 11, 12, 13, 14]] * 25, dtype=np.int64)
    home = np.ones(25, dtype=bool)
    points = np.array([2.0] * 20 + [0.0] * 5)
    hard = fit_unit_rapm(
        offense,
        defense,
        home,
        points,
        order=2,
        unit_penalty=10.0,
        minimum_exposure=10,
    )
    bucketed = fit_unit_rapm(
        offense,
        defense,
        home,
        points,
        order=2,
        unit_penalty=10.0,
        minimum_exposure=10,
        penalty_strategy="exposure_buckets",
    )
    assert bucketed.penalty_strategy == "exposure_buckets"
    assert len(bucketed.combinations) > len(hard.combinations)
