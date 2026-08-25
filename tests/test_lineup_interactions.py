import numpy as np
import pandas as pd

from nba_impact.models.lineup_interactions import (
    build_combination_vocabulary,
    build_interaction_design,
    fit_interaction_layer,
    game_margin_metrics,
    predict_interaction_layer,
)


def test_interaction_layer_recovers_repeated_pair_signal() -> None:
    offense = np.array(
        [[1, 2, 3, 4, 5]] * 100 + [[1, 6, 7, 8, 9]] * 100, dtype=np.int64
    )
    defense = np.array([[10, 11, 12, 13, 14]] * 200, dtype=np.int64)
    residual = np.array([1.0] * 100 + [0.0] * 100)
    fit = fit_interaction_layer(
        offense,
        defense,
        residual,
        order=2,
        penalty=1.0,
        minimum_exposure=2,
    )
    predicted = predict_interaction_layer(fit, offense, defense)
    assert predicted[:100].mean() > predicted[100:].mean()
    assert len(fit.combinations) > 0


def test_unknown_units_contribute_zero() -> None:
    keys = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
    vocabulary = build_combination_vocabulary(
        keys, np.array([10.0]), order=5, minimum_exposure=1
    )
    design = build_interaction_design(
        np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=np.int64),
        vocabulary,
        order=5,
    )
    assert design.nnz == 0


def test_game_margin_conservation() -> None:
    frame = pd.DataFrame(
        {"gameid": ["a", "a", "b"], "home_poss": [1, 0, 1], "pts": [2.0, 3.0, 1.0]}
    )
    metrics, games = game_margin_metrics(frame, np.array([2.0, 3.0, 1.0]))
    assert metrics["margin_rmse"] == 0.0
    assert games.set_index("gameid").loc["a", "actual_margin"] == -1.0
