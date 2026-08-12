from __future__ import annotations

from nba_impact.models.win_probability_stage1 import build_stage1_models


def test_stage1_candidates_are_fixed_and_bounded() -> None:
    models = build_stage1_models(seed=7)
    spline = models["spline_gam"].named_steps["splines"]
    gbm = models["hist_gbm"]
    assert spline.n_knots == 5
    assert spline.degree == 3
    assert gbm.max_iter == 200
    assert gbm.max_depth == 6
    assert gbm.max_leaf_nodes == 15
    assert gbm.early_stopping is False
