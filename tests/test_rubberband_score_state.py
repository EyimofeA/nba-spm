import numpy as np

from nba_impact.models.rapm import RapmConfig, build_design
from nba_impact.models.rubberband_score_state import (
    annotate_offense_margin_before,
    clock_margin_curve,
    clock_margin_design,
    fit_clock_margin_rapm,
    fit_score_state_rapm,
    predict_clock_margin_rapm,
    predict_score_state_rapm,
    score_state_curve,
    score_state_indices,
)


def _frame():
    import pandas as pd

    rows = []
    for index, margin in enumerate([-3, -2, -1, 0, 1, 2, 3] * 20):
        rows.append(
            {
                "gameid": str(index // 14),
                "season": 2025,
                "home_poss": index % 2,
                "pts": 1.1 - 0.02 * margin,
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            }
        )
    return pd.DataFrame(rows)


def test_score_state_indices_top_code_both_tails() -> None:
    result = score_state_indices(
        np.asarray([-70, -57, 0, 57, 70]), minimum=-57, maximum=57
    )
    assert result.tolist() == [0, 0, 57, 114, 114]


def test_five_point_score_buckets_are_centered_on_multiples_of_five() -> None:
    margins = np.asarray([-30, -23, -8, -3, -2, 0, 2, 3, 7, 8, 24, 30])
    indices = score_state_indices(
        margins,
        minimum=-25,
        maximum=25,
        bucket_width=5,
    )
    values = -25 + 5 * indices
    assert values.tolist() == [-25, -25, -10, -5, 0, 0, 0, 5, 5, 10, 25, 25]


def test_offense_margin_uses_only_points_before_the_possession() -> None:
    frame = _frame().iloc[:4].copy()
    frame["home_poss"] = [1, 0, 1, 0]
    frame["pts"] = [2, 3, 1, 0]
    frame["gameid"] = "one"
    frame["period"] = 1
    frame["num"] = [1, 2, 3, 4]
    result = annotate_offense_margin_before(frame)
    assert result["offense_margin_before"].tolist() == [0, -2, -1, 0]


def test_joint_fit_keeps_tie_reference_and_prediction_decomposition() -> None:
    frame = _frame()
    design = build_design(frame)
    margins = np.asarray([-3, -2, -1, 0, 1, 2, 3] * 20)
    config = RapmConfig(seasons=(2025,), lambda_off=10, lambda_def=10)
    fit = fit_score_state_rapm(
        design, margins, config, minimum=-3, maximum=3, state_penalty=0.1
    )
    curve = score_state_curve(fit)
    assert curve.loc[curve["margin"].eq(0), "effect_points_per_100_vs_tie"].item() == 0
    neutral = predict_score_state_rapm(
        fit, design, margins, include_score_state=False
    )
    conditional = predict_score_state_rapm(fit, design, margins)
    indices = score_state_indices(margins, minimum=-3, maximum=3)
    np.testing.assert_allclose(
        conditional - neutral, fit.state_coefficients[indices], atol=1e-15
    )


def test_joint_clock_margin_fit_keeps_points_target_and_context_decomposition() -> None:
    frame = _frame()
    frame["six_minute_bucket"] = np.arange(len(frame)) % 8
    frame["offense_margin_before"] = (np.arange(len(frame)) % 31) - 15
    frame["regulation"] = True
    frame["pts"] = (
        1.1
        - 0.03
        * frame["offense_margin_before"]
        * (frame["six_minute_bucket"] + 1)
        / 8
    )
    design = build_design(frame)
    config = RapmConfig(seasons=(2025,), lambda_off=10, lambda_def=10)
    fit = fit_clock_margin_rapm(
        design,
        frame,
        config,
        time_buckets=8,
        margin_clip=15,
        context_penalty=0.1,
    )
    neutral = predict_clock_margin_rapm(
        fit, design, frame, include_context=False
    )
    conditional = predict_clock_margin_rapm(fit, design, frame)
    context = clock_margin_design(frame, time_buckets=8, margin_clip=15)
    np.testing.assert_allclose(
        conditional - neutral,
        np.asarray(context @ fit.context_coefficients).ravel() - fit.context_mean,
    )
    curve = clock_margin_curve(fit)
    assert np.allclose(
        curve.loc[curve["offense_margin_before"].eq(0), "effect_points_per_100_vs_tie"],
        0.0,
    )
