import numpy as np
import pytest

from nba_impact.models.age_adjusted_rapm import build_age_design
from nba_impact.models.rapm import RapmConfig, build_design
from nba_impact.models.score_context_rapm import (
    clipped_linear_score_design,
    fit_context_rapm,
    predict_context_rapm,
    signed_score_bucket_design,
    spline_age_design,
    spline_score_design,
)


def _frame():
    import pandas as pd

    rows = []
    margins = np.tile(np.arange(-25, 26), 8)
    for index, margin in enumerate(margins):
        rows.append(
            {
                "gameid": str(index // 20),
                "season": 2025,
                "home_poss": index % 2,
                "pts": 1.1 - 0.01 * margin,
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            }
        )
    return pd.DataFrame(rows), margins.astype(float)


def test_signed_score_buckets_keep_tie_as_reference() -> None:
    margins = np.asarray([-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30])
    matrix, labels = signed_score_bucket_design(margins)
    assert matrix.shape == (11, 10)
    assert matrix[5].nnz == 0
    assert labels[0] == "trailing_1_5"
    assert labels[-1] == "leading_21_plus"
    assert np.asarray(matrix.sum(axis=1)).ravel().tolist() == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    assert matrix[4].nonzero()[1].tolist() == [0]
    assert matrix[6].nonzero()[1].tolist() == [5]


def test_signed_score_buckets_reject_fractional_margins() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        signed_score_bucket_design(np.asarray([-0.5, 0.0, 0.5]))


def test_spline_knots_use_training_rows_and_tie_is_zero() -> None:
    margins = np.arange(-30, 31, dtype=float)
    train = margins <= 20
    matrix, labels = spline_score_design(
        margins, train_mask=train, clip=30, n_knots=4
    )
    assert matrix.shape[1] == len(labels)
    np.testing.assert_allclose(matrix[margins == 0].toarray(), 0.0, atol=1e-12)


def test_generic_context_fit_preserves_neutral_conditional_decomposition() -> None:
    frame, margins = _frame()
    design = build_design(frame)
    context, _ = clipped_linear_score_design(margins, clip=20)
    fit = fit_context_rapm(
        design,
        context,
        RapmConfig(seasons=(2025,), lambda_off=10, lambda_def=10),
        context_penalty=1,
        row_mask=np.ones(len(frame), dtype=bool),
    )
    mask = np.ones(len(frame), dtype=bool)
    neutral = predict_context_rapm(
        fit, design, context, row_mask=mask, include_context=False
    )
    conditional = predict_context_rapm(
        fit, design, context, row_mask=mask, include_context=True
    )
    np.testing.assert_allclose(
        conditional - neutral,
        np.asarray(context @ fit.context_coefficients).ravel() - fit.context_mean,
        atol=1e-15,
    )


def test_age_spline_uses_reference_age_as_zero() -> None:
    import pandas as pd

    frame, _ = _frame()
    players = sorted(
        set(frame[[f"a{slot}" for slot in range(1, 6)]].to_numpy().ravel())
        | set(frame[[f"h{slot}" for slot in range(1, 6)]].to_numpy().ravel())
    )
    ages = pd.DataFrame(
        {"PLAYER_ID": players, "Season": 2025, "AGE": 27}
    )
    categorical = build_age_design(frame, ages)
    smooth, labels = spline_age_design(categorical, n_knots=4)
    assert smooth.shape[1] == len(labels)
    np.testing.assert_allclose(smooth.toarray(), 0.0, atol=1e-12)
