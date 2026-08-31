import json

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from nba_impact.models.spm_precision_weights import (
    analytic_ridge_label_variance,
    bounded_inverse_variance_weights,
    recentered_variance_diagonal,
)


def test_recentered_variance_matches_linear_transform() -> None:
    covariance = np.asarray([[2.0, 0.5], [0.5, 1.0]])
    weights = np.asarray([1.0, 3.0])
    transform = np.eye(2) - np.ones((2, 1)) @ (weights / weights.sum())[None, :]
    expected = np.diag(transform @ covariance @ transform.T)
    assert np.allclose(recentered_variance_diagonal(covariance, weights), expected)


def test_bounded_weights_are_finite_and_mean_one() -> None:
    import pandas as pd

    weights = bounded_inverse_variance_weights(pd.Series([0.1, 0.2, 1.0, 100.0]))
    assert np.isfinite(weights).all()
    assert np.isclose(weights.mean(), 1.0)
    assert weights.iloc[0] > weights.iloc[-1]


def test_analytic_variance_reads_stored_matrix_contract(tmp_path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    xtx = csr_matrix(np.diag([8.0, 6.0, 7.0, 5.0, 10.0]))
    save_npz(root / "train_xtx.npz", xtx)
    np.save(root / "train_xty_centered.npy", np.asarray([1.0, 2.0, -1.0, 1.0, 0.5]))
    np.save(root / "player_ids.npy", np.asarray([11, 22]))
    np.save(root / "train_off_possessions.npy", np.asarray([8.0, 6.0]))
    np.save(root / "train_def_possessions.npy", np.asarray([7.0, 5.0]))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "train": {"centered_y_sum_squares": 100.0, "possession_rows": 100},
                "metadata": {"window_end": 2024},
            }
        )
    )
    result = analytic_ridge_label_variance(
        root, lambda_offense=3.0, lambda_defense=3.0, lambda_home=1.0
    )
    assert result["PLAYER_ID"].tolist() == [11, 22]
    assert result["Window_End"].eq(2024).all()
    assert (result[["label_variance_offense", "label_variance_defense"]] > 0).all().all()
