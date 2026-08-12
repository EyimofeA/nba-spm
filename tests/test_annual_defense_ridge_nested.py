import pandas as pd

from nba_impact.models.annual_defense_ridge_nested import _select_alpha


def test_select_alpha_uses_mean_rmse_and_stronger_tie_break() -> None:
    metrics = pd.DataFrame(
        {
            "alpha": [300.0, 300.0, 1000.0, 1000.0, 3000.0, 3000.0],
            "weighted_rmse": [1.0, 1.2, 1.05, 1.05, 1.1, 1.0],
        }
    )
    assert _select_alpha(metrics) == 3000.0
