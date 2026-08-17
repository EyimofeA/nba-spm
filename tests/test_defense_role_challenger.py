from __future__ import annotations

import pandas as pd

from nba_impact.models.defense_role_challenger import _select_variant


def test_defense_challenger_selection_uses_mean_rmse_then_fewer_features() -> None:
    metrics = pd.DataFrame(
        [
            {"variant": "baseline", "added_features": 0, "weighted_rmse": 1.00},
            {"variant": "baseline", "added_features": 0, "weighted_rmse": 1.10},
            {"variant": "large", "added_features": 10, "weighted_rmse": 0.90},
            {"variant": "large", "added_features": 10, "weighted_rmse": 1.00},
            {"variant": "small", "added_features": 2, "weighted_rmse": 0.95},
            {"variant": "small", "added_features": 2, "weighted_rmse": 0.95},
        ]
    )
    assert _select_variant(metrics) == "small"
