import pandas as pd

from nba_impact.models.annual_defense_features_nested import _select_variant


def test_select_variant_uses_rmse_then_smaller_block_then_name() -> None:
    metrics = pd.DataFrame(
        {
            "variant": ["large", "large", "z_small", "z_small", "a_small", "a_small"],
            "added_features": [5, 5, 2, 2, 2, 2],
            "weighted_rmse": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    assert _select_variant(metrics) == "a_small"
