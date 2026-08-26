from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.hand_selected_sparse_spm import (
    FEATURE_SPECS,
    _pool_metric,
    hand_selected_features,
)


def test_contract_contains_exactly_twelve_unique_features() -> None:
    names = hand_selected_features()
    assert len(names["offense"]) == 8
    assert len(names["defense"]) == 4
    assert len(set((*names["offense"], *names["defense"]))) == 12
    assert sum(len(specs) for specs in FEATURE_SPECS.values()) == 12


def test_five_year_pool_never_uses_future_seasons() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1],
            "Season": [2019, 2020, 2021],
            "metric": [1.0, 3.0, 1000.0],
            "weight": [1.0, 3.0, 100.0],
        }
    )
    pooled = _pool_metric(
        annual,
        metric="metric",
        weight="weight",
        window_ends=(2020,),
    )
    assert np.isclose(pooled.iloc[0]["metric"], 2.5)
    assert pooled.iloc[0]["source_seasons"] == 2
