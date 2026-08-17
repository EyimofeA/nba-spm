from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.side_roles import RoleMapConfig, _fit_role_map


def test_side_role_map_selects_stable_count_and_namespaces_features() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for season in range(2014, 2020):
        for cluster in range(3):
            for player in range(30):
                rows.append(
                    {
                        "PLAYER_ID": 1000 * cluster + player,
                        "Season": season,
                        "usage_a": cluster * 4.0 + rng.normal(scale=0.15),
                        "usage_b": (2 - cluster) * 3.0 + rng.normal(scale=0.15),
                        "usage_c": cluster + rng.normal(scale=0.15),
                    }
                )
    result = _fit_role_map(
        pd.DataFrame(rows),
        ("usage_a", "usage_b", "usage_c"),
        RoleMapConfig(
            prefix="off_role",
            development_seasons=(2014, 2015, 2016),
            n_axes=2,
            candidate_clusters=(3, 4),
            minimum_cluster_share=0.01,
            maximum_cluster_share=0.50,
        ),
    )
    assignments = result["assignments"]

    assert result["metrics"]["selected_clusters"] == 3
    assert result["metrics"]["median_seed_adjusted_rand"] >= 0.90
    assert {"off_role_axis_1", "off_role_axis_2", "off_role_affinity_0"}.issubset(
        assignments.columns
    )
    assert assignments["off_role_cluster"].str.startswith("off_role_").all()
    affinity_columns = [
        column for column in assignments if column.startswith("off_role_affinity_")
    ]
    assert np.allclose(assignments[affinity_columns].sum(axis=1), 1.0)
    assert "off_role_affinity_2" not in result["model_features"]
