from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.role_stabilization import _smooth_assignments


def test_role_stabilization_is_forward_only_and_resets_after_gap() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2],
            "Season": [2020, 2021, 2023, 2021],
            "off_role_cluster": ["off_role_0", "off_role_1", "off_role_1", "off_role_0"],
            "off_role_affinity_0": [0.8, 0.2, 0.1, 0.9],
            "off_role_affinity_1": [0.2, 0.8, 0.9, 0.1],
        }
    )
    output, prediction = _smooth_assignments(frame, "off_role", 0.7)
    row_2021 = output.loc[
        output["PLAYER_ID"].eq(1) & output["Season"].eq(2021)
    ].iloc[0]
    np.testing.assert_allclose(
        [row_2021["off_role_stable_affinity_0"], row_2021["off_role_stable_affinity_1"]],
        [0.38, 0.62],
    )
    row_2023 = output.loc[
        output["PLAYER_ID"].eq(1) & output["Season"].eq(2023)
    ].iloc[0]
    np.testing.assert_allclose(
        [row_2023["off_role_stable_affinity_0"], row_2023["off_role_stable_affinity_1"]],
        [0.1, 0.9],
    )
    assert prediction[["PLAYER_ID", "Season"]].to_dict(orient="records") == [
        {"PLAYER_ID": 1, "Season": 2021}
    ]
