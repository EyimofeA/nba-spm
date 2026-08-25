import numpy as np
import pandas as pd

from nba_impact.models.aging_resolution import (
    build_annual_transitions,
    evaluate_aging_resolutions,
    kernel_age_change,
)


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    age_rows = []
    for season in range(2014, 2022):
        for player_id, base_age in ((1, 20), (2, 25), (3, 30), (4, 35)):
            age = base_age + season - 2014
            offense = 5.0 - 0.2 * (age - 27.0) ** 2
            defense = 1.0 - 0.02 * (age - 29.0) ** 2
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Season": season,
                    "target_offense": offense,
                    "target_defense": defense,
                    "target_net": offense + defense,
                    "Poss_Off": 1000 + player_id,
                    "Poss_Def": 1000 + player_id,
                }
            )
            age_rows.append({"PLAYER_ID": player_id, "Season": season, "AGE": age})
    return pd.DataFrame(target_rows), pd.DataFrame(age_rows)


def test_kernel_age_change_recovers_local_values() -> None:
    predicted = kernel_age_change(
        np.array([20.0, 25.0, 30.0]),
        np.array([1.0, 0.0, -1.0]),
        np.ones(3),
        np.array([20.0, 30.0]),
        bandwidth=0.1,
    )
    np.testing.assert_allclose(predicted, [1.0, -1.0], atol=1e-10)


def test_transition_builder_uses_adjacent_seasons() -> None:
    targets, ages = _panel()
    transitions = build_annual_transitions(targets, ages)
    assert set(transitions["Season"]) == set(range(2014, 2021))
    row = transitions.loc[
        transitions["PLAYER_ID"].eq(1) & transitions["Season"].eq(2014)
    ].iloc[0]
    assert row["next_net"] == targets.loc[
        targets["PLAYER_ID"].eq(1) & targets["Season"].eq(2015), "target_net"
    ].iloc[0]


def test_walk_forward_resolution_has_no_future_target() -> None:
    targets, ages = _panel()
    result = evaluate_aging_resolutions(
        targets,
        ages,
        bandwidths=(0.5, 1.0),
        trailing_windows=(1, 3),
        minimum_training_origins=2,
    )
    assert not result.metrics.empty
    assert (result.metrics["target_season"] == result.metrics["origin_season"] + 1).all()
    assert result.predictions["target_season"].max() == 2021
    assert result.quality["subannual_age_resolution_supported"] is False
