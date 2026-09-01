from __future__ import annotations

import pandas as pd

from research.run_spm_final_prior_stack import combine_prior_deltas


def test_combine_prior_deltas_adds_each_increment_once() -> None:
    box = pd.DataFrame({"PLAYER_ID": [1], "prior_offense": [2.0], "prior_defense": [1.0]})
    consensus = pd.DataFrame({"PLAYER_ID": [1], "prior_offense": [4.0], "prior_defense": [3.0]})
    residual = pd.DataFrame({"PLAYER_ID": [1], "prior_defense": [5.0]})

    result = combine_prior_deltas(
        box,
        consensus,
        residual,
        consensus_offense_weight=0.5,
        consensus_defense_weight=0.25,
        defense_residual_weight=0.5,
    ).iloc[0]

    assert result["prior_offense"] == 3.0
    assert result["prior_defense"] == 3.5
    assert result["prior_net"] == 6.5


def test_zero_consensus_and_full_residual_reproduces_residual_prior() -> None:
    box = pd.DataFrame({"PLAYER_ID": [1, 2], "prior_offense": [2.0, -1.0], "prior_defense": [1.0, 0.0]})
    consensus = pd.DataFrame({"PLAYER_ID": [1, 2], "prior_offense": [4.0, 2.0], "prior_defense": [3.0, -2.0]})
    residual = pd.DataFrame({"PLAYER_ID": [1, 2], "prior_defense": [5.0, -1.0]})

    result = combine_prior_deltas(
        box,
        consensus,
        residual,
        consensus_offense_weight=0.0,
        consensus_defense_weight=0.0,
        defense_residual_weight=1.0,
    )

    assert result["prior_offense"].tolist() == box["prior_offense"].tolist()
    assert result["prior_defense"].tolist() == residual["prior_defense"].tolist()
