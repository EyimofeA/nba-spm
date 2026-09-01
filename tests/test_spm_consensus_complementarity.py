from __future__ import annotations

import numpy as np
import pandas as pd

from research.run_spm_consensus_complementarity import (
    _permuted_within_season,
    _player_bootstrap,
    cross_reference_correlations,
    recurrence_threshold,
    stable_game_partition,
)


def test_game_partition_is_deterministic_and_disjoint() -> None:
    games = np.asarray(["a", "b", "c", "d", "e"])
    first = stable_game_partition(games)
    second = stable_game_partition(games[::-1])[::-1]
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1}


def test_recurrence_threshold_respects_null_and_minimum() -> None:
    assert recurrence_threshold(0.6, np.asarray([0.1, 0.2]), 0.95) == 0.6
    assert recurrence_threshold(0.6, np.asarray([0.7, 0.9]), 0.5) == 0.8


def test_player_bootstrap_keeps_player_histories_together() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 2, 2],
            "Season": [2014, 2015, 2014, 2015],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    sample = _player_bootstrap(frame, np.random.default_rng(3))
    counts = sample.groupby("PLAYER_ID")["Season"].value_counts().unstack(fill_value=0)
    assert (counts[2014] == counts[2015]).all()


def test_target_permutation_stays_within_season() -> None:
    frame = pd.DataFrame(
        {
            "Season": [2014, 2014, 2015, 2015],
            "target": [1.0, 2.0, 10.0, 20.0],
        }
    )
    permuted = _permuted_within_season(frame, "target", np.random.default_rng(1))
    assert set(permuted[:2]) == {1.0, 2.0}
    assert set(permuted[2:]) == {10.0, 20.0}


def test_cross_reference_uses_opposite_reference_halves() -> None:
    players = np.arange(1, 7)
    priors = pd.concat(
        [
            pd.DataFrame(
                {
                    "PLAYER_ID": players,
                    "rating_season": 2020,
                    "design": "current_control",
                    "candidate": candidate,
                    "prior_offense": players + shift,
                    "prior_defense": players - shift,
                }
            )
            for candidate, shift in (("box15", 0.0), ("rich", 0.5))
        ],
        ignore_index=True,
    )
    annual = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "rating_season": 2020,
            "reference": "one_year",
            "target_offense": players * 0.8,
            "target_defense": players * 0.7,
            "target_net": players * 1.5,
            "Poss_Off": 1000,
            "Poss_Def": 1000,
        }
    )
    split = pd.concat(
        [
            pd.DataFrame(
                {
                    "PLAYER_ID": players,
                    "rating_season": 2020,
                    "split": split_id,
                    "target_offense": players * scale,
                    "target_defense": players * (scale - 0.1),
                    "Poss_Off": 500,
                    "Poss_Def": 500,
                }
            )
            for split_id, scale in ((0, 0.6), (1, 0.9))
        ],
        ignore_index=True,
    )
    result = cross_reference_correlations(priors, annual, split)
    assert len(result) == 4
    assert result[["correlation_a_b", "correlation_b_a"]].notna().all().all()
