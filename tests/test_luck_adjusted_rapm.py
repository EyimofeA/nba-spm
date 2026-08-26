from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.luck_adjusted_rapm import (
    ARMS,
    build_expected_outcome_frame,
    compose_arm_beta,
    history_rate_table,
    load_contract,
)


def test_history_rate_never_uses_target_or_future_season() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2, 2],
            "season": [2022, 2023, 2024, 2022, 2023, 2024],
            "FTM": [8, 9, 0, 6, 7, 10_000],
            "FTA": [10, 10, 1, 10, 10, 10_000],
        }
    )
    first, league_first = history_rate_table(
        annual, category="ft", target_season=2024, prior_attempts=10, half_life=2
    )
    annual.loc[annual["season"].eq(2024), ["FTM", "FTA"]] = [0, 50_000]
    second, league_second = history_rate_table(
        annual, category="ft", target_season=2024, prior_attempts=10, half_life=2
    )
    pd.testing.assert_frame_equal(first, second)
    assert league_first == league_second


def test_expected_target_replaces_only_mapped_conversion_points() -> None:
    base = pd.DataFrame({"possession_id": ["a", "b"], "pts": [3.0, 2.0]})
    ledger = pd.DataFrame(
        {
            "possession_id": ["a"],
            "actual_points": [3.0],
            "neutral_expected_points": [1.2],
            "skill_expected_points": [1.5],
        }
    )
    output = build_expected_outcome_frame(base, ledger)
    assert output["expected_pts"].tolist() == pytest.approx([1.2, 2.0])


def test_arm_composition_keeps_components_separate() -> None:
    normal = np.array([1.0, 2.0, 10.0, 20.0, 3.0])
    expected = np.array([4.0, 5.0, 40.0, 50.0, 6.0])
    bonus = np.array([0.25, -0.25])
    results = {
        arm: compose_arm_beta(
            normal, expected, n_players=2, arm=arm, shooting_bonus=bonus
        )
        for arm in ARMS
    }
    assert results["normal_realized_points"].tolist() == normal.tolist()
    assert results["opponent_luck_adjusted"].tolist() == [1, 2, 40, 50, 3]
    assert results["teammate_and_opponent_luck_adjusted"].tolist() == [4.25, 4.75, 40, 50, 6]
    assert results["full_expected_outcome"].tolist() == expected.tolist()


def test_contract_rejects_2027_before_data_access(tmp_path) -> None:
    source = "research/experiments/luck_adjusted_rapm_spm_v1.yml"
    text = open(source).read().replace("output_season: 2026", "output_season: 2027")
    path = tmp_path / "bad.yml"
    path.write_text(text)
    with pytest.raises(ValueError, match="2027"):
        load_contract(path)
