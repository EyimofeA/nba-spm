from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.run_spm_stabilization_ablation import (
    ARMS,
    MODEL_ORDER,
    _arm_features,
    _load_contract,
    _ratio_recovery,
    paired_game_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/experiments/spm_stabilization_ablation_v1.yml"
MANIFEST = ROOT / (
    "artifacts/research/full_feature_2014_2026/panels/"
    "full_spm_features_2014_2026_v1_4c77ae6acc/run.json"
)


def test_feature_arms_have_equal_counts_and_one_value_per_concept() -> None:
    manifest = json.loads(MANIFEST.read_text())
    selected = {
        side: tuple(manifest["feature_contract"][side])
        for side in ("offense", "defense")
    }
    contract = yaml.safe_load(CONTRACT.read_text())
    arms, pairs = _arm_features(selected, contract)

    assert set(arms) == set(ARMS)
    assert pairs.groupby("component").size().to_dict() == {
        "defense": 10,
        "offense": 37,
    }
    for side in ("offense", "defense"):
        assert len(arms["raw_spm"][side]) == len(arms["stabilized_spm"][side])
        assert len(set(arms["raw_spm"][side])) == len(arms["raw_spm"][side])
        assert len(set(arms["stabilized_spm"][side])) == len(
            arms["stabilized_spm"][side]
        )


def test_ratio_recovery_inverts_empirical_bayes_reliability() -> None:
    raw = pd.Series([2.0, -3.0, 0.5])
    exposure = pd.Series([100.0, 500.0, 900.0])
    prior = 200.0
    stabilized = raw * exposure / (exposure + prior)
    recovered = _ratio_recovery(stabilized, exposure, prior)
    np.testing.assert_allclose(recovered, raw)


def test_contract_forbids_2027(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    contract["information_cutoff"]["season_2027"] = "allowed"
    path = tmp_path / "contract.yml"
    path.write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match="Season 2027"):
        _load_contract(path)


def test_paired_bootstrap_requires_identical_complete_games() -> None:
    rows = []
    for candidate_index, candidate in enumerate(MODEL_ORDER):
        for game_id, actual in (("a", 2.0), ("b", -1.0)):
            rows.append(
                {
                    "test_season": 2022,
                    "game_id": game_id,
                    "candidate": candidate,
                    "actual_margin": actual,
                    "predicted_margin": actual + candidate_index * 0.1,
                }
            )
    games = pd.DataFrame(rows)
    models, pairs = paired_game_bootstrap(games, draws=20, seed=7)
    assert set(models["candidate"]) == set(MODEL_ORDER)
    assert pairs["primary_comparison"].sum() == 2

    broken = games.loc[
        ~(
            games["candidate"].eq("raw_spm")
            & games["game_id"].eq("a")
        )
    ]
    with pytest.raises(ValueError, match="identical complete outcomes|missed a scored game"):
        paired_game_bootstrap(broken, draws=5, seed=7)
