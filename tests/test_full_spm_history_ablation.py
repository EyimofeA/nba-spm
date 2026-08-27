from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_full_spm_history_ablation import (  # noqa: E402
    MODEL_ORDER,
    _load_contract,
    paired_game_bootstrap,
)


def test_contract_freezes_thirteen_history_limited_defense_fields() -> None:
    contract = _load_contract(
        ROOT / "research/experiments/full_spm_history_ablation_v1.yml"
    )
    removed = contract["removed_defense_features"]
    assert len(removed) == 13
    assert len(set(removed)) == len(removed)
    assert all("matchup_" in field or field in {
        "deflections_p100",
        "charges_drawn_p100",
        "contested_2pt_p100",
        "contested_3pt_p100",
        "def_loose_balls_recovered_p100",
    } for field in removed)


def test_paired_bootstrap_uses_identical_games() -> None:
    rows = []
    for season in (2022, 2023):
        for game_id, actual in ((1, 3.0), (2, -2.0)):
            for index, candidate in enumerate(MODEL_ORDER):
                rows.append(
                    {
                        "test_season": season,
                        "game_id": game_id,
                        "candidate": candidate,
                        "actual_margin": actual,
                        "predicted_margin": actual + 0.1 * index,
                    }
                )
    games = pd.DataFrame(rows)
    models, pairs = paired_game_bootstrap(games, draws=50, seed=7)
    assert set(models["candidate"]) == set(MODEL_ORDER)
    assert np.isfinite(models["equal_season_rmse"]).all()
    assert len(pairs) == len(MODEL_ORDER) * (len(MODEL_ORDER) - 1) // 2
