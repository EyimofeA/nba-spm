from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_full_spm_history_ablation import (  # noqa: E402
    MODEL_ORDER,
    _load_contract,
    _select_box_alpha_rolling_origin,
    paired_game_bootstrap,
)


def test_v2_contract_removes_late_source_fields_and_availability_flags() -> None:
    contract = _load_contract(
        ROOT / "research/experiments/full_spm_history_ablation_v2.yml"
    )
    removed = contract["removed_defense_features"]
    assert len(removed) == 17
    assert len(set(removed)) == len(removed)
    assert {
        "has_hustle_tracking",
        "has_matchup_tracking",
        "has_dfg_tracking",
        "has_rim_defense_tracking",
    }.issubset(removed)
    assert all("matchup_" in field or field in {
        "deflections_p100",
        "charges_drawn_p100",
        "contested_2pt_p100",
        "contested_3pt_p100",
        "def_loose_balls_recovered_p100",
        "has_hustle_tracking",
        "has_matchup_tracking",
        "has_dfg_tracking",
        "has_rim_defense_tracking",
    } for field in removed)


def test_box_alpha_selection_uses_only_earlier_window_ends(monkeypatch) -> None:
    import run_full_spm_history_ablation as module

    train = pd.DataFrame(
        {
            "Window_End": [2018, 2019, 2020],
            "feature": [1.0, 2.0, 3.0],
            "target": [0.0, 0.0, 0.0],
            "sample_weight": [1.0, 1.0, 1.0],
        }
    )
    calls = []

    class Model:
        def __init__(self, train_end: int):
            self.train_end = train_end

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            calls.append((self.train_end, int(frame["feature"].iloc[0] + 2017)))
            return np.zeros(len(frame))

    def fake_fit(frame, features, target, alpha):
        return Model(int(frame["Window_End"].max()))

    monkeypatch.setattr(module, "_fit_box", fake_fit)
    _select_box_alpha_rolling_origin(
        train, ("feature",), "target", (10.0, 100.0)
    )

    assert calls
    assert all(train_end < validation_end for train_end, validation_end in calls)


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
