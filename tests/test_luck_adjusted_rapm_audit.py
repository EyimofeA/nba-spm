from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "research/analyze_luck_adjusted_rapm.py"
SPEC = importlib.util.spec_from_file_location("luck_adjusted_rapm_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_paired_bootstrap = MODULE._paired_bootstrap
_weighted_corr = MODULE._weighted_corr


def test_weighted_corr_respects_weights() -> None:
    actual = np.array([0.0, 1.0, 2.0])
    predicted = np.array([0.0, 1.0, -5.0])
    assert _weighted_corr(actual, predicted, np.array([100.0, 100.0, 0.01])) > 0.9


def test_paired_bootstrap_uses_identical_games() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["a", "b", "a", "b"],
            "actual_margin": [1.0, -2.0, 1.0, -2.0],
            "predicted_margin": [0.0, 0.0, 1.0, -2.0],
            "arm": [
                "normal_realized_points",
                "normal_realized_points",
                "challenger",
                "challenger",
            ],
            "test_season": [2025, 2025, 2025, 2025],
        }
    )
    result = _paired_bootstrap(games, challenger="challenger", test_season=2025, draws=100)
    assert result["rmse_delta"] < 0
    assert result["probability_better"] == 1.0
