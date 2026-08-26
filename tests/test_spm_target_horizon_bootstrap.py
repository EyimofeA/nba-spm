"""Paired bootstrap checks for the target-horizon comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "research/analyze_spm_target_horizon_full.py"
SPEC = importlib.util.spec_from_file_location("analyze_spm_target_horizon_full", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
paired_rmse_bootstrap = MODULE.paired_rmse_bootstrap


def test_paired_bootstrap_detects_better_challenger() -> None:
    rows = []
    for season in (2020, 2021):
        for game in range(20):
            actual = float(game % 5)
            for horizon, candidate, offset in (
                ("5y", "zero_prior", 2.0),
                ("1y", "spm_centered_aio", 0.5),
            ):
                rows.append(
                    {
                        "test_season": season,
                        "game_id": f"{season}_{game}",
                        "horizon": horizon,
                        "candidate": candidate,
                        "actual_margin": actual,
                        "predicted_margin": actual + offset,
                    }
                )

    result = paired_rmse_bootstrap(
        pd.DataFrame(rows), selected="5y__zero_prior", repetitions=200, seed=7
    ).iloc[0]

    assert result["challenger"] == "1y__spm_centered_aio"
    assert result["probability_challenger_better"] == 1.0
    assert result["fold_wins_challenger"] == 2


def test_paired_bootstrap_rejects_mismatched_games() -> None:
    frame = pd.DataFrame(
        {
            "test_season": [2020, 2020, 2020],
            "game_id": ["a", "b", "a"],
            "horizon": ["5y", "5y", "1y"],
            "candidate": ["zero_prior", "zero_prior", "spm_centered_aio"],
            "actual_margin": [0.0, 0.0, 0.0],
            "predicted_margin": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="same games"):
        paired_rmse_bootstrap(frame, selected="5y__zero_prior", repetitions=10)
