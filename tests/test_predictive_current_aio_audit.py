from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1] / "research/analyze_predictive_current_aio.py"
)
SPEC = importlib.util.spec_from_file_location("predictive_current_aio_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {SCRIPT}.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
paired_bootstrap_rows = MODULE.paired_bootstrap_rows


def test_paired_bootstrap_uses_same_games_and_detects_better_arm() -> None:
    rows = []
    for season in (2020, 2021):
        for game in range(20):
            rows.extend(
                [
                    {
                        "test_season": season,
                        "game_id": f"{season}-{game}",
                        "arm": "selected",
                        "squared_error": 1.0,
                    },
                    {
                        "test_season": season,
                        "game_id": f"{season}-{game}",
                        "arm": "baseline",
                        "squared_error": 4.0,
                    },
                ]
            )
    output = paired_bootstrap_rows(
        pd.DataFrame(rows),
        selected="selected",
        comparators=("baseline",),
        scopes={"development": (2020, 2021)},
        draws=100,
        seed=7,
    )
    assert output.iloc[0]["mse_delta_selected_minus_comparator"] == -3.0
    assert output.iloc[0]["probability_selected_better"] == 1.0
