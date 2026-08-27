from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "research/run_pipm_breaker.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_pipm_breaker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_starter_share_is_starts_over_games_squared_and_future_safe() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 6,
            "Season": list(range(2019, 2025)),
            "MIN": [100.0] * 6,
            "GP": [10.0] * 6,
            "games": [10.0] * 6,
            "games_started": [5.0] * 5 + [10.0],
            "OffPoss": [200.0] * 6,
            "position_adjusted_oreb": [0.01] * 5 + [0.90],
        }
    )
    before = MODULE._rolling_context(annual, range(2023, 2024)).iloc[0]
    changed = annual.copy()
    changed.loc[changed["Season"].eq(2024), "games_started"] = 0.0
    changed.loc[changed["Season"].eq(2024), "position_adjusted_oreb"] = -0.90
    after = MODULE._rolling_context(changed, range(2023, 2024)).iloc[0]

    assert before["starter_share_squared_5y"] == 0.25
    assert before["starter_share_squared_5y"] == after["starter_share_squared_5y"]
    assert before["position_adjusted_oreb_5y"] == after["position_adjusted_oreb_5y"]


def test_spacing_value_uses_volume_times_shot_value_above_league_efg() -> None:
    panel = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Window_End": [2023, 2023],
            "courtsignal_exposure": [100.0, 100.0],
            "FG2A_p100": [50.0, 50.0],
            "FG2M_p100": [25.0, 25.0],
            "FG3A_p100": [10.0, 10.0],
            "FG3M_p100": [4.0, 2.0],
            "fg3_pct_eb": [0.40, 0.20],
        }
    )
    result = MODULE._add_spacing(panel)
    expected_efg = (25.0 + 25.0 + 1.5 * (4.0 + 2.0)) / 120.0

    assert np.allclose(result["league_efg"], expected_efg)
    assert np.isclose(
        result.loc[0, "spacing_value_above_average_p100"],
        10.0 * (1.5 * 0.40 - expected_efg),
    )


def test_event_and_lineup_parsers_are_deterministic() -> None:
    assert MODULE._lineup("3|1|2|nan") == (1, 2, 3)
    assert MODULE._free_throws_made("Free Throw 1 of 2\nMISS Free Throw 2 of 2") == 1
    assert {spec.family for spec in MODULE._model_grid()} == {
        "ridge",
        "elastic_net",
        "histogram_gbm",
    }
