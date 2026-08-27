from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "research/run_rim_assist_spm_challenger.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "run_rim_assist_spm_challenger", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_annual_rim_assist_center_uses_only_same_season(tmp_path: Path) -> None:
    for season, assists in ((2020, [1.0, 3.0]), (2021, [100.0, 100.0])):
        pd.DataFrame(
            {
                "PLAYER_ID": [1, 2],
                "AtRimAssists": assists,
                "OffPoss": [100.0, 100.0],
            }
        ).to_parquet(tmp_path / f"{season}.parquet", index=False)

    annual, coverage = MODULE.annual_rim_assists(tmp_path, range(2020, 2022))
    first = annual.loc[
        annual["Season"].eq(2020) & annual["PLAYER_ID"].eq(1),
        "rim_assists_p100_eb",
    ].item()
    expected = 100.0 * (1.0 + 500.0 * (4.0 / 200.0)) / 600.0

    assert first == pytest.approx(expected)
    assert coverage.loc[coverage["Season"].eq(2020), "rim_assists"].item() == 4.0


def test_pool_uses_only_the_explicit_five_year_window() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 6,
            "Season": list(range(2014, 2020)),
            "OffPoss": [100.0] * 6,
            "rim_assists_p100_eb": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )

    pooled = MODULE.pool_rim_assists(annual, range(2018, 2019))

    assert pooled[MODULE.RIM_ASSIST_FEATURE].item() == pytest.approx(2.0)


def test_missing_players_receive_same_window_neutral_value() -> None:
    panel = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Window_End": [2023, 2023, 2023],
            "courtsignal_exposure": [1000.0, 1000.0, 10.0],
            MODULE.RIM_ASSIST_FEATURE: [1.0, 3.0, None],
        }
    )

    filled, coverage = MODULE.fill_unobserved_rim_assists(panel)

    assert filled.loc[filled["PLAYER_ID"].eq(3), MODULE.RIM_ASSIST_FEATURE].item() == 2.0
    assert coverage["observed_possession_rate"] == pytest.approx(2000.0 / 2010.0)
