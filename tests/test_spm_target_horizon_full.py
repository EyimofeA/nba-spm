"""Contract and window checks for the full target-horizon comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


MODULE_PATH = Path(__file__).parents[1] / "research/run_spm_target_horizon_full.py"
SPEC = importlib.util.spec_from_file_location("run_spm_target_horizon_full", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DEFAULT_CONTRACT = MODULE.DEFAULT_CONTRACT
_load_contract = MODULE._load_contract
_window_seasons = MODULE._window_seasons


def test_full_contract_stops_before_reused_diagnostics() -> None:
    contract = _load_contract(DEFAULT_CONTRACT)

    assert tuple(contract["test_seasons"]) == (2020, 2021, 2022, 2023, 2024)
    assert max(contract["window_ends"]) == 2023


def test_rolling_and_expanding_windows_are_distinct() -> None:
    contract = _load_contract(DEFAULT_CONTRACT)

    assert _window_seasons(contract["horizons"]["6y"], 2014) == tuple(
        range(2009, 2015)
    )
    assert _window_seasons(contract["horizons"]["expanding"], 2014) == (2014,)
    assert _window_seasons(contract["horizons"]["expanding"], 2017) == (
        2014,
        2015,
        2016,
        2017,
    )


def test_full_contract_rejects_post_2024_development(tmp_path) -> None:
    contract = yaml.safe_load(DEFAULT_CONTRACT.read_text())
    contract["test_seasons"].append(2025)
    path = tmp_path / "contract.yml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False))

    with pytest.raises(ValueError, match="stop before Season 2025"):
        _load_contract(path)
