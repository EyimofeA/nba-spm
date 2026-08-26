"""Contract checks for the matched-window target-horizon pilot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


MODULE_PATH = Path(__file__).parents[1] / "research/run_spm_target_horizon_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_spm_target_horizon_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DEFAULT_CONTRACT = MODULE.DEFAULT_CONTRACT
_load_contract = MODULE._load_contract


def test_repository_pilot_contract_is_frozen_and_pre_2025() -> None:
    contract = _load_contract(DEFAULT_CONTRACT)

    assert tuple(contract["test_seasons"]) == (2023, 2024)
    assert set(contract["horizons"]) == {"1y", "5y"}


def test_pilot_contract_rejects_2027_before_data_access(tmp_path) -> None:
    contract = yaml.safe_load(DEFAULT_CONTRACT.read_text())
    contract["test_seasons"] = [2023, 2027]
    path = tmp_path / "contract.yml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False))

    with pytest.raises(ValueError, match="stop at 2024"):
        _load_contract(path)
