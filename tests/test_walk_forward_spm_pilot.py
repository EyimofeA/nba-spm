import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import run_walk_forward_spm_pilot as pilot  # noqa: E402


def test_pilot_uses_one_strictly_future_season_and_four_requested_models() -> None:
    contract = yaml.safe_load(pilot.CONTRACT.read_text())

    assert contract["information_cutoff"]["rating_season"] == 2024
    assert contract["information_cutoff"]["test_season"] == 2025
    assert set(pilot.MODEL_ORDER) >= {
        "full_spm",
        "box_pipm",
        "full_spm_aio",
        "box_pipm_aio",
    }
    assert contract["evaluation"]["identical_games"] == "required"
