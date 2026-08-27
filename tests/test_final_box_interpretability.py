import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_final_box_interpretability import (  # noqa: E402
    BOX_GROUPS,
    _validate_groups,
)


def test_box_groups_cover_each_side_once():
    features = {
        side: tuple(
            feature
            for group in BOX_GROUPS.values()
            for feature in group[side]
        )
        for side in ("offense", "defense")
    }

    _validate_groups(features)


def test_box_groups_reject_missing_feature():
    features = {
        side: tuple(
            feature
            for group in BOX_GROUPS.values()
            for feature in group[side]
        )
        for side in ("offense", "defense")
    }
    features["offense"] = features["offense"][:-1]

    with pytest.raises(ValueError, match="do not cover Box15"):
        _validate_groups(features)
