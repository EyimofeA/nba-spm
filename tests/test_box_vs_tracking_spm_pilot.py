import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_box_vs_tracking_spm_pilot import _tracking_features  # noqa: E402


def test_tracking_contract_uses_stabilized_ratios_and_excludes_box15() -> None:
    selected = {
        "offense": (
            "PTS_p100",
            "catch_shoot_fga_p100",
            "catch_shoot_accuracy",
            "catch_shoot_accuracy_eb",
            "shot_quality_average_relative",
        ),
        "defense": (
            "DREB_p100",
            "rebound_contests_p100",
            "rim_points_saved_p100",
            "matchup_blocks_p100",
            "has_rim_defense_tracking",
        ),
    }

    features = _tracking_features(selected)

    assert "PTS_p100" not in features["offense"]
    assert "DREB_p100" not in features["defense"]
    assert "catch_shoot_accuracy_eb" in features["offense"]
    assert "catch_shoot_accuracy" not in features["offense"]
    assert "rim_points_saved_p100" in features["defense"]
