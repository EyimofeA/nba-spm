from pathlib import Path

import yaml

from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def test_followup_features_are_small_and_do_not_duplicate_box15():
    contract = yaml.safe_load(
        (ROOT / "research/experiments/box15_top_feature_followup_v1.yml").read_text()
    )
    box = set(BOX_PIPM_STYLE_FEATURES)
    assert contract["information_cutoff"]["season_2027"] == "forbidden"
    assert len(contract["features"]["offense"]) == 5
    assert len(contract["features"]["defense"]) == 5
    assert not box.intersection(contract["features"]["offense"])
    assert not box.intersection(contract["features"]["defense"])
