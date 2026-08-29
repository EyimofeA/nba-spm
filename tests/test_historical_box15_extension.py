from pathlib import Path

import yaml

from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def test_historical_contract_preserves_box15_and_2027_boundary():
    contract = yaml.safe_load(
        (ROOT / "research/experiments/historical_box15_extension_v1.yml").read_text()
    )
    assert tuple(contract["box15"]["features"]) == BOX_PIPM_STYLE_FEATURES
    assert contract["scope"]["raw_seasons"] == [1997, 2026]
    assert contract["scope"]["complete_five_year_window_ends"] == [2001, 2026]
    assert contract["scope"]["season_2027"] == "forbidden"
