from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_historical_validation_has_a_fixed_retention_gate():
    contract = yaml.safe_load(
        (ROOT / "research/experiments/historical_box15_validation_v1.yml").read_text()
    )
    assert contract["retention_gate"]["maximum_equal_season_rmse_increase"] == 0.05
    assert (
        contract["retention_gate"]["maximum_mean_margin_correlation_decline"]
        == 0.01
    )
    assert contract["information_cutoff"]["season_2027"] == "forbidden"
