import numpy as np
import pandas as pd

from nba_impact.models.current_spm_diagnostics import (
    _exposure_metrics,
    _feature_drift,
)


def test_exposure_metrics_preserve_component_and_bin_contract() -> None:
    frame = pd.DataFrame(
        {
            "Poss_Off": [400, 800, 1500, 2500],
            "Poss_Def": [400, 800, 1500, 2500],
            **{
                f"{kind}_{component}": np.array([0.0, 1.0, 2.0, 3.0])
                for kind in ("target", "spm")
                for component in ("offense", "defense", "net")
            },
        }
    )
    result = _exposure_metrics(frame)
    assert len(result) == 12
    assert set(result["exposure_bin"]) == {
        "under_500",
        "500_to_999",
        "1000_to_1999",
        "2000_plus",
    }
    assert set(result["component"]) == {"offense", "defense", "net"}


def test_feature_drift_labels_defensive_families() -> None:
    reference = pd.DataFrame(
        {"deflections_p100": [1.0, 2.0, 3.0], "dfg_attempts_p100": [4.0, 5.0, 6.0]}
    )
    current = pd.DataFrame(
        {"deflections_p100": [3.0, 4.0], "dfg_attempts_p100": [6.0, 7.0]}
    )
    result = _feature_drift(
        reference,
        current,
        ["deflections_p100", "dfg_attempts_p100"],
        {"deflections_p100", "dfg_attempts_p100"},
    ).set_index("feature")
    assert result.loc["deflections_p100", "family"] == "hustle"
    assert result.loc["dfg_attempts_p100", "family"] == "dfg_rim"
    assert result.loc["dfg_attempts_p100", "current_outside_historical_range_fraction"] == 0.5
