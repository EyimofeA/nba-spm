import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_final_box_feature_ladder import (  # noqa: E402
    _candidate_features,
    _select_candidate,
)


def test_candidate_features_build_cumulative_sides():
    contract = {
        "feature_families": [
            {"family": "shooting", "side": "offense", "features": ["zts"]},
            {"family": "rim", "side": "defense", "features": ["rim_saved"]},
        ],
        "ladder": {
            "eligible_steps": ["box_plus_shooting", "box_plus_rim"]
        },
    }
    selected = {
        "offense": ("zts",),
        "defense": ("rim_saved",),
    }
    candidates, families = _candidate_features(contract, selected)

    assert "zts" in candidates["box_plus_shooting"]["offense"]
    assert "zts" in candidates["box_plus_rim"]["offense"]
    assert "rim_saved" not in candidates["box_plus_shooting"]["defense"]
    assert "rim_saved" in candidates["box_plus_rim"]["defense"]
    assert families["rim"]["side"] == "defense"


def test_selection_requires_better_mse_and_correlation_guard():
    contract = {
        "ladder": {
            "eligible_steps": ["box_plus_a", "box_plus_b", "box_plus_c"]
        }
    }
    intervals = pd.DataFrame(
        {
            "candidate": [
                "box_15_aio",
                "box_plus_a_aio",
                "box_plus_b_aio",
                "box_plus_c_aio",
            ],
            "equal_season_mse": [200.0, 199.0, 198.0, 201.0],
        }
    )
    summary = pd.DataFrame(
        {
            "candidate": [
                "box_15_aio",
                "box_plus_a_aio",
                "box_plus_b_aio",
                "box_plus_c_aio",
            ],
            "mean_margin_correlation": [0.36, 0.355, 0.34, 0.37],
        }
    )

    selected, table = _select_candidate(contract, intervals, summary)

    assert selected == "box_plus_a"
    assert not table.loc[
        table["candidate"].eq("box_plus_b_aio"), "passes_correlation_guard"
    ].iloc[0]
