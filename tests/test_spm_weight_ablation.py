import json

import pandas as pd

from nba_impact.models.spm_weight_ablation import build_feature_catalog, describe_feature


def test_feature_catalog_marks_shared_and_side_specific_inputs(tmp_path) -> None:
    run = tmp_path / "source"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "models": {
                    "offense": {"features": ["PTS_p100", "zts_pct_points"]},
                    "defense": {"features": ["PTS_p100", "dfg_diff_pct_eb"]},
                }
            }
        )
    )
    catalog = build_feature_catalog(run).set_index("feature")
    assert catalog.loc["PTS_p100", "side"] == "both"
    assert catalog.loc["zts_pct_points", "side"] == "offense"
    assert catalog.loc["dfg_diff_pct_eb", "side"] == "defense"
    assert catalog["description"].str.len().gt(10).all()


def test_feature_descriptions_explain_stabilization_and_relative_values() -> None:
    assert "Empirical-Bayes" in describe_feature("fg3_pct_eb")
    assert "league average" in describe_feature("PTS_p100_relative")
    assert "play-type mix" in describe_feature("zts_pct_points")
