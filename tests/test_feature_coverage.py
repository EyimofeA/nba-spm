from __future__ import annotations

import pandas as pd

from nba_impact.data.feature_coverage import audit_feature_coverage


def test_coverage_audit_does_not_count_neutral_external_fill_as_observed() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Window_End": [2018, 2018],
            "OffPoss": [100.0, 100.0],
            "DefPoss": [100.0, 100.0],
            "PTS_p100": [20.0, 10.0],
            "zts_pct_points": [0.0, 0.0],
        }
    )
    five_year = annual.copy()
    selected = {"offense": ("PTS_p100", "zts_pct_points"), "defense": ("PTS_p100",)}
    playtype = pd.DataFrame({"PLAYER_ID": [1], "Season": [2018]})

    summary, _ = audit_feature_coverage(
        annual,
        five_year,
        selected,
        {
            "playtype": playtype,
            "dfg": playtype,
            "rim_dfg": playtype,
            "hustle": playtype,
            "matchup_defense": playtype,
        },
    )

    annual_zts = summary.query("panel == 'annual' and feature == 'zts_pct_points'").iloc[0]
    assert annual_zts["coverage_fraction"] == 0.5
    assert annual_zts["reason_code"] == "source_eligibility"


def test_external_source_row_does_not_hide_missing_feature_value() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "Window_End": [2020],
            "zts_pct_points": [float("nan")],
        }
    )
    five_year = annual.copy()
    playtype = pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]})
    summary, _ = audit_feature_coverage(
        annual,
        five_year,
        {"offense": ("zts_pct_points",), "defense": ()},
        {"playtype": playtype},
    )
    assert summary["observed_rows"].eq(0).all()
