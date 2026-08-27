from __future__ import annotations

import pandas as pd

from nba_impact.data.full_spm_features import build_rolling_five_year_features


def test_build_rolling_five_year_features_pools_external_fields(tmp_path) -> None:
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    annual_rows = []
    for season in range(2014, 2019):
        pd.DataFrame(
            {
                "PLAYER_ID": [1], "OffPoss": [100], "DefPoss": [100],
                "PTS": [season - 2000], "AST": [1], "TOV": [1], "STL": [1],
                "BLK": [1], "OREB": [1], "SelfOReb": [0], "DREB": [1],
                "PF": [1], "PFD": [1], "FTA": [1], "FTM": [1], "FG2A": [1],
                "FG2M": [1], "FG3A": [1], "FG3M": [1], "FGA": [2],
            }
        ).to_parquet(sheets / f"{season}.parquet", index=False)
        annual_rows.append(
            {"PLAYER_ID": 1, "Window_End": season, "OffPoss": 100, "DefPoss": 100,
             "PTS_p100": 100 + season - 2014, "zts_pct_points": season - 2010,
             "rim_points_saved_p100": 2 * (season - 2010)}
        )
    annual = pd.DataFrame(annual_rows)
    selected = {
        "offense": ("PTS_p100", "zts_pct_points"),
        "defense": ("rim_points_saved_p100",),
    }

    result = build_rolling_five_year_features(
        annual, sheets, selected, window_ends=(2018,)
    ).iloc[0]

    assert result["PTS_p100"] == 102
    assert result["zts_pct_points"] == 6
    assert result["rim_points_saved_p100"] == 12
    assert result["Window_End"] == 2018
