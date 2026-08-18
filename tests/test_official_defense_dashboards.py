from __future__ import annotations

import pandas as pd

from nba_impact.data.official_defense_dashboards import _to_source


def test_to_source_converts_overall_and_rim_percentages() -> None:
    overall = pd.DataFrame({
        "CLOSE_DEF_PERSON_ID": [1], "PLAYER_NAME": ["Player"], "D_FGM": [4], "D_FGA": [10],
        "D_FG_PCT": [0.4], "NORMAL_FG_PCT": [0.5], "PCT_PLUSMINUS": [-0.1],
    })
    rim = pd.DataFrame({
        "CLOSE_DEF_PERSON_ID": [1], "PLAYER_NAME": ["Player"], "FGM_LT_06": [3], "FGA_LT_06": [5],
        "LT_06_PCT": [0.6], "NS_LT_06_PCT": [0.7], "PLUSMINUS": [-0.1],
    })
    dfg = _to_source(overall, 2026, rim=False)
    rim_dfg = _to_source(rim, 2026, rim=True)
    assert dfg.loc[0, "DIFF%"] == -10
    assert dfg.loc[0, "FG%"] == 50
    assert rim_dfg.loc[0, "DFGA"] == 5
    assert rim_dfg.loc[0, "DFG%"] == 60
