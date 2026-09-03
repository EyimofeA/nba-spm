import pandas as pd
import pytest

from research.rapm_lab import run_pipm_reconstruction as pipm


def test_duplicate_source_rows_do_not_change_season_centers(monkeypatch):
    sheet = pd.DataFrame({
        "PLAYER_ID": [1, 2], "PLAYER_NAME": ["Alpha", "Beta"],
        "TEAM_ABBREVIATION": ["AAA", "BBB"], "MIN": [1000, 500],
        "GP": [30, 20], "PACE": [100, 99], "PTS": [500, 300],
        **{column: [100, 50] for column in [
            "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "FTA", "FG2A", "FG3A",
        ]},
    })
    context = pd.DataFrame({
        "PLAYER_ID": [1, 2], "o_onoff": [3., 1.], "d_onoff": [1., 2.],
        "o_oncourt_vs_avg": [2., -1.], "d_oncourt_vs_avg": [1., -2.],
    })
    monkeypatch.setattr(pipm, "lineup_context", lambda _: context)
    monkeypatch.setattr(pipm, "starter_counts", lambda _: pd.DataFrame({"PLAYER_ID": [1, 2], "GS": [20, 10]}))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: sheet.copy())
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sheet.copy())
    expected = pipm.season_features(2024)
    sheet = pd.concat([sheet, pd.concat([sheet.iloc[:1]] * 63)], ignore_index=True)
    pd.testing.assert_frame_equal(pipm.season_features(2024), expected)
    sheet.loc[2, "PTS"] = 999
    with pytest.raises(ValueError, match="unambiguous"):
        pipm.season_features(2024)
