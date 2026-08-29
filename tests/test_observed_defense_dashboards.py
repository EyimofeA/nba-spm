from __future__ import annotations

import pandas as pd

from nba_impact.data.observed_defense_dashboards import build_observed_defense_dashboards


def test_build_observed_defense_dashboards_aggregates_player_stints(tmp_path) -> None:
    source = tmp_path / "sheets"
    source.mkdir()
    pd.DataFrame({
        "PLAYER_ID": [1, 1, 2], "PLAYER_NAME": ["One", "One", "Two"],
        "D_FGM": [4, 6, 2], "D_FGA": [10, 20, 5], "NORMAL_FG_PCT": [50, 60, 40],
        "less_6ft_def_FGM_LT_06": [2, 4, 1], "less_6ft_def_FGA_LT_06": [4, 8, 2],
        "less_6ft_def_NS_LT_06_PCT": [55, 65, 45],
    }).to_csv(source / "2026.csv", index=False)
    run = build_observed_defense_dashboards(source, tmp_path / "out", seasons=(2026,))
    dfg = pd.read_csv(run["dfg_path"])
    rim = pd.read_csv(run["rim_dfg_path"])
    one = dfg.loc[dfg["PLAYER_ID"] == 1].iloc[0]
    assert one["DFGA"] == 30
    assert one["DFGM"] == 10
    assert one["FG%"] == (10 * 50 + 20 * 60) / 30
    assert set(rim["PLAYER_ID"]) == {1, 2}


def test_build_observed_defense_dashboards_reads_parquet_player_sheet(tmp_path) -> None:
    source = tmp_path / "sheets"
    source.mkdir()
    pd.DataFrame({
        "PLAYER_ID": [1], "PLAYER_NAME": ["One"],
        "D_FGM": [4], "D_FGA": [10], "NORMAL_FG_PCT": [50],
        "less_6ft_def_FGM_LT_06": [2], "less_6ft_def_FGA_LT_06": [4],
        "less_6ft_def_NS_LT_06_PCT": [55],
    }).to_parquet(source / "2026.parquet", index=False)

    run = build_observed_defense_dashboards(source, tmp_path / "out", seasons=(2026,))

    assert pd.read_csv(run["dfg_path"])["PLAYER_ID"].tolist() == [1]
    assert pd.read_csv(run["rim_dfg_path"])["PLAYER_ID"].tolist() == [1]


def test_build_observed_defense_dashboards_normalizes_proportion_baseline(tmp_path) -> None:
    source = tmp_path / "sheets"
    source.mkdir()
    pd.DataFrame({
        "PLAYER_ID": [1], "PLAYER_NAME": ["One"],
        "D_FGM": [4], "D_FGA": [10], "NORMAL_FG_PCT": [0.45],
        "less_6ft_def_FGM_LT_06": [6], "less_6ft_def_FGA_LT_06": [10],
        "less_6ft_def_NS_LT_06_PCT": [0.62],
    }).to_parquet(source / "2026.parquet", index=False)

    run = build_observed_defense_dashboards(source, tmp_path / "out", seasons=(2026,))
    rim = pd.read_csv(run["rim_dfg_path"]).iloc[0]

    assert rim["DFG%"] == 60.0
    assert rim["FG%"] == 62.0
    assert rim["DIFF%"] == -2.0
