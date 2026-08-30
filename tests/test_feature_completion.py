import pandas as pd
import pytest

from nba_impact.data.feature_completion import complete_selected_feature_panel


def test_semantic_completion_uses_zero_eb_and_zts_tiers():
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Window_End": [2020, 2020, 2020],
            "OffPoss": [100.0, 100.0, 100.0],
            "DefPoss": [100.0, 100.0, 100.0],
            "travels_p100": [1.0, None, None],
            "corner3_accuracy": [0.4, None, None],
            "true_shooting_pct": [0.60, 0.55, 0.50],
            "zts_pct_points": [5.0, 9.0, 9.0],
            "deflections_p100": [2.0, 9.0, 9.0],
            "matchup_opponent_adjusted_points_saved_p100_eb": [1.0, 9.0, 9.0],
            "dfg_attempts_p100": [3.0, 9.0, 9.0],
            "rim_points_saved_p100": [0.5, 9.0, 9.0],
        }
    )
    enriched = annual[["PLAYER_ID", "Window_End"]].assign(
        corner3_accuracy_eb=[0.41, 0.37, 0.36]
    )
    selected = {
        "offense": (
            "travels_p100",
            "corner3_accuracy",
            "true_shooting_pct",
            "zts_pct_points",
        ),
        "defense": (
            "deflections_p100",
            "matchup_opponent_adjusted_points_saved_p100_eb",
            "dfg_attempts_p100",
            "rim_points_saved_p100",
        ),
    }
    strict = pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]})
    loose = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Season": [2020, 2020],
            "zts_pct_points": [5.0, 2.0],
            "playtype_expected_ts_pct": [55.0, 54.0],
            "synergy_possessions": [100.0, 50.0],
        }
    )
    sources = {
        "hustle": pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]}),
        "matchup_defense": pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]}),
        "dfg": pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]}),
        "rim_dfg": pd.DataFrame({"PLAYER_ID": [1], "Season": [2020]}),
    }
    complete, expanded, ledger, quality = complete_selected_feature_panel(
        annual,
        enriched,
        selected,
        strict_playtype=strict,
        loose_playtype=loose,
        source_keys=sources,
    )

    assert complete["travels_p100"].tolist() == [1.0, 0.0, 0.0]
    assert complete["corner3_accuracy"].tolist() == [0.4, 0.37, 0.36]
    assert complete["zts_source_tier"].tolist() == [2.0, 1.0, 0.0]
    assert complete.loc[2, "zts_pct_points"] == -4.666666666666664
    assert complete["deflections_p100"].tolist() == [2.0, 0.0, 0.0]
    assert complete["matchup_opponent_adjusted_points_saved_p100_eb"].tolist() == [1.0, 0.0, 0.0]
    assert complete["dfg_attempts_p100"].tolist() == [3.0, 0.0, 0.0]
    assert complete["rim_points_saved_p100"].tolist() == [0.5, 0.0, 0.0]
    assert complete[list(expanded["offense"] + expanded["defense"])].isna().sum().sum() == 0
    assert ledger["completed_missing_rows"].sum() == 0
    assert set(ledger["feature"]) == set(expanded["offense"] + expanded["defense"])
    assert {
        "zts_source_tier",
        "has_hustle_tracking",
        "has_matchup_tracking",
        "has_dfg_tracking",
        "has_rim_defense_tracking",
    }.issubset(set(ledger["feature"]))
    assert quality["missing_values_after"] == 0


def test_semantic_completion_repairs_impossible_true_shooting_before_zts():
    annual = pd.DataFrame({
        "PLAYER_ID": [1, 2], "Window_End": [2026, 2026],
        "OffPoss": [100.0, 100.0], "DefPoss": [100.0, 100.0],
        "true_shooting_pct": [0.60, 7.95], "zts_pct_points": [1.0, 999.0],
    })
    selected = {"offense": ("true_shooting_pct", "zts_pct_points"), "defense": ()}
    loose = pd.DataFrame({
        "PLAYER_ID": [1, 2], "Season": [2026, 2026],
        "zts_pct_points": [1.0, 999.0],
        "playtype_expected_ts_pct": [59.0, 59.0],
        "synergy_possessions": [100.0, 1.0],
    })
    sources = {
        name: pd.DataFrame({"PLAYER_ID": [], "Season": []})
        for name in ("hustle", "matchup_defense", "dfg", "rim_dfg")
    }

    complete, _, _, quality = complete_selected_feature_panel(
        annual, annual[["PLAYER_ID", "Window_End"]], selected,
        strict_playtype=loose.rename(columns={"Season": "Window_End"}),
        loose_playtype=loose, source_keys=sources,
    )

    assert complete.loc[1, "true_shooting_pct"] == 0.60
    assert complete.loc[1, "zts_pct_points"] == 1.0
    assert quality["invalid_true_shooting_rows_repaired"] == 1
    assert quality["invalid_playtype_true_shooting_rows_repaired"] == 1


def test_semantic_completion_rejects_scoring_inconsistent_true_shooting():
    annual = pd.DataFrame({
        "PLAYER_ID": [1, 2], "Window_End": [2026, 2026],
        "OffPoss": [100.0, 100.0], "DefPoss": [100.0, 100.0],
        "PTS_p100": [30.0, 40.0],
        "FG2A_p100": [15.0, 10.0], "FG2M_p100": [10.0, 5.0],
        "FG3A_p100": [5.0, 5.0], "FG3M_p100": [2.0, 2.0],
        "FTA_p100": [5.0, 5.0], "FTM_p100": [4.0, 4.0],
        "true_shooting_pct": [0.60, 0.80], "zts_pct_points": [1.0, 21.0],
    })
    selected = {"offense": ("true_shooting_pct", "zts_pct_points"), "defense": ()}
    loose = pd.DataFrame({
        "PLAYER_ID": [1, 2], "Season": [2026, 2026],
        "zts_pct_points": [1.0, 21.0],
        "playtype_expected_ts_pct": [59.0, 59.0],
        "synergy_possessions": [100.0, 100.0],
    })
    sources = {
        name: pd.DataFrame({"PLAYER_ID": [], "Season": []})
        for name in ("hustle", "matchup_defense", "dfg", "rim_dfg")
    }

    complete, _, _, quality = complete_selected_feature_panel(
        annual, annual[["PLAYER_ID", "Window_End"]], selected,
        strict_playtype=loose.rename(columns={"Season": "Window_End"}),
        loose_playtype=loose, source_keys=sources,
    )

    assert complete.loc[1, "true_shooting_pct"] == 0.60
    assert complete.loc[1, "zts_pct_points"] == 1.0
    assert quality["scoring_inconsistent_true_shooting_rows_repaired"] == 1
    assert quality["invalid_playtype_true_shooting_rows_repaired"] == 1


def test_semantic_completion_stabilizes_true_shooting_within_season():
    annual = pd.DataFrame({
        "PLAYER_ID": [1, 2, 3], "Window_End": [2026, 2026, 2025],
        "OffPoss": [100.0, 100.0, 100.0], "DefPoss": [100.0, 100.0, 100.0],
        "PTS_p100": [100.0, 3.0, 200.0],
        "FG2A_p100": [100.0, 0.0, 100.0], "FG2M_p100": [50.0, 0.0, 100.0],
        "FG3A_p100": [0.0, 1.0, 0.0], "FG3M_p100": [0.0, 1.0, 0.0],
        "FTA_p100": [0.0, 0.0, 0.0], "FTM_p100": [0.0, 0.0, 0.0],
        "true_shooting_pct": [0.50, 1.50, 1.00],
        "zts_pct_points": [0.0, 100.0, 50.0],
    })
    selected = {"offense": ("true_shooting_pct", "zts_pct_points"), "defense": ()}
    loose = pd.DataFrame({
        "PLAYER_ID": [1, 2, 3], "Season": [2026, 2026, 2025],
        "zts_pct_points": [0.0, 100.0, 50.0],
        "playtype_expected_ts_pct": [50.0, 50.0, 50.0],
        "synergy_possessions": [100.0, 1.0, 100.0],
    })
    sources = {
        name: pd.DataFrame({"PLAYER_ID": [], "Season": []})
        for name in ("hustle", "matchup_defense", "dfg", "rim_dfg")
    }

    complete, _, _, quality = complete_selected_feature_panel(
        annual, annual[["PLAYER_ID", "Window_End"]], selected,
        strict_playtype=pd.DataFrame({"PLAYER_ID": [], "Season": []}),
        loose_playtype=loose, source_keys=sources,
    )

    season_prior = (0.50 * 100.0 + 1.50) / 101.0
    expected_tail = (1.50 + 100.0 * season_prior) / 101.0
    assert complete.loc[1, "true_shooting_pct"] == pytest.approx(expected_tail)
    assert complete.loc[1, "zts_pct_points"] == pytest.approx(100.0 * expected_tail - 50.0)
    assert complete.loc[2, "true_shooting_pct"] == 1.0
    assert quality["true_shooting_eb_prior_attempts"] == 100.0
