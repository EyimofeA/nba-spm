from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.data.defensive_tracking_features import (
    build_defensive_tracking_features,
    compute_defensive_tracking_features,
)


def test_build_defensive_tracking_features_resolves_names_and_rates(tmp_path: Path) -> None:
    box_dir = tmp_path / "box"
    box_dir.mkdir()
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER_NAME": "Nikola Jokić", "DefPoss": 1000},
            {"PLAYER_ID": 2, "PLAYER_NAME": "Player Two", "DefPoss": 1000},
        ]
    ).to_csv(box_dir / "2018.csv", index=False)
    dfg_path = tmp_path / "dfg.csv"
    rim_path = tmp_path / "rim.csv"
    hustle_path = tmp_path / "hustle.csv"
    pd.DataFrame(
        [
            {"PLAYER": "Nikola Jokic", "year": 2018, "DFGA": 400, "DIFF%": -5},
            {"PLAYER": "Player Two", "year": 2018, "DFGA": 200, "DIFF%": 5},
        ]
    ).to_csv(dfg_path, index=False)
    pd.DataFrame(
        [
            {"PLAYER": "Nikola Jokic", "year": 2018, "DFGA": 200, "DIFF%": -10},
            {"PLAYER": "Player Two", "year": 2018, "DFGA": 100, "DIFF%": 10},
        ]
    ).to_csv(rim_path, index=False)
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER_NAME": "Nikola Jokic", "year": 2018,
             "DEFLECTIONS": 20, "CHARGES_DRAWN": 2, "CONTESTED_SHOTS_2PT": 100,
             "CONTESTED_SHOTS_3PT": 50, "DEF_LOOSE_BALLS_RECOVERED": 10},
            {"PLAYER_ID": 2, "PLAYER_NAME": "Player Two", "year": 2018,
             "DEFLECTIONS": 10, "CHARGES_DRAWN": 1, "CONTESTED_SHOTS_2PT": 50,
             "CONTESTED_SHOTS_3PT": 25, "DEF_LOOSE_BALLS_RECOVERED": 5},
        ]
    ).to_csv(hustle_path, index=False)
    run = build_defensive_tracking_features(
        dfg_path, rim_path, hustle_path, box_dir,
        artifact_root=tmp_path, seasons=(2018,),
    )
    features = pd.read_parquet(run["features_path"]).set_index("PLAYER_ID")
    assert features.loc[1, "dfg_attempts_p100"] == pytest.approx(40.0)
    assert features.loc[1, "dfg_diff_pct_eb"] == pytest.approx(-10 / 3)
    assert features.loc[1, "rim_diff_pct_eb"] == pytest.approx(-20 / 3)
    assert features.loc[1, "rim_points_saved_p100_raw"] == pytest.approx(4.0)
    assert features.loc[1, "rim_points_saved_p100"] == pytest.approx(8 / 3)
    assert features.loc[1, "dfg_two_point_equivalent_saved_p100"] == pytest.approx(8 / 3)
    assert features.loc[1, "rim_matchup_attempt_share"] == pytest.approx(0.5)
    assert features.loc[1, "contested_3pt_share"] == pytest.approx(1 / 3)
    assert features.loc[1, "deflections_p100"] == pytest.approx(2.0)
    assert run["quality"]["source_join_quality"]["dfg"]["match_rate"] == 1.0
    assert run["quality"]["nonfinite_values"] == 0
    assert run["quality"]["neutral_fill_policy"] == "same_season_median_then_zero"


def test_defensive_tracking_never_uses_another_seasons_center() -> None:
    box = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER_NAME": "One", "DefPoss": 1000, "Season": 2017, "_name_key": "one", "_name_ambiguous": False},
            {"PLAYER_ID": 2, "PLAYER_NAME": "Two", "DefPoss": 1000, "Season": 2018, "_name_key": "two", "_name_ambiguous": False},
        ]
    )
    dfg = pd.DataFrame(
        [{"PLAYER": "Two", "year": 2018, "DFGA": 200, "DIFF%": -5.0}]
    )
    rim = pd.DataFrame(
        [{"PLAYER": "Two", "year": 2018, "DFGA": 100, "DIFF%": -10.0}]
    )
    hustle = pd.DataFrame(
        [{
            "PLAYER_NAME": "Two",
            "year": 2018,
            "DEFLECTIONS": 20,
            "CHARGES_DRAWN": 2,
            "CONTESTED_SHOTS_2PT": 100,
            "CONTESTED_SHOTS_3PT": 50,
            "DEF_LOOSE_BALLS_RECOVERED": 10,
        }]
    )

    features, quality = compute_defensive_tracking_features(box, dfg, rim, hustle)
    missing_season = features.loc[features["Season"].eq(2017)].iloc[0]

    assert missing_season["dfg_attempts_p100"] == 0.0
    assert missing_season["rim_points_saved_p100_raw"] == 0.0
    assert missing_season["rim_points_saved_p100"] == 0.0
    assert missing_season["deflections_p100"] == 0.0
    assert quality["neutral_fill_policy"] == "same_season_median_then_zero"


def test_defensive_tracking_recomputes_difference_across_percentage_units() -> None:
    box = pd.DataFrame([{
        "PLAYER_ID": 1, "PLAYER_NAME": "One", "DefPoss": 1000,
        "Season": 2026, "_name_key": "one", "_name_ambiguous": False,
    }])
    dfg = pd.DataFrame([{
        "PLAYER_ID": 1, "PLAYER": "One", "year": 2026,
        "DFGA": 200, "DFG%": 45.0, "FG%": 0.47, "DIFF%": 44.53,
    }])
    rim = pd.DataFrame([{
        "PLAYER_ID": 1, "PLAYER": "One", "year": 2026,
        "DFGA": 100, "DFG%": 60.0, "FG%": 0.62, "DIFF%": 59.38,
    }])
    hustle = pd.DataFrame([{
        "PLAYER_ID": 1, "PLAYER_NAME": "One", "year": 2026,
        "DEFLECTIONS": 0, "CHARGES_DRAWN": 0, "CONTESTED_SHOTS_2PT": 0,
        "CONTESTED_SHOTS_3PT": 0, "DEF_LOOSE_BALLS_RECOVERED": 0,
    }])

    features, _ = compute_defensive_tracking_features(box, dfg, rim, hustle)
    row = features.iloc[0]

    assert row["dfg_diff_pct_eb"] == pytest.approx(-1.0)
    assert row["rim_diff_pct_eb"] == pytest.approx(-1.0)
    assert row["rim_points_saved_p100"] == pytest.approx(0.2)
