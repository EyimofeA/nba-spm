import numpy as np
import pandas as pd

from nba_impact.models.aging_projection import (
    _design,
    _historical_player_projections,
    _weighted_metrics,
    build_aging_projection,
)


def test_aging_designs_are_finite_and_distinct() -> None:
    frame = pd.DataFrame(
        {
            "AGE": [21.0, 27.0, 35.0],
            "MIN": [400.0, 1800.0, 2800.0],
            "filtered_offense": [-1.0, 0.5, 3.0],
        }
    )
    linear = _design(frame, "linear_age", "offense")
    spline = _design(frame, "spline_age_impact", "offense")
    assert linear.shape == (3, 1)
    assert spline.shape[1] > linear.shape[1]
    assert np.isfinite(spline).all()


def test_weighted_metrics_report_true_r2() -> None:
    actual = np.array([-1.0, 1.0, 3.0])
    predicted = np.array([-1.0, 1.0, 3.0])
    metrics = _weighted_metrics(actual, predicted, np.ones(3))
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["correlation"] == 1.0


def test_historical_player_projections_preserve_causal_fold_labels() -> None:
    rows = []
    for origin in (2018, 2022):
        for player_id in (1, 2):
            rows.append(
                {
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "TEAM_ABBREVIATION": "AAA",
                    "origin_season": origin,
                    "target_season": origin + 1,
                    "AGE": 25.0,
                    "MIN": 1500.0,
                    "filtered_net": 1.0,
                    "evaluation_weight": 10.0,
                    "actual_offense": 0.5,
                    "actual_defense": 0.2,
                    "actual_net": 0.7,
                    "projected_offense": 0.4,
                    "projected_defense": 0.1,
                    "projected_net": 0.5,
                    "method": "spline_age_minutes",
                }
            )
    exported = _historical_player_projections(
        pd.DataFrame(rows),
        selected_method="spline_age_minutes",
        selection_origins=(2018,),
        diagnostic_origins=(2022,),
    )
    assert exported["target_season"].tolist() == [2019, 2019, 2023, 2023]
    assert set(exported["projection_kind"]) == {"walk_forward_backtest"}
    assert set(exported.loc[exported["origin_season"].eq(2018), "evidence_status"]) == {
        "selection_reused"
    }
    assert set(exported.loc[exported["origin_season"].eq(2022), "evidence_status"]) == {
        "diagnostic_reused"
    }


def test_aging_projection_exports_walk_forward_players_and_future_only_teams(tmp_path) -> None:
    trajectory_rows = []
    target_rows = []
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    for season in range(2014, 2027):
        sheet_rows = []
        for player_id in range(1, 5):
            offense = (player_id - 2.5) * 0.2 + (season - 2014) * 0.03
            defense = (2.5 - player_id) * 0.1
            trajectory_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "Season": season,
                    "filtered_offense": offense,
                    "filtered_defense": defense,
                    "filtered_net": offense + defense,
                    "Poss_Off": 1500.0,
                    "Poss_Def": 1500.0,
                    "phi": 0.9,
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Season": season,
                    "target_offense": offense + 0.05,
                    "target_defense": defense - 0.02,
                    "Poss_Off": 1500.0,
                    "Poss_Def": 1500.0,
                }
            )
            sheet_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "TEAM_ABBREVIATION": "AAA" if player_id < 3 else "BBB",
                    "AGE": 22 + player_id + season - 2014,
                    "MIN": 1500 + 100 * player_id,
                }
            )
        pd.DataFrame(sheet_rows).to_csv(sheets / f"{season}.csv", index=False)
    trajectories = tmp_path / "trajectories.parquet"
    targets = tmp_path / "targets.parquet"
    pd.DataFrame(trajectory_rows).to_parquet(trajectories, index=False)
    pd.DataFrame(target_rows).to_parquet(targets, index=False)

    run = build_aging_projection(
        trajectories,
        targets,
        sheets,
        artifact_root=tmp_path / "artifacts",
        selection_origins=(2018,),
        diagnostic_origins=(2019,),
        projection_origin=2026,
        minimum_training_origins=3,
    )
    output = run["artifact_path"]
    historical = pd.read_parquet(f"{output}/historical_player_projections.parquet")
    current = pd.read_parquet(f"{output}/player_projections.parquet")
    teams = pd.read_parquet(f"{output}/team_projections.parquet")

    assert set(historical["target_season"]) == {2019, 2020}
    assert set(historical["evidence_status"]) == {"selection_reused", "diagnostic_reused"}
    assert set(current["projection_season"]) == {2027}
    assert set(current["projection_kind"]) == {"forecast"}
    assert set(teams["projection_season"]) == {2027}
