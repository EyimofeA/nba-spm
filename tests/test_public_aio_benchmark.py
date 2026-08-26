from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.public_aio_benchmark import (
    build_pairwise_correlations,
    build_team_win_benchmark,
    load_epm_ratings,
    load_lebron_ratings,
    load_mamba_ratings,
    load_site_aio_ratings,
    validate_rating_panel,
)


def _ratings() -> pd.DataFrame:
    rows = []
    for season in (2021, 2022):
        for player in range(1, 13):
            value = (player - 6) / 2 + (season - 2021) * 0.1
            for metric, multiplier in (("a", 1.0), ("b", 2.0)):
                rows.append(
                    {
                        "PLAYER_ID": player,
                        "Season": season,
                        "metric": metric,
                        "metric_label": metric.upper(),
                        "category": "test",
                        "offense": value * multiplier * 0.6,
                        "defense": value * multiplier * 0.4,
                        "net": value * multiplier,
                    }
                )
    return pd.DataFrame(rows)


def _minutes() -> pd.DataFrame:
    rows = []
    for season in (2021, 2022, 2023):
        for player in range(1, 13):
            rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "team_id": 10 if player <= 6 else 20,
                    "minutes": 500 + player,
                }
            )
    return pd.DataFrame(rows)


def test_pairwise_correlations_use_identical_qualified_rows() -> None:
    correlations = build_pairwise_correlations(
        validate_rating_panel(_ratings()),
        _minutes(),
        seasons=(2021, 2022),
        minimum_minutes=250,
    )
    net = correlations.query(
        "component == 'net' and left_metric == 'a' and right_metric == 'b'"
    ).iloc[0]
    assert net["rows"] == 24
    assert np.isclose(net["pearson"], 1.0)
    assert np.isclose(net["spearman"], 1.0)


def test_team_win_benchmark_uses_next_season_team_minutes() -> None:
    games = []
    for season in (2022, 2023):
        for team, wins in ((10, 2), (20, 8)):
            games.extend(
                {"Season": season, "team_id": team, "won": index < wins}
                for index in range(10)
            )
    folds, summary, coverage = build_team_win_benchmark(
        validate_rating_panel(_ratings()),
        _minutes(),
        pd.DataFrame(games),
        rating_seasons=(2021, 2022),
        minimum_metric_minutes=250,
        replacement_values=(-2.0,),
        minimum_teams=2,
    )
    assert len(folds) == 4
    assert set(folds["teams"]) == {2}
    assert np.isclose(folds["pearson"], 1.0).all()
    assert np.isclose(summary["mean_r_squared"], 1.0).all()
    assert np.isclose(coverage["minute_coverage"], 1.0).all()


def test_rating_panel_rejects_duplicate_keys() -> None:
    ratings = _ratings()
    duplicate = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    try:
        validate_rating_panel(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("Duplicate rating keys should fail validation.")


def test_lebron_and_mamba_identity_mapping(tmp_path) -> None:
    lebron_path = tmp_path / "lebron.csv"
    pd.DataFrame(
        {
            "nba_id": [1, 2],
            "Player": ["Alpha One", "Beta Two"],
            "Season": [2024, 2024],
            "LEBRON": [3.0, -1.0],
            "O-LEBRON": [2.0, -0.5],
            "D-LEBRON": [1.0, -0.5],
        }
    ).to_csv(lebron_path, index=False)
    mamba_path = tmp_path / "mamba.csv"
    pd.DataFrame(
        {
            "Player": ["Alpha One", "Missing Player"],
            "Offense": [1.5, 0.0],
            "Defense": [0.5, 0.0],
            "Ovr": [2.0, 0.0],
            "Season": [2024, 2024],
        }
    ).to_csv(mamba_path, index=False)
    lebron, identity = load_lebron_ratings(lebron_path)
    mamba, unmatched = load_mamba_ratings(mamba_path, identity)
    assert len(lebron) == 2
    assert mamba.iloc[0]["PLAYER_ID"] == 1
    assert unmatched.iloc[0]["Player"] == "Missing Player"


def test_epm_loader_uses_season_end_and_exact_ids(tmp_path) -> None:
    path = tmp_path / "epm.csv"
    pd.DataFrame(
        {
            "EPM_season": ["2022-23", "2023-24"],
            "EPM_player_id": [1, 2],
            "EPM_off": [2.0, -0.5],
            "EPM_def": [1.0, 0.25],
            "EPM_tot": [3.0, -0.25],
        }
    ).to_csv(path, index=False)
    ratings = load_epm_ratings(path)
    assert ratings[["PLAYER_ID", "Season"]].to_dict("records") == [
        {"PLAYER_ID": 1, "Season": 2023},
        {"PLAYER_ID": 2, "Season": 2024},
    ]
    assert ratings["metric"].eq("epm").all()


def test_site_aio_loader_uses_exact_website_rows(tmp_path) -> None:
    pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Season": [2024, 2024],
            "aio_offense": [2.0, -0.5],
            "aio_defense": [1.0, 0.25],
            "aio_net": [3.0, -0.25],
        }
    ).to_json(tmp_path / "leaderboard-2024.json", orient="records")
    ratings = load_site_aio_ratings(tmp_path, (2024,))
    assert ratings[["PLAYER_ID", "Season", "net"]].to_dict("records") == [
        {"PLAYER_ID": 1, "Season": 2024, "net": 3.0},
        {"PLAYER_ID": 2, "Season": 2024, "net": -0.25},
    ]
    assert ratings["metric"].eq("site_aio").all()
