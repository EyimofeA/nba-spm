from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.impact_validation_suite import (
    build_adjacent_annual_metrics,
    chronological_game_halves,
    composite_ranking,
    weighted_correlation,
)


def test_chronological_game_halves_never_splits_a_game() -> None:
    frame = pd.DataFrame(
        {
            "gameid": ["3", "1", "2", "4", "1", "3", "2", "4"],
            "date": [
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-04",
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
                "2024-01-04",
            ],
        }
    )
    first, second, games = chronological_game_halves(frame)
    assert games.loc[games["half"].eq("first"), "gameid"].tolist() == ["1", "2"]
    assert set(frame.loc[first, "gameid"]) == {"1", "2"}
    assert set(frame.loc[second, "gameid"]) == {"3", "4"}
    assert not np.any(first & second)


def test_weighted_correlation_respects_weights() -> None:
    actual = np.asarray([0.0, 1.0, 2.0, 3.0])
    predicted = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert np.isclose(weighted_correlation(actual, predicted, np.ones(4)), 1.0)


def test_composite_uses_direction_and_reports_weight_sensitivity() -> None:
    scores = pd.DataFrame(
        [
            {"test_id": "a", "candidate": "one", "fold": 1, "value": 1.0, "higher_is_better": False},
            {"test_id": "a", "candidate": "two", "fold": 1, "value": 2.0, "higher_is_better": False},
            {"test_id": "b", "candidate": "one", "fold": 1, "value": 0.4, "higher_is_better": True},
            {"test_id": "b", "candidate": "two", "fold": 1, "value": 0.6, "higher_is_better": True},
        ]
    )
    summary, ranked = composite_ranking(scores, weights={"a": 0.75, "b": 0.25})
    result = summary.set_index("candidate")
    assert result.loc["one", "weighted_rank"] == 1
    assert result.loc["one", "weighted_percentile_score"] == 0.75
    assert result.loc["two", "weighted_percentile_score"] == 0.25
    assert set(ranked["percentile_score"]) == {0.0, 1.0}


def test_composite_rejects_weights_that_do_not_sum_to_one() -> None:
    scores = pd.DataFrame(
        [{"test_id": "a", "candidate": "one", "fold": 1, "value": 1.0, "higher_is_better": False}]
    )
    try:
        composite_ranking(scores, weights={"a": 0.9})
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid weights should fail.")


def test_adjacent_metrics_keep_aging_rows_when_some_ages_are_missing() -> None:
    target_rows = []
    rating_rows = []
    age_rows = []
    for season in range(2014, 2022):
        for player in range(1, 5):
            target_rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "sample_weight": 1.0,
                    "target_offense": float(player + season - 2014),
                    "target_defense": float(player) / 2,
                    "target_net": float(player + season - 2014 + player / 2),
                }
            )
            rating_rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "candidate": "candidate",
                    "offense": float(player),
                    "defense": float(player) / 2,
                    "net": float(1.5 * player),
                }
            )
            if not (season == 2020 and player == 4):
                age_rows.append(
                    {"PLAYER_ID": player, "Season": season, "AGE": 20 + player + season - 2014}
                )
    metrics, _ = build_adjacent_annual_metrics(
        pd.DataFrame(rating_rows),
        pd.DataFrame(target_rows),
        candidates=("candidate",),
        ages=pd.DataFrame(age_rows),
    )
    adjusted = metrics.loc[
        metrics["test_id"].eq("forward_annual_impact")
        & metrics["variant"].eq("aging_adjusted")
        & metrics["season"].eq(2020)
    ]
    assert len(adjusted) == 3
    assert set(adjusted["rows"]) == {3}
