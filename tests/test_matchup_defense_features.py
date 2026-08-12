from __future__ import annotations

import pandas as pd

from nba_impact.data.matchup_defense_features import (
    MATCHUP_DEFENSE_FEATURES,
    compute_matchup_defense_features,
)


def test_matchup_features_are_centered_and_shrunk() -> None:
    rows = []
    # Defender 10 holds both scorers below their performance against defender 20.
    for game_id, scorer, defender, points, fgm in (
        (1, 1, 10, 8, 4),
        (2, 1, 20, 12, 6),
        (3, 2, 10, 6, 3),
        (4, 2, 20, 14, 7),
    ):
        rows.append(
            {
                "game_id": game_id,
                "person_id": scorer,
                "matchups_person_id": defender,
                "partial_possessions": 10,
                "player_points": points,
                "matchup_assists": 2,
                "matchup_turnovers": 1,
                "matchup_field_goals_made": fgm,
                "matchup_field_goals_attempted": 8,
                "matchup_three_pointers_made": 0,
                "matchup_three_pointers_attempted": 2,
                "matchup_free_throws_made": 0,
                "shooting_fouls": 1,
            }
        )
    features, quality = compute_matchup_defense_features(
        pd.DataFrame(rows), 2025, defender_prior_possessions=20
    )
    indexed = features.set_index("PLAYER_ID")

    assert indexed.loc[10, "matchup_opponent_adjusted_points_saved_p100"] > 0
    assert indexed.loc[20, "matchup_opponent_adjusted_points_saved_p100"] < 0
    assert abs(
        indexed.loc[10, "matchup_opponent_adjusted_points_saved_p100_eb"]
    ) < abs(indexed.loc[10, "matchup_opponent_adjusted_points_saved_p100"])
    assert indexed.loc[
        10, "matchup_shotmaking_points_saved_vs_scorer_p100_eb"
    ] > 0
    assert indexed.loc[
        20, "matchup_shotmaking_points_saved_vs_scorer_p100_eb"
    ] < 0
    centered_total = (
        features["matchup_opponent_adjusted_points_saved_p100"]
        * features["matchup_possessions"]
        / 100
    ).sum()
    assert abs(centered_total) < 1e-12
    assert features[list(MATCHUP_DEFENSE_FEATURES)].notna().all().all()
    assert quality["point_reconstruction_mismatches"] == 0
    assert max(abs(value) for value in quality["factor_residual_centered_sums"].values()) < 1e-12


def test_matchup_features_reject_point_conservation_failure() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": 1,
                "person_id": 1,
                "matchups_person_id": 10,
                "partial_possessions": 10,
                "player_points": 9,
                "matchup_assists": 1,
                "matchup_turnovers": 1,
                "matchup_field_goals_made": 4,
                "matchup_field_goals_attempted": 8,
                "matchup_three_pointers_made": 0,
                "matchup_three_pointers_attempted": 2,
                "matchup_free_throws_made": 0,
                "shooting_fouls": 1,
            }
        ]
    )
    try:
        compute_matchup_defense_features(frame, 2025)
    except ValueError as error:
        assert "point_mismatches=1" in str(error)
    else:
        raise AssertionError("Expected point conservation failure.")
