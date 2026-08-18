from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.matchup_elo import BASE_ELO, fit_matchup_elo


def _row(game_id: str, scorer: int, defender: int, points: float) -> dict:
    return {
        "game_id": game_id,
        "person_id": scorer,
        "matchups_person_id": defender,
        "partial_possessions": 10.0,
        "player_points": points,
    }


def test_matchup_elo_recovers_simple_scorer_and_defender_order() -> None:
    # Scorer 1 is always better than scorer 2. Defender 10 suppresses both
    # scorers relative to defender 20. Repeating game IDs keeps the source
    # contract at one scorer-defender row per game.
    rows = []
    for game in range(24):
        rows.extend(
            [
                _row(f"g{game}_a", 1, 10, 10.0),
                _row(f"g{game}_b", 1, 20, 14.0),
                _row(f"g{game}_c", 2, 10, 6.0),
                _row(f"g{game}_d", 2, 20, 10.0),
            ]
        )
    ratings, audit = fit_matchup_elo(pd.DataFrame(rows), 2026, ridge_penalty=10.0)
    by_id = ratings.set_index("PLAYER_ID")
    assert by_id.loc[1, "offense_elo"] > by_id.loc[2, "offense_elo"]
    assert by_id.loc[10, "defense_elo"] > by_id.loc[20, "defense_elo"]
    assert np.isclose(audit["weighted_offense_log_rate_mean"], 0.0)
    assert np.isclose(audit["weighted_defense_log_rate_mean"], 0.0)
    assert np.isclose(
        np.average(by_id["offense_elo"], weights=by_id["offense_matchup_possessions"]),
        BASE_ELO,
    )


def test_matchup_elo_is_invariant_to_source_row_order() -> None:
    rows = [
        _row("a", 1, 10, 8.0),
        _row("b", 1, 20, 12.0),
        _row("c", 2, 10, 6.0),
        _row("d", 2, 20, 10.0),
    ]
    first, _ = fit_matchup_elo(pd.DataFrame(rows), 2025, ridge_penalty=10.0)
    second, _ = fit_matchup_elo(pd.DataFrame(rows[::-1]), 2025, ridge_penalty=10.0)
    pd.testing.assert_frame_equal(first.sort_values("PLAYER_ID").reset_index(drop=True), second.sort_values("PLAYER_ID").reset_index(drop=True))
