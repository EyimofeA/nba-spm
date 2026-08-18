from nba_impact.data.gabriel_fallback_repairs import (
    _bridge_lineup,
    _lineup_from_players,
    _ordinal_stint_ids,
)


def test_lineup_requires_exactly_five_players_for_each_team() -> None:
    players = tuple(range(1, 11))
    teams = {player_id: 10 if player_id <= 5 else 20 for player_id in players}

    assert _lineup_from_players(players, teams, 10, 20) == players

    teams[5] = 20
    assert _lineup_from_players(players, teams, 10, 20) is None


def test_unobserved_outcome_requires_two_sided_lineup_continuity() -> None:
    lineup = tuple(range(1, 11))
    other = tuple(range(11, 21))
    source_actions = [10, 20]

    assert _bridge_lineup(15, "fieldGoal", source_actions, {10: lineup, 20: lineup}) == (
        lineup,
        "two_sided_continuity",
    )
    assert _bridge_lineup(15, "fieldGoal", source_actions, {10: lineup, 20: other}) == (
        None,
        "unmapped",
    )


def test_boundary_can_use_adjacent_observed_lineup_but_no_outcome_can() -> None:
    lineup = tuple(range(1, 11))
    assert _bridge_lineup(1, "period", [10], {10: lineup}) == (lineup, "boundary_continuity")
    assert _bridge_lineup(1, "violation", [10], {10: lineup}) == (None, "unmapped")


def test_returning_lineup_creates_a_new_ordinal_stint() -> None:
    first = tuple(range(1, 11))
    second = tuple(range(11, 21))

    assert _ordinal_stint_ids("0022300535", [first, first, second, first]) == [
        "0022300535:g001",
        "0022300535:g001",
        "0022300535:g002",
        "0022300535:g003",
    ]
