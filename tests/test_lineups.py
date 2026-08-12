from __future__ import annotations

import pandas as pd

from nba_impact.data.lineups import _elapsed_seconds, _normalize_name, _substitution_groups


def test_elapsed_seconds_handles_period_boundaries_and_overtime() -> None:
    assert _elapsed_seconds(1, "PT12M00.00S") == 0
    assert _elapsed_seconds(2, "PT12M00.00S") == 720
    assert _elapsed_seconds(4, "PT00M00.00S") == 2880
    assert _elapsed_seconds(5, "PT05M00.00S") == 2880


def test_name_normalization_distinguishes_abbreviated_williamses() -> None:
    assert _normalize_name("Jal. Williams") == "jalwilliams"
    assert _normalize_name("Jay. Williams") == "jaywilliams"


def test_substitution_groups_apply_simultaneous_net_changes() -> None:
    frame = pd.DataFrame(
        [
            {"game_id": "1", "period": 1, "clock": "PT06M00.00S", "team_id": 10, "player_id": 1, "direction": "out"},
            {"game_id": "1", "period": 1, "clock": "PT06M00.00S", "team_id": 10, "player_id": 6, "direction": "in"},
            {"game_id": "1", "period": 1, "clock": "PT06M00.00S", "team_id": 10, "player_id": 6, "direction": "out"},
            {"game_id": "1", "period": 1, "clock": "PT06M00.00S", "team_id": 10, "player_id": 7, "direction": "in"},
        ]
    )
    groups = _substitution_groups(frame)
    team_id, outs, ins = groups["1"][360.0][0]
    assert team_id == 10
    assert outs == {1, 6}
    assert ins == {6, 7}
