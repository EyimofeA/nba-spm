from __future__ import annotations

from nba_impact.data.normalize import classify_game_type


def test_game_type_uses_id_prefix_not_date() -> None:
    assert classify_game_type("0022300001") == "regular"
    assert classify_game_type("0042300001") == "playoffs"
    assert classify_game_type("0052300001") == "play_in"
    assert classify_game_type("0062300001") == "other"
    assert classify_game_type("unexpected") == "other"
