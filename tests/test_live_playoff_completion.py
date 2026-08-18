from pathlib import Path

from nba_impact.data.live_playoff_completion import _read_live_game


def test_read_live_game_requires_identity_and_possession_fields(tmp_path: Path) -> None:
    path = tmp_path / "live_0042500001.json"
    path.write_text(
        '{"game":{"gameId":"0042500001","actions":['
        '{"actionNumber":1,"orderNumber":10000,"period":1,"clock":"PT12M00.00S",'
        '"possession":0,"scoreHome":"0","scoreAway":"0"}]}}'
    )

    frame = _read_live_game(path, "0042500001")

    assert len(frame) == 1
    assert frame.loc[0, "game_id"] == "0042500001"
    assert int(frame.loc[0, "orderNumber"]) == 10000


def test_read_live_game_rejects_wrong_game_identity(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(
        '{"game":{"gameId":"0042500002","actions":['
        '{"actionNumber":1,"orderNumber":10000,"period":1,"clock":"PT12M00.00S",'
        '"possession":0,"scoreHome":"0","scoreAway":"0"}]}}'
    )

    try:
        _read_live_game(path, "0042500001")
    except ValueError as exc:
        assert "expected game" in str(exc)
    else:
        raise AssertionError("wrong game identity should fail")
