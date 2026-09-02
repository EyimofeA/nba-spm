from research.build_canonical_lineup_stints import elapsed_seconds, lineup_ids


def test_elapsed_seconds_handles_regulation_and_overtime() -> None:
    assert elapsed_seconds(1, "12:00") == 0
    assert elapsed_seconds(2, "12:00") == 720
    assert elapsed_seconds(4, "00:00") == 2880
    assert elapsed_seconds(5, "05:00") == 2880
    assert elapsed_seconds(6, "00:00") == 3480
    assert elapsed_seconds(1, "PT11M30.50S") == 29.5


def test_lineup_parser_returns_five_sorted_ids() -> None:
    value = "23 A, 7 B, 19 C, 2 D, 11 E"
    assert lineup_ids(value) == (2, 7, 11, 19, 23)
