from __future__ import annotations

import pandas as pd

from nba_impact.data.contracts import POSSESSION_COLUMNS
from nba_impact.data.quality import audit_possession_frame, quarantine_invalid_games


def _row() -> dict:
    return dict(
        zip(
            POSSESSION_COLUMNS,
            [True, 2, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 2024, "2023-10-24", 1, 1, "g1"],
        )
    )


def test_empty_partition_is_critical() -> None:
    report = audit_possession_frame(pd.DataFrame(columns=POSSESSION_COLUMNS), expected_season=2025)
    assert not report.passed
    assert {issue.code for issue in report.issues} == {"empty_partition"}


def test_valid_possession_passes() -> None:
    report = audit_possession_frame(pd.DataFrame([_row()]), expected_season=2024)
    assert report.passed
    assert report.game_count == 1
    assert report.game_type_counts == {"other": 1}


def test_duplicate_key_fails() -> None:
    report = audit_possession_frame(pd.DataFrame([_row(), _row()]), expected_season=2024)
    assert not report.passed
    assert "duplicate_possession_keys" in {issue.code for issue in report.issues}


def test_quarantine_removes_entire_bad_game() -> None:
    first = _row()
    second = {**_row(), "num": 2}
    second["h2"] = second["h1"]
    other = {**_row(), "gameid": "g2", "num": 1}
    valid, rejected, counts = quarantine_invalid_games(pd.DataFrame([first, second, other]))
    assert valid["gameid"].tolist() == ["g2"]
    assert len(rejected) == 2
    assert counts["games"] == 1
