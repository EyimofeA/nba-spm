from __future__ import annotations

import pandas as pd

from research.build_dated_box15_history import _game_dates


def test_game_dates_require_one_date_per_game() -> None:
    frame = pd.DataFrame(
        {
            "gameid": ["1", "1", "2"],
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        }
    )
    result = _game_dates(frame)
    assert result["game_id"].tolist() == ["0000000001", "0000000002"]
    assert result["game_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-02",
    ]


def test_game_dates_parse_numeric_yyyymmdd() -> None:
    frame = pd.DataFrame(
        {"gameid": ["225000001", "225000002"], "date": [20251021.0, 20251022.0]}
    )
    result = _game_dates(frame)
    assert result["game_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-10-21",
        "2025-10-22",
    ]
