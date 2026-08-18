from pathlib import Path

import pandas as pd

from nba_impact.data.legacy_possession_migration import (
    _game_quality,
    _migrate_accepted_rows,
    migrate_legacy_possession_cache,
)


def _cache_rows(game_id: str = "0021600001") -> pd.DataFrame:
    rows = []
    for period, home_poss, points, event in ((1, 1, 2, 1), (2, 0, 3, 1), (3, 1, 2, 1), (4, 0, 2, 1)):
        rows.append(
            {
                "home_poss": home_poss, "pts": points,
                "a1": 1, "a2": 2, "a3": 3, "a4": 4, "a5": 5,
                "h1": 6, "h2": 7, "h3": 8, "h4": 9, "h5": 10,
                "season": 2017, "date": "2016-10-25", "period": period, "num": event,
                "gameid": game_id,
            }
        )
    return pd.DataFrame(rows)


def _official(game_id: str = "0021600001", home_score: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "project_season": 2017, "season_type": "regular", "game_id": game_id,
            "game_date": "2016-10-25", "home_team_id": 100, "away_team_id": 200,
            "home_score": home_score, "away_score": 5,
        }]
    )


def test_migration_preserves_one_terminal_segment_per_legacy_row() -> None:
    cache = _cache_rows()
    quality = _game_quality(cache, _official(), 2017)
    possessions, segments = _migrate_accepted_rows(cache, quality, 2017)

    assert quality["passed"].tolist() == [True]
    assert len(possessions) == len(cache) == len(segments)
    assert possessions["legacy_event_num"].tolist() == [1, 1, 1, 1]
    assert possessions["possession_number"].tolist() == [1, 2, 3, 4]
    assert segments["lineup_assignment"].eq("legacy_terminal_lineup").all()
    assert segments[[f"home_player_{number}" for number in range(1, 6)]].iloc[0].tolist() == [6, 7, 8, 9, 10]
    assert segments[[f"away_player_{number}" for number in range(1, 6)]].iloc[0].tolist() == [1, 2, 3, 4, 5]


def test_migration_rejects_a_game_without_exact_final_score_conservation() -> None:
    quality = _game_quality(_cache_rows(), _official(home_score=5), 2017)

    assert quality["passed"].tolist() == [False]
    assert "score_not_conserved" in quality.loc[0, "issues"]


def test_full_builder_writes_only_the_passing_games(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    good = _cache_rows()
    bad = _cache_rows("0021600002")
    bad.loc[0, "pts"] = 1
    pd.concat([good, bad], ignore_index=True).to_parquet(cache_dir / "matchups_2017.parquet", index=False)
    official_path = tmp_path / "official.parquet"
    pd.concat([_official(), _official("0021600002")], ignore_index=True).to_parquet(official_path, index=False)
    output = tmp_path / "possessions.parquet"
    segments = tmp_path / "segments.parquet"
    identity = tmp_path / "identity.parquet"
    quality = tmp_path / "quality.parquet"
    report = tmp_path / "report.json"

    result = migrate_legacy_possession_cache(
        cache_dir, official_path, output, segments, identity, quality, report, seasons=(2017,)
    )

    assert result["complete"] is False
    assert result["quality"]["accepted_games"] == 1
    assert pd.read_parquet(output)["game_id"].unique().tolist() == ["0021600001"]
    assert pd.read_parquet(quality)["passed"].sum() == 1
