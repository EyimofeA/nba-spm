from __future__ import annotations

from nba_impact.data.download import DownloadTask, _validate_file


def test_validate_csv_contract(tmp_path) -> None:
    path = tmp_path / "players.csv"
    path.write_text("PLAYER_ID,year,value\n1,2026,3.2\n2,2026,4.1\n")
    task = DownloadTask(
        name="players",
        url="https://example.invalid/players.csv",
        destination="players.csv",
        provider="fixture",
        license="fixture",
        expected_min_rows=2,
        required_columns=("PLAYER_ID", "year"),
    )
    result = _validate_file(path, task)
    assert result["rows"] == 2
    assert result["columns"] == ["PLAYER_ID", "year", "value"]
