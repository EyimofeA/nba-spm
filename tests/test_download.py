from __future__ import annotations

import pandas as pd

from nba_impact.data.download import DownloadTask, _validate_file, ingest_task


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


def test_ingest_promotes_complete_partial_without_network(tmp_path) -> None:
    destination = tmp_path / "source" / "sample.parquet"
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.parent.mkdir(parents=True)
    pd.DataFrame({"game_id": [1], "value": [2]}).to_parquet(partial, index=False)
    task = DownloadTask(
        name="complete_partial",
        url="https://unused.invalid/sample.parquet",
        destination="source/sample.parquet",
        provider="test",
        license="test",
        required_columns=("game_id",),
    )

    result = ingest_task(task, root=tmp_path)

    assert result["status"] == "downloaded"
    assert destination.exists()
    assert not partial.exists()
