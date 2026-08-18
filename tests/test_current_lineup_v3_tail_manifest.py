import json
from pathlib import Path

from nba_impact.data.download import load_tasks


def test_v3_playoff_tail_manifest_is_pinned_and_lineup_capable() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/ingest/current_lineup_v3_playoff_tail_2026.json"

    manifest, tasks = load_tasks(path)

    assert manifest["dataset_revision"] == "dfa8fa43f89ae2ca6c18db524edc2050a6bb2286"
    assert len(manifest["verified_game_ids_missing_from_cdnnba"]) == 25
    assert manifest["verified_game_ids_missing_from_cdnnba"][0] == "0042500204"
    assert manifest["verified_game_ids_missing_from_cdnnba"][-1] == "0042500405"
    assert len(tasks) == 1
    task = tasks[0]
    assert task.expected_bytes == 1010592
    assert task.expected_min_rows == 43000
    assert {"gameId", "actionId", "description", "personId", "teamId"}.issubset(
        task.required_columns
    )
    assert "po_2025.parquet" in task.url


def test_v3_playoff_tail_manifest_keeps_the_verified_ids_as_strings() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/ingest/current_lineup_v3_playoff_tail_2026.json").read_text()
    )

    assert all(value.startswith("00425") and len(value) == 10 for value in manifest["verified_game_ids_missing_from_cdnnba"])
