import json
from pathlib import Path

from nba_impact.data.download import load_tasks


def test_gabriel_playoff_tail_manifest_is_complete_and_quarantined() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/ingest/gabriel_2026_playoff_tail_lineups.json"

    manifest, tasks = load_tasks(path)

    missing = manifest["verified_game_ids_missing_from_cdnnba"]
    mapped = [
        game_id
        for task_name in manifest["target_game_ids_by_task"]
        for game_id in manifest["target_game_ids_by_task"][task_name]
    ]
    assert len(missing) == 25
    assert len(mapped) == len(set(mapped)) == 25
    assert set(mapped) == set(missing)
    assert set(manifest["target_game_ids_by_task"]) == {task.name for task in tasks}
    assert manifest["license"] == "not_declared_research_only"
    assert all(task.license == "not_declared_research_only" for task in tasks)


def test_gabriel_playoff_tail_manifest_pins_lineup_not_possession_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/ingest/gabriel_2026_playoff_tail_lineups.json"
    manifest = json.loads(path.read_text())
    _, tasks = load_tasks(path)

    assert manifest["dataset_revision"] == "6e077a0f62153e72db300ba1f0a45b30584fd3d2"
    assert len(tasks) == 4
    for task in tasks:
        assert task.expected_bytes is not None
        assert task.expected_sha256 is not None
        assert {"players_on", "off_players_on", "def_players_on"}.issubset(
            task.required_columns
        )
        assert "possession" not in task.required_columns
        assert "orderNumber" not in task.required_columns
