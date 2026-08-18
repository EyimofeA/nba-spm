import json
from pathlib import Path


def test_cdnnba_2026_playoff_manifest_records_the_source_cutoff() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/ingest/cdnnba_2026_playoffs_patch.json").read_text()
    )

    missing = manifest["verified_game_ids_missing_from_source"]
    assert manifest["verified_game_count"] == 60
    assert manifest["expected_game_count"] == 85
    assert len(missing) == len(set(missing)) == 25
    assert all(game_id.startswith("00425") and len(game_id) == 10 for game_id in missing)
