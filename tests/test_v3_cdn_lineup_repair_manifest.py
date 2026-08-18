from __future__ import annotations

import json
from pathlib import Path


def test_v3_cdn_repair_manifest_is_pinned_and_strict() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs" / "ingest" / "current_lineup_v3_cdn_repair_candidates.json").read_text()
    )

    assert manifest["dataset_revision"] == "dfa8fa43f89ae2ca6c18db524edc2050a6bb2286"
    assert manifest["target_game_ids"] == ["0022301210", "0022300339", "0022400061", "0022500264"]
    assert "Clock-only joins are forbidden" in manifest["alignment_policy"]
    assert len(manifest["tasks"]) == 3
    for task in manifest["tasks"]:
        assert "/resolve/dfa8fa43f89ae2ca6c18db524edc2050a6bb2286/" in task["url"]
        assert len(task["expected_sha256"]) == 64
        assert {"gameId", "actionId", "period", "clock", "description", "personId", "teamId"}.issubset(
            task["required_columns"]
        )
