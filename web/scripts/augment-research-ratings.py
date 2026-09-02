#!/usr/bin/env python3
"""Add the explicitly labeled Rich SPM research extension to the web snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web/public/data"
RICH = ROOT / "artifacts/research/full_spm_history_ablation/full_spm_history_ablation_v1_34725a86aa/ratings.parquet"


def _write(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _records(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _drop_wp_pulse(row: dict) -> None:
    for key in [key for key in row if key.startswith("wp_pulse_")]:
        del row[key]


def _add_rich_values(row: dict, values: pd.Series) -> None:
    for key, value in values.items():
        row[key] = None if pd.isna(value) else round(float(value), 4)


def _augment_board(additions: pd.DataFrame) -> tuple[list[dict], dict[str, dict]]:
    path = PUBLIC / "leaderboard-2026.json"
    board = _records(path)
    for row in board:
        _drop_wp_pulse(row)
        if row["PLAYER_ID"] in additions.index:
            _add_rich_values(row, additions.loc[row["PLAYER_ID"]])
    return board, {path.name: _write(path, board)}


def _augment_shards(
    additions: pd.DataFrame,
    board: list[dict],
    changed: dict[str, dict],
) -> None:
    index = {int(row["id"]): int(row["shard"]) for row in _records(PUBLIC / "players.json")}
    touched = sorted({index[int(row["PLAYER_ID"])] for row in board if int(row["PLAYER_ID"]) in index})
    for shard in touched:
        filename = f"ratings-{shard:02d}.json"
        path = PUBLIC / filename
        payload = json.loads(path.read_text())
        for player_id, player in payload.items():
            for annual in player.get("annual", []):
                _drop_wp_pulse(annual)
            if int(player_id) not in additions.index:
                continue
            for annual in player.get("annual", []):
                if annual.get("Season") == 2026:
                    _add_rich_values(annual, additions.loc[int(player_id)])
        changed[filename] = _write(path, payload)


def _update_catalog(changed: dict[str, dict]) -> None:
    path = PUBLIC / "catalog.json"
    catalog = json.loads(path.read_text())
    models = catalog["catalog"]["models"]
    rich_model = next(model for model in models if model["id"] == "rich_spm")
    rich_model["seasons"] = sorted(set(rich_model["seasons"]) | {2026})
    models[:] = [model for model in models if model["id"] != "wp_pulse"]
    catalog.setdefault("methods", {})["research_rating_extensions"] = {
        "rich_spm_2026": "Full-feature research fit; not a promoted PULSE prior.",
    }
    changed[path.name] = _write(path, catalog)


def _update_manifest(changed: dict[str, dict]) -> None:
    path = PUBLIC / "snapshot-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].update(changed)
    manifest.setdefault("lineage", {})["research_rating_extensions"] = [RICH.parent.name]
    _write(path, manifest)


def main() -> int:
    rich = pd.read_parquet(RICH)
    rich = rich.loc[
        (rich["rating_season"] == 2026) & (rich["candidate"] == "full_spm"),
        ["PLAYER_ID", "offense", "defense", "net"],
    ].rename(columns={side: f"rich_spm_{side}" for side in ("offense", "defense", "net")})
    if rich.duplicated("PLAYER_ID").any():
        raise ValueError("Research rating extension contains duplicate player ids")
    if not np.allclose(rich["rich_spm_net"], rich["rich_spm_offense"] + rich["rich_spm_defense"]):
        raise ValueError("Rich SPM side identity failed")
    additions = rich.set_index("PLAYER_ID")
    board, changed = _augment_board(additions)
    _augment_shards(additions, board, changed)
    _update_catalog(changed)
    _update_manifest(changed)
    print(json.dumps({"files": len(changed), "rich_rows": len(rich)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
