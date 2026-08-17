"""Build a static, derived-data snapshot for the ratings web client."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.api.ratings import ROLE_LABELS, RatingsApiConfig, RatingsStore


MODEL_CATALOG = [
    {
        "id": "aio",
        "label": "AIO",
        "scopes": ["1y"],
        "components": ["net", "offense", "defense"],
        "uncertainty": False,
    },
    {
        "id": "spm",
        "label": "SPM",
        "scopes": ["1y"],
        "components": ["net", "offense", "defense"],
        "uncertainty": False,
    },
    {
        "id": "rapm",
        "label": "RAPM",
        "scopes": ["1y", "3y", "5y", "2022-24", "2025"],
        "components": ["net", "offense", "defense"],
        "uncertainty": True,
    },
]


def build_web_snapshot(
    config_path: str | Path,
    artifact_root: str | Path,
    aging_curve_path: str | Path,
    output_dir: str | Path,
    *,
    shards: int = 32,
) -> dict:
    """Write a small index plus sharded derived player ratings."""
    if shards < 1:
        raise ValueError("shards must be positive.")
    config = RatingsApiConfig.from_json(config_path)
    store = RatingsStore(config, artifact_root)
    player_ids = sorted(
        set(store.annual["PLAYER_ID"].astype(int))
        | set(store.rolling["PLAYER_ID"].astype(int))
        | set(store.current["player_id"].astype(int))
    )
    players: dict[str, dict] = {}
    index = []
    for player_id in player_ids:
        player = store.player(player_id)
        if player is None:
            continue
        player["peaks"] = []
        player["matchup_defense_factors"] = []
        players[str(player_id)] = player
        index.append(
            {"id": player_id, "name": player["PLAYER_NAME"], "shard": player_id % shards}
        )
    index.sort(key=lambda item: (item["name"].casefold(), item["id"]))

    aging = pd.read_csv(aging_curve_path)
    aging = aging.astype(object).where(aging.notna(), None)
    aging_rows = aging.to_dict(orient="records")
    catalog = {
        "schema_version": "nba_impact_web_snapshot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": {
            "models": MODEL_CATALOG,
            "role_labels": ROLE_LABELS,
            "interval_scopes": {
                "2022-24": "trailing_2022_2024",
                "2025": "single_season_2025",
            },
        },
        "lineage": {
            "annual_run_id": config.annual_run_id,
            "rolling_run_id": config.rolling_run_id,
            "current_rapm_run_id": config.current_rapm_run_id,
            "side_roles_run_id": config.side_roles_run_id,
            "uncertainty_run_ids": config.normal_rapm_uncertainty_run_ids,
        },
        "aging": {
            "status": "exploratory_descriptive",
            "label": "Observed aging curve",
            "rows": aging_rows,
        },
        "shards": shards,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def write(name: str, value: object) -> tuple[int, str]:
        serialized = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        destination = output / name
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(serialized)
        temporary.replace(destination)
        return len(serialized.encode()), hashlib.sha256(serialized.encode()).hexdigest()

    files = {}
    files["catalog.json"] = write("catalog.json", catalog)
    files["players.json"] = write("players.json", index)
    for shard in range(shards):
        name = f"ratings-{shard:02d}.json"
        rows = {
            player_id: player
            for player_id, player in players.items()
            if int(player_id) % shards == shard
        }
        files[name] = write(name, rows)
    total_bytes = sum(size for size, _ in files.values())
    return {
        "output_dir": str(output.resolve()),
        "players": len(players),
        "shards": shards,
        "bytes": total_bytes,
        "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in files.items()},
    }
