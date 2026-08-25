"""Fit 2024-26 teammate-effect and observable play-channel RAPMs."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.observable_play_channels import (
    POSSESSION_CHANNELS,
    build_observable_channel_ledger,
    normalize_gabriel_events,
)
from nba_impact.models.channel_rapm import (
    fit_possession_channels,
    fit_teammate_channels,
    teammate_ratings_table,
)
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, load_current_player_names


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/rapm_lab/external/external/merged_playbyplay/old_data"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/teammate_play_channels"
SEASONS = (2024, 2025, 2026)
EVENT_COLUMNS = [
    "game_id", "actionNumber", "actionType", "description", "qualifier",
    "shotResult", "person_id", "assister_id", "stealPersonId",
    "blockPersonId", "teamId",
]


def _source_files() -> dict[int, list[Path]]:
    files = {
        season: sorted(SOURCE.glob(f"*_{season}_rs.parquet")) for season in SEASONS
    }
    for season, paths in files.items():
        if len(paths) != 30:
            raise ValueError(f"Expected 30 Gabriel team files for {season}, found {len(paths)}")
    return files


def _source_digest(files: dict[int, list[Path]]) -> str:
    digest = hashlib.sha256()
    for season in SEASONS:
        for path in files[season]:
            digest.update(path.name.encode())
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _load_events(files: dict[int, list[Path]]) -> pd.DataFrame:
    outputs = []
    for season in SEASONS:
        mirrored = pd.concat(
            [pd.read_parquet(path, columns=EVENT_COLUMNS) for path in files[season]],
            ignore_index=True,
        )
        outputs.append(normalize_gabriel_events(mirrored, season=season))
    return pd.concat(outputs, ignore_index=True)


def _merge_teammate_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    output = tables[0]
    for table in tables[1:]:
        common = [column for column in ("player_name", "opportunities") if column in table]
        output = output.merge(
            table.drop(columns=common), on="player_id", how="outer", validate="one_to_one"
        )
    return output.sort_values("player_id").reset_index(drop=True)


def run() -> dict:
    files = _source_files()
    source_digest = _source_digest(files)
    identity_payload = {
        "runner": sha256_file(Path(__file__)),
        "channel_model": sha256_file(ROOT / "src/nba_impact/models/channel_rapm.py"),
        "ledger_model": sha256_file(ROOT / "src/nba_impact/data/observable_play_channels.py"),
        "possessions": sha256_file(ROOT / "data/lake/silver/possessions.parquet"),
        "segments": sha256_file(ROOT / "data/lake/silver/possession_lineup_segments.parquet"),
        "gabriel_team_files_sha256": source_digest,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"teammate_play_channels_v1_{identity}"
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    possessions = pd.read_parquet(ROOT / "data/lake/silver/possessions.parquet")
    segments = pd.read_parquet(
        ROOT / "data/lake/silver/possession_lineup_segments.parquet"
    )
    base = canonical_terminal_frame(possessions, segments, seasons=SEASONS)
    events = _load_events(files)
    ledger = build_observable_channel_ledger(base, events)
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet"
    )

    teammate_config = {
        "focal_penalty": 3000.0,
        "nuisance_penalty": 12000.0,
        "opponent_penalty": 15000.0,
        "home_penalty": 300.0,
        "batch_size": 50_000,
    }
    offense_targets = ("teammate_scoring", "teammate_turnovers", "teammate_assists")
    offense_fit = fit_teammate_channels(
        ledger.possessions,
        focal_side="offense",
        team_targets=ledger.possessions,
        contributions=ledger.offensive_contributions,
        targets=offense_targets,
        **teammate_config,
    )
    offense_table = teammate_ratings_table(
        offense_fit, names=names, invert=("teammate_turnovers",)
    ).rename(columns={"opportunities": "possession_opportunities"})

    steal_fit = fit_teammate_channels(
        ledger.possessions,
        focal_side="defense",
        team_targets=ledger.possessions,
        contributions=ledger.steal_contributions,
        targets=("teammate_steals",),
        **teammate_config,
    )
    steal_table = teammate_ratings_table(steal_fit, names=names).rename(
        columns={"opportunities": "steal_opportunities"}
    )
    block_fit = fit_teammate_channels(
        ledger.shots,
        focal_side="defense",
        team_targets=ledger.shots,
        contributions=ledger.block_contributions,
        targets=("teammate_blocks",),
        **teammate_config,
    )
    block_table = teammate_ratings_table(block_fit, names=names).rename(
        columns={"opportunities": "block_opportunities"}
    )
    oreb_fit = fit_teammate_channels(
        ledger.rebounds,
        focal_side="offense",
        team_targets=ledger.rebounds,
        contributions=ledger.rebound_contributions,
        targets=("teammate_oreb",),
        **teammate_config,
    )
    oreb_table = teammate_ratings_table(oreb_fit, names=names).rename(
        columns={"opportunities": "oreb_opportunities"}
    )
    dreb_fit = fit_teammate_channels(
        ledger.rebounds,
        focal_side="defense",
        team_targets=ledger.rebounds,
        contributions=ledger.rebound_contributions,
        targets=("teammate_dreb",),
        **teammate_config,
    )
    dreb_table = teammate_ratings_table(dreb_fit, names=names).rename(
        columns={"opportunities": "dreb_opportunities"}
    )
    teammate = _merge_teammate_tables(
        [offense_table, steal_table, block_table, oreb_table, dreb_table]
    )
    teammate.to_parquet(output / "teammate_effect_ratings.parquet", index=False)

    channel_config = RapmConfig(
        seasons=SEASONS,
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="observable_play_channel_rapm_2024_2026",
    )
    channels = fit_possession_channels(
        ledger.possessions,
        targets=POSSESSION_CHANNELS,
        config=channel_config,
        names=names,
    )
    channels.to_parquet(output / "observable_play_channel_ratings.parquet", index=False)

    manifest = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "seasons": list(SEASONS),
        "identity": identity_payload,
        "teammate_estimand": (
            "Points or events per 100 valid opportunities produced by the other four "
            "same-side players, adjusted for their identities and the five opponents."
        ),
        "teammate_penalties": teammate_config,
        "channel_penalties": {
            "offense": channel_config.lambda_off,
            "defense": channel_config.lambda_def,
            "home": channel_config.lambda_home,
        },
        "channel_definitions": {
            "rim": "Two-point shot at 4 feet or closer, plus explicit dunk, layup, alley-oop, or tip finishes.",
            "midrange_frequency": "Non-rim two-point attempts per 100 possessions.",
            "transition": "Points on events carrying the NBA play-by-play fastbreak qualifier.",
            "observable_playtype": (
                "Mutually exclusive shot-finish labels derived from fastbreak, putback, "
                "cutting, driving, pull-up, post-like, and jump-shot descriptions."
            ),
        },
        "quality": ledger.quality,
        "paths": {
            "teammate_effect_ratings": "teammate_effect_ratings.parquet",
            "observable_play_channel_ratings": "observable_play_channel_ratings.parquet",
        },
        "forbidden_interpretation": (
            "These are descriptive lineup-adjusted associations, not causal teammate effects. "
            "The observable shot-finish labels are not Synergy possession play types."
        ),
    }
    write_json_atomic(manifest, run_path)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
