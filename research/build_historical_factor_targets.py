#!/usr/bin/env python3
"""Build annual and rolling five-year TS and opponent-OREB factor targets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.historical_factor_targets import (
    build_historical_factor_ledger,
    fit_historical_factor_ratings,
    load_gabriel_events,
)
from nba_impact.models.rapm import load_current_player_names


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/lake/bronze/gabriel_merged_playbyplay"
EVENT_ROOT = SOURCE / "old_data"
NAMES = ROOT / "rapm/data/all_names.csv"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
SCHEDULE_ROOT = ROOT / "data/lake/bronze/official_game_schedule_1997_2026"
CURRENT_GAME_DIM = ROOT / "data/lake/silver/game_dim.parquet"
OUTPUT_ROOT = ROOT / "artifacts/research/historical_factor_targets"
LEDGER_CACHE = OUTPUT_ROOT / "ledger_cache_6e077a0f_v1"
SEASONS = tuple(range(2014, 2027))
WINDOW_ENDS = tuple(range(2018, 2027))
EXPECTED_REVISION = "6e077a0f62153e72db300ba1f0a45b30584fd3d2"


def _revision() -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _game_dim(season: int, event_games: pd.Series) -> tuple[pd.DataFrame, dict]:
    if season <= 2024:
        path = SCHEDULE_ROOT / f"schedule_{season}.parquet"
        games = pd.read_parquet(path)[["game_id", "home_team_id", "away_team_id"]]
        source = str(path.relative_to(ROOT))
    else:
        games = pd.read_parquet(CURRENT_GAME_DIM)
        games = games.loc[
            games["source_season"].eq(season - 1),
            ["game_id", "home_team_id", "away_team_id"],
        ]
        source = str(CURRENT_GAME_DIM.relative_to(ROOT))
    games = games.copy()
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    games = games.drop_duplicates("game_id")
    observed = set(event_games.astype(str))
    scheduled = set(games["game_id"])
    return games, {
        "game_identity_source": source,
        "scheduled_games": len(scheduled),
        "event_games_missing_schedule_identity": len(observed - scheduled),
        "scheduled_games_missing_events": len(scheduled - observed),
    }


def main() -> None:
    revision = _revision()
    if revision != EXPECTED_REVISION:
        raise ValueError(f"Expected Gabriel revision {EXPECTED_REVISION}, found {revision}.")
    if max(SEASONS) >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    source_hashes = {
        "builder": sha256_file(Path(__file__)),
        "factor_model": sha256_file(
            ROOT / "src/nba_impact/models/historical_factor_targets.py"
        ),
    }
    config = {
        "experiment_id": "historical_factor_targets_v1",
        "status": "frozen_research_build",
        "seasons": list(SEASONS),
        "five_year_window_ends": list(WINDOW_ENDS),
        "lambda_off": 3000.0,
        "lambda_def": 3000.0,
        "lambda_home": 300.0,
        "gabriel_revision": revision,
        "minimum_valid_lineup_fraction": 0.95,
        "ledger_cache_contract": str(LEDGER_CACHE.relative_to(ROOT)),
        "source_hashes": source_hashes,
        "game_identity_sources": {
            **{
                str(season): sha256_file(SCHEDULE_ROOT / f"schedule_{season}.parquet")
                for season in range(2014, 2025)
            },
            "current_game_dim": sha256_file(CURRENT_GAME_DIM),
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"historical_factor_targets_v1_{identity}"
    if (output / "run.json").exists():
        print(output)
        return

    output.mkdir(parents=True, exist_ok=True)
    ledger_root = LEDGER_CACHE
    annual_root = output / "annual"
    rolling_root = output / "five_year"
    ledger_root.mkdir(parents=True, exist_ok=True)
    annual_root.mkdir(exist_ok=True)
    rolling_root.mkdir(exist_ok=True)
    quality_rows = []
    for season in SEASONS:
        season_root = ledger_root / f"season={season}"
        shooting_path = season_root / "shooting.parquet"
        rebound_path = season_root / "opponent_oreb.parquet"
        quality_path = season_root / "quality.json"
        if not (shooting_path.exists() and rebound_path.exists() and quality_path.exists()):
            events, source_quality = load_gabriel_events(EVENT_ROOT, season)
            game_dim, game_quality = _game_dim(season, events["game_id"])
            ledger = build_historical_factor_ledger(events, game_dim)
            season_root.mkdir(exist_ok=True)
            ledger.shooting.to_parquet(shooting_path, index=False)
            ledger.opponent_oreb.to_parquet(rebound_path, index=False)
            write_json_atomic(
                {**source_quality, **game_quality, **ledger.quality}, quality_path
            )
        quality_rows.append(json.loads(quality_path.read_text()))
    quality = pd.DataFrame(quality_rows).sort_values("season")
    if quality["games_missing_home_team"].sum() != 0:
        raise ValueError("One or more source games lack canonical home-team identity.")
    if quality["team_game_score_match_rate"].min() < 0.99:
        raise ValueError("A source season fails the 99% team-score conservation gate.")
    invalid_shot_rate = quality["invalid_shooting_lineups"] / (
        quality["shooting_rows"] + quality["invalid_shooting_lineups"]
    )
    invalid_rebound_rate = quality["invalid_rebound_lineups"] / (
        quality["resolved_misses"] + quality["invalid_rebound_lineups"]
    )
    if invalid_shot_rate.max() > 0.05 or invalid_rebound_rate.max() > 0.05:
        raise ValueError("A source season fails the 95% valid-lineup gate.")

    def load_ledger(season: int):
        from nba_impact.models.historical_factor_targets import HistoricalFactorLedger

        season_root = ledger_root / f"season={season}"
        return HistoricalFactorLedger(
            pd.read_parquet(season_root / "shooting.parquet"),
            pd.read_parquet(season_root / "opponent_oreb.parquet"),
            json.loads((season_root / "quality.json").read_text()),
        )

    annual_rows = []
    for season in SEASONS:
        path = annual_root / f"season={season}.parquet"
        if not path.exists():
            ratings = fit_historical_factor_ratings({season: load_ledger(season)}, (season,))
            ratings["Season"] = season
            ratings.to_parquet(path, index=False)
        annual_rows.append(pd.read_parquet(path))
    annual = pd.concat(annual_rows, ignore_index=True)

    rolling_rows = []
    for window_end in WINDOW_ENDS:
        window = tuple(range(window_end - 4, window_end + 1))
        path = rolling_root / f"window_end={window_end}.parquet"
        if not path.exists():
            ledgers = {season: load_ledger(season) for season in window}
            ratings = fit_historical_factor_ratings(ledgers, window)
            ratings["Window_Start"] = window[0]
            ratings["Window_End"] = window_end
            ratings.to_parquet(path, index=False)
        rolling_rows.append(pd.read_parquet(path))
    rolling = pd.concat(rolling_rows, ignore_index=True)

    names = load_current_player_names(NAMES, PLAYER_GAMES).rename(
        columns={"PLAYER_ID": "player_id"}
    )
    annual = annual.merge(names, on="player_id", how="left", validate="many_to_one")
    rolling = rolling.merge(names, on="player_id", how="left", validate="many_to_one")
    annual["PLAYER_NAME"] = annual["PLAYER_NAME"].fillna(annual["player_id"].astype(str))
    rolling["PLAYER_NAME"] = rolling["PLAYER_NAME"].fillna(rolling["player_id"].astype(str))

    annual.to_parquet(output / "annual_factor_targets.parquet", index=False)
    rolling.to_parquet(output / "five_year_factor_targets.parquet", index=False)
    quality.to_parquet(output / "source_quality.parquet", index=False)
    write_json_atomic(
        {
            "run_id": output.name,
            "status": "research_targets",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "quality": {
                "annual_rows": int(len(annual)),
                "five_year_rows": int(len(rolling)),
                "annual_seasons": list(SEASONS),
                "five_year_window_ends": list(WINDOW_ENDS),
                "minimum_team_game_score_match_rate": float(
                    quality["team_game_score_match_rate"].min()
                ),
                "maximum_invalid_shooting_lineup_rate": float(invalid_shot_rate.max()),
                "maximum_invalid_rebound_lineup_rate": float(invalid_rebound_rate.max()),
                "minimum_valid_shooting_lineup_fraction": float(1.0 - invalid_shot_rate.max()),
                "minimum_valid_rebound_lineup_fraction": float(1.0 - invalid_rebound_rate.max()),
                "season_2027_loaded": False,
            },
            "paths": {
                "annual_factor_targets": "annual_factor_targets.parquet",
                "five_year_factor_targets": "five_year_factor_targets.parquet",
                "source_quality": "source_quality.parquet",
            },
            "forbidden_interpretation": (
                "These are research-only lineup-adjusted factor targets. They do not "
                "identify a unique shot defender or a unique box-out assignment."
            ),
        },
        output / "run.json",
    )
    print(output)


if __name__ == "__main__":
    main()
