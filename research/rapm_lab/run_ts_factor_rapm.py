"""Replace the eFG shooting factor with denominator-weighted true shooting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.possession_outcome_rapm import (
    build_factor_ledger,
    fit_factor_ratings,
    fit_weighted_factor_ratings,
)
from nba_impact.models.rapm import RapmConfig


ROOT = Path(__file__).resolve().parents[2]
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
EVENTS = ROOT / "data/lake/silver/event_states.parquet"
OUTPUT = ROOT / "research/rapm_lab/outputs/ts_factor_rapm"


def run() -> dict:
    ledger = build_factor_ledger(
        pd.read_parquet(POSSESSIONS),
        pd.read_parquet(SEGMENTS),
        pd.read_parquet(EVENTS),
        seasons=(2024, 2025, 2026),
    )
    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="canonical_three_season_ts_factor_rapm",
    )
    factors = {
        "shooting_ts": fit_weighted_factor_ratings(
            ledger.ts_opportunities,
            "ts_value",
            "ts_attempt_weight",
            factor="shooting_ts",
            config=config,
        ),
        "turnover": fit_factor_ratings(
            ledger.possessions,
            "turnover",
            factor="turnover",
            higher_is_good_for_offense=False,
            config=config,
        ),
        "offensive_rebound": fit_factor_ratings(
            ledger.rebound_opportunities,
            "offensive_rebound",
            factor="offensive_rebound",
            higher_is_good_for_offense=True,
            config=config,
        ),
    }
    ratings = None
    for name, frame in factors.items():
        columns = [
            "player_id",
            f"{name}_offense",
            f"{name}_defense",
            f"{name}_net",
            f"{name}_off_exposure",
            f"{name}_def_exposure",
        ]
        ratings = frame[columns] if ratings is None else ratings.merge(
            frame[columns], on="player_id", how="outer", validate="one_to_one"
        )
    identity = hashlib.sha256(
        json.dumps(
            {
                "possessions": sha256_file(POSSESSIONS),
                "segments": sha256_file(SEGMENTS),
                "events": sha256_file(EVENTS),
                "runner": sha256_file(Path(__file__)),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"ts_factor_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output / "factor_ratings.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "descriptive_research",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": [2024, 2025, 2026],
        "penalties": {"offense": 3000, "defense": 3000, "home": 300},
        "factor_estimands": {
            "shooting_ts": "lineup effect on points divided by 2 times FGA plus 0.44 FTA, weighted by TS attempt denominator",
            "turnover": "turnover avoidance and forcing per 100 possessions",
            "offensive_rebound": "offensive rebound conversion and prevention per 100 resolved missed-FG rebound opportunities"
        },
        "factor_sign": "positive is good on offense and defense",
        "quality": ledger.quality,
        "artifact": "factor_ratings.parquet",
        "forbidden_interpretation": "An additive decomposition of points RAPM or a production all-in-one rating.",
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
