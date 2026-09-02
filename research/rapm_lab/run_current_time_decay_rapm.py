#!/usr/bin/env python3
"""Build the selected current five-year time-decayed RAPM leaderboard."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import AgeDesign, fit_age_adjusted_rapm, season_decay_weights
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    load_current_player_names,
    load_unified_terminal_possessions,
    ratings_table,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / "research/rapm_lab/outputs/time_decay_actual_age_5y_rapm/time_decay_actual_age_5y_rapm_v1_d60aedf265/run.json"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/current_time_decay_rapm"
SEASONS = (2022, 2023, 2024, 2025, 2026)


def run() -> Path:
    source = json.loads(SOURCE_RUN.read_text())
    selected = source["selected"]
    payload = {
        "source_selection_run": sha256_file(SOURCE_RUN),
        "runner": sha256_file(Path(__file__)),
        "seasons": SEASONS,
        "half_life_years": selected["half_life_years"],
        "player_penalty": selected["player_penalty"],
        "lambda_home": 300.0,
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"current_time_decay_rapm_v1_{identity}"
    if (output / "run.json").exists():
        return output
    output.mkdir(parents=True, exist_ok=True)

    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        SEASONS,
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    weights = season_decay_weights(
        design.seasons,
        window_end=max(SEASONS),
        half_life_years=float(selected["half_life_years"]),
    )
    empty_age = AgeDesign(
        X=sparse.csr_matrix((len(frame), 0), dtype=float),
        ages=np.array([], dtype=int),
        reference_age=27,
        known_slots=0,
        total_slots=0,
    )
    config = RapmConfig(
        seasons=SEASONS,
        lambda_off=float(selected["player_penalty"]),
        lambda_def=float(selected["player_penalty"]),
        lambda_home=300.0,
        data_scope="five_year_time_decay_no_age",
    )
    fit = fit_age_adjusted_rapm(
        design,
        empty_age,
        config,
        age_penalty=0.0,
        row_weights=weights,
    )
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet"
    )
    ratings = ratings_table(design, fit.player_coefficients, names=names).rename(columns={
        "offense_per_100": "offense",
        "defense_per_100": "defense",
        "net_per_100": "net",
    })
    ratings["window_start"] = min(SEASONS)
    ratings["window_end"] = max(SEASONS)
    ratings["half_life_years"] = float(selected["half_life_years"])
    ratings.to_parquet(output / "ratings.parquet", index=False)
    identity_error = float(np.max(np.abs(ratings["offense"] + ratings["defense"] - ratings["net"])))
    write_json_atomic({
        "run_id": output.name,
        "status": "research_leaderboard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": "current_five_year_time_decay_rapm_v1",
        "config": payload,
        "quality": {
            "games": int(frame["gameid"].nunique()),
            "possessions": int(len(frame)),
            "players": int(len(ratings)),
            "maximum_side_identity_error": identity_error,
        },
        "selection_warning": (
            "The half-life and penalty come from the frozen actual-age selection run. "
            "That candidate did not pass its later diagnostic gate."
        ),
        "files": {"ratings": "ratings.parquet"},
    }, output / "run.json")
    return output


if __name__ == "__main__":
    print(run())
