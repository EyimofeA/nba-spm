"""Canonical current annual terminal-lineup zero-prior normal-RAPM targets.

This module intentionally does not reuse the legacy possession cache.  It fits
one regular-season model per completed canonical season and produces the same
target schema as the historical annual-label builder.  The two sources remain
separate until their overlap has been audited.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    load_current_possessions,
    ratings_table,
)


def build_current_single_season_rapm_targets(
    possessions_path: str | Path,
    segments_path: str | Path,
    legacy_names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = (2024, 2025, 2026),
    lambda_off: float = 3000.0,
    lambda_def: float = 4500.0,
    lambda_home: float = 300.0,
) -> dict:
    """Fit one canonical current normal RAPM target table per season.

    ``Season`` follows the NBA season-ending convention.  For example, 2024 is
    the 2023-24 season.  The fit is terminal-lineup, regular season, and
    zero-prior to exactly match the frozen normal-RAPM specification.
    """
    requested = tuple(sorted({int(season) for season in seasons}))
    if not requested:
        raise ValueError("At least one current season is required.")

    frame = load_current_possessions(
        possessions_path,
        segments_path,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    available = {int(value) for value in frame["season"].unique()}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            f"Requested canonical annual normal-RAPM seasons are unavailable: {missing}."
        )
    names = load_current_player_names(legacy_names_path, player_games_path)

    rows: list[pd.DataFrame] = []
    season_quality: list[dict] = []
    for season in requested:
        season_frame = frame.loc[frame["season"].eq(season)].copy()
        if season_frame.empty:
            raise AssertionError(f"Current annual RAPM season {season} is empty.")
        design = build_design(season_frame)
        config = RapmConfig(
            seasons=(season,),
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
            game_types=("regular",),
            data_scope="canonical_current_single_season_terminal_lineup",
        )
        beta, _ = fit_coefficients(design, config)
        ratings = ratings_table(design, beta, names=names).rename(
            columns={
                "player_id": "PLAYER_ID",
                "offense_per_100": "target_offense",
                "defense_per_100": "target_defense",
                "net_per_100": "target_net",
                "off_possessions": "Poss_Off",
                "def_possessions": "Poss_Def",
            }
        )
        ratings["Season"] = season
        rows.append(
            ratings[
                [
                    "PLAYER_ID",
                    "Season",
                    "target_offense",
                    "target_defense",
                    "target_net",
                    "Poss_Off",
                    "Poss_Def",
                ]
            ]
        )
        season_quality.append(
            {
                "season": season,
                "possession_rows": int(len(season_frame)),
                "games": int(season_frame["gameid"].nunique()),
                "players": int(len(design.players)),
                "missing_player_names": int(ratings["player_name"].isna().sum()),
                "minimum_offensive_possessions": int(design.off_possessions.min()),
                "minimum_defensive_possessions": int(design.def_possessions.min()),
            }
        )

    targets = pd.concat(rows, ignore_index=True)
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Canonical annual RAPM targets have duplicate player-season keys.")
    numeric = ["target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    if not np.isfinite(targets[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Canonical annual RAPM targets contain non-finite values.")
    if (targets[["Poss_Off", "Poss_Def"]] <= 0).any().any():
        raise ValueError("Canonical annual RAPM targets require positive side possessions.")
    component_error = np.abs(
        targets["target_net"] - targets["target_offense"] - targets["target_defense"]
    )
    if float(component_error.max()) > 1e-10:
        raise AssertionError("Canonical annual RAPM targets do not satisfy net = offense + defense.")

    run_id = f"current_single_season_rapm_targets_v1_{uuid.uuid4().hex[:10]}"
    output = (
        Path(artifact_root)
        / "models"
        / "current_single_season_rapm_targets"
        / run_id
    )
    output.mkdir(parents=True, exist_ok=False)
    targets.to_parquet(output / "targets.parquet", index=False)
    pd.DataFrame(season_quality).to_parquet(output / "season_quality.parquet", index=False)
    source_hashes = {
        "possessions": sha256_file(possessions_path),
        "possession_lineup_segments": sha256_file(segments_path),
        "player_games": sha256_file(player_games_path),
    }
    legacy_names = Path(legacy_names_path)
    if legacy_names.exists():
        source_hashes["legacy_player_names"] = sha256_file(legacy_names)
    run = {
        "run_id": run_id,
        "model_family": "canonical_current_single_season_zero_prior_normal_rapm_targets",
        "estimand": "single_regular_season_offense_defense_and_net_points_per_100",
        "status": "research_canonical_training_labels",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seasons": list(requested),
            "season_label_convention": "season-ending year",
            "builder_sha256": sha256_file(Path(__file__)),
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_home": lambda_home,
            "lineup_policy": "terminal",
            "prior": "zero",
            "game_types": ["regular"],
            "source_hashes": source_hashes,
        },
        "quality": {
            "rows": int(len(targets)),
            "players": int(targets["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "maximum_component_identity_error": float(component_error.max()),
            "minimum_games_per_season": int(min(item["games"] for item in season_quality)),
            "maximum_missing_player_names": int(
                max(item["missing_player_names"] for item in season_quality)
            ),
        },
        "metrics": {"season_quality": season_quality},
        "targets_path": str((output / "targets.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "One-season RAPM is noisy and is a target for research, not ground truth.",
            "Canonical current targets must pass an overlap comparison before they are joined to legacy annual targets.",
            "Games that fail canonical lineup-quality gates are excluded.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
