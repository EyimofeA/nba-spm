#!/usr/bin/env python3
"""Interpret the frozen BoxPIPM winner without changing model selection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

import run_full_spm_history_ablation as base
from run_final_box_feature_ladder import (
    FEATURE_RUN,
    MATRIX_ROOT,
    PLAYER_SHEET_2026,
    POSSESSION_CACHE,
    TARGETS,
    _group_permutation_importance,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT
    / "artifacts/research/final_box_feature_ladder"
    / "final_box_feature_ladder_v1_8bb26f12e7"
)
EXPERIMENT_ID = "final_box_interpretability_v1"
SELECTED_CANDIDATE = "box_15"
REPEATS = 5

BOX_GROUPS = {
    "shooting_scoring": {
        "offense": (
            "PTS_p100",
            "FTA_p100",
            "FTM_p100",
            "FG2A_p100",
            "FG2M_p100",
            "FG3A_p100",
            "FG3M_p100",
        ),
        "defense": (
            "PTS_p100",
            "FTA_p100",
            "FTM_p100",
            "FG2A_p100",
            "FG2M_p100",
            "FG3A_p100",
            "FG3M_p100",
        ),
    },
    "creation_security": {
        "offense": ("AST_p100", "TOV_p100"),
        "defense": ("AST_p100", "TOV_p100"),
    },
    "rebounding": {
        "offense": ("OREB_p100", "DREB_p100"),
        "defense": ("OREB_p100", "DREB_p100"),
    },
    "disruption_fouls": {
        "offense": ("STL_p100", "BLK_p100", "PF_p100", "PFD_p100"),
        "defense": ("STL_p100", "BLK_p100", "PF_p100", "PFD_p100"),
    },
}


def _validate_groups(features: dict[str, tuple[str, ...]]) -> None:
    for side in ("offense", "defense"):
        assigned = [
            feature
            for group in BOX_GROUPS.values()
            for feature in group[side]
        ]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"The {side} interpretation groups overlap.")
        if set(assigned) != set(features[side]):
            raise ValueError(f"The {side} interpretation groups do not cover Box15.")


def _load_models() -> dict:
    models = {}
    for season in base.RATING_SEASONS:
        for side in ("offense", "defense"):
            path = SOURCE_RUN / "models" / f"{season}_{SELECTED_CANDIDATE}_{side}.joblib"
            models[(season, SELECTED_CANDIDATE, side)] = joblib.load(path)
    return models


def _coefficient_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = pd.read_parquet(
        SOURCE_RUN / "selected_standardized_coefficients.parquet"
    ).drop(columns="group")
    mapping = {
        (side, feature): group
        for group, sides in BOX_GROUPS.items()
        for side, fields in sides.items()
        for feature in fields
    }
    coefficients["group"] = [
        mapping[(side, feature)]
        for side, feature in coefficients[["side", "feature"]].itertuples(index=False)
    ]
    summary = (
        coefficients.groupby(["side", "group", "feature"], as_index=False)
        .agg(
            mean_standardized_coefficient=("standardized_coefficient", "mean"),
            mean_absolute_coefficient=(
                "standardized_coefficient",
                lambda values: float(np.mean(np.abs(values))),
            ),
            positive_folds=(
                "standardized_coefficient",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            negative_folds=(
                "standardized_coefficient",
                lambda values: int(np.sum(np.asarray(values) < 0)),
            ),
            folds=("rating_season", "nunique"),
        )
        .sort_values(
            ["side", "mean_absolute_coefficient"],
            ascending=[True, False],
            kind="stable",
        )
    )
    return coefficients, summary


def _active_leaderboard() -> pd.DataFrame:
    ratings = pd.read_parquet(SOURCE_RUN / "ratings.parquet")
    active = ratings.loc[
        ratings["rating_season"].eq(2026)
        & ratings["candidate"].isin(
            (SELECTED_CANDIDATE, f"{SELECTED_CANDIDATE}_aio")
        )
        & ratings["Poss_Off"].gt(0)
        & ratings["Poss_Def"].gt(0)
    ].copy()
    identity = pd.read_parquet(
        PLAYER_SHEET_2026,
        columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"],
    ).drop_duplicates()
    if identity["PLAYER_ID"].duplicated().any():
        raise ValueError("The 2026 identity table contains conflicting players.")
    active = active.merge(identity, on="PLAYER_ID", how="left", validate="many_to_one")
    if active["PLAYER_NAME"].isna().any():
        missing = active.loc[active["PLAYER_NAME"].isna(), "PLAYER_ID"].tolist()
        raise ValueError(f"Active 2026 players lack names: {missing[:10]}")
    active["rank"] = active.groupby("candidate")["net"].rank(
        method="min", ascending=False
    )
    return active.sort_values(
        ["candidate", "rank", "PLAYER_ID"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    source_manifest = json.loads((SOURCE_RUN / "run.json").read_text())
    if source_manifest["selected_candidate"] != SELECTED_CANDIDATE:
        raise ValueError("The source run did not select Box15.")

    candidate_table = pd.read_parquet(SOURCE_RUN / "candidate_features.parquet")
    features = {
        side: tuple(
            candidate_table.loc[
                candidate_table["candidate"].eq(SELECTED_CANDIDATE)
                & candidate_table["side"].eq(side)
            ]
            .sort_values("feature_order", kind="stable")["feature"]
            .tolist()
        )
        for side in ("offense", "defense")
    }
    _validate_groups(features)

    panel, _ = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    models = _load_models()
    annual, _ = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    games = pd.read_parquet(SOURCE_RUN / "game_predictions.parquet")
    detail, group_summary = _group_permutation_importance(
        selected_candidate=SELECTED_CANDIDATE,
        groups=BOX_GROUPS,
        candidate_features=features,
        panel=panel,
        models=models,
        annual=annual,
        games=games,
        repeats=REPEATS,
    )
    coefficients, coefficient_summary = _coefficient_tables()
    leaderboard = _active_leaderboard()

    config = {
        "experiment_id": EXPERIMENT_ID,
        "source_run": source_manifest["run_id"],
        "source_run_sha256": sha256_file(SOURCE_RUN / "run.json"),
        "runner_sha256": sha256_file(Path(__file__)),
        "selected_candidate": SELECTED_CANDIDATE,
        "repeats": REPEATS,
        "permutation_unit": "player row within rating season",
        "score": "downstream next-season AIO game-margin MSE",
        "groups": BOX_GROUPS,
        "leaderboard_filter": "Poss_Off > 0 and Poss_Def > 0 in 2026",
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        ROOT
        / "artifacts/research/final_box_interpretability"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "group_permutation_detail.parquet": detail,
        "group_permutation_summary.parquet": group_summary,
        "selected_standardized_coefficients.parquet": coefficients,
        "coefficient_summary.parquet": coefficient_summary,
        "active_2026_leaderboard.parquet": leaderboard,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": source_manifest["run_id"],
        "selected_candidate": SELECTED_CANDIDATE,
        "config": config,
        "quality": {
            "box_features_per_side": len(features["offense"]),
            "feature_group_overlap": False,
            "feature_group_coverage_complete": True,
            "active_leaderboard_name_coverage": float(
                leaderboard["PLAYER_NAME"].notna().mean()
            ),
            "inactive_rows_removed": int(
                len(
                    pd.read_parquet(SOURCE_RUN / "final_2026_leaderboards.parquet")
                )
                - len(leaderboard)
            ),
            "model_selection_reopened": False,
        },
        "files": {},
        "forbidden_interpretation": [
            "Permutation importance is model dependence, not causal importance.",
            "The corrected leaderboard is descriptive and is not a forecast.",
            "This post-selection audit must not change the frozen model selection.",
        ],
    }
    for name, frame in outputs.items():
        manifest["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(manifest, output / "run.json")
    print(output)
    print(group_summary.to_string(index=False))
    print("\nActive 2026 Box15 AIO leaders")
    print(
        leaderboard.loc[leaderboard["candidate"].eq("box_15_aio")]
        .head(15)[["rank", "PLAYER_NAME", "TEAM_ABBREVIATION", "offense", "defense", "net"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
