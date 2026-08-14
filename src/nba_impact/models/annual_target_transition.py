"""Controlled transition from legacy to canonical annual normal-RAPM targets."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import load_current_player_names


REQUIRED_COLUMNS = (
    "PLAYER_ID",
    "Season",
    "target_offense",
    "target_defense",
    "target_net",
    "Poss_Off",
    "Poss_Def",
)
COMPONENTS = ("offense", "defense", "net")


def _validate_targets(targets: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(targets.columns))
    if missing:
        raise ValueError(f"{source} annual targets are missing columns: {missing}.")
    frame = targets.loc[:, REQUIRED_COLUMNS].copy()
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    frame["Season"] = pd.to_numeric(frame["Season"], errors="raise").astype(int)
    numeric = [column for column in REQUIRED_COLUMNS if column not in {"PLAYER_ID", "Season"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise").astype(float)
    if frame.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError(f"{source} annual targets have duplicate player-season keys.")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError(f"{source} annual targets contain non-finite values.")
    if (frame[["Poss_Off", "Poss_Def"]] <= 0).any().any():
        raise ValueError(f"{source} annual targets require positive side possessions.")
    component_error = np.abs(
        frame["target_net"] - frame["target_offense"] - frame["target_defense"]
    )
    if float(component_error.max()) > 1e-9:
        raise ValueError(f"{source} annual targets violate net = offense + defense.")
    return frame.sort_values(["Season", "PLAYER_ID"], kind="stable").reset_index(drop=True)


def _exclude_invalid_side_exposure(targets: pd.DataFrame, *, source: str) -> tuple[pd.DataFrame, int]:
    missing = sorted({"Poss_Off", "Poss_Def"} - set(targets.columns))
    if missing:
        raise ValueError(f"{source} annual targets are missing columns: {missing}.")
    exposure = targets[["Poss_Off", "Poss_Def"]].apply(pd.to_numeric, errors="raise")
    invalid = (exposure <= 0).any(axis=1)
    return targets.loc[~invalid].copy(), int(invalid.sum())


def _overlap_metrics(legacy: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    merged = legacy.merge(
        current,
        on=["PLAYER_ID", "Season"],
        how="inner",
        suffixes=("_legacy", "_canonical"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Legacy and canonical annual targets have no overlap.")
    rows = []
    for season, frame in merged.groupby("Season", sort=True):
        for component in COMPONENTS:
            legacy_values = frame[f"target_{component}_legacy"]
            canonical_values = frame[f"target_{component}_canonical"]
            legacy_sd = float(legacy_values.std(ddof=0))
            rows.append(
                {
                    "season": int(season),
                    "component": component,
                    "matched_players": int(len(frame)),
                    "legacy_coverage": float(len(frame) / len(legacy.loc[legacy["Season"].eq(season)])),
                    "canonical_coverage": float(len(frame) / len(current.loc[current["Season"].eq(season)])),
                    "pearson": float(legacy_values.corr(canonical_values)),
                    "spearman": float(legacy_values.corr(canonical_values, method="spearman")),
                    "rmse": float(np.sqrt(np.mean((canonical_values - legacy_values) ** 2))),
                    "mean_delta_canonical_minus_legacy": float(
                        (canonical_values - legacy_values).mean()
                    ),
                    "legacy_sd": legacy_sd,
                    "canonical_sd": float(canonical_values.std(ddof=0)),
                    "canonical_to_legacy_sd_ratio": float(
                        canonical_values.std(ddof=0) / legacy_sd
                    )
                    if legacy_sd > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_canonical_annual_target_panel(
    legacy_targets_path: str | Path,
    canonical_targets_path: str | Path,
    legacy_names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
    transition_season: int = 2024,
    minimum_overlap_correlation: float = 0.95,
    minimum_overlap_coverage: float = 0.95,
    minimum_sd_ratio: float = 0.80,
    maximum_sd_ratio: float = 1.25,
    maximum_absolute_mean_delta: float = 0.10,
) -> dict:
    """Audit and join annual normal-RAPM sources at one explicit season boundary.

    Canonical rows replace legacy rows from ``transition_season`` onward.  The
    overlap is an engineering compatibility check, not an estimate of player
    accuracy.  The function fails closed before a source join that would make a
    trajectory show a source-induced discontinuity.
    """
    legacy_path = Path(legacy_targets_path)
    canonical_path = Path(canonical_targets_path)
    legacy_raw, legacy_excluded = _exclude_invalid_side_exposure(
        pd.read_parquet(legacy_path), source="Legacy"
    )
    canonical_raw, canonical_excluded = _exclude_invalid_side_exposure(
        pd.read_parquet(canonical_path), source="Canonical"
    )
    legacy = _validate_targets(legacy_raw, source="Legacy")
    canonical = _validate_targets(canonical_raw, source="Canonical")
    if transition_season not in set(legacy["Season"]):
        raise ValueError("Transition season is absent from legacy annual targets.")
    if transition_season not in set(canonical["Season"]):
        raise ValueError("Transition season is absent from canonical annual targets.")

    overlap = _overlap_metrics(
        legacy.loc[legacy["Season"].ge(transition_season)],
        canonical.loc[canonical["Season"].ge(transition_season)],
    )
    transition_overlap = overlap.loc[overlap["season"].eq(transition_season)].copy()
    failures = transition_overlap.loc[
        transition_overlap["pearson"].lt(minimum_overlap_correlation)
        | transition_overlap[["legacy_coverage", "canonical_coverage"]]
        .min(axis=1)
        .lt(minimum_overlap_coverage)
        | transition_overlap["canonical_to_legacy_sd_ratio"].lt(minimum_sd_ratio)
        | transition_overlap["canonical_to_legacy_sd_ratio"].gt(maximum_sd_ratio)
        | transition_overlap["mean_delta_canonical_minus_legacy"]
        .abs()
        .gt(maximum_absolute_mean_delta)
    ]
    if not failures.empty:
        failing = ", ".join(failures["component"].tolist())
        raise ValueError(
            f"Annual target source transition failed at season {transition_season}: {failing}."
        )

    panel = pd.concat(
        [
            legacy.loc[legacy["Season"].lt(transition_season)].assign(
                annual_target_source="legacy"
            ),
            canonical.loc[canonical["Season"].ge(transition_season)].assign(
                annual_target_source="canonical_current"
            ),
        ],
        ignore_index=True,
    ).sort_values(["Season", "PLAYER_ID"], kind="stable")
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise AssertionError("Transition panel has duplicate player-season keys.")
    expected = set(range(int(panel["Season"].min()), int(panel["Season"].max()) + 1))
    actual = set(panel["Season"])
    if missing := sorted(expected - actual):
        raise ValueError(f"Transition panel has missing annual seasons: {missing}.")
    names = load_current_player_names(legacy_names_path, player_games_path)

    run_id = f"canonical_annual_target_panel_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "canonical_annual_target_panel" / run_id
    output.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "targets.parquet", index=False)
    overlap.to_parquet(output / "overlap_metrics.parquet", index=False)
    names.to_csv(output / "player_names.csv", index=False)
    source_hashes = {
        "legacy_targets": sha256_file(legacy_path),
        "canonical_targets": sha256_file(canonical_path),
        "legacy_player_names": sha256_file(legacy_names_path),
        "player_games": sha256_file(player_games_path),
    }
    run = {
        "run_id": run_id,
        "model_family": "annual_normal_rapm_source_transition",
        "estimand": "annual_normal_rapm_target_panel_with_audited_source_transition",
        "status": "research_source_transition_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "transition_season": int(transition_season),
            "season_label_convention": "season-ending year",
            "compatibility_gates": {
                "minimum_overlap_correlation": minimum_overlap_correlation,
                "minimum_overlap_coverage": minimum_overlap_coverage,
                "minimum_sd_ratio": minimum_sd_ratio,
                "maximum_sd_ratio": maximum_sd_ratio,
                "maximum_absolute_mean_delta": maximum_absolute_mean_delta,
            },
            "source_hashes": source_hashes,
            "builder_sha256": sha256_file(Path(__file__)),
            "invalid_side_exposure_rows_excluded": {
                "legacy": legacy_excluded,
                "canonical": canonical_excluded,
            },
        },
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "first_season": int(panel["Season"].min()),
            "last_season": int(panel["Season"].max()),
            "duplicate_keys": 0,
            "transition_components_passed": int(len(transition_overlap)),
        },
        "metrics": {
            "transition_overlap": transition_overlap.to_dict(orient="records"),
        },
        "artifact_path": str(output.resolve()),
        "targets_path": str((output / "targets.parquet").resolve()),
        "names_path": str((output / "player_names.csv").resolve()),
        "caveats": [
            "The overlap gate tests source compatibility, not rating accuracy.",
            "The canonical source replaces legacy data at and after the transition season.",
            "This panel remains research input until the consuming model passes its own evaluation.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
