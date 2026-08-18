"""Build a compact, lazy-loaded derived-data snapshot for the web client."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_impact.api.player_profiles import PROFILE_AXES, build_player_skill_profiles
from nba_impact.api.ratings import ROLE_LABELS, RatingsApiConfig, RatingsStore


COMPONENTS = ("offense", "defense", "net")

# Public model selector.  `prefix` is the exported column prefix and `source`
# is the column prefix inside the pinned annual rating run.
MODEL_CATALOG = [
    {
        "id": "aio",
        "label": "AIO",
        "prefix": "aio_",
        "source": "aio_",
        "note": "SPM center plus the RAPM update.",
    },
    {
        "id": "rapm",
        "label": "Normal RAPM",
        "prefix": "normal_rapm_",
        "source": "normal_rapm_",
        "note": "Zero-prior one-season ridge on possessions.",
    },
    {
        "id": "spm",
        "label": "SPM",
        "prefix": "spm_",
        "source": "spm_center_",
        "note": "Held-out statistical prediction that centers the RAPM fit.",
    },
]

# Verified external agreement, transcribed from frozen benchmark runs.  These
# are diagnostics against public metrics, never labels or ground truth.
EXTERNAL_BENCHMARK = {
    "note": (
        "Basketball Reference BPM and xrapm.com are external diagnostics. "
        "They are not training labels and not ground truth."
    ),
    "rows": [
        {
            "scope": "Three-season SPM windows",
            "exposure": "3,000+ possessions per side",
            "players": 2295,
            "component": "net",
            "bpm": 0.876,
            "xrapm": 0.756,
            "run_id": "external_impact_benchmark_v1_bab43a4087",
        },
        {
            "scope": "Three-season SPM windows",
            "exposure": "3,000+ possessions per side",
            "players": 2295,
            "component": "offense",
            "bpm": None,
            "xrapm": 0.831,
            "run_id": "external_impact_benchmark_v1_bab43a4087",
        },
        {
            "scope": "Three-season SPM windows",
            "exposure": "3,000+ possessions per side",
            "players": 2295,
            "component": "defense",
            "bpm": None,
            "xrapm": 0.630,
            "run_id": "external_impact_benchmark_v1_bab43a4087",
        },
        {
            "scope": "Annual SPM baseline",
            "exposure": "1,000+ possessions per side",
            "players": 2860,
            "component": "net",
            "bpm": 0.897,
            "xrapm": 0.762,
            "run_id": "single_season_spm_v1_51adc53061",
        },
        {
            "scope": "Annual SPM baseline",
            "exposure": "1,000+ possessions per side",
            "players": 2860,
            "component": "defense",
            "bpm": 0.803,
            "xrapm": 0.590,
            "run_id": "single_season_spm_v1_51adc53061",
        },
        {
            "scope": "Annual SPM plus tracking",
            "exposure": "1,000+ possessions per side",
            "players": None,
            "component": "defense",
            "bpm": 0.690,
            "xrapm": 0.701,
            "run_id": "single_season_spm_v1_bff6060df6",
        },
    ],
    "pinned_model_note": (
        "The pinned rating model adds scorer-adjusted matchup features. Its "
        "defense agreement improves against xRAPM and declines against BPM; "
        "exact values are not published."
    ),
}


def _compact_memberships(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not items:
        return []
    return [
        {
            "role_id": item["role_id"],
            "label": item["label"],
            "affinity": round(float(item["affinity"]), 4),
        }
        for item in sorted(items, key=lambda item: float(item["affinity"]), reverse=True)[:4]
    ]


def _compact_role(role: dict[str, Any] | None) -> dict[str, Any] | None:
    """Public roles are raw only.  Stabilized labels stay out of the snapshot."""
    if not role:
        return None
    return {
        "primary_role": role["primary_role"],
        "confidence": round(float(role["confidence"]), 4),
        "memberships": _compact_memberships(role.get("memberships")),
    }


def _model_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map available source rating columns to their exported names."""
    return {
        f"{model['source']}{component}": f"{model['prefix']}{component}"
        for model in MODEL_CATALOG
        for component in COMPONENTS
        if f"{model['source']}{component}" in frame.columns
    }


def _team_age_panel(player_sheets_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = player_sheets_dir / f"{season}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda column: column
            in {"PLAYER_ID", "TEAM_ABBREVIATION", "AGE", "year"},
        ).rename(columns={"year": "Season"})
        frame["Season"] = season
        frame = frame.dropna(subset=["PLAYER_ID"]).drop_duplicates(
            ["PLAYER_ID", "Season"], keep="last"
        )
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "TEAM_ABBREVIATION", "AGE"])
    return pd.concat(frames, ignore_index=True)


def _current_normal_rapm_rows(
    run_path: Path,
    player_sheets_dir: Path,
    player_games_path: Path | None,
    published_seasons: list[int],
) -> tuple[pd.DataFrame, str]:
    """Read the pinned current target run as Normal-RAPM-only annual rows.

    The historical annual artifact remains the source of AIO, SPM, player name,
    and team.  Its overlapping Normal RAPM and possession columns may be
    replaced only after the caller establishes an exact player-season match.
    Later complete seasons are added from the canonical terminal-lineup,
    zero-prior target run.
    """
    manifest_path = run_path / "run.json"
    targets_path = run_path / "targets.parquet"
    if not manifest_path.exists() or not targets_path.exists():
        raise FileNotFoundError(
            "Current Normal RAPM requires run.json and targets.parquet."
        )
    manifest = json.loads(manifest_path.read_text())
    expected = "canonical_current_single_season_zero_prior_normal_rapm_targets"
    if manifest.get("model_family") != expected:
        raise ValueError("Current Normal RAPM run has the wrong model family.")
    config = manifest.get("config", {})
    if (
        config.get("lineup_policy") != "terminal"
        or config.get("prior") != "zero"
        or config.get("game_types") != ["regular"]
    ):
        raise ValueError("Current Normal RAPM run must be terminal-lineup, zero-prior regular season.")
    if any(float(config.get(name, float("nan"))) != value for name, value in (
        ("lambda_off", 3000.0), ("lambda_def", 3000.0), ("lambda_home", 300.0)
    )):
        raise ValueError("Current Normal RAPM run must use 3000/3000/300 penalties.")

    targets = pd.read_parquet(targets_path)
    required = {
        "PLAYER_ID", "Season", "target_offense", "target_defense", "target_net",
        "Poss_Off", "Poss_Def",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Current Normal RAPM targets are missing columns: {sorted(missing)}")
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Current Normal RAPM targets have duplicate player-seasons.")
    numeric = ["target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    if not np.isfinite(targets[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Current Normal RAPM targets contain non-finite values.")
    if (targets[["Poss_Off", "Poss_Def"]] <= 0).any().any():
        raise ValueError("Current Normal RAPM targets require positive side possessions.")
    if not np.allclose(
        targets["target_net"], targets["target_offense"] + targets["target_defense"], atol=1e-10
    ):
        raise ValueError("Current Normal RAPM targets violate offense + defense = net.")

    new_targets = targets.loc[~targets["Season"].isin(published_seasons)].copy()
    if new_targets.empty:
        rows = targets.rename(
            columns={
                "target_offense": "normal_rapm_offense",
                "target_defense": "normal_rapm_defense",
                "target_net": "normal_rapm_net",
            }
        )[[
            "PLAYER_ID", "Season", "Poss_Off", "Poss_Def",
            "normal_rapm_offense", "normal_rapm_defense", "normal_rapm_net",
        ]]
        return rows, str(manifest.get("run_id", run_path.name))

    target_seasons = sorted(int(value) for value in new_targets["Season"].unique())
    sheet_frames = []
    for season in target_seasons:
        path = player_sheets_dir / f"{season}.csv"
        if not path.exists():
            continue
        sheet = pd.read_csv(
            path,
            usecols=lambda column: column in {
                "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
            },
        )
        if {"PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"}.issubset(sheet.columns):
            sheet["Season"] = season
            sheet_frames.append(sheet)
    sheet_metadata = pd.concat(sheet_frames, ignore_index=True) if sheet_frames else pd.DataFrame(
        columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season"]
    )
    if not sheet_metadata.empty:
        sheet_metadata = sheet_metadata.dropna(
            subset=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
        ).drop_duplicates(["PLAYER_ID", "Season"], keep="last")

    game_metadata = pd.DataFrame(
        columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season"]
    )
    if player_games_path is not None and player_games_path.exists():
        games = pd.read_parquet(
            player_games_path,
            columns=["season_end", "player_id", "player_name", "team_tricode", "game_date"],
        )
        games = games.loc[games["season_end"].isin(target_seasons)].copy()
        games["Season"] = games.pop("season_end").astype(int)
        games = games.rename(
            columns={
                "player_id": "PLAYER_ID", "player_name": "PLAYER_NAME",
                "team_tricode": "TEAM_ABBREVIATION",
            }
        ).dropna(subset=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"])
        game_metadata = games.sort_values(
            ["Season", "game_date", "PLAYER_ID"], kind="stable"
        ).drop_duplicates(["PLAYER_ID", "Season"], keep="last")[
            ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season"]
        ]

    metadata = pd.concat([game_metadata, sheet_metadata], ignore_index=True)
    metadata = metadata.drop_duplicates(["PLAYER_ID", "Season"], keep="last")
    new_targets = new_targets.merge(
        metadata, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    )
    if new_targets[["PLAYER_NAME", "TEAM_ABBREVIATION"]].isna().any().any():
        missing_rows = new_targets.loc[
            new_targets[["PLAYER_NAME", "TEAM_ABBREVIATION"]].isna().any(axis=1),
            ["PLAYER_ID", "Season"],
        ]
        raise ValueError(
            f"Current Normal RAPM metadata is incomplete for {len(missing_rows)} player-seasons."
        )
    overlap_rows = targets.loc[targets["Season"].isin(published_seasons)].rename(
        columns={
            "target_offense": "normal_rapm_offense",
            "target_defense": "normal_rapm_defense",
            "target_net": "normal_rapm_net",
        }
    )[[
        "PLAYER_ID", "Season", "Poss_Off", "Poss_Def",
        "normal_rapm_offense", "normal_rapm_defense", "normal_rapm_net",
    ]]
    new_rows = new_targets.rename(
        columns={
            "target_offense": "normal_rapm_offense",
            "target_defense": "normal_rapm_defense",
            "target_net": "normal_rapm_net",
        }
    )[[
        "PLAYER_ID", "PLAYER_NAME", "Season", "TEAM_ABBREVIATION", "Poss_Off", "Poss_Def",
        "normal_rapm_offense", "normal_rapm_defense", "normal_rapm_net",
    ]]
    rows = pd.concat([overlap_rows, new_rows], ignore_index=True, sort=False)
    return rows, str(manifest.get("run_id", run_path.name))


def _calibration_summary(oof_path: Path) -> dict[str, Any]:
    if not oof_path.exists():
        return {}
    frame = pd.read_parquet(oof_path)
    summary: dict[str, Any] = {}
    for component in ("offense", "defense", "net"):
        prediction = frame[f"spm_{component}"].to_numpy(dtype=float)
        target = frame[f"target_{component}"].to_numpy(dtype=float)
        weights = frame["sample_weight"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(frame)), prediction])
        root_weight = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None
        )
        quantiles = np.quantile(prediction, [0.01, 0.05, 0.5, 0.95, 0.99])
        summary[component] = {
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
            "display_range_01_99": [float(quantiles[0]), float(quantiles[-1])],
            "quantiles": [float(value) for value in quantiles],
        }
    return summary


def _rapm_aging_rows(aging_curve_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(aging_curve_path).sort_values("Age").reset_index(drop=True)
    for source, target in (
        ("f_total", "change_net"),
        ("f_off", "change_offense"),
        ("f_def", "change_defense"),
    ):
        frame[target] = frame[source].shift(-1) - frame[source]
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _aio_aging_rows(annual: pd.DataFrame, team_age: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "PLAYER_ID", "Season", "Poss_Off", "Poss_Def",
        "aio_net", "aio_offense", "aio_defense",
    ]
    frame = annual[columns].merge(
        team_age[["PLAYER_ID", "Season", "AGE"]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    ).sort_values(["PLAYER_ID", "Season"], kind="stable")
    grouped = frame.groupby("PLAYER_ID", sort=False)
    shifted = [
        "Season", "AGE", "Poss_Off", "Poss_Def",
        "aio_net", "aio_offense", "aio_defense",
    ]
    for column in shifted:
        frame[f"next_{column}"] = grouped[column].shift(-1)
    eligible = frame.loc[
        frame["next_Season"].eq(frame["Season"] + 1)
        & frame["next_AGE"].eq(frame["AGE"] + 1)
        & frame[["Poss_Off", "Poss_Def", "next_Poss_Off", "next_Poss_Def"]]
        .min(axis=1).ge(1000)
    ].copy()
    if eligible.empty:
        return []
    current_exposure = eligible[["Poss_Off", "Poss_Def"]].min(axis=1)
    next_exposure = eligible[["next_Poss_Off", "next_Poss_Def"]].min(axis=1)
    eligible["weight"] = 2.0 / (1.0 / current_exposure + 1.0 / next_exposure)
    for component in ("net", "offense", "defense"):
        eligible[f"change_{component}"] = (
            eligible[f"next_aio_{component}"] - eligible[f"aio_{component}"]
        )
    rows = []
    for age, group in eligible.groupby(eligible["AGE"].round().astype(int)):
        row: dict[str, Any] = {"Age": int(age), "n": int(len(group))}
        for component in ("net", "offense", "defense"):
            row[f"change_{component}"] = float(
                np.average(group[f"change_{component}"], weights=group["weight"])
            )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("Age")
    for component in ("net", "offense", "defense"):
        result[f"change_{component}"] = result[f"change_{component}"].rolling(
            3, center=True, min_periods=1
        ).mean()
    return result.to_dict(orient="records")


def _weighted_r2(actual: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
    valid = actual.notna() & predicted.notna() & weights.notna() & weights.gt(0)
    y = actual.loc[valid].to_numpy(dtype=float)
    p = predicted.loc[valid].to_numpy(dtype=float)
    w = weights.loc[valid].to_numpy(dtype=float)
    if len(y) < 2:
        return float("nan")
    mean = float(np.average(y, weights=w))
    denominator = float(np.sum(w * np.square(y - mean)))
    return float(1.0 - np.sum(w * np.square(y - p)) / denominator) if denominator > 0 else float("nan")


def _walk_forward_summary(run_path: Path, oof_path: Path) -> list[dict[str, Any]]:
    if not (run_path / "priors.parquet").exists() or not oof_path.exists():
        return []
    priors = pd.read_parquet(run_path / "priors.parquet")
    targets = pd.read_parquet(oof_path)
    frame = priors.merge(
        targets[[
            "PLAYER_ID", "Season", "target_offense", "target_defense", "target_net",
            "sample_weight",
        ]],
        on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one",
    )
    rows = []
    for component in ("offense", "defense", "net"):
        fold_rows = []
        for season, group in frame.groupby("Season", sort=True):
            actual = group[f"target_{component}"]
            predicted = group[f"prior_{component}_per_100"]
            weights = group["sample_weight"]
            fold_rows.append(
                {
                    "season": int(season),
                    "rmse": float(np.sqrt(np.average(np.square(actual - predicted), weights=weights))),
                    "correlation": float(actual.corr(predicted)),
                    "r2": _weighted_r2(actual, predicted, weights),
                }
            )
        rows.append(
            {
                "component": component,
                "seasons": f"{min(row['season'] for row in fold_rows)}–{max(row['season'] for row in fold_rows)}",
                "folds": len(fold_rows),
                "rmse": float(np.mean([row["rmse"] for row in fold_rows])),
                "correlation": float(np.mean([row["correlation"] for row in fold_rows])),
                "r2": float(np.mean([row["r2"] for row in fold_rows])),
            }
        )
    return rows


def _walk_backward_summary(run_path: Path) -> list[dict[str, Any]]:
    path = run_path / "predictions.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    rows = []
    for direction in sorted(frame["direction"].unique()):
        group = frame.loc[frame["direction"].eq(direction)]
        for component in ("offense", "defense", "net"):
            actual = group[f"raw_target_{component}"]
            predicted = group[f"spm_{component}"]
            weights = group["evaluation_weight"]
            rows.append(
                {
                    "direction": str(direction),
                    "component": component,
                    "rows": int(len(group)),
                    "rmse": float(np.sqrt(np.average(np.square(actual - predicted), weights=weights))),
                    "correlation": float(actual.corr(predicted)),
                    "r2": _weighted_r2(actual, predicted, weights),
                }
            )
    return rows


def _aging_projection_summary(run_path: Path) -> dict[str, Any]:
    run_file = run_path / "run.json"
    if not run_file.exists():
        return {}
    run = json.loads(run_file.read_text())
    result: dict[str, Any] = {
        "selected_method": run["config"]["selected_method"],
        "selection_seasons": run["config"]["selection_origins"],
        "diagnostic_seasons": run["config"]["diagnostic_origins"],
    }
    for label, filename in (
        ("selection", "selection_summary.parquet"),
        ("diagnostic", "diagnostic_summary.parquet"),
        ("subgroups", "subgroup_metrics.parquet"),
    ):
        frame = pd.read_parquet(run_path / filename)
        result[label] = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return result


def build_web_snapshot(
    config_path: str | Path,
    artifact_root: str | Path,
    aging_curve_path: str | Path,
    output_dir: str | Path,
    *,
    spm_run_path: str | Path | None = None,
    player_sheets_dir: str | Path | None = None,
    features_path: str | Path | None = None,
    walk_forward_run_path: str | Path | None = None,
    walk_backward_run_path: str | Path | None = None,
    aging_projection_run_path: str | Path | None = None,
    current_normal_rapm_run_path: str | Path | None = None,
    current_player_games_path: str | Path | None = None,
    shards: int = 128,
) -> dict:
    """Write indexes plus season-specific tables and role maps."""
    if shards < 1:
        raise ValueError("shards must be positive.")
    config = RatingsApiConfig.from_json(config_path)
    store = RatingsStore(config, artifact_root)
    historical_seasons = sorted(int(value) for value in store.annual["Season"].unique())
    project_root = Path(__file__).resolve().parents[3]
    sheets = Path(player_sheets_dir or project_root / "data/raw/playersheets/year_totals")
    current_run_id: str | None = None
    current_rows = pd.DataFrame(columns=["PLAYER_ID", "Season"])
    if current_normal_rapm_run_path is not None:
        current_rows, current_run_id = _current_normal_rapm_rows(
            Path(current_normal_rapm_run_path),
            sheets,
            Path(current_player_games_path) if current_player_games_path else None,
            historical_seasons,
        )
    historical = store.annual.copy()
    overlap = current_rows.loc[current_rows["Season"].isin(historical_seasons)].copy()
    if not overlap.empty:
        overlap_seasons = sorted(int(value) for value in overlap["Season"].unique())
        historical_overlap = historical.loc[historical["Season"].isin(overlap_seasons)]
        current_keys = set(zip(overlap["PLAYER_ID"].astype(int), overlap["Season"].astype(int)))
        historical_keys = set(zip(
            historical_overlap["PLAYER_ID"].astype(int), historical_overlap["Season"].astype(int)
        ))
        if current_keys != historical_keys:
            raise ValueError(
                "Current Normal RAPM overlap keys do not exactly match the historical annual panel."
            )
        override_columns = [
            "normal_rapm_offense", "normal_rapm_defense", "normal_rapm_net", "Poss_Off", "Poss_Def",
        ]
        if overlap[override_columns].isna().any().any():
            raise ValueError("Current Normal RAPM overlap has missing override values.")
        overrides = overlap.set_index(["PLAYER_ID", "Season"])[override_columns]
        overlap_mask = historical["Season"].isin(overlap_seasons)
        historical_overlap_index = pd.MultiIndex.from_frame(
            historical.loc[overlap_mask, ["PLAYER_ID", "Season"]]
        )
        for column in override_columns:
            # Do not touch historical identity, SPM, or AIO fields.  This is a
            # narrow replacement of the validated current RAPM observation.
            historical.loc[overlap_mask, column] = overrides[column].reindex(
                historical_overlap_index
            ).to_numpy()
    new_current_rows = current_rows.loc[~current_rows["Season"].isin(historical_seasons)]
    annual = pd.concat([historical, new_current_rows], ignore_index=True, sort=False)
    if annual.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Web snapshot annual panel has duplicate player-seasons.")
    seasons = sorted(int(value) for value in annual["Season"].unique())
    team_age = _team_age_panel(sheets, seasons)
    annual = annual.merge(
        team_age[["PLAYER_ID", "Season", "TEAM_ABBREVIATION"]],
        on=["PLAYER_ID", "Season"], how="left", suffixes=("", "_sheet"), validate="one_to_one",
    )
    if "TEAM_ABBREVIATION_sheet" in annual:
        if "TEAM_ABBREVIATION" in annual:
            annual["TEAM_ABBREVIATION"] = annual["TEAM_ABBREVIATION"].combine_first(
                annual["TEAM_ABBREVIATION_sheet"]
            )
        else:
            annual["TEAM_ABBREVIATION"] = annual["TEAM_ABBREVIATION_sheet"]
        annual = annual.drop(columns="TEAM_ABBREVIATION_sheet")
    team_lookup = {
        key: value if pd.notna(value) else None
        for key, value in annual.set_index(["PLAYER_ID", "Season"])["TEAM_ABBREVIATION"].to_dict().items()
    }
    profiles = pd.DataFrame(columns=["PLAYER_ID", "Season", *PROFILE_AXES])
    if features_path and Path(features_path).exists():
        profiles = build_player_skill_profiles(pd.read_parquet(features_path), historical_seasons)
    profile_lookup = {
        int(player_id): group.drop(columns="PLAYER_ID").round(1).astype(object).where(group.notna(), None).to_dict(orient="records")
        for player_id, group in profiles.groupby("PLAYER_ID", sort=False)
    }

    annual_model_columns = _model_columns(annual)
    player_ids = sorted(set(annual["PLAYER_ID"].astype(int)))
    players: dict[str, dict] = {}
    index = []
    for player_id in player_ids:
        player = store.player(player_id)
        player_rows = annual.loc[annual["PLAYER_ID"].eq(player_id)].sort_values("Season")
        if player_rows.empty:
            continue
        name = str(player_rows["PLAYER_NAME"].dropna().iloc[-1])
        public_player = {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": name,
            "annual": [
                {
                    "Season": int(row["Season"]),
                    "TEAM_ABBREVIATION": team_lookup.get((player_id, int(row["Season"]))),
                    "Poss_Off": int(row["Poss_Off"]),
                    "Poss_Def": int(row["Poss_Def"]),
                    **{
                        exported: round(float(row[source]), 4)
                        for source, exported in annual_model_columns.items()
                        if pd.notna(row.get(source))
                    },
                }
                for row in player_rows.to_dict(orient="records")
            ],
            "roles": [
                {
                    "Season": int(row["Season"]),
                    **{side: compact for side in ("offense", "defense") if (compact := _compact_role(row.get(side)))},
                }
                for row in (player["roles"] if player is not None else [])
            ],
            "profiles": profile_lookup.get(player_id, []),
        }
        players[str(player_id)] = public_player
        index.append({"id": player_id, "name": public_player["PLAYER_NAME"], "shard": player_id % shards})
    index.sort(key=lambda item: (item["name"].casefold(), item["id"]))

    role_frames: dict[str, pd.DataFrame] = {}
    for side, frame, prefix in (
        ("offense", store.offense_roles, "off"),
        ("defense", store.defense_roles, "def"),
    ):
        if frame is None:
            continue
        selected = frame.copy()
        selected[f"{side}_role"] = selected[f"{prefix}_role_cluster"].map(ROLE_LABELS[side])
        role_frames[side] = selected

    spm_path = Path(spm_run_path) if spm_run_path else None
    forward_path = Path(walk_forward_run_path) if walk_forward_run_path else None
    backward_path = Path(walk_backward_run_path) if walk_backward_run_path else None
    projection_path = Path(aging_projection_run_path) if aging_projection_run_path else None
    public_models = [
        {
            **model,
            "seasons": sorted(
                int(value)
                for value in annual.loc[
                    annual[f"{model['source']}net"].notna(), "Season"
                ].unique()
            ),
        }
        for model in MODEL_CATALOG
    ]
    catalog = {
        "schema_version": "nba_impact_web_snapshot_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": {
            "models": public_models,
            "role_labels": ROLE_LABELS,
            "seasons": seasons,
            "role_seasons": {
                side: sorted(
                    int(value)
                    for value in frame.loc[frame["Season"].isin(seasons), "Season"].unique()
                )
                for side, frame in role_frames.items()
            },
        },
        "lineage": {
            "annual_run_id": config.annual_run_id,
            "rolling_run_id": config.rolling_run_id,
            "current_rapm_run_id": config.current_rapm_run_id,
            "current_normal_rapm_run_id": current_run_id,
            "side_roles_run_id": config.side_roles_run_id,
        },
        "methods": {
            "aio_equation": "AIO = SPM center + centered RAPM update",
            "rapm_update_note": "The update is the deviation of one joint centered ridge fit from its SPM center; it is not zero-prior RAPM added afterward.",
            "spm_calibration": _calibration_summary(spm_path / "oof_predictions.parquet") if spm_path else {},
        },
        "aging": {
            "rapm": {
                "status": "exploratory_descriptive", "coverage": "1997–2024",
                "rows": _rapm_aging_rows(Path(aging_curve_path)),
            },
            "aio": {
                "status": "exploratory_descriptive", "coverage": f"{seasons[0]}–{seasons[-1]}",
                "rows": _aio_aging_rows(store.annual, team_age),
            },
        },
        "validation": {
            "walk_forward": _walk_forward_summary(forward_path, spm_path / "oof_predictions.parquet")
            if forward_path and spm_path else [],
            "walk_backward": _walk_backward_summary(backward_path) if backward_path else [],
            "aging_projection": _aging_projection_summary(projection_path) if projection_path else {},
            "external_benchmark": EXTERNAL_BENCHMARK,
        },
        "shards": shards,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("leaderboard-*.json", "roles-*.json", "ratings-*.json"):
        for old_path in output.glob(pattern):
            old_path.unlink()

    def write(name: str, value: object) -> tuple[int, str]:
        serialized = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        destination = output / name
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(serialized)
        temporary.replace(destination)
        return len(serialized.encode()), hashlib.sha256(serialized.encode()).hexdigest()

    files = {
        "catalog.json": write("catalog.json", catalog),
        "players.json": write("players.json", index),
    }
    if projection_path:
        projection_players = pd.read_parquet(projection_path / "player_projections.parquet")
        projection_teams = pd.read_parquet(projection_path / "team_projections.parquet")
        historical_players_path = projection_path / "historical_player_projections.parquet"
        if historical_players_path.exists():
            historical_players = pd.read_parquet(historical_players_path).rename(
                columns={"target_season": "projection_season", "origin_season": "Season"}
            )
            projection_players = pd.concat(
                [historical_players, projection_players], ignore_index=True, sort=False
            )
        canonical_names = store.annual.sort_values("Season").drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"]
        projection_players["PLAYER_NAME"] = projection_players["PLAYER_ID"].map(canonical_names).fillna(projection_players["PLAYER_NAME"])
        files["projection-players.json"] = write(
            "projection-players.json",
            projection_players.astype(object).where(projection_players.notna(), None).to_dict(orient="records"),
        )
        files["projection-teams.json"] = write(
            "projection-teams.json",
            projection_teams.astype(object).where(projection_teams.notna(), None).to_dict(orient="records"),
        )
    base_columns = [
        "PLAYER_ID", "PLAYER_NAME", "Season", "TEAM_ABBREVIATION", "Poss_Off", "Poss_Def",
    ]
    rating_columns = _model_columns(annual)
    for season in seasons:
        frame = annual.loc[annual["Season"].eq(season)].copy()
        for column in base_columns:
            if column not in frame:
                frame[column] = None
        season_rating_columns = {
            source: exported
            for source, exported in rating_columns.items()
            if frame[source].notna().any()
        }
        selected = frame[base_columns + list(season_rating_columns)].rename(columns=season_rating_columns)
        exported = list(season_rating_columns.values())
        selected[exported] = selected[exported].round(4)
        selected = selected.astype(object).where(selected.notna(), None)
        name = f"leaderboard-{season}.json"
        files[name] = write(name, selected.to_dict(orient="records"))

    names = store.annual[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID", keep="last")
    for side, frame in role_frames.items():
        prefix = "off" if side == "offense" else "def"
        points = frame.merge(names, on="PLAYER_ID", how="left", validate="many_to_one").merge(
            team_age[["PLAYER_ID", "Season", "TEAM_ABBREVIATION"]],
            on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
        )
        points["raw_role"] = points[f"{prefix}_role_cluster"].map(ROLE_LABELS[side])
        points = points.rename(columns={f"{prefix}_role_axis_1": "x", f"{prefix}_role_axis_2": "y"})
        points[["x", "y"]] = points[["x", "y"]].round(5)
        columns = ["PLAYER_ID", "PLAYER_NAME", "Season", "TEAM_ABBREVIATION", "x", "y", "raw_role"]
        for season in sorted(
            int(value) for value in points.loc[points["Season"].isin(seasons), "Season"].unique()
        ):
            selected = points.loc[points["Season"].eq(season), columns]
            selected = selected.astype(object).where(selected.notna(), None)
            name = f"roles-{side}-{season}.json"
            files[name] = write(name, selected.to_dict(orient="records"))

    for shard in range(shards):
        name = f"ratings-{shard:02d}.json"
        rows = {player_id: player for player_id, player in players.items() if int(player_id) % shards == shard}
        files[name] = write(name, rows)
    total_bytes = sum(size for size, _ in files.values())
    return {
        "output_dir": str(output.resolve()), "players": len(players),
        "seasons": seasons, "shards": shards, "bytes": total_bytes,
        "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in files.items()},
    }
