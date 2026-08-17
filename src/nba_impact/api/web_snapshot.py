"""Build a compact, lazy-loaded derived-data snapshot for the web client."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_impact.api.ratings import ROLE_LABELS, RatingsApiConfig, RatingsStore


MODEL_CATALOG = [
    {"id": "aio", "label": "AIO", "components": ["net", "offense", "defense"]},
    {"id": "spm", "label": "SPM", "components": ["net", "offense", "defense"]},
    {"id": "rapm", "label": "RAPM", "components": ["net", "offense", "defense"]},
]


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


def build_web_snapshot(
    config_path: str | Path,
    artifact_root: str | Path,
    aging_curve_path: str | Path,
    output_dir: str | Path,
    *,
    spm_run_path: str | Path | None = None,
    player_sheets_dir: str | Path | None = None,
    shards: int = 32,
) -> dict:
    """Write indexes plus season-specific tables and role maps."""
    if shards < 1:
        raise ValueError("shards must be positive.")
    config = RatingsApiConfig.from_json(config_path)
    store = RatingsStore(config, artifact_root)
    seasons = sorted(int(value) for value in store.annual["Season"].unique())
    project_root = Path(__file__).resolve().parents[3]
    sheets = Path(player_sheets_dir or project_root / "data/raw/playersheets/year_totals")
    team_age = _team_age_panel(sheets, seasons)

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
        index.append({"id": player_id, "name": player["PLAYER_NAME"], "shard": player_id % shards})
    index.sort(key=lambda item: (item["name"].casefold(), item["id"]))

    annual = store.annual.merge(
        team_age[["PLAYER_ID", "Season", "TEAM_ABBREVIATION"]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    )
    role_frames: dict[str, pd.DataFrame] = {}
    for side, frame, prefix in (
        ("offense", store.offense_roles, "off"),
        ("defense", store.defense_roles, "def"),
    ):
        if frame is None:
            continue
        selected = frame.copy()
        selected[f"{side}_role"] = selected[f"{prefix}_role_cluster"].map(ROLE_LABELS[side])
        stable_column = f"{prefix}_role_stable_cluster"
        selected[f"{side}_stable_role"] = (
            selected[stable_column].map(ROLE_LABELS[side])
            if stable_column in selected else selected[f"{side}_role"]
        )
        role_frames[side] = selected
        annual = annual.merge(
            selected[["PLAYER_ID", "Season", f"{side}_role", f"{side}_stable_role"]],
            on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
        )

    stable_manifest = store.role_stabilization_manifest or {}
    spm_path = Path(spm_run_path) if spm_run_path else None
    catalog = {
        "schema_version": "nba_impact_web_snapshot_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": {
            "models": MODEL_CATALOG,
            "role_labels": ROLE_LABELS,
            "seasons": seasons,
            "role_seasons": {
                side: sorted(int(value) for value in frame["Season"].unique())
                for side, frame in role_frames.items()
            },
        },
        "lineage": {
            "annual_run_id": config.annual_run_id,
            "rolling_run_id": config.rolling_run_id,
            "current_rapm_run_id": config.current_rapm_run_id,
            "side_roles_run_id": config.side_roles_run_id,
            "role_stabilization_run_id": config.role_stabilization_run_id,
        },
        "methods": {
            "aio_equation": "AIO = SPM center + centered RAPM update",
            "rapm_update_note": "The update is the deviation of one joint centered ridge fit from its SPM center; it is not zero-prior RAPM added afterward.",
            "spm_calibration": _calibration_summary(spm_path / "oof_predictions.parquet") if spm_path else {},
            "role_stabilization": stable_manifest.get("metrics", {}),
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
        "shards": shards,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("leaderboard-*.json", "roles-*.json"):
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
    leaderboard_columns = [
        "PLAYER_ID", "PLAYER_NAME", "Season", "TEAM_ABBREVIATION", "Poss_Off", "Poss_Def",
        "spm_raw_offense", "spm_raw_defense", "spm_raw_net",
        "normal_rapm_offense", "normal_rapm_defense", "normal_rapm_net",
        "aio_offense", "aio_defense", "aio_net",
        "rapm_update_offense", "rapm_update_defense", "rapm_update_net",
        "offense_role", "offense_stable_role", "defense_role", "defense_stable_role",
    ]
    for season in seasons:
        frame = annual.loc[annual["Season"].eq(season)].copy()
        for column in leaderboard_columns:
            if column not in frame:
                frame[column] = None
        selected = frame[leaderboard_columns]
        selected = selected.astype(object).where(selected.notna(), None)
        name = f"leaderboard-{season}.json"
        files[name] = write(name, selected.to_dict(orient="records"))

    names = store.annual[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID", keep="last")
    for side, frame in role_frames.items():
        prefix = "off" if side == "offense" else "def"
        stable_column = f"{prefix}_role_stable_cluster"
        points = frame.merge(names, on="PLAYER_ID", how="left", validate="many_to_one").merge(
            team_age[["PLAYER_ID", "Season", "TEAM_ABBREVIATION"]],
            on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
        )
        points["raw_role"] = points[f"{prefix}_role_cluster"].map(ROLE_LABELS[side])
        points["stable_role"] = (
            points[stable_column].map(ROLE_LABELS[side]) if stable_column in points else points["raw_role"]
        )
        points = points.rename(columns={f"{prefix}_role_axis_1": "x", f"{prefix}_role_axis_2": "y"})
        columns = ["PLAYER_ID", "PLAYER_NAME", "Season", "TEAM_ABBREVIATION", "x", "y", "raw_role", "stable_role"]
        for season in sorted(int(value) for value in points["Season"].unique()):
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
