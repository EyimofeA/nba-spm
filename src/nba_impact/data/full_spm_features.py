"""Build the frozen full-SPM annual and rolling five-year feature panels."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic


def load_feature_contract(path: str | Path) -> dict[str, tuple[str, ...]]:
    manifest = json.loads(Path(path).read_text())
    raw = (
        manifest.get("features")
        or manifest.get("selected_features")
        or manifest.get("feature_contract")
    )
    if not isinstance(raw, dict):
        raise ValueError("Feature contract must contain offense and defense lists.")
    selected = {
        side: tuple(dict.fromkeys(str(value) for value in raw.get(side, ())))
        for side in ("offense", "defense")
    }
    if any(not values for values in selected.values()):
        raise ValueError("Feature contract must contain non-empty offense and defense lists.")
    return selected


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _pool_annual_field(
    window: pd.DataFrame,
    player_ids: pd.Series,
    field: str,
    weight_field: str,
) -> pd.Series:
    values = pd.to_numeric(window[field], errors="coerce")
    weights = pd.to_numeric(window[weight_field], errors="coerce").clip(lower=0)
    valid = values.notna() & weights.gt(0)
    numerator = (values.where(valid, 0.0) * weights.where(valid, 0.0)).groupby(
        window["PLAYER_ID"]
    ).sum()
    denominator = weights.where(valid, 0.0).groupby(window["PLAYER_ID"]).sum()
    return player_ids.map((numerator / denominator.replace(0.0, np.nan)).to_dict())


def build_rolling_five_year_features(
    annual: pd.DataFrame,
    player_sheet_dir: str | Path | None,
    selected: dict[str, tuple[str, ...]],
    *,
    window_ends: tuple[int, ...] = tuple(range(2018, 2027)),
) -> pd.DataFrame:
    """Possession-pool frozen annual estimates over exact five-season windows.

    ``player_sheet_dir`` remains in the signature for caller compatibility. The
    rolling panel must not re-engineer rates from pooled raw rows because that
    would change the annual stabilization target.
    """
    required_annual = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss"}
    selected_union = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    if missing := sorted((required_annual | set(selected_union)) - set(annual.columns)):
        raise ValueError(f"Annual feature panel is missing {missing}.")
    if annual.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual feature keys are not unique.")

    outputs: list[pd.DataFrame] = []
    for end in window_ends:
        window = annual.loc[annual["Window_End"].between(end - 4, end)].copy()
        player_ids = pd.Series(
            sorted(window["PLAYER_ID"].dropna().unique()), name="PLAYER_ID"
        )
        pooled_columns: dict[str, pd.Series | int] = {
            "PLAYER_ID": player_ids,
            "Window_End": int(end),
        }
        for weight_field in ("OffPoss", "DefPoss"):
            totals = pd.to_numeric(window[weight_field], errors="coerce").groupby(
                window["PLAYER_ID"]
            ).sum(min_count=1)
            pooled_columns[weight_field] = player_ids.map(totals)
        for field in selected_union:
            if field in selected["offense"] and field not in selected["defense"]:
                weight_field = "OffPoss"
            elif field in selected["defense"] and field not in selected["offense"]:
                weight_field = "DefPoss"
            else:
                weight_field = "OffPoss"
            pooled_columns[field] = _pool_annual_field(
                window, player_ids, field, weight_field
            )
        pooled = pd.DataFrame(pooled_columns)
        keep = ["PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *selected_union]
        outputs.append(pooled[keep])

    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year feature keys are not unique.")
    numeric = result[["OffPoss", "DefPoss", *selected_union]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Five-year feature panel contains infinite selected values.")
    return result.sort_values(["Window_End", "PLAYER_ID"]).reset_index(drop=True)


def build_full_spm_feature_panels(
    *,
    annual_features_path: str | Path,
    feature_contract_path: str | Path,
    player_sheet_dir: str | Path,
    coverage_paths: dict[str, str | Path],
    output_root: str | Path,
) -> dict:
    """Validate the annual contract, build five-year rows, and record coverage."""
    annual_path = Path(annual_features_path)
    contract_path = Path(feature_contract_path)
    annual = pd.read_parquet(annual_path)
    selected = load_feature_contract(contract_path)
    selected_union = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    required = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *selected_union}
    if missing := sorted(required - set(annual.columns)):
        raise ValueError(f"Annual full-SPM panel is missing {missing}.")
    if annual.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual full-SPM feature keys are not unique.")
    if sorted(annual["Window_End"].astype(int).unique()) != list(range(2014, 2027)):
        raise ValueError("Annual full-SPM panel must cover every season from 2014 through 2026.")
    annual_selected = annual[["PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *selected_union]].copy()
    annual_numeric = annual_selected[
        ["OffPoss", "DefPoss", *selected_union]
    ].to_numpy(dtype=float)
    if np.isinf(annual_numeric).any():
        raise ValueError("Annual full-SPM panel contains infinite selected values.")

    five_year = build_rolling_five_year_features(
        annual_selected,
        player_sheet_dir,
        selected,
    )

    family_frames: dict[str, pd.DataFrame] = {}
    for family, raw_path in coverage_paths.items():
        path = Path(raw_path)
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        season_column = "Season" if "Season" in frame else "year"
        required_columns = {"PLAYER_ID", season_column}
        if missing := sorted(required_columns - set(frame.columns)):
            raise ValueError(f"Coverage source {family} lacks {missing}.")
        keys = frame[["PLAYER_ID", season_column]].copy()
        keys.columns = ["PLAYER_ID", "Window_End"]
        for column in ("PLAYER_ID", "Window_End"):
            keys[column] = pd.to_numeric(keys[column], errors="coerce")
        family_frames[family] = keys.dropna().drop_duplicates().astype(int)

    coverage_rows: list[dict] = []
    for season, season_rows in annual_selected.groupby("Window_End", sort=True):
        base_keys = season_rows[["PLAYER_ID", "Window_End"]]
        base_count = len(base_keys)
        coverage_rows.append(
            {"season": int(season), "source_family": "player_sheet", "rated_rows": base_count,
             "observed_rows": base_count, "observed_fraction": 1.0}
        )
        for family, keys in family_frames.items():
            observed = len(base_keys.merge(keys, on=["PLAYER_ID", "Window_End"], how="inner"))
            coverage_rows.append(
                {"season": int(season), "source_family": family, "rated_rows": base_count,
                 "observed_rows": observed, "observed_fraction": observed / base_count}
            )
    coverage = pd.DataFrame(coverage_rows)

    input_hashes = {
        "annual_features": sha256_file(annual_path),
        "feature_contract": sha256_file(contract_path),
        **{f"coverage_{name}": sha256_file(path) for name, path in coverage_paths.items()},
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"full_spm_features_2014_2026_v1_{identity}"
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=True)
    annual_output = output / "annual_features.parquet"
    five_year_output = output / "five_year_features.parquet"
    coverage_output = output / "source_coverage.parquet"
    _atomic_parquet(annual_selected, annual_output)
    _atomic_parquet(five_year, five_year_output)
    _atomic_parquet(coverage, coverage_output)
    run = {
        "run_id": run_id,
        "dataset": "full_spm_features_2014_2026_v1",
        "status": "validated_research_input",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": {
            "offense_count": len(selected["offense"]),
            "defense_count": len(selected["defense"]),
            "offense": list(selected["offense"]),
            "defense": list(selected["defense"]),
        },
        "quality": {
            "annual_rows": len(annual_selected),
            "annual_seasons": list(range(2014, 2027)),
            "five_year_rows": len(five_year),
            "five_year_window_ends": list(range(2018, 2027)),
            "duplicate_keys": 0,
            "infinite_selected_values": 0,
            "annual_selected_missing_values": int(
                annual_selected[list(selected_union)].isna().sum().sum()
            ),
            "annual_max_selected_missing_fraction": float(
                annual_selected[list(selected_union)].isna().mean().max()
            ),
            "five_year_selected_missing_values": int(
                five_year[list(selected_union)].isna().sum().sum()
            ),
            "five_year_max_selected_missing_fraction": float(
                five_year[list(selected_union)].isna().mean().max()
            ),
            "season_2027_rows": 0,
        },
        "source_hashes": input_hashes,
        "paths": {
            "annual_features": annual_output.name,
            "five_year_features": five_year_output.name,
            "source_coverage": coverage_output.name,
        },
        "caveat": "Source-family absence is recorded separately. Player-level NaNs remain available for the fitted model's training-only median imputer; zero does not imply an observed source row.",
    }
    write_json_atomic(run, output / "run.json")
    return run
