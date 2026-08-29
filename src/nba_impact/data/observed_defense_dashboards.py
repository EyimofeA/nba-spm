"""Derive observed annual DFG and rim-DFG tables from canonical player sheets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values.loc[valid], weights=weights.loc[valid])) if valid.any() else float("nan")


def _percentage_points(values: pd.Series) -> pd.Series:
    """Normalize percentages stored as either proportions or percentage points."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.abs().gt(1.0), 100.0 * numeric)


def _annual_source(frame: pd.DataFrame, season: int, *, rim: bool) -> pd.DataFrame:
    required = {"PLAYER_ID", "PLAYER_NAME"}
    if rim:
        made, attempted, baseline = "less_6ft_def_FGM_LT_06", "less_6ft_def_FGA_LT_06", "less_6ft_def_NS_LT_06_PCT"
    else:
        made, attempted, baseline = "D_FGM", "D_FGA", "NORMAL_FG_PCT"
    required |= {made, attempted, baseline}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Season {season}: player sheet lacks observed defense columns {missing}")
    work = frame[["PLAYER_ID", "PLAYER_NAME", made, attempted, baseline]].copy()
    for column in ("PLAYER_ID", made, attempted, baseline):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work[baseline] = _percentage_points(work[baseline])
    work = work.dropna(subset=["PLAYER_ID", "PLAYER_NAME", attempted]).loc[lambda x: x[attempted].ge(0)].copy()
    rows = []
    for player_id, group in work.groupby("PLAYER_ID", sort=False):
        attempts = float(group[attempted].sum())
        makes = float(group[made].fillna(0.0).sum())
        actual_pct = 100.0 * makes / attempts if attempts > 0 else np.nan
        normal_pct = _weighted_mean(group[baseline], group[attempted])
        rows.append({"PLAYER_ID": int(player_id), "PLAYER": str(group["PLAYER_NAME"].iloc[0]), "DFGM": makes, "DFGA": attempts, "DFG%": actual_pct, "FG%": normal_pct, "DIFF%": actual_pct - normal_pct if np.isfinite(actual_pct) and np.isfinite(normal_pct) else np.nan, "year": int(season)})
    return pd.DataFrame(rows)


def build_observed_defense_dashboards(
    player_sheets_dir: str | Path,
    output_dir: str | Path,
    *,
    seasons: tuple[int, ...],
    historical_dfg_source: str | Path | None = None,
    historical_rim_dfg_source: str | Path | None = None,
) -> dict:
    """Create a uniform source table without filling or inventing observations."""
    source = Path(player_sheets_dir)
    dfg_rows: list[pd.DataFrame] = []
    rim_rows: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    historical_dfg = pd.read_csv(historical_dfg_source, low_memory=False) if historical_dfg_source else pd.DataFrame()
    historical_rim = pd.read_csv(historical_rim_dfg_source, low_memory=False) if historical_rim_dfg_source else pd.DataFrame()
    if historical_dfg_source:
        hashes[str(Path(historical_dfg_source).resolve())] = sha256_file(historical_dfg_source)
    if historical_rim_dfg_source:
        hashes[str(Path(historical_rim_dfg_source).resolve())] = sha256_file(historical_rim_dfg_source)
    for season in seasons:
        csv_path = source / f"{season}.csv"
        parquet_path = source / f"{season}.parquet"
        if csv_path.exists():
            path = csv_path
            frame = pd.read_csv(path, low_memory=False)
        elif parquet_path.exists():
            path = parquet_path
            frame = pd.read_parquet(path)
        else:
            raise FileNotFoundError(
                f"No player sheet for season {season}: expected {csv_path} or {parquet_path}."
            )
        try:
            dfg_rows.append(_annual_source(frame, season, rim=False))
        except ValueError:
            fallback = historical_dfg.loc[pd.to_numeric(historical_dfg.get("year"), errors="coerce") == season].copy()
            if fallback.empty:
                raise
            dfg_rows.append(fallback)
        try:
            rim_rows.append(_annual_source(frame, season, rim=True))
        except ValueError:
            fallback = historical_rim.loc[pd.to_numeric(historical_rim.get("year"), errors="coerce") == season].copy()
            if fallback.empty:
                raise
            rim_rows.append(fallback)
        hashes[str(path.resolve())] = sha256_file(path)
    dfg, rim = pd.concat(dfg_rows, ignore_index=True), pd.concat(rim_rows, ignore_index=True)
    for label, frame in (("dfg", dfg), ("rim_dfg", rim)):
        if frame["DFGA"].lt(0).any():
            raise ValueError(f"{label}: negative attempts")
        if not frame["DFG%"].dropna().between(0.0, 100.0).all():
            raise ValueError(f"{label}: DFG% is not in percentage points")
        if not frame["FG%"].dropna().between(0.0, 100.0).all():
            raise ValueError(f"{label}: FG% is not in percentage points")
        if not frame["DIFF%"].dropna().between(-100.0, 100.0).all():
            raise ValueError(f"{label}: DIFF% is not in percentage points")
    identity = hashlib.sha256(json.dumps({
        "seasons": list(seasons),
        "hashes": hashes,
        "builder_sha256": sha256_file(Path(__file__)),
    }, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"observed_defense_dashboards_v1_{identity}"
    root = Path(output_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    dfg_path, rim_path = root / "dfg.csv", root / "rim_dfg.csv"
    dfg.to_csv(dfg_path, index=False)
    rim.to_csv(rim_path, index=False)
    manifest = {"run_id": run_id, "dataset": "observed_annual_defense_dashboards", "created_at": datetime.now(timezone.utc).isoformat(), "seasons": list(seasons), "source_hashes": hashes, "dfg_path": str(dfg_path.resolve()), "rim_dfg_path": str(rim_path.resolve()), "row_counts": {"dfg": int(len(dfg)), "rim_dfg": int(len(rim))}, "note": "Rows come from observed NBA dashboard fields. Player sheets provide 2026; pinned dashboard extracts cover older unavailable player-sheet fields. No neutral filling occurs here."}
    write_json_atomic(manifest, root / "run.json")
    return manifest
