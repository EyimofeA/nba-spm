#!/usr/bin/env python3
"""Export selected RAPM research leaderboards as lazy JSON shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "web/public/data/rapm"
PULSE = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a"
CANONICAL_SUITE = ROOT / "artifacts/models/canonical_rapm_suite/canonical_rapm_suite_v1_ed9af098b6"


def _write(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _records(frame: pd.DataFrame) -> list[dict]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame = frame.copy()
    frame[numeric] = frame[numeric].round(4)
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _player_names() -> dict[int, str]:
    ratings = pd.read_parquet(PULSE / "ratings.parquet", columns=["PLAYER_ID", "PLAYER_NAME", "Season"])
    return (
        ratings.sort_values("Season").drop_duplicates("PLAYER_ID", keep="last")
        .set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()
    )


def _normalize_player(frame: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    aliases = {
        "player_id": "PLAYER_ID", "player_name": "PLAYER_NAME",
        "off_possessions": "Poss_Off", "def_possessions": "Poss_Def",
        "offense_per_100": "offense", "defense_per_100": "defense", "net_per_100": "net",
    }
    result = frame.rename(columns={key: value for key, value in aliases.items() if key in frame}).copy()
    if "PLAYER_ID" in result:
        result["PLAYER_ID"] = pd.to_numeric(result["PLAYER_ID"], errors="coerce").astype("Int64")
        if "PLAYER_NAME" not in result:
            result["PLAYER_NAME"] = result["PLAYER_ID"].map(names)
        else:
            result["PLAYER_NAME"] = result["PLAYER_NAME"].fillna(result["PLAYER_ID"].map(names))
    return result


def _add_grouped(
    catalog: list[dict], files: dict[str, dict], *, ident: str, title: str,
    frame: pd.DataFrame, group: str, columns: list[str], unit: str = "points per 100",
    note: str = "",
) -> None:
    periods: list[dict] = []
    for value, part in frame.groupby(group, sort=True):
        label = str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        filename = f"{ident}-{label}.json"
        selected = part[[column for column in columns if column in part]].copy()
        files[filename] = _write(OUTPUT / filename, _records(selected))
        periods.append({"id": label, "label": label, "url": f"/data/rapm/{filename}", "rows": len(selected)})
    catalog.append({"id": ident, "title": title, "unit": unit, "note": note, "periods": periods})


def main() -> int:
    names = _player_names()
    catalog: list[dict] = []
    files: dict[str, dict] = {}

    stint_note = "Score-conserving stint RAPM. Penalties 3000/4500/300."
    annual = _normalize_player(pd.read_parquet(PULSE / "ratings.parquet"), names)
    _add_grouped(
        catalog, files, ident="annual", title="Annual RAPM", frame=annual,
        group="Season", columns=["PLAYER_ID", "PLAYER_NAME", "Season", "rapm_offense", "rapm_defense", "rapm_net", "Poss_Off", "Poss_Def"],
        note=stint_note,
    )

    rolling = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "rolling_5y.parquet"), names)
    _add_grouped(
        catalog, files, ident="rolling-five", title="Rolling five-year RAPM", frame=rolling,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
        note=stint_note,
    )

    rolling_three = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "rolling_3y.parquet"), names)
    _add_grouped(
        catalog, files, ident="rolling-three", title="Rolling three-year RAPM", frame=rolling_three,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
        note=stint_note,
    )

    full_history = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "full_history.parquet"), names)
    filename = "full-history-1997-2026.json"
    full_columns = ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    files[filename] = _write(OUTPUT / filename, _records(full_history[[column for column in full_columns if column in full_history]]))
    catalog.append({
        "id": "full-history",
        "title": "Full-history RAPM",
        "unit": "points per 100",
        "note": f"One canonical fit over 1997–2026. {stint_note}",
        "periods": [{"id": "1997-2026", "label": "1997–2026", "url": f"/data/rapm/{filename}", "rows": len(full_history)}],
    })

    wp_source = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/rolling_5y_wp_rapm/rolling_5y_wp_rapm_v1_39800d31b3/ratings_repaired.parquet"
    )
    wp_source = wp_source.drop(columns=[
        column for column in ("offense_per_100", "defense_per_100", "net_per_100")
        if column in wp_source
    ])
    wp = _normalize_player(wp_source, names).rename(columns={
        "offense_wp_percentage_points_per_100": "offense",
        "defense_wp_percentage_points_per_100": "defense",
        "net_wp_percentage_points_per_100": "net",
    })
    _add_grouped(
        catalog, files, ident="win-probability", title="Win-probability RAPM", frame=wp,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
        unit="win-probability percentage points per 100",
        note="Conserved game-level win-probability credit. Not points impact.",
    )

    factor_root = ROOT / "artifacts/research/historical_factor_targets/historical_factor_targets_v2_6cd7e959eb"
    factor_specs = (
        ("shooting_ts", "True-shooting value", "true-shooting points per 100 shooting possessions"),
        ("turnover_avoidance", "Turnover value", "turnovers avoided per 100 possessions"),
        ("shot_volume", "Shot-volume value", "true-shot attempts per 100 possessions"),
        ("opponent_oreb_prevention", "Opponent offensive-rebound prevention", "offensive rebounds prevented per 100 misses"),
    )
    for horizon, source_name, group, start_name, end_name in (
        ("annual", "annual_factor_targets.parquet", "Season", "Season", "Season"),
        ("rolling-five", "five_year_factor_targets.parquet", "Window_End", "Window_Start", "Window_End"),
    ):
        factors = _normalize_player(pd.read_parquet(factor_root / source_name), names)
        for prefix, title, unit in factor_specs:
            one = factors.rename(columns={
                f"{prefix}_offense": "offense",
                f"{prefix}_defense": "defense",
                f"{prefix}_off_exposure": "Poss_Off",
                f"{prefix}_def_exposure": "Poss_Def",
            }).copy()
            one["net"] = one["offense"] + one["defense"]
            columns = list(dict.fromkeys([
                "PLAYER_ID", "PLAYER_NAME", start_name, end_name,
                "offense", "defense", "net", "Poss_Off", "Poss_Def",
            ]))
            _add_grouped(
                catalog,
                files,
                ident=f"factor-{prefix}-{horizon}",
                title=f"{title} · {horizon.replace('-', ' ')}",
                frame=one,
                group=group,
                columns=columns,
                unit=unit,
                note="Native factor units. Offense and defense are lineup-adjusted; net is their sum.",
            )

    release = {
        "schema": "courtsignal_rapm_release_v1",
        "lineage": {
            "pulse_run": PULSE.name,
            "canonical_rapm_suite": CANONICAL_SUITE.name,
            "other_estimands": "win_probability_rapm_and_factor_boards_only",
        },
        "estimands": catalog,
        "files": files,
        "forbidden": ["Tuning-sweep winner claims", "Causal interpretation", "Forecast interpretation"],
    }
    manifest = _write(OUTPUT / "catalog.json", release)
    print(json.dumps({"estimands": len(catalog), "shards": len(files), "catalog": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
