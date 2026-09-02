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

    annual = _normalize_player(pd.read_parquet(PULSE / "ratings.parquet"), names)
    _add_grouped(
        catalog, files, ident="annual", title="Annual RAPM", frame=annual,
        group="Season", columns=["PLAYER_ID", "PLAYER_NAME", "Season", "rapm_offense", "rapm_defense", "rapm_net", "Poss_Off", "Poss_Def"],
    )

    rolling = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "rolling_5y.parquet"), names)
    _add_grouped(
        catalog, files, ident="rolling-five", title="Rolling five-year RAPM", frame=rolling,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
    )

    rolling_three = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "rolling_3y.parquet"), names)
    _add_grouped(
        catalog, files, ident="rolling-three", title="Rolling three-year RAPM", frame=rolling_three,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
    )

    full_history = _normalize_player(pd.read_parquet(CANONICAL_SUITE / "full_history.parquet"), names)
    filename = "full-history-1997-2026.json"
    full_columns = ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    files[filename] = _write(OUTPUT / filename, _records(full_history[[column for column in full_columns if column in full_history]]))
    catalog.append({"id": "full-history", "title": "Full-history RAPM", "unit": "points per 100", "note": "One canonical fit over 1997–2026.", "periods": [{"id": "1997-2026", "label": "1997–2026", "url": f"/data/rapm/{filename}", "rows": len(full_history)}]})

    age = _normalize_player(pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/age_adjusted_rapm/age_adjusted_full_1997_2026_v1_1765feaffc/ratings.parquet"
    ), names)
    one = age.rename(columns={f"age27_{side}": side for side in ("offense", "defense", "net")})
    filename = "same-age-27-1997-2026.json"
    cols = ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    files[filename] = _write(OUTPUT / filename, _records(one[[c for c in cols if c in one]]))
    catalog.append({"id": "same-age-27", "title": "Full-history same-age RAPM", "unit": "points per 100", "note": "Player impact evaluated at age 27.", "periods": [{"id": "1997-2026", "label": "1997–2026", "url": f"/data/rapm/{filename}", "rows": len(one)}]})

    age_curve = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/age_adjusted_rapm/age_adjusted_full_1997_2026_v1_1765feaffc/age_curve.parquet"
    ).rename(columns={"offense": "age_offense", "defense": "age_defense", "net": "age_net"})
    age_rows = []
    for season in range(1997, 2027):
        if season <= 2013:
            source = ROOT / f"data/lake/bronze/historical_player_sheets/year_totals/{season}.csv"
            annual_age = pd.read_csv(source, usecols=["PLAYER_ID", "AGE"])
        else:
            source = ROOT / f"data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals/{season}.parquet"
            annual_age = pd.read_parquet(source, columns=["PLAYER_ID", "AGE"])
        annual_age = annual_age.dropna().drop_duplicates("PLAYER_ID")
        annual_age["PLAYER_ID"] = pd.to_numeric(annual_age["PLAYER_ID"], errors="raise").astype(int)
        annual_age["age"] = pd.to_numeric(annual_age["AGE"], errors="raise").round().clip(19, 43).astype(int)
        annual_age["Season"] = season
        age_rows.append(annual_age[["PLAYER_ID", "Season", "age"]])
    actual_age = pd.concat(age_rows, ignore_index=True).merge(age_curve, on="age", validate="many_to_one")
    same_age = age[["PLAYER_ID", "PLAYER_NAME", "age27_offense", "age27_defense"]]
    actual_age = actual_age.merge(same_age, on="PLAYER_ID", how="inner", validate="many_to_one")
    actual_age["offense"] = actual_age["age27_offense"] + actual_age["age_offense"]
    actual_age["defense"] = actual_age["age27_defense"] + actual_age["age_defense"]
    actual_age["net"] = actual_age["offense"] + actual_age["defense"]
    _add_grouped(
        catalog, files, ident="full-history-actual-age", title="Full-history age-conditioned RAPM",
        frame=actual_age, group="Season",
        columns=["PLAYER_ID", "PLAYER_NAME", "Season", "age", "offense", "defense", "net"],
        note="One full-history fit evaluated at each player-season age.",
    )

    decay = _normalize_player(pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/time_decay_actual_age_5y_rapm/time_decay_actual_age_5y_rapm_v1_009decdcfa/ratings.parquet"
    ), names)
    filename = "current-age-time-decay-2022-2026.json"
    decay_cols = ["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    files[filename] = _write(OUTPUT / filename, _records(decay[[c for c in decay_cols if c in decay]]))
    catalog.append({"id": "current-age-time-decay", "title": "Current age-conditioned time decay", "unit": "points per 100", "note": "Five recent seasons with chronology-tuned decay and age controls.", "periods": [{"id": "2022-2026", "label": "2022–2026", "url": f"/data/rapm/{filename}", "rows": len(decay)}]})

    normal_decay = _normalize_player(pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/current_time_decay_rapm/current_time_decay_rapm_v1_16f96bf312/ratings.parquet"
    ), names)
    filename = "current-time-decay-2022-2026.json"
    files[filename] = _write(OUTPUT / filename, _records(normal_decay[[c for c in decay_cols if c in normal_decay]]))
    catalog.append({"id": "current-time-decay", "title": "Current time-decayed RAPM", "unit": "points per 100", "note": "Five recent seasons with a five-year half-life. Research leaderboard only.", "periods": [{"id": "2022-2026", "label": "2022–2026", "url": f"/data/rapm/{filename}", "rows": len(normal_decay)}]})

    luck = _normalize_player(pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/luck_teammate_shooting_rapm/luck_teammate_shooting_rapm_v1_3641085323/luck_adjusted_ratings.parquet"
    ), names).rename(columns={"luck_offense": "offense", "luck_defense": "defense", "luck_net": "net"})
    filename = "luck-adjusted-2024-2026.json"
    luck_cols = ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    files[filename] = _write(OUTPUT / filename, _records(luck[[c for c in luck_cols if c in luck]]))
    catalog.append({"id": "luck-adjusted", "title": "Three-year luck-adjusted RAPM", "unit": "points per 100", "note": "Adjusts free-throw and three-point shooting residuals.", "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(luck)}]})

    coaches = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/full_coach_age_rapm/coach_age_full_1997_2026_v1_16d4f5e1e0/coach_ratings.parquet"
    ).rename(columns={"coach": "PLAYER_NAME", "coach_id": "PLAYER_ID"})
    filename = "coach-1997-2026.json"
    coach_cols = ["PLAYER_ID", "PLAYER_NAME", "seasons", "teams", "listed_games", "offense", "defense", "net"]
    files[filename] = _write(OUTPUT / filename, _records(coaches[coach_cols]))
    catalog.append({"id": "coach", "title": "Coach RAPM", "unit": "points per 100", "note": "Full-history coach effects with player-age controls.", "periods": [{"id": "1997-2026", "label": "1997–2026", "url": f"/data/rapm/{filename}", "rows": len(coaches)}]})

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
        unit="win-probability percentage points per 100", note="Conserved game-level win-probability credit.",
    )

    units = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/standalone_unit_rapm/standalone_unit_rapm_v1_460b34a3b1/diagnostic_unit_ratings.parquet"
    )
    units["PLAYER_NAME"] = units["players"].map(
        lambda value: " · ".join(names.get(int(item), str(item)) for item in str(value).split("|") if item)
    )
    units["PLAYER_ID"] = units["players"]
    units = units.rename(columns={"offense_per_100": "offense", "defense_per_100": "defense", "net_per_100": "net"})
    _add_grouped(
        catalog, files, ident="units", title="Unit RAPM", frame=units, group="order",
        columns=["PLAYER_ID", "PLAYER_NAME", "order", "offense", "defense", "net"],
        note="Standalone pair, trio, quartet, and five-player unit coefficients.",
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
            "other_estimands": "selected_research_artifacts_with_per-estimand_notes",
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
