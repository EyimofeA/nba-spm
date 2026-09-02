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
EXTERNAL_REPRODUCTION = ROOT / (
    "research/rapm_lab/outputs/external_reproduction_benchmark/"
    "external_reproduction_benchmark_v1_0a95702214"
)
POINT_CHANNELS = ROOT / (
    "research/rapm_lab/outputs/points_channel_rapm/"
    "points_channel_rapm_v1_4507aab97c"
)
WP_VS_PULSE = ROOT / (
    "research/rapm_lab/outputs/wp_rapm_vs_pulse/"
    "wp_rapm_vs_pulse_v1_3d2995995c"
)
LOG_ODDS_WP = ROOT / (
    "research/rapm_lab/outputs/log_odds_wp_rapm_lambda/"
    "log_odds_wp_rapm_lambda_v1_91e7dccec4"
)
TEAMMATE_EFFECTS = ROOT / (
    "research/rapm_lab/outputs/teammate_play_channels/"
    "teammate_play_channels_v1_9f5feb3641"
)
SHOOTING_LUCK = ROOT / (
    "research/rapm_lab/outputs/luck_teammate_shooting_rapm/"
    "luck_teammate_shooting_rapm_v1_3641085323"
)


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
    catalog.append({"id": "same-age-27", "title": "Full-history age-27 RAPM", "unit": "points per 100", "note": "One 1997–2026 fit with every player evaluated at age 27. This is not single-season RAPM.", "periods": [{"id": "1997-2026", "label": "1997–2026", "url": f"/data/rapm/{filename}", "rows": len(one)}]})

    age_curve = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/age_adjusted_rapm/age_adjusted_full_1997_2026_v1_1765feaffc/age_curve.parquet"
    ).rename(columns={"offense": "age_offense", "defense": "age_defense", "net": "age_net"})

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
        SHOOTING_LUCK / "luck_adjusted_ratings.parquet"
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
    wp = wp.loc[wp["window_end"].between(2024, 2026)].copy()
    _add_grouped(
        catalog, files, ident="win-probability", title="Win-probability RAPM", frame=wp,
        group="window_end", columns=["PLAYER_ID", "PLAYER_NAME", "window_start", "window_end", "offense", "defense", "net", "Poss_Off", "Poss_Def"],
        unit="win-probability percentage points per 100",
        note="Conserved game-level win-probability credit. Each board uses a rolling five-season fit; only the latest three endpoints are published.",
    )

    log_odds_wp = _normalize_player(
        pd.read_parquet(LOG_ODDS_WP / "public_ratings_2024_2026.parquet"), names
    )
    log_odds_wp = log_odds_wp.loc[
        log_odds_wp[["Poss_Off", "Poss_Def"]].min(axis=1).gt(0)
    ].copy()
    _add_grouped(
        catalog, files,
        ident="log-odds-win-probability",
        title="Log-odds WP-RAPM",
        frame=log_odds_wp,
        group="Season",
        columns=[
            "PLAYER_ID", "PLAYER_NAME", "Season", "offense", "defense", "net",
            "Poss_Off", "Poss_Def",
        ],
        unit="home-win log-odds per 100 possessions",
        note=(
            "Descriptive one-season leverage rating from clipped possession-to-possession "
            "home-win log-odds changes. The 2.5% clip and 150,000/150,000 penalties "
            "were selected on reused historical folds. This is not a forecast."
        ),
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

    # One compact six-factor board. The factor columns stay in their native
    # estimands; the final three columns are the point-scoring RAPM totals.
    # We do not add unlike factor units together or publish the discarded
    # factor-to-RAPM reconstruction.
    factors = _normalize_player(
        pd.read_parquet(factor_root / "annual_factor_targets.parquet"), names
    )
    rapm_totals = annual[[
        "PLAYER_ID", "Season", "rapm_offense", "rapm_defense", "rapm_net",
        "Poss_Off", "Poss_Def",
    ]]
    six = factors.merge(
        rapm_totals, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    ).rename(columns={
        "shooting_ts_offense": "off_ts",
        "turnover_avoidance_offense": "off_tov",
        "opponent_oreb_prevention_offense": "off_reb",
        "shooting_ts_defense": "def_ts",
        "turnover_avoidance_defense": "def_tov",
        "opponent_oreb_prevention_defense": "def_reb",
        "rapm_offense": "offense",
        "rapm_defense": "defense",
        "rapm_net": "net",
    })
    _add_grouped(
        catalog,
        files,
        ident="six-factor-annual",
        title="Six-factor RAPM",
        frame=six,
        group="Season",
        columns=[
            "PLAYER_ID", "PLAYER_NAME", "Season", "Poss_Off", "Poss_Def",
            "off_ts", "off_tov", "off_reb", "def_ts", "def_tov", "def_reb",
            "offense", "defense", "net",
        ],
        note="TS, turnover, and rebound factor ratings beside total offense, defense, and RAPM.",
    )

    game_pm = pd.read_parquet(EXTERNAL_REPRODUCTION / "matched_rows.parquet")
    game_pm = game_pm.loc[
        game_pm["source"].eq("CourtSignal reproduction")
        & game_pm["comparison"].eq("GPM-style game-level ridge")
        & game_pm["scope"].eq("2024-2026")
        & game_pm["component"].eq("net"),
        ["PLAYER_ID", "player_name", "reference"],
    ].rename(columns={"player_name": "PLAYER_NAME", "reference": "net"})
    game_pm["PLAYER_ID"] = pd.to_numeric(game_pm["PLAYER_ID"], errors="raise").astype(int)
    minutes = pd.read_parquet(
        ROOT / "data/lake/silver/player_games.parquet",
        columns=["season_end", "player_id", "played", "minutes_seconds"],
    )
    minutes = minutes.loc[
        minutes["season_end"].between(2024, 2026) & minutes["played"]
    ]
    minutes = (
        minutes.groupby("player_id", as_index=False)["minutes_seconds"].sum()
        .rename(columns={"player_id": "PLAYER_ID"})
    )
    minutes["minutes"] = minutes.pop("minutes_seconds") / 60
    game_pm = game_pm.merge(minutes, on="PLAYER_ID", how="left", validate="one_to_one")
    filename = "game-level-pm-2024-2026.json"
    files[filename] = _write(OUTPUT / filename, _records(game_pm))
    catalog.append({
        "id": "game-level-pm",
        "title": "Game-level plus-minus",
        "unit": "game margin points per 100",
        "note": "Ridge on final game margins and signed player minute shares, 2024–2026.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(game_pm)}],
    })

    point_channels = _normalize_player(
        pd.read_parquet(POINT_CHANNELS / "ratings.parquet"), names
    )
    filename = "point-channels-2024-2026.json"
    files[filename] = _write(OUTPUT / filename, _records(point_channels))
    catalog.append({
        "id": "point-channels",
        "title": "Point-channel RAPM",
        "unit": "points per 100",
        "note": "One-point, two-point, and three-plus-point channels add to total offense, defense, and RAPM.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(point_channels)}],
    })

    teammate_effects = _normalize_player(
        pd.read_parquet(TEAMMATE_EFFECTS / "teammate_effect_ratings.parquet"), names
    )
    filename = "teammate-effects-2024-2026.json"
    teammate_columns = [
        "PLAYER_ID", "PLAYER_NAME", "possession_opportunities",
        "teammate_scoring", "teammate_turnovers", "teammate_assists",
        "teammate_steals", "teammate_blocks", "teammate_oreb", "teammate_dreb",
    ]
    files[filename] = _write(
        OUTPUT / filename,
        _records(teammate_effects[teammate_columns]),
    )
    catalog.append({
        "id": "teammate-effects",
        "title": "Teammate outcome RAPM",
        "unit": "events per 100 opportunities",
        "note": "Lineup-adjusted associations with the other four teammates' outcomes. Descriptive, not causal.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(teammate_effects)}],
    })

    teammate_efg = _normalize_player(
        pd.read_parquet(SHOOTING_LUCK / "teammate_efg_ratings.parquet"), names
    )
    filename = "teammate-efg-2024-2026.json"
    teammate_efg_columns = [
        "PLAYER_ID", "PLAYER_NAME", "Poss_Off", "Poss_Def",
        "teammate_efg_offense", "shot_defense", "teammate_efg_net",
    ]
    files[filename] = _write(OUTPUT / filename, _records(teammate_efg[teammate_efg_columns]))
    catalog.append({
        "id": "teammate-efg",
        "title": "Teammate eFG and shot defense",
        "unit": "points per 100 shot attempts",
        "note": "The shooter is excluded from the offensive lineup. The defensive side estimates five-defender shot suppression.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(teammate_efg)}],
    })

    observable = _normalize_player(
        pd.read_parquet(TEAMMATE_EFFECTS / "observable_play_channel_ratings.parquet"), names
    )
    scoring_columns = [
        "PLAYER_ID", "PLAYER_NAME", "Poss_Off", "Poss_Def",
        "rim_assists_net", "transition_points_net", "three_point_points_net",
        "free_throw_points_net", "midrange_attempts_net", "rim_points_net",
    ]
    filename = "observable-scoring-channels-2024-2026.json"
    files[filename] = _write(OUTPUT / filename, _records(observable[scoring_columns]))
    catalog.append({
        "id": "observable-scoring-channels",
        "title": "Observable scoring channels",
        "unit": "channel value per 100 possessions",
        "note": "Lineup-adjusted rim assists, transition, three-point, free-throw, midrange-attempt, and rim-point channels.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(observable)}],
    })

    finish_columns = [
        "PLAYER_ID", "PLAYER_NAME", "Poss_Off", "Poss_Def",
        "playtype_transition_points_net", "playtype_putback_points_net",
        "playtype_cut_points_net", "playtype_drive_points_net",
        "playtype_pullup_points_net", "playtype_post_points_net",
        "playtype_spotup_points_net", "playtype_other_points_net",
    ]
    filename = "observable-finish-channels-2024-2026.json"
    files[filename] = _write(OUTPUT / filename, _records(observable[finish_columns]))
    catalog.append({
        "id": "observable-finish-channels",
        "title": "Observable shot-finish channels",
        "unit": "points per 100 possessions",
        "note": "Lineup-adjusted transition, putback, cut, drive, pull-up, post-like, jump-shot, and other finish channels.",
        "periods": [{"id": "2024-2026", "label": "2024–2026", "url": f"/data/rapm/{filename}", "rows": len(observable)}],
    })

    score_state_curve = pd.read_parquet(
        ROOT / "research/rapm_lab/outputs/rubberband_score_signal/"
        "rubberband_score_signal_v1_deac872ede/curve.parquet"
    )
    curve_filename = "research-curves.json"
    wp_vs_pulse = json.loads((WP_VS_PULSE / "run.json").read_text())
    files[curve_filename] = _write(OUTPUT / curve_filename, {
        "age": _records(age_curve.rename(columns={
            "age_offense": "offense",
            "age_defense": "defense",
            "age_net": "net",
        })),
        "score_state": _records(score_state_curve),
        "wp_rapm_vs_pulse": {
            key: wp_vs_pulse[key]
            for key in (
                "comparison", "target", "outcome_seasons", "games",
                "summary", "paired_comparisons", "warning",
            )
        },
    })

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
