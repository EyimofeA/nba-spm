"""Build compact localhost-only player-skill shards from the frozen artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "web/local-data/skills"
AUDIT = ROOT / "research/audits/predictive_player_skills_2026_v1/decision.json"
GAME_LOGS = ROOT / "data/lake/bronze/llimllib_nba_data/player_game_logs.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"

PROFILE_AXES: dict[str, tuple[str, ...]] = {
    "rim_finishing": ("rim_pct",),
    "three_point_shooting": ("three_point_pct", "catch_shoot_three_pct", "pull_up_three_pct", "corner_three_pct", "above_break_three_pct"),
    "free_throw_shooting": ("free_throw_pct",),
    "shotmaking": ("shotmaking_above_expected", "zts"),
    "shooting_context": ("three_point_pct", "catch_shoot_three_pct", "shot_quality"),
    "creation": ("assist_creation", "potential_assists", "drive_creation", "rim_pressure", "free_throw_pressure", "offensive_load"),
    "passing": ("rim_assists", "three_point_assists", "passing_efficiency"),
    "ball_security": ("turnover_rate", "live_ball_turnover_rate"),
    "rebounding": ("offensive_rebound_rate", "defensive_rebound_rate", "contested_rebound_conversion"),
    "rim_defense": ("rim_deterrence", "rim_points_saved"),
    "perimeter_defense": ("non_rim_shot_suppression", "matchup_adjusted_points_saved", "foul_discipline"),
    "disruption": ("turnovers_forced", "deflections", "recovered_blocks"),
}

GAME_SKILLS = {
    "free_throw_pct": ("freeThrowsMade", "freeThrowsAttempted"),
    "three_point_pct": ("threePointersMade", "threePointersAttempted"),
}


def _number(value: object, digits: int = 4) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    rounded = round(numeric, digits)
    return int(rounded) if rounded.is_integer() else rounded


def _write(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    temporary.replace(destination)


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return _number(float(np.mean(finite)), 1) if finite else None


def _game_observations(estimates: pd.DataFrame, active_ids: set[int]) -> dict[int, dict[str, list[dict]]]:
    logs = pd.read_parquet(
        GAME_LOGS,
        columns=["gameId", "personId", *{column for pair in GAME_SKILLS.values() for column in pair}],
    ).rename(columns={"personId": "PLAYER_ID"})
    schedule = pd.read_parquet(
        PLAYER_GAMES,
        columns=["game_id", "season_end", "season_type", "game_date", "player_id", "played"],
    ).rename(columns={"player_id": "PLAYER_ID"})
    logs["game_id"] = logs["gameId"].astype(str).str.zfill(10)
    schedule["game_id"] = schedule["game_id"].astype(str).str.zfill(10)
    schedule = schedule.loc[
        schedule["season_end"].eq(2026)
        & schedule["season_type"].eq("regular")
        & schedule["PLAYER_ID"].isin(active_ids)
    ]
    logs = logs.loc[logs["game_id"].isin(schedule["game_id"].unique())]
    logs = logs.drop_duplicates()
    if logs.duplicated(["game_id", "PLAYER_ID"]).any():
        raise ValueError("Current regular-season game logs have conflicting player-game rows.")
    games = schedule.merge(
        logs.drop(columns="gameId"),
        on=["game_id", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
    )
    games = games.sort_values(["PLAYER_ID", "game_date", "game_id"], kind="stable")
    current = estimates.loc[estimates["Season"].eq(2026)].set_index(["PLAYER_ID", "skill"])
    output: dict[int, dict[str, list[dict]]] = {}
    for player_id, player_games in games.groupby("PLAYER_ID", sort=False):
        skill_output: dict[str, list[dict]] = {}
        for skill, (makes_column, attempts_column) in GAME_SKILLS.items():
            key = (int(player_id), skill)
            if key not in current.index:
                continue
            annual = current.loc[key]
            prior_mean = float(annual["preseason_estimate"])
            prior_precision = float(annual["preseason_precision"])
            target_attempts = float(annual["opportunities"])
            target_makes = float(annual["raw_value"]) * target_attempts / 100.0
            if not all(np.isfinite(value) for value in (prior_mean, prior_precision, target_attempts, target_makes)):
                continue
            candidate = player_games.copy()
            candidate["_makes"] = pd.to_numeric(candidate[makes_column], errors="coerce").fillna(0.0)
            candidate["_attempts"] = pd.to_numeric(candidate[attempts_column], errors="coerce").fillna(0.0)
            candidate.loc[~candidate["played"].astype(bool), ["_makes", "_attempts"]] = 0.0
            candidate["_cumulative_makes"] = candidate["_makes"].cumsum()
            candidate["_cumulative_attempts"] = candidate["_attempts"].cumsum()
            reconciled = candidate.loc[
                np.isclose(candidate["_cumulative_makes"], target_makes, atol=1e-6)
                & np.isclose(candidate["_cumulative_attempts"], target_attempts, atol=1e-6)
            ]
            if reconciled.empty:
                continue
            cutoff_position = int(candidate.index.get_loc(reconciled.index[0]))
            candidate = candidate.iloc[: cutoff_position + 1]
            cumulative_makes = 0.0
            cumulative_attempts = 0.0
            rows: list[dict] = []
            played_number = 0
            for row in candidate.itertuples(index=False):
                played = bool(row.played)
                if played:
                    played_number += 1
                makes = float(getattr(row, makes_column)) if pd.notna(getattr(row, makes_column)) else 0.0
                attempts = float(getattr(row, attempts_column)) if pd.notna(getattr(row, attempts_column)) else 0.0
                if played and attempts > 0:
                    cumulative_makes += makes
                    cumulative_attempts += attempts
                stabilized = (
                    100.0 * (prior_precision * prior_mean / 100.0 + cumulative_makes) / (prior_precision + cumulative_attempts)
                    if played and prior_precision + cumulative_attempts > 0
                    else None
                )
                rows.append({
                    "date": pd.Timestamp(row.game_date).date().isoformat(),
                    "game": played_number if played else None,
                    "played": played,
                    "raw": _number(100.0 * makes / attempts if played and attempts > 0 else None),
                    "estimate": _number(stabilized),
                    "opportunities": int(attempts) if played else None,
                })
            final_played = next((row for row in reversed(rows) if row["played"]), None)
            if final_played is None or not np.isclose(
                float(final_played["estimate"]), float(annual["estimate"]), atol=1e-4
            ):
                raise ValueError(
                    f"Sequential {skill} posterior does not reconcile for PLAYER_ID={player_id}."
                )
            skill_output[skill] = rows
        output[int(player_id)] = skill_output
    return output


def build(run_id: str | None = None) -> dict:
    audit = json.loads(AUDIT.read_text())
    selected_run = run_id or str(audit["run_id"])
    run_dir = ROOT / "artifacts/models/predictive_player_skills" / selected_run
    manifest = json.loads((run_dir / "run.json").read_text())
    estimates = pd.read_parquet(run_dir / "skill_estimates.parquet")
    definitions = pd.read_parquet(run_dir / "skill_definitions.parquet")
    current = estimates.loc[estimates["Season"].eq(2026)]
    active_ids = set(map(int, current["PLAYER_ID"].unique()))
    game_data = _game_observations(estimates, active_ids)

    complete = current.groupby("PLAYER_ID")["raw_value"].count()
    complete_ids = set(map(int, complete.loc[complete.eq(len(definitions))].index))
    default_player = int(manifest["quality"]["default_player_id"])
    if default_player not in complete_ids:
        default_player = min(complete_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("player-*.json"):
        stale.unlink()
    players: list[dict] = []
    for player_id, player_rows in estimates.loc[estimates["PLAYER_ID"].isin(active_ids)].groupby("PLAYER_ID", sort=False):
        name = str(player_rows.loc[player_rows["Season"].eq(2026), "PLAYER_NAME"].iloc[0])
        team = player_rows.loc[player_rows["Season"].eq(2026), "TEAM_ABBREVIATION"].iloc[0]
        player_game_data = game_data.get(int(player_id), {})
        reconciled_cutoffs = [
            rows[-1]["date"] for rows in player_game_data.values() if rows
        ]
        source_cutoff = max(reconciled_cutoffs) if reconciled_cutoffs else None
        skills: dict[str, dict] = {}
        for skill, skill_rows in player_rows.groupby("skill", sort=False):
            skill_rows = skill_rows.sort_values("Season", kind="stable")
            latest = skill_rows.loc[skill_rows["Season"].eq(2026)].iloc[0]
            skills[str(skill)] = {
                "arm": latest["model_arm"],
                "halfLife": None if str(latest["half_life_years"]) == "none" else _number(latest["half_life_years"]),
                "prior": _number(latest["prior_strength"]),
                "rows": [
                    [
                        int(row.Season), _number(row.estimate), _number(row.raw_value),
                        _number(row.opportunities, 1), _number(row.percentile, 1),
                        _number(row.year_over_year_change), _number(row.standard_error),
                        source_cutoff
                        if int(row.Season) == 2026 and source_cutoff
                        else str(row.last_update_date),
                    ]
                    for row in skill_rows.itertuples(index=False)
                    if pd.notna(row.estimate) or pd.notna(row.raw_value)
                ],
            }
        profiles = []
        for season, season_rows in player_rows.groupby("Season", sort=True):
            lookup = season_rows.set_index("skill")["percentile"].to_dict()
            values = {
                axis: _finite_mean([lookup.get(skill, np.nan) for skill in members])
                for axis, members in PROFILE_AXES.items()
            }
            profiles.append({"season": int(season), **values})
        payload = {
            "id": int(player_id), "name": name, "team": None if pd.isna(team) else str(team),
            "complete2026": int(player_id) in complete_ids,
            "skills": skills, "profiles": profiles, "games": player_game_data,
        }
        _write(payload, OUTPUT / f"player-{int(player_id)}.json")
        players.append({"id": int(player_id), "name": name, "team": payload["team"], "complete2026": payload["complete2026"]})

    definition_records = []
    for row in definitions.itertuples(index=False):
        definition_records.append({
            "key": row.key, "label": row.label, "group": row.group, "unit": row.unit,
            "higherIsBetter": bool(row.higher_is_better), "definition": row.definition,
        })
    league_work = estimates.dropna(subset=["estimate"]).copy()
    league_work["weight"] = pd.to_numeric(league_work["opportunities"], errors="coerce").fillna(0).clip(lower=1)
    league_work["weighted_estimate"] = league_work["estimate"] * league_work["weight"]
    league_rows = league_work.groupby(["skill", "Season"], as_index=False).agg(
        weighted_estimate=("weighted_estimate", "sum"), weight=("weight", "sum")
    )
    league_rows["estimate"] = league_rows["weighted_estimate"] / league_rows["weight"]
    league = {
        str(skill): [[int(row.Season), _number(row.estimate)] for row in group.itertuples(index=False)]
        for skill, group in league_rows.groupby("skill", sort=False)
    }
    index = {
        "schema": "courtsignal_local_player_skills_v1", "scope": "localhost_only",
        "runId": selected_run, "defaultPlayerId": default_player,
        "seasons": sorted(map(int, estimates["Season"].unique())),
        "players": sorted(players, key=lambda row: (row["name"].casefold(), row["id"])),
        "definitions": definition_records,
        "profileAxes": {key: list(value) for key, value in PROFILE_AXES.items()},
        "gameSkills": list(GAME_SKILLS),
        "league": league,
    }
    _write(index, OUTPUT / "index.json")
    return {
        "run_id": selected_run, "players": len(players), "complete_2026": len(complete_ids),
        "default_player_id": default_player, "output": OUTPUT.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(build(args.run_id), indent=2))


if __name__ == "__main__":
    main()
