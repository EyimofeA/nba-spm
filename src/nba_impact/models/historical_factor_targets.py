"""Research-only TS and opponent-OREB factor targets from observed lineups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nba_impact.models.possession_outcome_rapm import (
    fit_factor_ratings,
    fit_weighted_factor_ratings,
)
from nba_impact.models.rapm import RapmConfig


SOURCE_COLUMNS = (
    "period",
    "clock",
    "minutes_left_in_game",
    "actionNumber",
    "actionType",
    "description",
    "qualifier",
    "scoreHome",
    "scoreAway",
    "shotResult",
    "assisted",
    "previous_action",
    "players_on",
    "person_id",
    "xLegacy",
    "yLegacy",
    "teamId",
    "game_id",
    "poc_ok",
    "date",
    "playoffs",
    "off_players_on",
    "def_players_on",
)
EVENT_KEY = (
    "game_id",
    "period",
    "actionNumber",
    "actionType",
    "person_id",
    "description",
)


@dataclass(frozen=True)
class HistoricalFactorLedger:
    shooting: pd.DataFrame
    opponent_oreb: pd.DataFrame
    quality: dict[str, object]


def _game_id(value: object) -> str:
    return f"{int(value):010d}"


def _lineup(value: object) -> tuple[int, ...]:
    players = tuple(sorted(int(player) for player in str(value).split("|") if player))
    return players if len(players) == 5 and len(set(players)) == 5 else ()


def load_gabriel_events(source_root: str | Path, season: int) -> tuple[pd.DataFrame, dict]:
    """Load one regular season and remove exact team-mirror duplicates."""
    paths = sorted(Path(source_root).glob(f"*_{season}_rs.parquet"))
    if len(paths) < 29:
        raise ValueError(f"Season {season} has only {len(paths)} Gabriel team files.")
    frames = []
    seen_games: set[str] = set()
    raw_rows = 0
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(
            path, columns=[column for column in SOURCE_COLUMNS if column in available]
        )
        if "poc_ok" not in frame:
            frame["poc_ok"] = True
        frame = frame.loc[~frame["playoffs"].fillna(False).astype(bool)].copy()
        raw_rows += len(frame)
        frame["game_id"] = frame["game_id"].map(_game_id)
        new_game = ~frame["game_id"].isin(seen_games)
        if new_game.any():
            selected = frame.loc[new_game].copy()
            frames.append(selected)
            seen_games.update(selected["game_id"].unique())
    raw = pd.concat(frames, ignore_index=True)
    raw["game_id"] = raw["game_id"].map(_game_id)
    raw["season"] = int(season)
    raw["teamId"] = pd.to_numeric(raw["teamId"], errors="coerce").astype("Int64")
    raw["person_id"] = pd.to_numeric(raw["person_id"], errors="coerce").astype("Int64")
    raw["actionNumber"] = pd.to_numeric(raw["actionNumber"], errors="raise").astype(int)
    raw["period"] = pd.to_numeric(raw["period"], errors="raise").astype(int)
    semantic = [*SOURCE_COLUMNS, "season"]
    events = raw.drop_duplicates(semantic, keep="first").copy()
    events = events.sort_values(
        ["game_id", "period", "actionNumber", "actionType", "person_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    duplicated_keys = int(events.duplicated(list(EVENT_KEY), keep=False).sum())
    return events, {
        "season": int(season),
        "source_files": len(paths),
        "raw_rows": int(raw_rows),
        "deduplicated_rows": int(len(events)),
        "mirror_rows_removed": int(raw_rows - len(events)),
        "games": int(events["game_id"].nunique()),
        "duplicate_event_keys": duplicated_keys,
    }


def derive_game_dim(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recover home/away identity from final scores and observed scoring events."""
    action = events["actionType"].fillna("").astype(str).str.lower()
    result = events["shotResult"].fillna("").astype(str).str.lower()
    scores = events.loc[
        action.isin(("2pt", "3pt", "freethrow"))
        & result.eq("made")
        & events["teamId"].notna()
    ].copy()
    score_action = scores["actionType"].astype(str).str.lower()
    scores["event_points"] = np.select(
        [score_action.eq("3pt"), score_action.eq("2pt")], [3, 2], default=1
    )
    observed = (
        scores.groupby(["game_id", "teamId"], as_index=False)["event_points"]
        .sum()
        .rename(columns={"event_points": "observed_points"})
    )
    finals = (
        events.groupby("game_id", as_index=False, sort=False)
        .agg(home_score=("scoreHome", "max"), away_score=("scoreAway", "max"))
    )
    candidates = observed.merge(finals, on="game_id", how="inner", validate="many_to_one")
    assignments_by_game: dict[str, tuple[int, int]] = {}
    vote_rows = 0
    consistent_vote_rows = 0
    for game_id, game in events.groupby("game_id", sort=False):
        ordered = game.sort_values(["period", "actionNumber"], kind="stable")
        home_change = pd.to_numeric(ordered["scoreHome"], errors="coerce").diff().fillna(0)
        away_change = pd.to_numeric(ordered["scoreAway"], errors="coerce").diff().fillna(0)
        team = pd.to_numeric(ordered["teamId"], errors="coerce")
        home_votes = team.loc[home_change.gt(0) & team.notna()].astype("int64")
        away_votes = team.loc[away_change.gt(0) & team.notna()].astype("int64")
        if home_votes.empty or away_votes.empty:
            continue
        home_id = int(home_votes.mode().iloc[0])
        away_id = int(away_votes.mode().iloc[0])
        if home_id == away_id:
            continue
        assignments_by_game[str(game_id)] = (home_id, away_id)
        vote_rows += len(home_votes) + len(away_votes)
        consistent_vote_rows += int(home_votes.eq(home_id).sum() + away_votes.eq(away_id).sum())

    assignments = [
        (game_id, home_id, away_id)
        for game_id, (home_id, away_id) in assignments_by_game.items()
    ]
    exact_games = 0
    approximate_games = 0
    maximum_error = 0.0
    for final in finals.itertuples(index=False):
        if final.game_id in assignments_by_game:
            continue
        game = candidates.loc[candidates["game_id"].eq(final.game_id)]
        if len(game) != 2:
            continue
        first, second = tuple(game.itertuples(index=False))
        direct = abs(first.observed_points - final.home_score) + abs(
            second.observed_points - final.away_score
        )
        reverse = abs(second.observed_points - final.home_score) + abs(
            first.observed_points - final.away_score
        )
        if direct == reverse:
            continue
        home, away, error = (
            (first, second, direct) if direct < reverse else (second, first, reverse)
        )
        assignments.append((final.game_id, int(home.teamId), int(away.teamId)))
        maximum_error = max(maximum_error, float(error))
        if error == 0:
            exact_games += 1
        else:
            approximate_games += 1
    assigned = pd.DataFrame(
        assignments, columns=("game_id", "home_team_id", "away_team_id")
    )
    games = finals.merge(assigned, on="game_id", how="left", validate="one_to_one")
    games[["home_team_id", "away_team_id"]] = games[
        ["home_team_id", "away_team_id"]
    ].astype("Int64")
    return games, {
        "games": int(len(games)),
        "games_with_score_derived_teams": int(len(assigned)),
        "games_with_exact_score_derived_teams": exact_games,
        "games_with_approximate_score_derived_teams": approximate_games,
        "games_without_score_derived_teams": int(len(games) - len(assigned)),
        "maximum_assignment_score_error": maximum_error,
        "score_change_team_consistency": (
            float(consistent_vote_rows / vote_rows) if vote_rows else 0.0
        ),
    }


def _lineup_rows(
    frame: pd.DataFrame, home_team: pd.Series
) -> tuple[pd.DataFrame, int, int, int]:
    work = frame.copy()
    work["off_lineup"] = work["off_players_on"].map(_lineup)
    work["def_lineup"] = work["def_players_on"].map(_lineup)
    all_players = work.get("players_on", pd.Series("", index=work.index))
    work["all_lineup"] = all_players.map(
        lambda value: tuple(sorted(int(player) for player in str(value).split("|") if player))
    )
    off_valid = work["off_lineup"].map(len).eq(5)
    def_valid = work["def_lineup"].map(len).eq(5)
    all_valid = work["all_lineup"].map(
        lambda players: len(players) == 10 and len(set(players)) == 10
    )
    complement_repairs = 0
    repair_off = ~off_valid & def_valid & all_valid
    if repair_off.any():
        repair_index = work.index[repair_off]
        work.loc[repair_index, "off_lineup"] = pd.Series(
            [
                tuple(sorted(set(all_players) - set(defense)))
                for all_players, defense in zip(
                    work.loc[repair_index, "all_lineup"],
                    work.loc[repair_index, "def_lineup"],
                    strict=True,
                )
            ],
            index=repair_index,
            dtype=object,
        )
        complement_repairs += len(repair_index)
    repair_def = off_valid & ~def_valid & all_valid
    if repair_def.any():
        repair_index = work.index[repair_def]
        work.loc[repair_index, "def_lineup"] = pd.Series(
            [
                tuple(sorted(set(all_players) - set(offense)))
                for all_players, offense in zip(
                    work.loc[repair_index, "all_lineup"],
                    work.loc[repair_index, "off_lineup"],
                    strict=True,
                )
            ],
            index=repair_index,
            dtype=object,
        )
        complement_repairs += len(repair_index)
    disjoint = pd.Series(
        [
            not set(offense).intersection(defense)
            for offense, defense in zip(
                work["off_lineup"], work["def_lineup"], strict=True
            )
        ],
        index=work.index,
    )
    structural = (
        work["off_lineup"].map(len).eq(5)
        & work["def_lineup"].map(len).eq(5)
        & disjoint
    )
    trusted = structural
    candidate = pd.Series(
        [
            (offense, defense) if is_trusted else None
            for offense, defense, is_trusted in zip(
                work["off_lineup"], work["def_lineup"], trusted, strict=True
            )
        ],
        index=work.index,
        dtype=object,
    )
    previous = candidate.groupby(work["game_id"]).ffill()
    following = candidate.groupby(work["game_id"]).bfill()
    bracketed = (
        ~trusted
        & previous.notna()
        & following.notna()
        & previous.eq(following)
    )
    if bracketed.any():
        repair_index = work.index[bracketed]
        work.loc[repair_index, "off_lineup"] = pd.Series(
            [value[0] for value in previous.loc[repair_index]],
            index=repair_index,
            dtype=object,
        )
        work.loc[repair_index, "def_lineup"] = pd.Series(
            [value[1] for value in previous.loc[repair_index]],
            index=repair_index,
            dtype=object,
        )
    repaired = int(bracketed.sum())
    valid = (
        (trusted | bracketed)
        & work["teamId"].notna()
        & home_team.notna()
    )
    valid = valid | (
        bracketed & work["teamId"].notna() & home_team.notna()
    )
    invalid = int((~valid).sum())
    work = work.loc[valid].copy()
    home = home_team.loc[valid].astype("int64")
    work["home_poss"] = work["teamId"].astype("int64").eq(home).astype(int)
    for index in range(5):
        work[f"h{index + 1}"] = [
            off[index] if is_home else defense[index]
            for off, defense, is_home in zip(
                work["off_lineup"], work["def_lineup"], work["home_poss"], strict=True
            )
        ]
        work[f"a{index + 1}"] = [
            defense[index] if is_home else off[index]
            for off, defense, is_home in zip(
                work["off_lineup"], work["def_lineup"], work["home_poss"], strict=True
            )
        ]
    work["gameid"] = work["game_id"]
    work["num"] = np.arange(len(work), dtype=np.int64)
    return work, invalid, repaired, complement_repairs


def build_historical_factor_ledger(
    events: pd.DataFrame,
    game_dim: pd.DataFrame,
) -> HistoricalFactorLedger:
    """Create event-weighted TS and resolved-miss opponent-OREB rows."""
    games = game_dim[["game_id", "home_team_id", "away_team_id"]].copy()
    games["game_id"] = games["game_id"].map(_game_id)
    games = games.drop_duplicates("game_id")
    merged = events.merge(games, on="game_id", how="left", validate="many_to_one")
    action = merged["actionType"].fillna("").astype(str).str.lower()
    result = merged["shotResult"].fillna("").astype(str).str.lower()

    shooting_source = merged.loc[action.isin(("2pt", "3pt", "freethrow"))].copy()
    shooting_action = shooting_source["actionType"].astype(str).str.lower()
    shooting_result = shooting_source["shotResult"].fillna("").astype(str).str.lower()
    shooting_source["ts_attempt_weight"] = np.where(
        shooting_action.eq("freethrow"), 0.44, 1.0
    )
    made = shooting_result.eq("made")
    shooting_source["ts_value"] = np.select(
        [made & shooting_action.eq("3pt"), made & shooting_action.eq("2pt"), made],
        [1.5, 1.0, 1.0 / 0.88],
        default=0.0,
    )
    shooting, invalid_shooting, repaired_shooting, complement_shooting = _lineup_rows(
        shooting_source, shooting_source["home_team_id"]
    )

    rebound_rows: list[dict[str, object]] = []
    unresolved_misses = 0
    for _, game in merged.groupby("game_id", sort=False):
        pending: dict[str, object] | None = None
        for row in game.sort_values(["period", "actionNumber"], kind="stable").itertuples():
            row_action = str(row.actionType).lower()
            row_result = str(row.shotResult).lower()
            if row_action in {"2pt", "3pt"} and row_result == "missed":
                if pending is not None:
                    unresolved_misses += 1
                pending = row._asdict()
            elif row_action in {"2pt", "3pt"} and row_result == "made":
                if pending is not None:
                    unresolved_misses += 1
                pending = None
            elif row_action == "rebound" and pending is not None:
                if pd.notna(row.teamId) and pd.notna(pending["teamId"]):
                    pending["offensive_rebound"] = float(int(row.teamId) == int(pending["teamId"]))
                    rebound_rows.append(pending)
                pending = None
            elif row_action in {"turnover", "period"}:
                if pending is not None:
                    unresolved_misses += 1
                pending = None
        if pending is not None:
            unresolved_misses += 1
    rebound_source = pd.DataFrame(rebound_rows)
    opponent_oreb, invalid_rebounds, repaired_rebounds, complement_rebounds = _lineup_rows(
        rebound_source, rebound_source["home_team_id"]
    )

    score_rows = merged.loc[action.isin(("2pt", "3pt", "freethrow")) & result.eq("made")].copy()
    score_rows["event_points"] = np.select(
        [score_rows["actionType"].astype(str).str.lower().eq("3pt")], [3], default=1
    )
    score_rows.loc[
        score_rows["actionType"].astype(str).str.lower().eq("2pt"), "event_points"
    ] = 2
    observed = score_rows.groupby(["game_id", "teamId"], dropna=False)["event_points"].sum()
    expected_rows = []
    game_scores = merged.groupby("game_id", as_index=False, sort=False).agg(
        home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"),
        scoreHome=("scoreHome", "max"),
        scoreAway=("scoreAway", "max"),
    )
    for record in game_scores.itertuples(index=False):
        if pd.notna(record.home_team_id):
            expected_rows.extend(
                [
                    (record.game_id, int(record.home_team_id), int(record.scoreHome)),
                    (record.game_id, int(record.away_team_id), int(record.scoreAway)),
                ]
            )
    expected = pd.Series(
        [points for _, _, points in expected_rows],
        index=pd.MultiIndex.from_tuples(
            [(game_id, team_id) for game_id, team_id, _ in expected_rows],
            names=("game_id", "teamId"),
        ),
    )
    aligned = expected.to_frame("expected").join(observed.rename("observed"), how="left")
    aligned["observed"] = aligned["observed"].fillna(0)
    score_match_rate = float(aligned["expected"].eq(aligned["observed"]).mean())
    quality = {
        "events": int(len(events)),
        "games": int(events["game_id"].nunique()),
        "games_missing_home_team": int(merged.loc[merged["home_team_id"].isna(), "game_id"].nunique()),
        "shooting_rows": int(len(shooting)),
        "invalid_shooting_lineups": invalid_shooting,
        "repaired_shooting_lineups": repaired_shooting,
        "complement_repaired_shooting_lineups": complement_shooting,
        "resolved_misses": int(len(opponent_oreb)),
        "unresolved_misses": int(unresolved_misses),
        "invalid_rebound_lineups": invalid_rebounds,
        "repaired_rebound_lineups": repaired_rebounds,
        "complement_repaired_rebound_lineups": complement_rebounds,
        "offensive_rebound_rate": float(opponent_oreb["offensive_rebound"].mean()),
        "team_game_score_match_rate": score_match_rate,
    }
    return HistoricalFactorLedger(shooting, opponent_oreb, quality)


def fit_historical_factor_ratings(
    ledgers: dict[int, HistoricalFactorLedger],
    seasons: tuple[int, ...],
    *,
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> pd.DataFrame:
    """Fit both factors jointly over one supplied season window."""
    shooting = pd.concat([ledgers[season].shooting for season in seasons], ignore_index=True)
    rebounds = pd.concat([ledgers[season].opponent_oreb for season in seasons], ignore_index=True)
    config = RapmConfig(
        seasons=seasons,
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=lambda_home,
        data_scope="historical_factor_targets_research_only",
    )
    ts = fit_weighted_factor_ratings(
        shooting,
        "ts_value",
        "ts_attempt_weight",
        factor="shooting_ts",
        config=config,
    )
    oreb = fit_factor_ratings(
        rebounds,
        "offensive_rebound",
        factor="opponent_oreb_prevention",
        higher_is_good_for_offense=True,
        config=config,
    )
    fields = [
        "player_id",
        "shooting_ts_offense",
        "shooting_ts_defense",
        "shooting_ts_off_exposure",
        "shooting_ts_def_exposure",
    ]
    other = [
        "player_id",
        "opponent_oreb_prevention_offense",
        "opponent_oreb_prevention_defense",
        "opponent_oreb_prevention_off_exposure",
        "opponent_oreb_prevention_def_exposure",
    ]
    return ts[fields].merge(oreb[other], on="player_id", how="outer", validate="one_to_one")
