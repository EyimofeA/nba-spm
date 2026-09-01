"""Leak-free player-game Box15 inputs for blocked-game validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.data.statistical_features import CORE_RATE_SPECS
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES


BOX_COUNTS = tuple(
    dict.fromkeys(CORE_RATE_SPECS[feature][0] for feature in BOX_PIPM_STYLE_FEATURES)
)
ESPN_COLUMNS = (
    "season",
    "game_id",
    "player_id",
    "pts",
    "assister",
    "tov1",
    "stlr",
    "blockplyr",
    "orebounder",
    "drebounder",
    "dfoulplyr",
    "ofoulplyr",
    "ftaplyr",
    "ftmplyr",
    "fgaplyr",
    "fgmplyr",
    "fg3aplyr",
    "fg3mplyr",
)
ESPN_RENAMES = {
    "player_id": "PLAYER_ID",
    "pts": "PTS",
    "assister": "AST",
    "tov1": "TOV",
    "stlr": "STL",
    "blockplyr": "BLK",
    "orebounder": "OREB",
    "drebounder": "DREB",
    "ftaplyr": "FTA",
    "ftmplyr": "FTM",
    "fg3aplyr": "FG3A",
    "fg3mplyr": "FG3M",
}
NBA_COLUMNS = (
    "gameId",
    "personId",
    "points",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "reboundsOffensive",
    "reboundsDefensive",
    "foulsPersonal",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "threePointersAttempted",
    "threePointersMade",
)
PBP_COLUMNS = (
    "game_id",
    "period",
    "actionNumber",
    "actionType",
    "description",
    "person_id",
    "foulDrawnPersonId",
    "playoffs",
)
GAME_FINAL_COLUMNS = (
    "game_id",
    "period",
    "scoreHome",
    "scoreAway",
    "playoffs",
)


def _game_id(value: object) -> str:
    return f"{int(value):010d}"


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def player_game_exposure(possessions: pd.DataFrame) -> pd.DataFrame:
    """Count offensive and defensive possession rows by game and player."""
    required = {
        "gameid",
        "home_poss",
        *AWAY_PLAYER_COLUMNS,
        *HOME_PLAYER_COLUMNS,
    }
    if missing := sorted(required - set(possessions.columns)):
        raise ValueError(f"Possession frame is missing {missing}.")
    away = possessions.loc[:, AWAY_PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    home = possessions.loc[:, HOME_PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    home_offense = possessions["home_poss"].astype(bool).to_numpy()
    offense = np.where(home_offense[:, None], home, away)
    defense = np.where(home_offense[:, None], away, home)
    game_ids = possessions["gameid"].astype(str).to_numpy()

    outputs = []
    for players, field in ((offense, "OffPoss"), (defense, "DefPoss")):
        rows = pd.DataFrame(
            {
                "game_id": np.repeat(game_ids, 5),
                "PLAYER_ID": players.ravel(),
                field: 1,
            }
        )
        outputs.append(
            rows.groupby(["game_id", "PLAYER_ID"], as_index=False, sort=False)[
                field
            ].sum()
        )
    exposure = outputs[0].merge(
        outputs[1], on=["game_id", "PLAYER_ID"], how="outer", validate="one_to_one"
    )
    exposure[["OffPoss", "DefPoss"]] = exposure[["OffPoss", "DefPoss"]].fillna(0)
    exposure["PLAYER_ID"] = exposure["PLAYER_ID"].astype("int64")
    if exposure.duplicated(["game_id", "PLAYER_ID"]).any():
        raise AssertionError("Player-game exposure keys must be unique.")
    return exposure.sort_values(["game_id", "PLAYER_ID"]).reset_index(drop=True)


def load_espn_player_game_box(
    path: str | Path,
    *,
    season: int,
    game_ids: Iterable[str],
    require_all_games: bool = True,
) -> pd.DataFrame:
    """Load raw player-game box counts for one exact game universe."""
    expected_games = {str(value) for value in game_ids}
    frame = pd.read_parquet(path, columns=list(ESPN_COLUMNS))
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)].copy()
    frame["game_id"] = frame["game_id"].map(_game_id)
    frame = frame.loc[frame["game_id"].isin(expected_games)].copy()
    observed_games = set(frame["game_id"])
    if require_all_games and (missing := sorted(expected_games - observed_games)):
        raise ValueError(f"ESPN player box is missing {len(missing)} games: {missing[:10]}.")
    if frame.duplicated(["game_id", "player_id"]).any():
        raise ValueError("ESPN player box has duplicate player-game keys.")

    numeric_source = set(ESPN_RENAMES) - {"player_id"}
    _numeric(frame, numeric_source | {"dfoulplyr", "ofoulplyr", "fgaplyr", "fgmplyr"})
    frame = frame.rename(columns=ESPN_RENAMES)
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(
        "int64"
    )
    frame["PF"] = frame["dfoulplyr"] + frame["ofoulplyr"]
    frame["FG2A"] = frame["fgaplyr"] - frame["FG3A"]
    frame["FG2M"] = frame["fgmplyr"] - frame["FG3M"]
    if (frame[["FG2A", "FG2M"]] < 0).any().any():
        raise ValueError("Derived two-point counts cannot be negative.")
    keep = ["game_id", "PLAYER_ID", *(field for field in BOX_COUNTS if field != "PFD")]
    return frame[keep].sort_values(["game_id", "PLAYER_ID"]).reset_index(drop=True)


def load_nba_player_game_box(
    path: str | Path, *, game_ids: Iterable[str]
) -> pd.DataFrame:
    """Load official-style box counts for an exact fallback game universe."""
    expected_games = {str(value) for value in game_ids}
    if not expected_games:
        return pd.DataFrame(
            columns=["game_id", "PLAYER_ID", *(field for field in BOX_COUNTS if field != "PFD")]
        )
    frame = pd.read_parquet(path, columns=list(NBA_COLUMNS))
    frame["game_id"] = frame["gameId"].map(_game_id)
    frame = frame.loc[frame["game_id"].isin(expected_games)].copy()
    if missing := sorted(expected_games - set(frame["game_id"])):
        raise ValueError(f"NBA player boxes are missing {len(missing)} games: {missing[:10]}.")
    frame = frame.drop_duplicates()
    if frame.duplicated(["game_id", "personId"]).any():
        raise ValueError("NBA player boxes have duplicate player-game keys.")
    source = [column for column in NBA_COLUMNS if column not in {"gameId", "personId"}]
    _numeric(frame, source)
    frame["PLAYER_ID"] = pd.to_numeric(frame["personId"], errors="raise").astype("int64")
    frame["PTS"] = frame["points"]
    frame["AST"] = frame["assists"]
    frame["TOV"] = frame["turnovers"]
    frame["STL"] = frame["steals"]
    frame["BLK"] = frame["blocks"]
    frame["OREB"] = frame["reboundsOffensive"]
    frame["DREB"] = frame["reboundsDefensive"]
    frame["PF"] = frame["foulsPersonal"]
    frame["FTA"] = frame["freeThrowsAttempted"]
    frame["FTM"] = frame["freeThrowsMade"]
    frame["FG3A"] = frame["threePointersAttempted"]
    frame["FG3M"] = frame["threePointersMade"]
    frame["FG2A"] = frame["fieldGoalsAttempted"] - frame["FG3A"]
    frame["FG2M"] = frame["fieldGoalsMade"] - frame["FG3M"]
    keep = ["game_id", "PLAYER_ID", *(field for field in BOX_COUNTS if field != "PFD")]
    if frame[keep[2:]].lt(0).any().any():
        raise ValueError("Derived NBA player-game box counts cannot be negative.")
    return frame[keep].sort_values(["game_id", "PLAYER_ID"]).reset_index(drop=True)


def load_player_game_fouls_drawn(
    source_root: str | Path,
    *,
    season: int,
    game_ids: Iterable[str],
) -> pd.DataFrame:
    """Count one foul drawn for every canonical foul event with a beneficiary."""
    expected_games = {str(value) for value in game_ids}
    paths = sorted(Path(source_root).glob(f"*_{season}_rs.parquet"))
    if len(paths) < 29:
        raise ValueError(f"Season {season} has only {len(paths)} Gabriel team files.")
    frames: list[pd.DataFrame] = []
    seen_games: set[str] = set()
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        required = {"game_id", "actionType", "foulDrawnPersonId"}
        if missing := sorted(required - available):
            raise ValueError(f"{path} is missing foul-drawn fields {missing}.")
        frame = pd.read_parquet(
            path, columns=[column for column in PBP_COLUMNS if column in available]
        )
        if "playoffs" in frame:
            frame = frame.loc[~frame["playoffs"].fillna(False).astype(bool)].copy()
        frame["game_id"] = frame["game_id"].map(_game_id)
        keep_game = frame["game_id"].isin(expected_games - seen_games)
        if keep_game.any():
            selected = frame.loc[keep_game].copy()
            frames.append(selected)
            seen_games.update(selected["game_id"].unique())
        if seen_games == expected_games:
            break
    if missing := sorted(expected_games - seen_games):
        raise ValueError(f"Gabriel PBP is missing {len(missing)} games: {missing[:10]}.")

    events = pd.concat(frames, ignore_index=True)
    keys = [
        column
        for column in (
            "game_id",
            "period",
            "actionNumber",
            "actionType",
            "person_id",
            "foulDrawnPersonId",
            "description",
        )
        if column in events
    ]
    events = events.drop_duplicates(keys, keep="first")
    action = events["actionType"].fillna("").astype(str).str.lower()
    player_id = pd.to_numeric(events["foulDrawnPersonId"], errors="coerce")
    fouls = events.loc[action.eq("foul") & player_id.notna(), ["game_id"]].copy()
    fouls["PLAYER_ID"] = player_id.loc[fouls.index].astype("int64")
    fouls["PFD"] = 1.0
    return (
        fouls.groupby(["game_id", "PLAYER_ID"], as_index=False, sort=False)["PFD"]
        .sum()
        .sort_values(["game_id", "PLAYER_ID"])
        .reset_index(drop=True)
    )


def load_gabriel_game_finals(
    source_root: str | Path,
    *,
    season: int,
    game_ids: Iterable[str],
) -> pd.DataFrame:
    """Load final scores and periods after checking mirrored team files agree."""
    expected_games = {str(value) for value in game_ids}
    paths = sorted(Path(source_root).glob(f"*_{season}_rs.parquet"))
    if len(paths) < 29:
        raise ValueError(f"Season {season} has only {len(paths)} Gabriel team files.")
    summaries: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        required = {"game_id", "period", "scoreHome", "scoreAway"}
        if missing := sorted(required - available):
            raise ValueError(f"{path} is missing game-final fields {missing}.")
        frame = pd.read_parquet(
            path, columns=[column for column in GAME_FINAL_COLUMNS if column in available]
        )
        if "playoffs" in frame:
            frame = frame.loc[~frame["playoffs"].fillna(False).astype(bool)].copy()
        frame["game_id"] = frame["game_id"].map(_game_id)
        frame = frame.loc[frame["game_id"].isin(expected_games)].copy()
        if frame.empty:
            continue
        frame["period"] = pd.to_numeric(frame["period"], errors="raise")
        frame["scoreHome"] = pd.to_numeric(frame["scoreHome"], errors="coerce")
        frame["scoreAway"] = pd.to_numeric(frame["scoreAway"], errors="coerce")
        local = frame.groupby("game_id", as_index=False).agg(
            scoreHome=("scoreHome", "max"),
            scoreAway=("scoreAway", "max"),
            official_max_period=("period", "max"),
        )
        if local[["scoreHome", "scoreAway"]].isna().any().any():
            raise ValueError(f"{path} has a game without a final score state.")
        summaries.append(local)
    if not summaries:
        raise ValueError("Gabriel PBP produced no game-final rows.")
    mirrored = pd.concat(summaries, ignore_index=True)
    value_columns = ["scoreHome", "scoreAway", "official_max_period"]
    if mirrored.groupby("game_id")[value_columns].nunique().gt(1).any().any():
        raise ValueError("Mirrored Gabriel game files disagree on a final score or period.")
    finals = mirrored.drop_duplicates("game_id", keep="first")
    if missing := sorted(expected_games - set(finals["game_id"])):
        raise ValueError(f"Gabriel PBP is missing {len(missing)} game finals: {missing[:10]}.")
    finals[["scoreHome", "scoreAway", "official_max_period"]] = finals[
        ["scoreHome", "scoreAway", "official_max_period"]
    ].astype("int64")
    return finals.sort_values("game_id").reset_index(drop=True)


def audit_legacy_game_target_parity(
    possessions: pd.DataFrame,
    game_finals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Identify regulation games whose cached possession points conserve score."""
    required = {"gameid", "period", "home_poss", "pts"}
    if missing := sorted(required - set(possessions.columns)):
        raise ValueError(f"Possession frame is missing {missing}.")
    work = possessions[["gameid", "period", "home_poss", "pts"]].copy()
    work["game_id"] = work.pop("gameid").astype(str)
    work["pts"] = pd.to_numeric(work["pts"], errors="raise")
    work["home_poss"] = pd.to_numeric(work["home_poss"], errors="raise")
    work["cache_home_points"] = np.where(work["home_poss"].eq(1), work["pts"], 0.0)
    work["cache_away_points"] = np.where(work["home_poss"].eq(0), work["pts"], 0.0)
    cached = work.groupby("game_id", as_index=False).agg(
        cache_rows=("pts", "size"),
        cache_max_period=("period", "max"),
        cache_home_points=("cache_home_points", "sum"),
        cache_away_points=("cache_away_points", "sum"),
    )
    parity = cached.merge(game_finals, on="game_id", how="left", validate="one_to_one")
    if parity[["scoreHome", "scoreAway", "official_max_period"]].isna().any().any():
        raise ValueError("Every cached game requires one Gabriel final score.")
    parity["complete_regulation"] = parity["cache_max_period"].eq(4) & parity[
        "official_max_period"
    ].eq(4)
    parity["score_conserved"] = parity["cache_home_points"].eq(
        parity["scoreHome"]
    ) & parity["cache_away_points"].eq(parity["scoreAway"])
    parity["strict_eligible"] = parity["complete_regulation"] & parity[
        "score_conserved"
    ]
    quality = {
        "cache_games": int(len(parity)),
        "complete_regulation_games": int(parity["complete_regulation"].sum()),
        "score_conserved_games": int(parity["score_conserved"].sum()),
        "strict_eligible_games": int(parity["strict_eligible"].sum()),
        "overtime_or_incomplete_games": int((~parity["complete_regulation"]).sum()),
        "complete_regulation_score_mismatch_games": int(
            (parity["complete_regulation"] & ~parity["score_conserved"]).sum()
        ),
        "strict_eligible_fraction": float(parity["strict_eligible"].mean()),
    }
    return parity.sort_values("game_id").reset_index(drop=True), quality


def build_player_game_box15_ledger(
    possessions: pd.DataFrame,
    *,
    espn_player_box_path: str | Path,
    gabriel_pbp_root: str | Path,
    season: int,
    nba_player_game_box_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Join exact game-level counts to the canonical RAPM exposure universe."""
    exposure = player_game_exposure(possessions)
    game_ids = tuple(sorted(exposure["game_id"].unique()))
    box = load_espn_player_game_box(
        espn_player_box_path,
        season=season,
        game_ids=game_ids,
        require_all_games=nba_player_game_box_path is None,
    )
    missing_box_games = sorted(set(game_ids) - set(box["game_id"]))
    if missing_box_games:
        fallback = load_nba_player_game_box(
            nba_player_game_box_path, game_ids=missing_box_games
        )
        box = pd.concat([box, fallback], ignore_index=True)
    fouls = load_player_game_fouls_drawn(
        gabriel_pbp_root, season=season, game_ids=game_ids
    )
    events = box.merge(
        fouls, on=["game_id", "PLAYER_ID"], how="outer", validate="one_to_one"
    )
    ledger = exposure.merge(
        events,
        on=["game_id", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_event_rows = int(ledger["_merge"].eq("left_only").sum())
    ledger = ledger.drop(columns="_merge")
    ledger[list(BOX_COUNTS)] = ledger[list(BOX_COUNTS)].fillna(0.0)
    if ledger[list(BOX_COUNTS)].lt(0).any().any():
        raise ValueError("Player-game box counts cannot be negative.")
    if ledger.duplicated(["game_id", "PLAYER_ID"]).any():
        raise AssertionError("Player-game Box15 ledger keys must be unique.")
    event_keys = events[["game_id", "PLAYER_ID"]]
    unmatched = event_keys.merge(
        exposure[["game_id", "PLAYER_ID"]],
        on=["game_id", "PLAYER_ID"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched.loc[unmatched["_merge"].eq("left_only")]
    quality = {
        "games": len(game_ids),
        "player_game_rows": len(ledger),
        "exposure_rows_without_box_events": missing_event_rows,
        "event_rows_outside_validated_lineups": len(unmatched),
        "player_game_source_join_coverage": float(1.0 - missing_event_rows / len(ledger)),
        "nba_fallback_games": len(missing_box_games),
    }
    return (
        ledger.sort_values(["game_id", "PLAYER_ID"]).reset_index(drop=True),
        quality,
    )


def aggregate_box15_features(
    ledger: pd.DataFrame,
    *,
    season: int,
    excluded_game_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate a player-game ledger after removing complete held-out games."""
    excluded = {str(value) for value in excluded_game_ids}
    work = ledger.loc[~ledger["game_id"].isin(excluded)].copy()
    if work.empty:
        raise ValueError("The training ledger cannot be empty.")
    sums = work.groupby("PLAYER_ID", as_index=False, sort=False)[
        ["OffPoss", "DefPoss", *BOX_COUNTS]
    ].sum()
    output: dict[str, pd.Series | int] = {
        "PLAYER_ID": sums["PLAYER_ID"].astype("int64"),
        "Window_End": int(season),
        "OffPoss": sums["OffPoss"],
        "DefPoss": sums["DefPoss"],
    }
    for feature in BOX_PIPM_STYLE_FEATURES:
        numerator, denominator = CORE_RATE_SPECS[feature]
        output[feature] = 100.0 * sums[numerator] / sums[denominator].replace(0, np.nan)
    frame = pd.DataFrame(output)
    keep = frame["OffPoss"].gt(0) | frame["DefPoss"].gt(0)
    return frame.loc[keep].sort_values("PLAYER_ID").reset_index(drop=True)


def audit_annual_source_parity(
    *,
    espn_player_box_path: str | Path,
    gabriel_pbp_root: str | Path,
    player_sheet_path: str | Path,
    season: int,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """Compare the game-level reconstruction with the frozen annual box source."""
    raw = pd.read_parquet(
        espn_player_box_path, columns=["season", "game_id"]
    )
    raw = raw.loc[pd.to_numeric(raw["season"], errors="coerce").eq(season)].copy()
    raw["game_id"] = raw["game_id"].map(_game_id)
    regular_games = tuple(sorted(raw.loc[raw["game_id"].str.startswith("002"), "game_id"].unique()))
    if not regular_games:
        raise ValueError(f"ESPN player box has no regular-season games for {season}.")
    box = load_espn_player_game_box(
        espn_player_box_path, season=season, game_ids=regular_games
    )
    fouls = load_player_game_fouls_drawn(
        gabriel_pbp_root, season=season, game_ids=regular_games
    )
    reconstructed = box.merge(
        fouls, on=["game_id", "PLAYER_ID"], how="outer", validate="one_to_one"
    )
    reconstructed[list(BOX_COUNTS)] = reconstructed[list(BOX_COUNTS)].fillna(0.0)
    reconstructed = reconstructed.groupby("PLAYER_ID", as_index=False)[
        list(BOX_COUNTS)
    ].sum()

    sheet = pd.read_parquet(player_sheet_path)
    required = {"PLAYER_ID", *BOX_COUNTS}
    if missing := sorted(required - set(sheet.columns)):
        raise ValueError(f"Annual player sheet is missing {missing}.")
    sheet = sheet[["PLAYER_ID", *BOX_COUNTS]].copy()
    _numeric(sheet, BOX_COUNTS)
    sheet["PLAYER_ID"] = pd.to_numeric(sheet["PLAYER_ID"], errors="raise").astype(
        "int64"
    )
    if sheet["PLAYER_ID"].duplicated().any():
        raise ValueError("Annual player sheet must be unique by PLAYER_ID.")
    joined = sheet.merge(
        reconstructed,
        on="PLAYER_ID",
        how="outer",
        suffixes=("_annual", "_game"),
        indicator=True,
    )
    matched = joined["_merge"].eq("both")
    rows = []
    for field in BOX_COUNTS:
        annual = joined[f"{field}_annual"].fillna(0.0).astype(float)
        game = joined[f"{field}_game"].fillna(0.0).astype(float)
        annual_total = float(annual.sum())
        game_total = float(game.sum())
        larger = max(abs(annual_total), abs(game_total), 1.0)
        agreement = 1.0 - abs(annual_total - game_total) / larger
        absolute_error_fraction = float(
            np.abs(annual - game).sum() / max(np.abs(annual).sum(), 1.0)
        )
        correlation = float(
            annual.loc[matched].corr(game.loc[matched])
            if matched.sum() >= 3
            else np.nan
        )
        rows.append(
            {
                "feature_numerator": field,
                "annual_total": annual_total,
                "game_level_total": game_total,
                "total_agreement": agreement,
                "absolute_error_fraction": absolute_error_fraction,
                "player_correlation": correlation,
                "passed": agreement >= 0.99
                and absolute_error_fraction <= 0.01,
            }
        )
    parity = pd.DataFrame(rows)
    join_coverage = float(matched.sum() / len(joined))
    quality = {
        "season": season,
        "regular_games": len(regular_games),
        "annual_players": len(sheet),
        "game_level_players": len(reconstructed),
        "matched_players": int(matched.sum()),
        "player_join_coverage": join_coverage,
        "minimum_total_agreement": float(parity["total_agreement"].min()),
        "maximum_absolute_error_fraction": float(
            parity["absolute_error_fraction"].max()
        ),
        "passed": bool(
            join_coverage >= 0.99 and parity["passed"].all()
        ),
    }
    return parity, quality
