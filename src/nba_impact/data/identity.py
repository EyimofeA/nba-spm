"""Build canonical player, team, alias, and observed-stint dimensions.

These dimensions are intentionally conservative.  They record identities that
are observed in the silver game and player-game layers; they do not claim a
complete historical roster or contract record.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .manifest import sha256_file, write_json_atomic


IDENTITY_SCHEMA_VERSION = "identity_dimensions_v1"
_NAME_TOKEN = re.compile(r"[^a-z0-9]+")


def normalize_player_name(value: object) -> str:
    """Create an audit key only; it is never a cross-game player identifier."""
    return _NAME_TOKEN.sub("", str(value).casefold())


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _source_record(label: str, path: Path) -> dict[str, object]:
    return {"label": label, "file_name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _canonical_players(player_games: pd.DataFrame, event_states: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = player_games.loc[
        player_games["player_name"].notna() & player_games["player_name"].astype(str).str.strip().ne(""),
        ["player_id", "player_name", "game_date", "game_id", "first_name", "family_name"],
    ].copy()
    names["alias_source"] = "player_games"
    if event_states is not None:
        event_names = event_states.loc[
            event_states["personId"].notna()
            & event_states["personId"].gt(0)
            & event_states["playerName"].notna()
            & event_states["playerName"].astype(str).str.strip().ne(""),
            ["personId", "playerName", "game_date", "game_id"],
        ].rename(columns={"personId": "player_id", "playerName": "player_name"})
        event_names["first_name"] = pd.NA
        event_names["family_name"] = pd.NA
        event_names["alias_source"] = "event_states"
        names = pd.concat([names, event_names[names.columns]], ignore_index=True)
    names["player_id"] = pd.to_numeric(names["player_id"], errors="raise").astype(int)
    names["game_date"] = pd.to_datetime(names["game_date"], errors="coerce")
    names["player_name"] = names["player_name"].astype(str).str.strip()
    names["normalized_alias"] = names["player_name"].map(normalize_player_name)
    names = names.loc[names["normalized_alias"].ne("")].copy()
    aliases = (
        names.groupby(["player_id", "player_name", "normalized_alias", "alias_source"], as_index=False)
        .agg(
            first_seen_game_date=("game_date", "min"),
            last_seen_game_date=("game_date", "max"),
            source_rows=("game_id", "size"),
            observed_games=("game_id", "nunique"),
        )
    )
    alias_counts = aliases.groupby("normalized_alias")["player_id"].nunique()
    aliases["player_count_for_normalized_alias"] = aliases["normalized_alias"].map(alias_counts).astype(int)
    aliases["is_ambiguous_normalized_alias"] = aliases["player_count_for_normalized_alias"].gt(1)
    aliases["alias_granularity"] = "full_name"
    aliases.loc[
        aliases["alias_source"].eq("event_states")
        & ~aliases["player_name"].str.contains(r"\s", regex=True),
        "alias_granularity",
    ] = "surname_or_single_token"
    aliases["requires_player_id_for_join"] = True

    # Player-game names are authoritative for this layer.  Event-only names are
    # retained as aliases but cannot silently replace a box-score name.
    priority = {"player_games": 0, "event_states": 1}
    canonical = names.assign(_priority=names["alias_source"].map(priority)).sort_values(
        ["player_id", "_priority", "game_date", "game_id", "player_name"],
        ascending=[True, True, False, False, True],
        kind="stable",
    ).drop_duplicates("player_id", keep="first")
    availability = player_games.groupby("player_id", as_index=False).agg(
        first_observed_game_date=("game_date", "min"),
        last_observed_game_date=("game_date", "max"),
        first_observed_season=("season_start", "min"),
        last_observed_season=("season_end", "max"),
        observed_games=("game_id", "nunique"),
    )
    players = canonical.merge(availability, on="player_id", how="outer", validate="one_to_one")
    players["canonical_name_source"] = players["alias_source"]
    players = players.rename(columns={"player_name": "canonical_player_name"})
    columns = [
        "player_id", "canonical_player_name", "first_name", "family_name",
        "canonical_name_source", "first_observed_game_date", "last_observed_game_date",
        "first_observed_season", "last_observed_season", "observed_games",
    ]
    return players[columns].sort_values("player_id", kind="stable"), aliases.sort_values(
        ["player_id", "alias_source", "player_name"], kind="stable"
    )


def _team_dimension(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["game_id", "game_date", "season_start", "season_end", "home_team_id", "home_team_tricode"]].rename(
        columns={"home_team_id": "team_id", "home_team_tricode": "team_tricode"}
    )
    home["game_side"] = "home"
    away = games[["game_id", "game_date", "season_start", "season_end", "away_team_id", "away_team_tricode"]].rename(
        columns={"away_team_id": "team_id", "away_team_tricode": "team_tricode"}
    )
    away["game_side"] = "away"
    observed = pd.concat([home, away], ignore_index=True)
    observed["team_id"] = pd.to_numeric(observed["team_id"], errors="raise").astype(int)
    observed["team_tricode"] = observed["team_tricode"].astype(str).str.strip()
    canonical = observed.sort_values(["team_id", "game_date", "game_id"], ascending=[True, False, False], kind="stable").drop_duplicates("team_id")
    aggregates = observed.groupby("team_id", as_index=False).agg(
        first_observed_game_date=("game_date", "min"),
        last_observed_game_date=("game_date", "max"),
        first_observed_season=("season_start", "min"),
        last_observed_season=("season_end", "max"),
        observed_games=("game_id", "nunique"),
        tricode_count=("team_tricode", "nunique"),
    )
    aliases = observed.groupby("team_id")["team_tricode"].agg(lambda values: "|".join(sorted(set(values)))).rename("tricode_aliases")
    teams = canonical[["team_id", "team_tricode"]].rename(columns={"team_tricode": "canonical_tricode"}).merge(
        aggregates, on="team_id", validate="one_to_one"
    )
    teams["tricode_aliases"] = teams["team_id"].map(aliases)
    return teams.sort_values("team_id", kind="stable")


def _observed_player_team_stints(player_games: pd.DataFrame) -> pd.DataFrame:
    rows = player_games[[
        "player_id", "team_id", "team_tricode", "game_id", "game_date", "season_start", "season_end", "season_type"
    ]].copy()
    rows["player_id"] = pd.to_numeric(rows["player_id"], errors="raise").astype(int)
    rows["team_id"] = pd.to_numeric(rows["team_id"], errors="raise").astype(int)
    if rows.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Observed player-team stints require unique player-game keys.")
    rows = rows.sort_values(["player_id", "game_date", "game_id", "team_id"], kind="stable")
    rows["_new_stint"] = rows["team_id"].ne(rows.groupby("player_id")["team_id"].shift())
    rows["stint_number"] = rows.groupby("player_id")["_new_stint"].cumsum().astype(int)
    stints = rows.groupby(["player_id", "stint_number", "team_id", "team_tricode"], as_index=False).agg(
        start_game_id=("game_id", "first"),
        end_game_id=("game_id", "last"),
        start_game_date=("game_date", "min"),
        end_game_date=("game_date", "max"),
        start_season=("season_start", "min"),
        end_season=("season_end", "max"),
        observed_games=("game_id", "nunique"),
        regular_games=("season_type", lambda values: int((values == "regular").sum())),
        playoff_games=("season_type", lambda values: int((values == "playoffs").sum())),
    )
    stints["stint_id"] = stints.apply(
        lambda row: f"{int(row.player_id)}_{int(row.stint_number):03d}", axis=1
    )
    return stints[[
        "stint_id", "player_id", "stint_number", "team_id", "team_tricode",
        "start_game_id", "end_game_id", "start_game_date", "end_game_date",
        "start_season", "end_season", "observed_games", "regular_games", "playoff_games",
    ]].sort_values(["player_id", "stint_number"], kind="stable")


def build_identity_dimensions(
    game_dim_path: str | Path,
    player_games_path: str | Path,
    output_dir: str | Path,
    manifest_dir: str | Path,
    *,
    event_states_path: str | Path | None = None,
) -> dict:
    """Create versioned dimensions from validated silver inputs."""
    game_source = Path(game_dim_path)
    player_source = Path(player_games_path)
    event_source = Path(event_states_path) if event_states_path is not None else None
    games = pd.read_parquet(game_source)
    player_games = pd.read_parquet(player_source)
    event_states = pd.read_parquet(event_source) if event_source is not None else None
    required_games = {"game_id", "game_date", "season_start", "season_end", "home_team_id", "home_team_tricode", "away_team_id", "away_team_tricode"}
    required_players = {"game_id", "game_date", "season_start", "season_end", "season_type", "team_id", "team_tricode", "player_id", "player_name", "first_name", "family_name"}
    if missing := sorted(required_games - set(games.columns)):
        raise ValueError(f"game_dim lacks columns: {missing}")
    if missing := sorted(required_players - set(player_games.columns)):
        raise ValueError(f"player_games lacks columns: {missing}")
    if event_states is not None:
        required_events = {"personId", "playerName", "game_id", "game_date"}
        if missing := sorted(required_events - set(event_states.columns)):
            raise ValueError(f"event_states lacks columns: {missing}")

    teams = _team_dimension(games)
    players, aliases = _canonical_players(player_games, event_states)
    stints = _observed_player_team_stints(player_games)
    game_ids = set(games["game_id"].astype(str))
    team_pairs = set(zip(games["game_id"].astype(str), games["home_team_id"].astype(int))) | set(zip(games["game_id"].astype(str), games["away_team_id"].astype(int)))
    issues = {
        "duplicate_game_ids": int(games.duplicated("game_id").sum()),
        "duplicate_player_game_keys": int(player_games.duplicated(["game_id", "player_id"]).sum()),
        "player_game_missing_game_dim": int((~player_games["game_id"].astype(str).isin(game_ids)).sum()),
        "player_game_team_not_in_game_dim": int(sum((str(row.game_id), int(row.team_id)) not in team_pairs for row in player_games[["game_id", "team_id"]].itertuples(index=False))),
        "players_without_canonical_name": int(players["canonical_player_name"].isna().sum()),
        "ambiguous_normalized_alias_rows": int(aliases["is_ambiguous_normalized_alias"].sum()),
        "duplicate_stint_ids": int(stints.duplicated("stint_id").sum()),
        "stint_games_not_reconciled": int(stints["observed_games"].sum() != len(player_games)),
    }
    critical = {"duplicate_game_ids", "duplicate_player_game_keys", "player_game_missing_game_dim", "player_game_team_not_in_game_dim", "players_without_canonical_name", "duplicate_stint_ids", "stint_games_not_reconciled"}
    passed = not any(issues[key] for key in critical)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "team_dim": output / "team_dim.parquet",
        "player_dim": output / "player_dim.parquet",
        "player_aliases": output / "player_aliases.parquet",
        "observed_player_team_stints": output / "observed_player_team_stints.parquet",
    }
    _write_parquet_atomic(teams, paths["team_dim"])
    _write_parquet_atomic(players, paths["player_dim"])
    _write_parquet_atomic(aliases, paths["player_aliases"])
    _write_parquet_atomic(stints, paths["observed_player_team_stints"])
    sources = [_source_record("game_dim", game_source), _source_record("player_games", player_source)]
    if event_source is not None:
        sources.append(_source_record("event_states", event_source))
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"identity_dimensions_{identity}",
        "dataset": "identity_dimensions",
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "grain": {
            "team_dim": "one NBA team ID observed in game_dim",
            "player_dim": "one NBA player ID observed in player_games or event_states",
            "player_aliases": "one source-qualified observed player name alias",
            "observed_player_team_stints": "one contiguous team assignment in observed player-game order",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "sources": sources,
        "row_counts": {label: int(len(frame)) for label, frame in {"team_dim": teams, "player_dim": players, "player_aliases": aliases, "observed_player_team_stints": stints}.items()},
        "issues": issues,
        "scope_note": "Stints are observed player-game assignments, not a complete roster, contract, or inactive-availability history.",
        "artifacts": {label: path.name for label, path in paths.items()},
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
