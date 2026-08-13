from __future__ import annotations

import pandas as pd

from nba_impact.data.identity import build_identity_dimensions, normalize_player_name


def test_normalize_player_name_is_only_a_name_audit_key() -> None:
    assert normalize_player_name("J. R. O'Neal") == "jroneal"


def test_build_identity_dimensions_preserves_aliases_and_observed_trade_stints(tmp_path) -> None:
    games = pd.DataFrame(
        {
            "game_id": ["0022500001", "0022500002"],
            "game_date": pd.to_datetime(["2025-10-21", "2025-11-01"]),
            "season_start": [2025, 2025],
            "season_end": [2026, 2026],
            "home_team_id": [10, 20],
            "home_team_tricode": ["HOM", "AWY"],
            "away_team_id": [20, 10],
            "away_team_tricode": ["AWY", "HOM"],
        }
    )
    player_games = pd.DataFrame(
        {
            "game_id": ["0022500001", "0022500001", "0022500002", "0022500002"],
            "game_date": pd.to_datetime(["2025-10-21", "2025-10-21", "2025-11-01", "2025-11-01"]),
            "season_start": [2025] * 4,
            "season_end": [2026] * 4,
            "season_type": ["regular"] * 4,
            "team_id": [10, 20, 20, 10],
            "team_tricode": ["HOM", "AWY", "AWY", "HOM"],
            "player_id": [1, 2, 1, 2],
            "player_name": ["Alpha Guard", "Beta Wing", "Alpha Guard", "Beta Wing"],
            "first_name": ["Alpha", "Beta", "Alpha", "Beta"],
            "family_name": ["Guard", "Wing", "Guard", "Wing"],
        }
    )
    events = pd.DataFrame(
        {
            "personId": [1, 1, 2],
            "playerName": ["A. Guard", "Alpha Guard", "Beta Wing"],
            "game_id": ["0022500001", "0022500002", "0022500001"],
            "game_date": pd.to_datetime(["2025-10-21", "2025-11-01", "2025-10-21"]),
        }
    )
    game_path = tmp_path / "game_dim.parquet"
    player_path = tmp_path / "player_games.parquet"
    event_path = tmp_path / "event_states.parquet"
    output_dir = tmp_path / "silver"
    games.to_parquet(game_path, index=False)
    player_games.to_parquet(player_path, index=False)
    events.to_parquet(event_path, index=False)

    snapshot = build_identity_dimensions(
        game_path, player_path, output_dir, tmp_path / "manifests", event_states_path=event_path
    )

    assert snapshot["passed"]
    teams = pd.read_parquet(output_dir / "team_dim.parquet")
    players = pd.read_parquet(output_dir / "player_dim.parquet")
    aliases = pd.read_parquet(output_dir / "player_aliases.parquet")
    stints = pd.read_parquet(output_dir / "observed_player_team_stints.parquet")
    assert teams["team_id"].tolist() == [10, 20]
    assert players.set_index("player_id").loc[1, "canonical_player_name"] == "Alpha Guard"
    assert set(aliases.loc[aliases["player_id"].eq(1), "player_name"]) == {"A. Guard", "Alpha Guard"}
    assert aliases["requires_player_id_for_join"].all()
    assert stints.loc[stints["player_id"].eq(1), "team_id"].tolist() == [10, 20]
    assert snapshot["issues"]["stint_games_not_reconciled"] == 0
