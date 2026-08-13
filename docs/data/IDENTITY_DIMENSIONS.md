# Identity Dimensions

Build the current canonical identity layer with:

```bash
nba-impact build-identity-dimensions
```

It writes four silver tables and one content-hashed manifest.

| Table | Grain | Safe use |
|---|---|---|
| `team_dim` | NBA team ID observed in `game_dim` | Game, team, and season joins |
| `player_dim` | NBA player ID observed in `player_games` or `event_states` | Player identity joins |
| `player_aliases` | Source-qualified observed name for one player ID | Audit and display only |
| `observed_player_team_stints` | Contiguous team assignment in observed player-game order | Played-game team history |

`player_aliases` is not a name-to-player crosswalk. Event sources can provide a
surname only. Every row has `requires_player_id_for_join=true`; aliases must
never resolve a player without the source player ID, game, and team context.

`observed_player_team_stints` is not a roster, injury, transaction, salary, or
availability table. It only records the team in games where the player appears
in the canonical player-game source.
