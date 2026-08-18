# Historical player-game source audit (2017–2023)

Status: **audited 2026-08-18; no historical production table written.**

## Question

Can the pinned local sources make a verified `player_games` table with one row
per player-game, official game and team IDs, starter status, and played
minutes for project seasons 2017 through 2023?

The answer is **yes for 2019 and 2021 only**. The resulting rows would be
research-only because their upstream ESPN mirror has no declared licence.
The remaining seasons need more official box-score coverage or individual game
repair. Do not weaken the current player-game gates to make them pass.

Project season means the season end year. For example, project season 2019 is
the 2018-19 NBA season. Official game counts include regular season and
playoffs, but not Play-In games.

## Sources inspected

| Source | Local material and pinned provenance | Required fields | Result |
| --- | --- | --- | --- |
| ESPN player boxes | `bronze/llimllib_nba_data/espn/player_box.parquet`; 213,970 rows, 9,021,318 bytes, SHA-256 `85be23af3026992d3a680f556e8bceb01ce94de5e15bed9df66e29a9aff06ead`; `llimllib/nba_data` revision `3519bb36e8f70a8bb61bfbf4b6c37e1fbb0f9c2c`; licence `not_declared_research_only` | `game_id`, `player_id`, `team`, home/away flag, `starter`, `minutes_played`, `played`; `team_id` is null in 2022-23 | The only local source with native starter flags and player minutes. `home` can map to the official home or away team ID. Research-only; do not redistribute its rows. |
| NBA Stats player-game mirror | `bronze/llimllib_nba_data/player_game_logs.parquet`; 92,841 rows, 4,278,492 bytes; same revision and licence status | `gameId`, `teamId`, `personId`, position and minutes | Not a historical solution. It has sparse, incomplete 2017-23 game coverage and no explicit starter flag. The current code's `position != ""` rule is a fallback convention, not official starter evidence. |
| Official final scores | `bronze/official_game_scores/official_game_scores.parquet`; 12,812 verified games | `game_id`, date, home/away team IDs, final scores | Gives the authoritative game universe and a safe home/away-to-team-ID bridge. It does not contain player rows or starters. |
| Licensed matchup archive | `bronze/shufinskiy_nba_data/revision=e829d46/matchups`; `shufinskiy/nba_data` revision `e829d4678be1e075f99e5d41a1c5f97089be446b`, Apache-2.0 | game, home/away, team, player and opponent IDs, matchup minutes | Useful for identity and matchup checks. It has no total player minutes or starter flag; summing matchup minutes over opponents would overcount. It cannot build `player_games`. |
| Scoring event panel | `silver/scoring_events_2017_2026`; validated 2017-26 final-score event panel | game ID, period, score state, event team/player IDs | It supplies the game period needed to validate minute totals. It is not a player box score or lineup source. |

The ESPN schema also contains `team_id`, but it is null for every inspected
2022 and 2023 row. This does not prevent a deterministic ID assignment: map
`home = 1` to `official_game_scores.home_team_id`, and `home = 0` to
`away_team_id`, after an exact game-ID join.

## Local coverage and quality checks

For every ESPN-covered official game, the audit checked two team sides, ten
starters, five starters per side, and the sum of player minutes against five
players for regulation plus the observed overtime periods. Minute tolerance is
five seconds, matching the existing builder.

| Project season | Official games | ESPN-covered games | Games with two teams / ten starters | Team sides with 5 starters | Team sides within 5 seconds | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2017 | 1,309 | 0 | — | — | — | No local player-box source. |
| 2018 | 1,312 | 0 | — | — | — | No local player-box source. |
| 2019 | 1,312 | 1,312 | 1,312 / 1,312 | 2,624 / 2,624 | 2,624 / 2,624 (max error 3 s) | **Complete candidate available.** |
| 2020 | 1,142 | 971 | 971 / 971 | 1,942 / 1,942 | 1,942 / 1,942 (max error 5 s) | Incomplete: 171 games absent. |
| 2021 | 1,165 | 1,165 | 1,165 / 1,165 | 2,330 / 2,330 | 2,330 / 2,330 (max error 3 s) | **Complete candidate available.** |
| 2022 | 1,317 | 1,317 | 1,317 / 1,313 | 2,630 / 2,634 | 2,585 / 2,634 (max error 241 s) | Quarantine 4 starter-invalid games and 49 minute-invalid team sides. |
| 2023 | 1,314 | 1,314 | 1,314 / 1,313 | 2,627 / 2,628 | 2,581 / 2,628 (max error 154 s) | Quarantine 1 starter-invalid game and 47 minute-invalid team sides. |

Sample identity check: game `0021800512` is Boston at Houston in the official
score table (Boston `1610612738`, Houston `1610612745`). The ESPN rows mark
Boston `home = 0` and Houston `home = 1`, so its side mapping gives the same
team IDs. All 1,312 project-2019 official game IDs have this exact ESPN game
coverage.

## Smallest safe path

1. Build **separate, research-only candidate** `player_games` outputs for
   **2019 and 2021**. Join ESPN rows to the official game table by exact
   `game_id`; derive team IDs from the ESPN home flag; use ESPN starter and
   minute fields; validate the same existing gates and keep a source manifest.
2. Do not overwrite the canonical 2023-26 table and do not add these
   unlicensed-derived rows to a public release bundle or API.
3. Retain 2022-23 malformed games in quarantine. Repair them only with pinned
   official `BoxScoreTraditionalV3` JSON. Do not expand the minute tolerance.
4. For 2017-18 and the 171 missing 2020 games, acquire a separately pinned,
   rights-reviewed official box-score source before construction. The event and
   matchup sources cannot substitute for starters and total minutes.

The existing `build_player_games` function already implements the relevant
ESPN fallback join, including home/away team-ID mapping. A historical candidate
also needs a complete historical `game_dim` carrying team tricodes and maximum
period. The official score table plus the scoring-event panel provide the
necessary pieces, but no historical candidate is written by this audit.
