# Historical player-game source audit (2017–2023)

Status: **strict separate candidate rebuilt 2026-08-18; no historical production table written.**

## Question

Can the pinned local sources make a verified `player_games` table with one row
per player-game, official game and team IDs, starter status, and played
minutes for project seasons 2017 through 2023?

The current answer is **yes for every official game in project seasons
2018--2023** after a 3,260-game official BoxScoreTraditionalV3 repair cache.
Project season 2017 remains blocked: its 1,309 cached official boxes do not
provide a valid five-starter state under the unchanged contract. The separate
candidate is still research-only and is not the canonical current player-game
table. Do not weaken the current gates to make 2017 pass.

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

## Current local coverage and quality checks

For every ESPN-covered official game, the audit checked two team sides, ten
starters, five starters per side, and the sum of player minutes against five
players for regulation plus the observed overtime periods. Minute tolerance is
five seconds, matching the existing builder.

| Project season | Official games | Strict accepted | Decision |
| --- | ---: | ---: | --- |
| 2017 | 1,309 | 0 | Cached official rows fail the five-starter gate; blocked. |
| 2018 | 1,312 | 1,312 | **Complete candidate.** |
| 2019 | 1,312 | 1,312 | **Complete candidate.** |
| 2020 | 1,142 | 1,142 | **Complete candidate.** |
| 2021 | 1,165 | 1,165 | **Complete candidate.** |
| 2022 | 1,317 | 1,317 | **Complete candidate.** |
| 2023 | 1,314 | 1,314 | **Complete candidate.** |

The rebuilt table has 168,428 player-game rows across 7,562 accepted games.
Every accepted team-game has five starters and satisfies the five-second team
minute gate. All 1,309 rejected games are from project season 2017 and fail both
home and away starter-count checks.

Sample identity check: game `0021800512` is Boston at Houston in the official
score table (Boston `1610612738`, Houston `1610612745`). The ESPN rows mark
Boston `home = 0` and Houston `home = 1`, so its side mapping gives the same
team IDs. All 1,312 project-2019 official game IDs have this exact ESPN game
coverage.

## Remaining safe path

1. Finish the resumable `--all-games` official cache so accepted games no longer
   depend on the unlicensed ESPN fallback.
2. Rebuild the same separate table and require identical 2018--2023 coverage.
3. Keep 2017 blocked until a source supplies trustworthy starters; do not infer
   them from minutes or relax the gate.
4. Do not overwrite canonical current tables or redistribute raw official or
   ESPN rows in the public release bundle.

The existing `build_player_games` function implements the relevant ESPN fallback
join, including home/away team-ID mapping. The separate historical candidate
uses the official score table plus the scoring-event panel for game identity,
team tricodes, and maximum period. It remains outside the canonical current
table and outside public release bundles.
