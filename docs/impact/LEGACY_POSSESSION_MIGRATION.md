# Legacy possession migration

## Result

The 2017--2023 legacy RAPM cache now has a strict, loader-compatible migration
path. It is separate from the current canonical event lake.

The migration emits only a game when these local checks pass:

- one official home/away game identity exists;
- the cache date equals the official date;
- the cache home and away point sums equal the official final score;
- every legacy `(game, period, event number)` key is unique;
- every row has a nonnegative integer point result, a valid offense side, and
  ten distinct player IDs;
- the game begins in period 1 and reaches period 4.

The local source pair is the immutable legacy cache and the verified local
official-game-score table. The official table supplies the game date, home and
away team IDs, and final score. The cache supplies the home/away lineup sides
and the terminal lineup stored for each row.

## Full backfill, 2026-08-18

| Season end | Accepted games | Accepted rows |
|---|---:|---:|
| 2017 | 514 | 99,437 |
| 2018 | 552 | 107,388 |
| 2019 | 524 | 104,836 |
| 2020 | 452 | 90,341 |
| 2021 | 511 | 101,224 |
| 2022 | 553 | 108,299 |
| 2023 | 525 | 104,197 |
| **Total** | **3,631** | **715,722** |

Of those games, 3,343 are regular season and 288 are playoffs. All 715,722
accepted cache rows produce exactly one possession and one lineup segment. The
output has no duplicate IDs, no invalid ten-player segment, no lost source row,
no score-conservation failure, and no possession-to-segment point mismatch.

The strict migration rejects 5,239 games whose cache point sums do not equal
the verified final score. It also rejects 19 cache games with no official game
identity and 13 games that do not cover four periods. Smaller isolated failures
include one invalid lineup, one duplicate period/event key, and one missing
cache game. These are data exclusions, not repairs.

## Lineup boundary

The old cache does not preserve substitution timing inside a possession. The
migration therefore writes one `legacy_terminal_lineup` segment per source row.
It does not estimate start lineups, stint duration, clock timing, or fractional
exposure. The synthetic order field preserves `(period, event number)` order;
`legacy_event_num` retains the original event number.

Use terminal assignment only with these tables. Do not merge them into
`possessions.parquet` or claim that they have the current CDN action-level
lineup contract.

## Reproduce

```bash
uv run python -m nba_impact.cli migrate-legacy-possessions \
  --seasons 2017,2018,2019,2020,2021,2022,2023
```

This writes separate `legacy_possessions.parquet`,
`legacy_possession_lineup_segments.parquet`, `legacy_game_identity.parquet`,
and `legacy_possession_migration_quality.parquet` files under `data/lake/silver/`.
The report records source hashes, accepted coverage, and every blocking issue.

The result is suitable for historical terminal-lineup RAPM research. It is not
a replacement for a complete historical action-level possession build.
