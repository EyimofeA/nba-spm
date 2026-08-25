# Shot-level defender-source audit

**Date:** 2026-08-19
**Decision:** no permissible shot-level primary-defender source has been added.

## What we checked

| Source | Has shot outcome/location? | Has offensive–defensive pairing? | Same row is a shot? | Decision |
|---|---:|---:|---:|---|
| Local NBA V3 + official shot detail + CDN | Yes | No | Yes | Use for player-neutral expected shot quality only |
| NBA `BoxScoreMatchupsV3` | No | Yes | No; it is game/player-pair aggregate | Keep as exploratory matchup exposure/outcome data |
| Public NBA tracking dashboards | Aggregate splits only | Sometimes closest-defender aggregates | No | Useful feature source, not a shot-to-defender join |
| Second Spectrum optical tracking | Yes in principle | Yes in principle | Yes in principle | Requires a rights-reviewed direct licence or research agreement |

The NBA matchup endpoint documents game-level pair fields such as matchup
minutes, partial possessions, FGA/FGM, threes, free throws, assists, blocks and
turnovers. It does **not** expose shot event ID, shot location, or a primary
defender at release. See the
[endpoint schema](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/boxscorematchupsv3.md).

The NBA identifies Second Spectrum as its optical tracking provider and says it
works with teams and league partners on custom tracking-data solutions; that is
the appropriate route for a granular defender-at-shot dataset. See the
[NBA's partnership announcement](https://pr.nba.com/nba-announces-multiyear-partnership-sportradar-second-spectrum/).

## Consequence

We cannot honestly calculate `expected shot quality versus defender` by joining
the current two sources. Allocation by matchup minutes, possession overlap, or
pair FGA would create a synthetic defender label and produce circular
suppression estimates.

The current implementation boundary is therefore:

1. Publish a **shooter-only** location-based expected-shot decomposition.
2. Keep pair matchup Elo exploratory and do not call defense Elo a shot-quality
   or contest rating.
3. Reopen defender-specific rim/non-rim models only after a permitted
   shot-level file has `(game_id, shot_event_id, shooter_id, defender_id)` plus
   a clear defensive-credit definition.

## Acceptance test for any future source

Before ingestion, require all of the following:

- one unique primary/credited defender field or an explicit multi-defender
  credit rule at each shot;
- event identity that joins to location and result with at least 99% coverage;
- method/coverage documentation, licence/provenance, and season scope;
- a reproducible alignment audit with no aggregation backfill;
- exact treatment of switches, help, contests, blocks, and team rebounds.
