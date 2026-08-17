# RAPM input coverage

## Contract

Normal RAPM requires one possession outcome and one ordinal ten-player lineup
state. A score-only event table does not meet this contract. A game fails when
the possession table or its lineup-segment child table is missing, has a
duplicate key, or has a lineup other than five players per team.

Run this check before any current RAPM fit:

```bash
nba-impact audit-rapm-inputs --seasons 2017,2018,2019,2020,2021,2022,2023,2024,2025,2026
```

The command returns nonzero until every requested regular-season and playoff
partition passes. It does not run a model.

## Current audit, 2026-08-18

| Project season | Canonical RAPM-ready games | Status |
|---|---:|---|
| 2017--2023 | 0 | The clean canonical table has not been built. A legacy possession cache exists, but it is a separate historical input contract. |
| 2024 regular | 1,227 / 1,230 | Three games are quarantined by lineup-minute QA. |
| 2024 playoffs | 82 / 82 | Passes. |
| 2025 regular | 1,226 / 1,230 | Four games are quarantined by lineup-minute QA. |
| 2025 playoffs | 84 / 84 | Passes. |
| 2026 regular | 1,228 / 1,230 | Two games are quarantined by lineup-minute QA. |
| 2026 playoffs | 60 / 85 | One game is quarantined and 24 games lack the CDN ordinal event partition. |

Do not silently fill these gaps with clock joins or inferred lineups.

The legacy `rapm/data/possession_cache/` has possession rows with ten player
columns for 2017--2024. It is the input used by the historical model work. It
is not a substitute for the canonical source contract: the 2024 cache is
incomplete and has duplicate-player lineup rows, the 2025 cache is empty, and
there is no 2026 cache.

## Immediate, bounded repair

`configs/ingest/cdnnba_2026_playoffs_patch.json` downloads one pinned 1.49 MB
CDN playoff partition. It can restore the source event coverage for the 24
2026 playoff games after the normal game, player-box, lineup, and possession
builders are rerun. It does not bypass their minute-reconciliation gate.

Gabriel Adebayo's `merged_playbyplay` repository is a targeted fallback for
the nine regular-season quarantines and the remaining failed playoff game.
Its 2024 pilot had unique action numbers, a matching final score, and ten
unique players on every non-substitution event. Substitution rows have
transitional player lists by design. A fallback builder must join the source
to canonical player-game team identities, retain the upstream action order,
and reject a game unless its final score, all outcome-action lineups, and
player-minute totals reconcile. Do not use the raw fallback directly as a
RAPM table.

## Boundary

This page measures input availability, not model quality. The public 2017--24
ratings remain frozen. No rating is updated until the repaired data passes the
same audit and a separately declared model run.
