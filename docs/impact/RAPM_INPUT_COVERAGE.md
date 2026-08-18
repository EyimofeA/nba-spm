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
| 2017--2023 | 7,250 / 8,289 regular-season games in separate strict V3 research candidates | Possession ownership, action order, terminal ordinal lineups, official score, and player minutes pass. The completed 8,871-game official box-score cache is the starter/minute source. The independent strict legacy migration remains available for 3,343 regular-season games. The V3 candidates are not canonical or public. |
| 2024 regular | 1,229 / 1,230 | One game remains quarantined after Gabriel and V3 repairs. |
| 2024 playoffs | 82 / 82 | Passes. |
| 2025 regular | 1,227 / 1,230 | Three games remain quarantined after V3 repair. |
| 2025 playoffs | 84 / 84 | Passes. |
| 2026 regular | 1,228 / 1,230 | Two games are quarantined by lineup-minute QA. |
| 2026 playoffs | 85 / 85 | Official Live completion passes every possession and lineup gate. |

Do not silently fill these gaps with clock joins or inferred lineups.

The legacy `rapm/data/possession_cache/` has possession rows with ten player
columns for 2017--2024. `migrate-legacy-possessions` now makes a separate,
strict 2017--2023 terminal-lineup table from its 3,631 score-conserved games.
It is not a substitute for the canonical action-level source contract: it has
no within-possession substitution timing. The 2024 cache is incomplete and has
duplicate-player lineup rows, the 2025 cache is empty, and there is no 2026
cache. See `LEGACY_POSSESSION_MIGRATION.md`.

## Immediate, bounded repair

`configs/ingest/cdnnba_2026_playoffs_patch.json` pins the available 1.49 MB CDN
playoff partition. Direct inspection confirms that it has 60 of 85 games. It
does **not** contain the 25-game tail, so rerunning the standard builders cannot
restore those possessions.

`configs/ingest/current_lineup_v3_playoff_tail_2026.json` pins a licensed V3
file with all 85 games. V3 supplies event order and substitutions, but no native
possession owner. It is a lineup source only.

`configs/ingest/gabriel_2026_playoff_tail_lineups.json` pins four research-only
files that cover the 25-game tail. All 25 game IDs are present, event keys are
unique, maximum final scores match the official game table, and every
non-substitution row has exact five-versus-five lineups. The upstream repository
does not declare a license. These files remain quarantined and cannot create
canonical possessions without an independently validated possession source.

### Official Live completion

`nba-impact build-live-playoff-completion` downloads the official NBA Live JSON
with 20 retries and writes one schema-identical completed CDN partition. The
official source contains `orderNumber`, possession owner, scores, and
substitutions. It is not a possession inference.

The 60 games shared with the archived CDN source are identical on all 34,579
event keys and on action number, period, clock, possession owner, score, and
action type. The completed partition has 49,727 actions across 85 games with no
duplicate event keys, score mismatch, or invalid possession owner.

The normal lineup and possession builders then pass on all 85 games:

- 2,940 minute-reconciled lineup stints;
- 16,648 possessions;
- 20,639 ordinal lineup segments;
- zero invalid ten-player segments or point-conservation failures.

Raw NBA rows are internal research inputs and must not enter a public release
bundle.

Gabriel Adebayo's `merged_playbyplay` repository is a targeted fallback for
the nine regular-season quarantines and the remaining failed playoff game.
Its 2024 pilot had unique action numbers, a matching final score, and ten
unique players on every non-substitution event. Substitution rows have
transitional player lists by design.

`nba-impact build-gabriel-fallback-repairs` is a separate, strict adapter. It
uses canonical CDN action order and possession ownership, and observed Gabriel
on-court states only. It maps players to the canonical player-game team, keeps
only exact five-versus-five states, assigns non-outcome boundaries from an
adjacent observed state, and permits an unobserved outcome only when identical
observed lineups bracket it. The adapter rejects a candidate unless final
score, possession and segment keys, every ten-player segment, and official
player-minute totals all reconcile. Do not use the raw fallback directly as a
RAPM table.

### Targeted repair pilot, 2026-08-18

The ten downloaded fallback files were evaluated without changing the
canonical tables. One game passed all gates:

| Game | Result | Effect on RAPM-ready coverage |
|---|---|---|
| `0022300535` | Repaired | 2024 regular: 1,227 / 1,230 to **1,228 / 1,230** |
| `0022300339`, `0022400061`, `0022400821` | Blocked | An upstream substitution cannot be mapped to a direct observed post-substitution state. |
| `0022400771` | Blocked | An outcome-adjacent violation lacks two-sided lineup continuity. |
| `0022301210`, `0022400223`, `0022500264`, `0022500643` | Blocked | Official player-minute error is 136.0, 115.0, 78.4, and 48.0 seconds respectively; the 5-second gate is retained. |
| `0042500205` | Superseded | The official Live completion now supplies the canonical possession and substitution rows. |

All other audited 2024--26 partitions were unchanged. The adapter writes a
separate merged candidate (`possessions_repaired.parquet` and
`possession_lineup_segments_repaired.parquet`); it never overwrites the
canonical inputs.

### Pinned V3 substitution repair

`nba-impact build-v3-cdn-lineup-repair` aligns a pinned V3 substitution to a
CDN ordinal event only when game, period, clock, team, incoming player, and
substitution direction agree. CDN remains the source for event order, scores,
and possession ownership.

Two additional regular-season games pass every gate:

| Game | Project season | Maximum official-minute error |
|---|---:|---:|
| `0022301210` | 2024 | 0.4 seconds |
| `0022400061` | 2025 | 0.1 seconds |

Games `0022300339` and `0022500264` remain quarantined because one or more V3
substitutions cannot be mapped under the full ordinal key. The candidate keeps
separate outputs and does not weaken the production gate.

### Official Live retry of the six remaining regular games

The current official NBA Live feed was also tested directly for all six
remaining regular-season quarantines. It supplied 3,477 valid ordered actions,
but none of the six passed the unchanged lineup contract:

| Game | Transition errors | Maximum official-minute error |
|---|---:|---:|
| `0022300339` | 0 | 156.1 seconds |
| `0022400223` | 0 | 115.0 seconds |
| `0022400771` | 6 | 611.2 seconds |
| `0022400821` | 2 | 25.0 seconds |
| `0022500264` | 0 | 93.0 seconds |
| `0022500643` | 1 | 6.9 seconds |

Current Live events therefore do not repair these games by themselves. Keep
all six quarantined. The next acceptable repair needs a better observed lineup
source or a separately validated ordinal reconstruction; rerunning the same
clock-based substitution replay is not useful.

A one-game-per-season availability probe found nonempty official Live actions
for project seasons 2020--2023, while the sampled 2017--2019 games returned
HTTP 403. This is only an availability probe, not a season-completeness claim.
Do not start a large historical Live download until historical starter and
official player-minute inputs are pinned: play-by-play alone is not lineup-ready.

### Historical V3 possession candidate

The retired PlayByPlayV2 endpoint now returns empty data, so the local complete
V3 archive is the historical event source. The frozen V3 owner state machine
passes independent regular-season CDN validation in project seasons 2024 and 2025. Core
action-owner agreement is 99.93% in both seasons; the exact full owner sequence
matches in 93.6% and 91.9% of games.

The separate all-type 2017--2023 possession build accepts 8,863 games and
1,768,472 possession rows. Eight 2017 regular-season games fail exact team-
score conservation and remain rejected. The strict regular-season lineup
attachment then accepts 7,250 games, 1,448,146 possessions, and 1,756,230
ordinal lineup segments. Every accepted action maps once, each segment has ten
unique players, and official player minutes reconcile within five seconds.

The matched legacy comparison uses identical regular-season games, one
persisted chronological split, and official final margins. Net-rating Pearson
correlation is 0.971--0.981; it remains 0.969--0.982 after requiring 2,000
possessions per side in both sources. V3 reconstructs official margins more
accurately, but its RAPM has lower held-out prediction RMSE in only two of seven
seasons. Keep it as a validated research input, not exact ground truth or a
public source replacement. See `HISTORICAL_V3_POSSESSIONS.md`,
`HISTORICAL_V3_LINEUPS.md`, and `HISTORICAL_MATCHED_RAPM.md`.

Playoff transfer remains research-only. The 2025 playoff validation passes,
but the 2024 check misses the within-two-possession gate in 2 of 82 games even
with 99.98% core action-owner agreement. Historical playoff rows are excluded
from the first Normal RAPM fit.

## Integrated current candidate

`possessions_complete.parquet` combines the canonical base, the one strict
Gabriel repair, the two strict V3 repairs, and the completed 2026 playoff slice.
It has 793,122 possessions, 953,691 lineup segments, and 3,935 games. All IDs,
lineups, scores, and segment points reconcile.

Normal RAPM run `current_single_season_rapm_targets_v1_8f2a6f2e0a` uses only
regular-season rows from that table. It has 1,229 / 1,230 games in 2024,
1,227 / 1,230 in 2025, and 1,228 / 1,230 in 2026. Against the previous current
run, net-rating correlation is 0.999841 in 2024 and 0.999846 in 2025; mean
absolute change is 0.0057 and 0.0071 points per 100. The 2026 regular fit is
unchanged.

The change is too small to justify another SPM/AIO model-selection pass. Keep
the validated 2017--24 SPM/AIO and treat the newer SPM refresh as a null.

## Boundary

This page measures input availability, not model quality. The public 2017--24
SPM/AIO ratings remain frozen. RAPM can use the integrated current run for
2024--26. The 2017--23 V3 fits remain versioned research artifacts while the
immutable official player-game cache and official-preferred reproducibility
check finish.
