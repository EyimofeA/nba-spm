# Historical first-five starter candidate

## Decision

Use `official_box_first_five_v1` only for old NBA
`BoxScoreTraditionalV3` payloads that do not have exactly five non-empty
`position` markers for a team. The fallback selects the first five response
rows. It requires a complete consecutive row order, at least five rows, and
unique player IDs. It records its source on every player-game row.

Do not use this as a canonical player-game source. It is a separate historical
research candidate.

## Validation before use

The 2016--17 payloads are the target for project season 2017. They do not
encode starter status in `position`: all 2,618 team records have 6--13 marked
players. All have five unique first response rows.

The same first-five rule was checked on the later historical cache. At the
audit snapshot, NBA position markers exactly matched the first five rows for
all 6,522 team records from game-code years 2017--22. The 3,551 team records
with an independent local ESPN native-starter row also matched exactly. That
is 1,778 games and zero disagreements.

This is validation of a response-order contract, not direct 2016--17 starter
ground truth. The candidate remains research-only.

## 2017 candidate result

The separate build used pinned official scores, V3 game/team identities, and
the historical official box cache. It accepted all 1,309 2016--17 games:
1,230 regular-season and 79 playoff games. It emitted 33,610 player-game
rows. Each of the 2,618 teams had five selected starters and passed the
existing five-second minute-reconciliation gate.

Candidate output:

`data/lake/silver/candidates/historical_2017_first_five_candidate/`

The output contains `starter_inference_source` and the quality ledger contains
`official_starter_inference_sources`. A malformed source row order, duplicate
player ID, fewer than five players, or failed identity/minute check rejects the
game. There is no ESPN fallback after an official box is selected.
