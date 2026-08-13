# Current event-source coverage

This is the checked local coverage contract for the raw `nba_data_archive`
snapshot. It prevents a downstream model from treating every source as equally
complete or silently selecting a convenient but incomplete one.

## Source selection

Use **NBA Stats V3** as the primary raw event source for seasons 2023–25. It
has all 1,230 regular-season games and all locally expected playoff games in
each season. CDN NBA is a useful possession-control reference where available,
but it is not a complete 2025 playoff source. PBPStats is an independent 2023–
24 validation source, not a 2025 fallback.

| Source | 2023 | 2024 | 2025 | Contract |
|---|---:|---:|---:|---|
| NBA Stats V3 event rows | 598,705 regular + 38,307 playoff; 1,230 + 82 games | 606,538 + 41,772; 1,230 + 84 | 621,887 + 43,289; 1,230 + 85 | Primary event source. |
| CDN NBA event rows | 674,937 + 43,452; 1,230 + 82 | 686,008 + 47,569; 1,230 + 84 | 708,268 + 34,579; 1,230 + 60 | Possession-tag reference only; 25 2025 playoff games are absent. |
| PBPStats possession rows | 478,625 + 30,532 | 483,126 + 32,939 | absent | Independent possession validator for 2023–24 only. |
| Shot detail | 218,701 + 13,840; 1,230 + 82 | 219,527 regular only; 1,230 games | 219,160 + 14,473; 1,230 + 85 | Shot-context source; 2024 playoffs are absent. |
| Matchups | 230,703 + 13,600; 1,227 + 82 | 231,961 regular only; 1,229 games | 240,839 + 15,909; 1,230 + 85 | Individual-matchup research input; do not assume regular-season completeness before QA. |

The season label is the start year: `2025` is the 2025–26 season. Counts were
read from each file's verified local manifest and distinct game IDs on
2026-08-13.

## Non-negotiable guards

- A current RAPM, possession, or win-probability build must use V3 for the
  2025 playoff tail; it must never substitute absent CDN games with zero rows.
- CDN event ordering must use `orderNumber`, not editable `actionNumber`.
- A 2024 playoff shot-detail or matchup analysis is invalid until its missing
  source slice is acquired and verified; do not make it look complete by
  joining regular-season rows.
- Every bronze file has a sidecar manifest with source revision, SHA-256, row
  count, validation columns, and retrieval time. New files enter only through
  a manifest-driven ingest.

This document states data availability, not source rights. Release bundles
remain derived-only unless the source licence permits raw redistribution.
