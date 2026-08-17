# Verified scoring events, 2017–2026

## Result

The local scoring-event panel covers every regular-season and playoff game from
2016–17 through 2025–26. It is a points-only event layer for score-state and
scoring research. It is not a lineup, possession, or RAPM input.

| Check | Result |
|---|---:|
| Source partitions | 20 |
| Games | 12,812 |
| Raw event rows | 6,297,872 |
| Score-change rows | 1,496,326 |
| Score-correction rows | 955 |
| Missing reference games | 0 |
| Unexpected games | 0 |
| Final-score mismatches | 0 |

## Sources and contract

- Primary event source: `cdechoch/nba-data-archive`, NBA Stats V3 files, pinned
  to revision `dfa8fa43f89ae2ca6c18db524edc2050a6bb2286`.
- Manifest: `configs/ingest/nbastatsv3_scoring_2017_2026.json`.
- Raw files: `data/lake/bronze/nba_data_archive_scoring/`.
- Verified output: `data/lake/silver/scoring_events_2017_2026/`.
- Extraction version: `score_state_verified_with_fallback_v4`.

Events preserve source physical order with `actionId`. Do not sort by
`actionNumber`: it is an editable identifier and can backtrack. The silver
table retains the event identity, clock, actor, action, score, score delta, and
correction flag needed to audit every point.

Final scores were checked against 10,183 games downloaded from NBA
`LeagueGameLog` and 2,629 games from the separately validated canonical
`game_dim`. Five source files contained one or two stale trailing replay rows;
only those rows were excluded. Game `0021700025` had a corrupted primary score
state and was replaced from the pinned `datanba` archive using
`configs/ingest/datanba_scoring_fallbacks.json`.

## Reproduce

```bash
nba-impact download-data --manifest configs/ingest/nbastatsv3_scoring_2017_2026.json
nba-impact build-scoring-events \
  --manifest configs/ingest/nbastatsv3_scoring_2017_2026.json \
  --official-scores data/lake/bronze/official_game_scores/official_game_scores.parquet
```

The downloader and builder are resumable and hash-validated. The data files are
local artifacts and are intentionally not committed to Git.

## Boundary

This layer supports scoring timelines, final-score QA, and future expected-shot
or points-credit work. Normal RAPM still requires validated possessions and
ordinal lineup assignments. Rebounds, turnovers, substitutions, and other raw
actions exist in the bronze files, but this silver contract deliberately does
not claim that they have been canonicalized or verified.
