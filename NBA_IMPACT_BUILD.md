# NBA Impact Lab — Active Build

This is the clean implementation path. Legacy analyses and outputs remain available
for reference, but they are not imported as evidence or production dependencies.

## Current vertical slice

```text
source-native Parquet ──> immutable bronze + checksum sidecars
                              │
                              └──> source contracts + cross-source game reconciliation

legacy 17-column possessions ──> structural quarantine ──> independent zero-prior RAPM
                                                               │
                                                               └──> immutable run + DuckDB registry
```

Owned paths:

- `src/nba_impact/`: clean package and CLI
- `configs/ingest/`: declarative download manifests
- `data/lake/bronze/`: immutable source-native downloads
- `data/lake/manifests/`: content-addressed quality snapshots
- `artifacts/models/`: immutable model and comparison runs
- `artifacts/registry/nba_impact.duckdb`: dataset/model registry

## Reproduce

```bash
python3 -m pip install --user -e .
nba-impact ingest --manifest configs/ingest/nba_data_archive_bootstrap.json
nba-impact audit-events
nba-impact audit-possessions --seasons 2022,2023,2024,2025
nba-impact fit-rapm --seasons 2021,2022,2023,2024 \
  --snapshot-id legacy_possessions_58ee15becffc55e1
nba-impact compare-rapm --seasons 2021,2022,2023,2024 \
  --snapshot-id legacy_possessions_58ee15becffc55e1
python3 -m pytest -q
```

Quality commands deliberately return a nonzero status when a partition is unsafe.
The 2025 legacy possession partition is empty, and the downloaded 2024 matchup-detail
partition is missing one game, so those snapshots currently fail by design.

## Modeling labels

- `research_baseline_unverified`: a reproducible estimator, not a trusted ranking.
- `research_diagnostic_unverified`: a comparison used to find problems or formulate
  hypotheses; it cannot select a champion from one holdout.
- `lineup_conditioned_retrodiction`: ratings applied to observed future lineups. It
  is not a pregame forecast.

The RAPM convention is points per 100 possessions. Positive offense is better;
positive defense means points prevented. Both blocks are centered on the
possession-weighted average player, with an exactly compensating intercept shift.

## Next build order

1. Convert V3 events into canonical games, score states, stints, and possessions;
   reconcile every game to final scores and team/player minutes.
2. Add expanding chronological RAPM folds, decay, garbage-time weighting as a
   training-only variant, and game-cluster uncertainty.
3. Build a state-only calibrated win-probability baseline, then Net Points-style
   value conservation and WP-RAPM as separate estimands.
4. Build time-safe box/tracking/role priors and compare AIO families only after the
   RAPM target and folds pass.
5. Serve versioned artifacts through an API; build the website after metric contracts
   and data freshness indicators are stable.

