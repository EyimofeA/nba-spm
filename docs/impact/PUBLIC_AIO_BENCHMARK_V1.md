# Public all-in-one benchmark v1

## Result

This is a matched four-season comparison, not a model-promotion test. MAMBA has
the highest mean next-season team-win R² (`0.674`). xRAPM follows (`0.644`). The
old and new CourtSignal AIO models are effectively tied (`0.641` and `0.640`).
Their player-season net ratings correlate at `0.999`, so the selected feature
additions have not materially changed the final AIO posterior.

Run: `public_aio_benchmark_v1_67a99b5e1e`.

| Metric | Mean R² | Mean Pearson | Mean rank correlation | Minimum minute coverage |
|---|---:|---:|---:|---:|
| MAMBA | 0.674 | 0.820 | 0.782 | 84.1% |
| xRAPM | 0.644 | 0.802 | 0.762 | 84.5% |
| Old AIO | 0.641 | 0.799 | 0.769 | 84.5% |
| New AIO | 0.640 | 0.799 | 0.768 | 84.5% |
| BoxPIPM-style | 0.607 | 0.777 | 0.716 | 84.5% |
| BPM 2.0 | 0.604 | 0.774 | 0.738 | 84.5% |
| LEBRON | 0.600 | 0.773 | 0.721 | 84.5% |
| Annual SPM | 0.595 | 0.770 | 0.746 | 84.5% |
| RAPM | 0.586 | 0.764 | 0.721 | 84.5% |

## Team-win contract

For rating season `Y`, a player's metric is assigned to every team for which
the player records minutes in `Y+1`. A player receives a replacement rating of
`-2.0` when their `Y` sample is below 250 minutes or the metric is missing.
The team rating is

```text
team rating = 5 * sum(player rating_Y * team minutes_Y+1)
                  / sum(team minutes_Y+1)
```

The benchmark correlates that rating with team win percentage in `Y+1`.
Pearson measures linear agreement, Spearman measures rank agreement, and R² is
the squared within-season Pearson correlation. The headline is the equal-weight
mean across rating seasons 2021 through 2024, predicting 2022 through 2025.
Every included metric is scored on the same four outcome seasons.

This uses observed next-season minutes. It therefore knows future injuries,
trades, availability, and rotations. It is an **oracle-minutes retrodiction**,
not a preseason forecast. A true out-of-sample version requires archived
minutes projections created before each season; none were supplied.

Replacement sensitivity was run at `-3.0`, `-2.5`, `-2.0`, and `-1.5`. MAMBA
ranked first in all four settings. The full table is stored in
`team_win_summary.parquet`.

## Pairwise agreement contract

Pearson and Spearman correlations are calculated for offense, defense, and net
on pairwise-complete player-seasons. Each player must have at least 250 minutes
in that season. The common scope is 2021 through 2024. Pairwise sample sizes are
stored with every correlation because public files do not have identical
coverage.

New AIO net correlations are `0.999` with Old AIO, `0.923` with RAPM, `0.884`
with MAMBA, `0.873` with xRAPM, `0.795` with Annual SPM, `0.774` with LEBRON,
`0.723` with BPM, and `0.702` with BoxPIPM-style. Correlation measures agreement,
not accuracy or causal validity.

## What the metrics mean

- **New AIO:** the five-year-target statistical prior with the selected
  same-season stabilized features, updated by one season of possession RAPM.
- **Old AIO:** the original five-year-target statistical prior, updated by one
  season of possession RAPM.
- **RAPM:** zero-prior ridge regression on possession points with separate
  offensive and defensive player coefficients.
- **Annual SPM:** box, play-by-play, and tracking features predicting annual
  zero-prior RAPM without lineup outcomes as model inputs.
- **BoxPIPM-style:** a transparent leave-one-season-out ridge using 15
  traditional per-100 box rates to predict annual RAPM. It is not full PIPM.
- **BPM 2.0:** official Basketball-Reference values. BPM uses box-score rates,
  estimated position and offensive role, then a team-efficiency adjustment.
  This run ingests and verifies the published values; it does not claim an
  independent formula recreation.
- **xRAPM:** a statistical prior combined with regularized adjusted plus-minus.
  It shares RAPM structure with the CourtSignal hybrids.
- **LEBRON:** BBall Index's role-stabilized BoxPIPM prior combined with
  luck-adjusted RAPM.
- **MAMBA:** a current-season statistical prior combined with time-decayed
  multi-year RAPM and small shooting-luck adjustments. The author labels it a
  proof of concept.
- **EPM:** estimated skills feed an SPM prior and RAPM update. It is described
  in the UI but not scored because a complete historical export was not
  available. The public page exposed only a five-player preview.

## Interpretation limits

- Four folds are too few for a production promotion decision.
- MAMBA's multi-year time decay is structurally aligned with the next-season
  target; this may explain part of its lead.
- Metrics estimate different objects and use overlapping source data.
- Team aggregation cannot identify which player rating is individually right.
- Correlation and R² do not measure calibration in points per 100 possessions.
- Full PIPM was not recreated because its complete historical inputs and
  luck-adjusted on/off contract are not public. The box-only baseline is labeled
  accordingly.

## Reproduction

The builder is `src/nba_impact/models/public_aio_benchmark.py`; the source
adapter is `research/run_public_aio_benchmark.py`; unit tests are in
`tests/test_public_aio_benchmark.py`. The manifest stores hashes for every input
file and the builder. Unmatched public-name identities are retained separately.
Third-party player-level rating tables are not copied into the committed
artifact.

Method references: [LEBRON](https://www.bball-index.com/lebron-introduction/),
[EPM](https://dunksandthrees.com/about/epm),
[MAMBA](https://www.teemohoop.com/mamba/Blog%20Post%20Title%20One-mm8gk-cy9wh),
and [BPM 2.0](https://www.basketball-reference.com/about/bpm2.html).
