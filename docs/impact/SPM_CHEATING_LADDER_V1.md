# SPM cheating ladder v1

## Decision

Do not replace the five-year SPM with the five-year BoxPIPM-style model. Their
mean next-season team-win R-squared is effectively tied (`0.51816` versus
`0.51811`), while BoxPIPM-style is much worse at reconstructing five-year RAPM
(`1.7838` versus `1.4541` weighted net RMSE).

Keep age, minutes, and listed position out. They lower mean team-win R-squared
and barely change the player target fit. Legacy AuPM also loses.

Raw on/off is the most useful full-coverage impure addition. It raises mean
team-win R-squared by `0.0162`, wins all three folds, and lowers five-year RAPM
net RMSE from `1.4541` to `1.2869`. This is expected because it imports lineup
outcomes into a model whose public purpose is to remain statistical-only. Keep
it as a hybrid research arm, not as SPM.

A BPM-style team reconciliation has the largest team-win gain (`+0.0599` mean
R-squared) but slightly worsens player RAPM RMSE. It is carrying persistent team
strength, not demonstrating better individual attribution. It belongs in a
team forecast layer, not the player metric.

Run: `spm_cheating_ladder_v1_fff340f6b6`.

## Frozen test

- Rating seasons: 2022, 2023, and 2024.
- Outcome seasons: 2023, 2024, and 2025.
- Primary score: equal-season mean correlation-squared between team ratings and
  next-season win percentage.
- Team rating: five times the player-rating average weighted by observed
  next-season minutes.
- Replacement policy: players missing a qualified 250-minute rating receive
  `-2.0`; sensitivity covers `-3.0`, `-2.5`, `-2.0`, and `-1.5`.
- Secondary score: weighted RMSE and correlation against rolling five-year
  zero-prior RAPM on identical player rows.
- The exact saved `selected_combined` five-year SPM is the baseline.
- Every new feature for a window ending in season Y uses seasons no later than
  Y. Season 2027 is untouched.

This is an oracle-minutes retrodiction, not a preseason forecast. The retained
year totals assign traded players to one primary team. Recomputing the published
annual-SPM check with those rows raises mean R-squared by `0.0297` relative to
the exact player-team-stint benchmark. Every arm here uses the same approximate
rows, but the absolute R-squared values must not be mixed with the public
benchmark.

## Team-win result

| Arm | Mean R-squared | Delta | Fold wins | Paired team bootstrap 95% interval for delta |
|---|---:|---:|---:|---:|
| SPM + team reconciliation | 0.5781 | +0.0599 | 2/3 | [-0.0140, +0.1317] |
| SPM + RAPTOR on/off | 0.5372 | +0.0190 | 2/3 | [-0.0029, +0.0456] |
| SPM + raw on/off | 0.5344 | +0.0162 | 3/3 | [-0.0028, +0.0337] |
| SPM + all full-coverage cheats | 0.5334 | +0.0152 | 3/3 | [-0.0078, +0.0362] |
| Five-year SPM | 0.5182 | — | — | — |
| Five-year BoxPIPM-style | 0.5181 | -0.0001 | 2/3 | [-0.0697, +0.0636] |
| SPM + age/minutes/position | 0.5161 | -0.0021 | 1/3 | [-0.0077, +0.0035] |
| SPM + legacy AuPM | 0.5158 | -0.0024 | 1/3 | [-0.0239, +0.0193] |

No challenger has a paired bootstrap interval entirely above zero. Three folds
are not enough for promotion.

The raw-on/off fold deltas are positive in all three seasons. BoxPIPM-style is
unstable: it trails by `0.0917` R-squared in 2022, leads by `0.0850` in 2023,
and leads by `0.0066` in 2024. Its zero mean advantage is not a hidden upgrade.

## Player-target result

| Arm | Five-year RAPM net RMSE | Net correlation |
|---|---:|---:|
| SPM + all full-coverage cheats | 1.2651 | 0.7650 |
| SPM + raw on/off | 1.2869 | 0.7658 |
| SPM + RAPTOR on/off | 1.3425 | 0.6078 |
| SPM + legacy AuPM | 1.3579 | 0.7506 |
| SPM + age/minutes/position | 1.4479 | 0.6982 |
| Five-year SPM | 1.4541 | 0.7026 |
| SPM + team reconciliation | 1.4569 | 0.6991 |
| Five-year BoxPIPM-style | 1.7838 | 0.5136 |

The full-coverage cheat set contains endpoint age, five-year minutes, three
multi-hot listed-position flags, five-year raw on/off, and five-year AuPM. A
fixed standardized ridge predicts only the residual left by the exact saved SPM
using earlier rating windows. It does not refit or rename the public SPM.

## What each addition means

- **BoxPIPM-style:** forward-chained ridge on the 15 traditional per-100 box
  fields, trained directly against the same rolling five-year RAPM target. This
  is the reproducible box component, not full PIPM.
- **Age/minutes/position:** age at the window endpoint, log total minutes in the
  trailing five seasons, and multi-hot G/F/C listed-position flags.
- **Raw on/off:** player on-court net efficiency minus derived off-court net
  efficiency, possession-weighted across five seasons. Against the retained
  RAPTOR/WOWY raw-on/off table at 1,000 possessions, the derived value has
  Pearson `0.8891`, rank correlation `0.9137`, and mean absolute difference
  `1.47` points per 100.
- **Legacy AuPM:** the archived local AuPM reproduction, possession-weighted
  across five seasons. It is not claimed as a canonical Ben Taylor AuPM.
- **RAPTOR on/off:** official FiveThirtyEight on/off components, weighted across
  the available trailing seasons. The source stops in 2022; mean five-season
  coverage falls from `54.7%` in the 2022 window to `43.2%` in 2024. Its result
  is therefore stale and not a fair production candidate.
- **Team reconciliation:** for each team-season, add one constant to every
  player's net rating so the minute-weighted five-player sum equals observed
  team net efficiency. This follows the central reconciliation idea in
  [BPM 2.0](https://www.basketball-reference.com/about/bpm2.html), without
  claiming an exact reproduction of BPM's lead correction or offense/defense
  split.

## Reproduction

- Runner: `research/run_spm_cheating_ladder.py`
- Tests: `tests/test_spm_cheating_ladder.py`
- Artifact: `artifacts/research/spm_cheating_ladder/spm_cheating_ladder_v1_fff340f6b6`
- Main tables: `team_win_folds.parquet`, `team_win_summary.parquet`,
  `team_bootstrap.parquet`, `player_metrics.parquet`, and
  `player_summary.parquet`.

The manifest stores input and runner hashes, source coverage, on/off validation,
the primary-team allocation drift, and the exact `offense + defense = net`
check.
