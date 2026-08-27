# Full feature coverage and corrected refit

## Decision

The full 127-offense and 68-defense feature contract remains the research SPM.
The corrected data do not establish that its one-season RAPM posterior beats
the 15-feature BoxPIPM-style posterior. Neither model should replace the public
2017–24 AIO from reused evidence.

## Coverage gate

Run `full_feature_coverage_v1_3de4ec8954` audits all 170 unique selected
features before model imputation. Coverage means an observed upstream value.
Median and zero fills do not count.

The Gabriel player sheets omitted `OffPoss` and `DefPoss` for 506
player-seasons. The rebuild filled 1,012 missing exposure cells from the
canonical zero-prior RAPM target panel. It did not overwrite observed exposure
or add impact outcomes as features. The repair reduced sub-99% features from
108 to 75 in the annual panel and from 170 to 77 in the five-year panel.

Every remaining sub-99% feature has an explicit reason in
`artifacts/research/full_feature_coverage/full_feature_coverage_v1_3de4ec8954/coverage_report.md`.
The machine-readable table is `features_below_99pct.csv` in the same run.

| Cause | Annual fields | Five-year fields | Meaning |
|---|---:|---:|---|
| Zero opportunity or missing rate input | 36 | 36 | A player had no qualifying attempt, touch, drive, turnover, or shot context. |
| Missing player-sheet event input | 21 | 21 | The source omitted that event family for some player-seasons. |
| Matchup history or exposure | 8 | 8 | Scorer-defender matchup data start in 2018; some later players have no positive assignment. |
| Hustle history or row absence | 5 | 5 | Hustle data start in 2018; a few later rows are absent. |
| Rim defended-shot row absent | 3 | 3 | The source has no observed rim-defense row for that player-season. |
| Closest-defender row absent | 0 | 2 | No observed defended-shot row appears in any season of that five-year window. |
| Playtype eligibility | 1 | 1 | zTS requires 250 minutes and 50 qualifying playtype possessions. |
| Weighted-average input absent | 1 | 1 | The source value or its natural opportunity weight is unavailable. |

Undefined percentages should remain missing until training-fold imputation.
Treating a player with zero corner threes as an observed 0% shooter would change
the estimand and encode a false performance value.

## Corrected refit

Run `full_spm_history_ablation_v1_2eb5eb428c` uses the rebuilt annual estimates
and pools those frozen values across each five-year window. It no longer mixes
one-season stabilized fields with features re-engineered from raw five-year
rows.

| Candidate | Mean next-season margin RMSE | Correlation |
|---|---:|---:|
| BoxPIPM-style + one-season RAPM | 14.379 | .3610 |
| Full SPM + one-season RAPM | 14.402 | .3660 |
| History-complete SPM + one-season RAPM | 14.428 | .3576 |
| Zero-prior one-season RAPM | 14.570 | .3219 |
| Full SPM alone | 14.583 | .3238 |
| BoxPIPM-style alone | 14.673 | .2962 |
| History-complete SPM alone | 14.701 | .2939 |

The full standalone SPM beats BoxPIPM-style by `-2.750` equal-season MSE. The
95% whole-game interval is `[-4.821, -0.701]`. After the RAPM update, full SPM
trails BoxPIPM-style by `+0.681` MSE, but the interval `[-0.227, +1.587]`
crosses zero. The AIO comparison is unresolved.

The full defense contract beats the history-complete defense contract after
RAPM by `-0.756` MSE with interval `[-1.362, -0.151]`. Keep the late-start
hustle and matchup fields. Missing-history indicators and fold-only imputation
must remain explicit.

## Prior-strength audit

Run `aio_prior_scale_audit_v1_aeca5715b3` selects prior-center scale using only
strictly earlier stored next-season games. The grid is `.25, .50, .75, 1.00`.
It selects `.75` for full SPM and `1.00` for BoxPIPM-style in every scored
fold. Scaling full SPM from `1.00` to `.75` changes equal-season MSE by only
`-0.014`; the 95% interval is `[-0.543, +0.515]`. Fixed unit scale was not the
reason the full posterior failed to beat BoxPIPM-style.

Across 2023–26, BoxPIPM-style minus tuned full SPM is `-0.665` MSE with interval
`[-1.470, +0.139]`. The result still does not resolve a winner.

## Validation contracts

Retrospective impact and prediction require separate promotion gates.

| Product | Primary question | Valid gate |
|---|---|---|
| Retrospective RAPM/AIO | What happened during this season after lineup adjustment? | coefficient recovery, source agreement, within-season blocked-game stability, lineup/source sensitivity, and identity checks |
| Predictive prior | What player strength transfers to future games? | chronological future-game MSE with projected availability and minutes |
| Oracle exposure diagnostic | How do ratings perform if actual future lineups are supplied? | paired next-season game MSE, labeled as an oracle-exposure test |

The current next-season test supplies actual future lineups as exposure
weights. Unknown player slots rise from 8.28% in 2023 to 10.74% in 2026, and
more than 1,000 games per season contain at least one unknown player. This is a
fair paired diagnostic because every candidate scores the same rows. It is not
a deployable forecast and must not select the public retrospective flagship.

Season 2027 remains an untouched confirmation. It can confirm a frozen choice.
It cannot erase the adaptivity created by the many reused 2022–26 experiments.

## Public lineage

The static site snapshot now uses `annual_aio_ratings_v1_b52b5aecd9` for AIO,
SPM, and RAPM through 2024. It adds RAPM-only rows for 2025–26 from
`current_single_season_rapm_targets_v1_9c0cdda919`. The research-only 2014–26
SPM/AIO refresh is not published. `snapshot-manifest.json` records relative
artifact paths and hashes the exported row set.
