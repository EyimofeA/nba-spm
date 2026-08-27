# Semantic feature completion

## Decision

The research SPM now has a fully finite input panel. The annual panel contains
6,942 player-seasons from 2014 through 2026. The rolling five-year panel
contains 8,620 rows. All 175 final model inputs have values.

This result does not mean that every value was observed. Five new availability
fields preserve the distinction between observed data and a declared fallback.
The public model remains unchanged.

## Completion rules

| Input type | Completion rule | Reason |
|---|---|---|
| Event count per 100 | Zero when possession exposure exists | An absent event means zero observed events. |
| Accuracy, frequency, or share | Same-season empirical-Bayes estimate | Zero would mean observed failure, not no opportunity. |
| Weighted level metric | Same-season median | The field has no natural zero. |
| Hustle, matchup, closest-defender, and rim-defense value | Zero plus a source-availability field | Zero is the neutral fallback. The flag tells the model that the source was absent. |
| zTS with 250 minutes and 50 playtype possessions | Strict playtype estimate | This is the original qualified estimate. |
| zTS with some playtype data below either threshold | Estimate from every observed playtype possession | The estimate remains noisy but observed. |
| zTS without a playtype row | Player TS minus the season's possession-weighted mean expected TS | This reduces the fallback to season-relative shooting efficiency. |

The final contract adds `zts_source_tier`, `has_hustle_tracking`,
`has_matchup_tracking`, `has_dfg_tracking`, and
`has_rim_defense_tracking`.

## Coverage evidence

The original observed-data audit found 75 selected annual fields below 99%
coverage. Every field now has a reason and a completion method in
`artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f/features_below_99pct_completion.csv`.
No listed field has an unexplained cause, a missing method, or a remaining
missing model value.

| Cause | Selected fields | Completion |
|---|---:|---|
| Undefined rate or missing rate input | 36 | 35 empirical-Bayes estimates and one season median |
| Missing event or possession input | 21 | Zero event rate after exposure repair |
| Matchup source history or assignment | 8 | Zero and matchup availability |
| Hustle source history or row absence | 5 | Zero and hustle availability |
| No rim defended-shot row | 3 | Zero and rim-defense availability |
| Playtype eligibility | 1 | Low-sample estimate or TS-based fallback |
| Missing weighted-average input | 1 | Same-season median |

The source gaps are explicit:

* Hustle has no rows from 2014 through 2017. It lacks two ultra-low-exposure
  rows in 2025 and one in 2026.
* Matchup data have no rows from 2014 through 2017. They lack one row in 2019
  and 15 rows in 2020.
* Closest-defender data lack 46 player-seasons across 2014 through 2026.
* Rim-defense data lack 123 player-seasons across 2014 through 2026.
* Strict zTS covers 5,140 rows. Low-sample playtype data add 414 rows. The
  TS-based fallback covers the remaining 1,388 rows.

## Predictive check

Run `semantic_feature_completion_comparison_v1_235b4dea34` scores five future
seasons on identical games. It compares the completed SPM, the completed SPM
without eight matchup fields, the previous missing-data implementation, and
BoxPIPM-style. Each prior also receives the same one-season RAPM update.

| Candidate | Equal-season RMSE | Mean fold correlation |
|---|---:|---:|
| BoxPIPM-style plus RAPM | 14.402 | .3616 |
| Previous SPM plus RAPM | 14.430 | .3660 |
| Completed SPM plus RAPM | 14.431 | .3661 |
| Completed SPM without matchups plus RAPM | 14.443 | .3603 |
| Completed SPM | 14.601 | .3246 |
| Previous SPM | 14.610 | .3238 |
| Completed SPM without matchups | 14.695 | .3004 |
| BoxPIPM-style | 14.707 | .2960 |

The completed standalone SPM beats BoxPIPM-style by `-3.104` paired MSE. Its
95% whole-game interval is `[-5.219, -0.982]`. It also beats the no-matchup SPM
by `-2.757`, with interval `[-3.984, -1.534]`.

The completed AIO and previous AIO are tied. Completed minus previous equals
`+0.030` MSE, with interval `[-0.153, +0.212]`. BoxPIPM-style AIO has the best
point estimate. Completed minus BoxPIPM-style equals `+0.837` MSE, with
interval `[-0.119, +1.799]`. The interval crosses zero.

## Model status

Keep all matchup fields. Removing them causes a clear standalone loss and a
possible AIO loss. Use the completed panel for new research because it has an
explicit value and provenance state for every selected input. Keep
BoxPIPM-style and the completed SPM as separate AIO challengers. Season 2027
remains the untouched confirmation.
