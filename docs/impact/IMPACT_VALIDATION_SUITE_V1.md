# Impact validation suite v1

Run `impact_validation_suite_v1_4f2ad7cdd8` compares three frozen statistical
priors under five tests. It also keeps zero-prior RAPM as a game-prediction
baseline. Every season is reused evidence. Season 2027 stays untouched.

## What matters most

The tests are reported in this order:

| Priority | Test | Weight | Why it is here |
| ---: | --- | ---: | --- |
| 1 | Next-season game margin | 50% | Tests the season-t AIO estimate on identical season-t+1 games and observed lineups. |
| 2 | Midseason adaptation | 25% | Uses the season-t-1 statistical prior, updates on the first half of season t, and predicts the second half. |
| 3 | Forward annual impact | 15% | Compares a statistical rating with next-season one-year RAPM after an earlier-only aging adjustment. |
| 4 | Reverse annual impact | 5% | Checks whether the metric tracks the previous season as well as the next. It is diagnostic only. |
| 5 | Same-season RAPM fit | 5% | Measures descriptive agreement with the noisy label used to train SPM. |

Raw MSE, RMSE, and correlations are never averaged. Each candidate receives a
within-fold percentile rank. The composite averages those ranks using the
declared weights. An equal-weight result is reported beside it.

## Results

### 1. Next-season game margins

Five folds cover test seasons 2022 through 2026.

| Prior used in AIO | Mean MSE | Mean RMSE | Mean correlation |
| --- | ---: | ---: | ---: |
| BoxPIPM-style | 206.476 | 14.342 | .3664 |
| Selected five-year SPM | 208.128 | 14.398 | .3656 |
| Five-year SPM | 208.175 | 14.400 | .3654 |
| Zero prior | 213.206 | 14.570 | .3219 |

BoxPIPM-style lowers equal-season mean game MSE by `1.652` against selected
five-year SPM. The paired whole-game bootstrap interval is `[-2.478, -0.836]`.
It is the strongest result in this run.

### 2. Midseason adaptation

Three folds cover 2022 through 2024. Every season is split by whole games in
date and game-ID order.

| Prior from the prior season | Mean MSE | Mean RMSE | Mean correlation |
| --- | ---: | ---: | ---: |
| Selected five-year SPM | 184.335 | 13.557 | .4563 |
| Five-year SPM | 184.579 | 13.566 | .4550 |
| BoxPIPM-style | 185.253 | 13.594 | .4526 |
| Zero prior | 195.773 | 13.973 | .4057 |

Selected five-year SPM beats the base five-year SPM by `0.244` MSE with a
paired interval of `[0.069, 0.419]` when the difference is base minus selected.
Its `0.919` MSE advantage over BoxPIPM-style is not resolved. The interval for
Box minus selected is `[-0.765, 2.556]`.

### 3. Annual player tests

These tables use the net component. Forward and reverse targets remove the
age change estimated from earlier origin seasons only.

| Test | Five-year SPM corr | Selected five-year SPM corr | BoxPIPM-style corr |
| --- | ---: | ---: | ---: |
| Forward annual impact | .4760 | .4732 | .4425 |
| Reverse annual impact | .5225 | .5237 | .4474 |
| Same-season RAPM fit | .5498 | .5487 | .4800 |

BoxPIPM-style has the lowest forward annual RMSE, `1.851` versus `1.922` and
`1.927`, while the five-year SPMs have higher correlation. BoxPIPM is better
calibrated in RAPM units but worse at ordering players. That distinction is why
the suite keeps RMSE and correlation visible.

## Composite and decision

| Candidate | Weighted score | Weighted rank | Equal-weight score | Equal-weight rank |
| --- | ---: | ---: | ---: | ---: |
| Selected five-year SPM | .577 | 1 | .633 | 2 |
| BoxPIPM-style | .483 | 2 | .227 | 3 |
| Five-year SPM | .440 | 3 | .640 | 1 |

There is no defensible overall winner. Changing from decision weights to equal
weights changes the ordering. The composite is useful as a warning against
optimizing one number, not as a promotion statistic.

The model decision remains narrower:

- use BoxPIPM-style as the frozen research prior for the AIO because it wins the
  primary next-season game-margin test on all five historical folds in the
  saved bake-off;
- retain selected five-year SPM as the stronger midseason update prior and keep
  the five-year SPM family for standalone statistical ratings;
- do not replace one model with the other across every use case;
- wait for the complete 2027 season before making a public promotion decision.

## Reproduction

The contract is `research/experiments/impact_validation_suite_v1.yml`. The
runner is `research/run_impact_validation_suite.py`. The reusable calculations
are in `src/nba_impact/models/impact_validation_suite.py`. The artifact stores
the per-fold scores, game predictions, annual matches, coverage, paired
whole-game intervals, source hashes, and both composite rankings.

The first artifact, `impact_validation_suite_v1_021be06a12`, is invalid. Its
age join required every matched player to have an age and silently omitted both
adjacent-season tests. The corrected code keeps the valid age-matched rows and
the final artifact has all five tests at full declared weight. Intermediate
artifact `impact_validation_suite_v1_07c7b85efc` is complete but superseded. It
weighted adjacent seasons by target exposure alone. The final run uses the
smaller of origin and target exposure, matching the transition contract.
