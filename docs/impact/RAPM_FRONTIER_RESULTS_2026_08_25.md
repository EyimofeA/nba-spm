# RAPM frontier experiments, 2026-08-25

## Decision

Keep terminal-lineup, zero-prior `3000 / 3000 / 300` RAPM as the reference.
None of the predictive challengers cleared an out-of-sample promotion gate.
Age translation, factor ratings, and win-probability credit remain useful in
separate, explicitly descriptive or forecasting lanes.

Season 2027 was not loaded. Selection used earlier seasons and all reported
2026 comparisons are reused diagnostics rather than untouched confirmation.

## Results

| Experiment | Exact test | Result | Decision |
|---|---|---:|---|
| Age translation | Forecast next annual RAPM from trailing 1, 3, or 5 annual ratings; Gaussian age bandwidths 0.1, 0.5, 1, and 2 years | One-year bandwidth lowered net weighted RMSE by 0.014, 0.027, and 0.036 for the 1/3/5-season inputs | Use as a forecast translation; do not put age into retrospective RAPM |
| Pair residual layer | Fit after one-player RAPM; select penalty on 2025, check 2026 | 2026 RMSE change +0.0345 | Reject |
| Trio residual layer | Same | +0.0702 | Reject |
| Four-player residual layer | Same | +0.0362 | Reject |
| Five-player residual layer | Same | -0.0039; correlation +0.0006 | Statistical null; do not promote |
| Pair-only RAPM | Five-season train; select on 2025, check 2026 against one-player RAPM | RMSE change +0.471; paired 95% interval [0.260, 0.688] | Reject |
| Trio-only RAPM | Same | +0.780; [0.516, 1.043] | Reject |
| Four-man-only RAPM | Same | +1.022; [0.728, 1.329] | Reject |
| Lineup-only RAPM | Same | +1.226; [0.884, 1.565] | Reject |
| Clock-adjusted RAPM | Remove the fitted 2024--25 six-minute score-margin slope from the target; check neutral player-only coefficients on identical 2026 games | RMSE change +0.018, 95% interval [-0.043, 0.079]; correlation change +0.009 | Reject as player rating; keep descriptive curve |
| Possession-progress adjusted RAPM | Same, but use fixed 25-possession progress bins with no final-game-length input | RMSE change +0.026, 95% interval [-0.036, 0.087]; correlation change +0.009 | Reject as player rating; validates the clock result as a same-row context proxy |
| Six-sided factors | Shooting eFG, turnover, and OREB ratings on offense and defense, 2024--26 | 743,946 possessions; 98.43% of relevant events mapped | Keep as descriptive mechanism surfaces |
| Multinomial RAPM | 0/1/2/3+ softmax; alpha selected on 2025, checked on 2026 | Margin RMSE 15.508 vs 15.473 for linear points RAPM; log loss 1.1109 vs 1.1096 for constant class rates | Reject as predictor |
| Win-probability RAPM | Conserved possession-to-possession WP change, 2025--26 | 497,177 possessions; game conservation error `1.11e-16`; net correlation with points RAPM .738; annual net stability .125 | Keep as descriptive leverage credit |
| Coach RAPM | Joint player and coach offense/defense effects, 2017--26 | Selected coach penalty 100,000; 2026 RMSE change +0.0147 and correlation change -0.0012 | Reject |

Negative RMSE changes are improvements. Baselines and candidates score the same
games within each comparison.

## Aging details

The annual target panel contains 6,942 player-seasons from 2014--26 and 5,053
adjacent-season transitions for 1,240 players. The stored age field has integer
resolution. A 0.1-year kernel is therefore a narrow smoothing choice over
integer ages, not evidence about within-year aging. True 0.1-year testing needs
dates of birth and exact observation dates.

The one-year smoothing bandwidth won net RMSE for all three trailing windows:

| Input history | No-age net RMSE | Age-adjusted net RMSE | No-age correlation | Age-adjusted correlation |
|---:|---:|---:|---:|---:|
| 1 season | 2.0045 | 1.9904 | .3959 | .3919 |
| 3 seasons | 1.7707 | 1.7442 | .4354 | .4467 |
| 5 seasons | 1.7440 | 1.7079 | .4626 | .4792 |

The 1-season result trades a small correlation decline for lower weighted
error. The 3- and 5-season results improve both measures. The descriptive age
curve remains offense-led and peaks around ages 25--27; defense is much flatter.

## What each new model estimates

### Residual unit interactions

The interaction model first fits ordinary player RAPM. It then fits one
zero-centered ridge layer to the possession residual using either all offensive
and defensive pairs, trios, four-player units, or five-player units. This is a
conditional lineup association. It is not causal chemistry and it is not a
replacement for player RAPM.

### Standalone unit RAPM

The corrected unit estimand removes individual player columns entirely. A pair
model has one offense and one points-allowed defense coefficient for each
eligible unordered pair. Trio, four-man, and lineup models use only units of
their stated size. Every model also has one signed home-offense column and fits
raw possession points directly. None uses player RAPM predictions or residuals.

The comparison uses rolling five-season training windows. Seasons 2020--24
select one ridge penalty per order on 2025. Seasons 2021--25 are then refit and
checked on the same 1,228 games in reused 2026. Whole-game paired bootstrap
intervals use 2,000 draws. The one-player reference has RMSE 15.296. Pair,
trio, four-man, and lineup models have RMSE 15.768, 16.076, 16.319, and 16.522.

The test also exposes the main structural problem. Under training-only exposure
floors of 500 pair possessions, 250 trio possessions, 100 four-man possessions,
and 50 lineup possessions, 2026 unit-slot coverage is 41.4%, 21.7%, 10.5%, and
3.7%. Unseen units receive zero coefficients. This is a valid result for the
frozen specification, not proof that every possible partial-pooling unit model
must fail.

### Six-sided factor RAPM

The factor run fits three mechanisms separately on each side of the ball:

1. shooting eFG value per 100 shot attempts;
2. turnover avoidance or forcing per 100 possessions;
3. offensive-rebound conversion or prevention per 100 resolved missed-field-goal opportunities.

The six ratings have different opportunity sets, so they do not add to net
points. The already-built conserved one-point, two-point, and three-plus point
channels are the additive decomposition of ordinary RAPM.

### Rubber-band adjusted RAPM

The score curve is fit after whole-game cross-fitted lineup expectations. One
version uses exact possession-start game clock; the other uses the number of
completed regulation possessions in fixed 25-possession bins. The proxy does
not divide by final game length, so it does not look ahead. Their eight fitted
margin slopes correlate 0.971.

Each candidate subtracts only the fitted score-margin slope term from observed
points before the standard `3000 / 3000 / 300` player fit. The time-segment
intercept is not subtracted. Both variants improved 2026 game-margin correlation
slightly but worsened RMSE, and both paired intervals include zero. They remain
local research ratings. Adding the observed score path back to predictions is
invalid for the primary gate and performed much worse because the score path is
partly caused by the players being rated.

### Multinomial RAPM

The softmax model estimates the probabilities of 0, 1, 2, and 3-plus points on
a possession from the same player lineup. Expected points are reconstructed
from the predicted class probabilities. The richer outcome distribution did
not improve held-out game margins or class log loss enough to justify replacing
linear points RAPM.

### Win-probability RAPM

The target is the change from the current possession-start home win probability
to the next possession-start probability, with the final possession receiving
the jump to the observed result. The WP surface used for each season was trained
only on the prior season and contains game state, not player identity. Credits
sum to the game result change to floating-point precision.

This estimates leverage-weighted retrospective credit. It does not estimate
how many points a player adds, and the low annual stability rules out using it
as the main player-strength rating.

### Coach RAPM

Coach offense and defense columns were added beside all player columns. The
source contains 325 coach-season rows, 72 coaches, and full assignments for
11,969 modeled games from 2017--26. Strong shrinkage could not make coach
effects improve held-out predictions. The design cannot cleanly separate a
coach from roster, franchise, assistants, and organizational context.

## Previously answered ideas

- Infinite RAPM lost: repeated prior updates compounded degradation.
- Learned age/recency buckets lost to a simple exponential decay curve and are
  parked. Time decay was not rerun because the principal parked it.
- Global home advantage remains useful. Team-specific home effects, hard
  garbage-time removal, rubber-band controls, clock-state fatigue, adaptive
  player penalties, and alternative offense/defense penalties failed the
  frozen five-year evaluation.
- Bayesian and neural RAPM, playoffs, start-lineup versus terminal-lineup, and
  defender-assignment structural models remain parked by explicit scope, not
  silently counted as completed.

## Reproduction map

- `research/rapm_lab/run_aging_resolution.py`
- `research/rapm_lab/run_rubberband_adjustment.py`
- `research/rapm_lab/run_rubberband_progress_rapm.py`
- `research/rapm_lab/run_lineup_interactions.py`
- `research/rapm_lab/run_standalone_unit_rapm.py`
- `research/rapm_lab/run_possession_outcome_rapm.py`
- `research/rapm_lab/run_win_probability_rapm.py`
- `research/rapm_lab/run_coach_rapm.py`

Each runner writes a local content-addressed artifact with its configuration,
quality checks, comparisons, flat tables, and caveats. Generated research data
and coach source pages remain local and are not committed.
