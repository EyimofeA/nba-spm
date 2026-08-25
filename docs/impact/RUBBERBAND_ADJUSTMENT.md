# Rubber-band scoring adjustment

## Result

The score effect is small early and large late. On regulation possessions from
2024--26, an offense playing with a lead scores below its cross-fitted lineup
expectation. The selected curve uses actual six-minute game-time bins and clips
the pre-possession margin at 15 points.

This is an estimate of game-state scoring context. A frozen follow-up also
tested actual-clock and possession-progress target corrections in RAPM. Neither
cleared its future-game gate. The curve does not identify effort or garbage
time.

## Data and split

- 743,946 regular-season possessions from 3,681 of 3,690 games in 2024--26;
- 739,969 regulation possessions with actual start time;
- exact terminal lineups and pre-possession score reconstructed from the same
  possession ledger;
- 2024 for shape development;
- 2025 for selecting the time resolution and margin cap;
- 2026 as reused diagnostic evidence;
- Season 2027 never loaded.

The earlier context study used possession order within a quarter as its clock
proxy. This study replaces that proxy with actual elapsed game seconds.

## Estimator

First, fit ordinary `3000 / 3000 / 300` lineup RAPM within each season using
five whole-game folds. A possession receives a prediction only from a fit that
excluded its game. Define the residual

\[
r_i = y_i - \widehat{y}_{i,\mathrm{lineup, OOF}}.
\]

For each actual game-time bucket `b`, fit

\[
r_i = \alpha_b + \beta_b\,\mathrm{clip}(m_i,-c,c) + \epsilon_i,
\]

where `m_i` is the offense's lead before the possession. The candidate grid
compared one, four, and eight time buckets plus margin caps of 10, 15, 20, 25,
30, and no cap. The fit uses regulation possessions; the first contract assigns
zero adjustment in overtime.

The proposed tie-state normalization for a later RAPM test is

\[
y_i^* = y_i - \widehat{\beta}_{b(i)}
\mathrm{clip}(m_i,-15,15).
\]

Do not subtract the time intercept. It is not part of the lead-versus-trail
contrast.

## Estimated slopes

Slopes are points per 100 possessions for each point of pre-possession offense
margin. Negative means leading offenses score less than their lineup expectation.
Intervals use a game-cluster sandwich covariance.

| Minutes elapsed | Slope | 95% interval |
| ---: | ---: | ---: |
| 0--6 | +0.003 | -0.224 to +0.230 |
| 6--12 | -0.056 | -0.189 to +0.077 |
| 12--18 | -0.199 | -0.310 to -0.088 |
| 18--24 | -0.176 | -0.275 to -0.076 |
| 24--30 | -0.044 | -0.141 to +0.052 |
| 30--36 | -0.147 | -0.238 to -0.056 |
| 36--42 | -0.173 | -0.263 to -0.083 |
| 42--48 | -0.524 | -0.608 to -0.439 |

Example: with ten points of lead in the final six minutes, the estimated
adjustment is `-5.24` points per 100 possessions. To normalize the observed
possession toward a tie state, a later target experiment would add `5.24` points
per 100 to that possession outcome before fitting player coefficients.

## Validation

The 15-point cap won the 2025 selection loss, but only barely. Its paired
game-level MSE improvement over the 10-point cap was `0.0000097`, with a 95%
interval of `-0.0000283` to `0.0000486`. Treat 10 and 15 as tied.

After refitting the selected shape on 2024--25, the 2026 possession residual
RMSE changed from `1.191592` to `1.191427`. MSE fell by 0.028%. All 2,000
whole-game resamples favored including the score curve; the MSE improvement
interval was `0.000180` to `0.000602`. The minimum pairwise correlation between
the eight annual slope vectors was 0.691.

The effect predicts a real but tiny fraction of possession variance. Its value
is removing a systematic context association before player fitting, not making
individual possession outcomes easy to predict.

## Possession-progress replication

The exact-clock field is available on the canonical 2024--26 rows, so the main
study uses it directly. The legacy 2022--23 RAPM cache has no safe clock field.
Historical event possessions do not share the same possession boundaries, so
we did not force a fuzzy join.

As a same-row check, each game was also divided by the count of completed
regulation possessions before the current possession. Segments are fixed at
1--25, 26--50, 51--75, 76--100, 101--125, 126--150, 151--175, and 176-plus.
The definition never uses the final number of possessions in the game. The
clock and possession-progress slope vectors correlate `0.971`. In the final
segment, a ten-point lead implies `-5.24` points per 100 under the clock model
and `-5.03` under the possession-progress model.

## Adjusted RAPM test

Both candidates use the same terminal-lineup design and `3000 / 3000 / 300`
penalties as normal RAPM. The score curves are fit on 2024--25 out-of-fold
lineup residuals. The player target is

\[
y_i^* = y_i - \widehat{\beta}_{b(i)}
\operatorname{clip}(m_i,-15,15).
\]

Only the signed-margin term is removed. The segment intercept is not. Player
coefficients are then scored on the same 1,228 reused 2026 games without using
their observed score path.

| Player model | Margin RMSE | Correlation | RMSE change vs normal |
| --- | ---: | ---: | ---: |
| Normal RAPM | 15.473 | 0.334 | 0.000 |
| Clock-adjusted RAPM | 15.491 | 0.344 | +0.018 |
| Possession-adjusted RAPM | 15.499 | 0.343 | +0.026 |

The paired 95% RMSE intervals are `[-0.043, +0.079]` for clock and
`[-0.036, +0.087]` for possession progress. Both include zero. Adding the
observed score-context path back into game predictions is much worse because
score margin is endogenous: RMSE rises to 17.933 and 18.020.

On the descriptive 2024--26 refit, adjusted and normal net ratings correlate
0.991. Mean absolute net movement is 0.239 points per 100. The movement is real
enough to inspect, but the predictive gate does not support promotion.

The local RAPM Lab shows the two curves, exact evaluation table, adjusted
leaderboards, and every other saved RAPM-test leaderboard. The missing nine
games still fail the canonical lineup-quality panel; this study did not weaken
those gates.

Runs: `rubberband_adjustment_v1_34be1ee621` and
`rubberband_progress_rapm_v2_b72716c2fb` under the ignored local
`research/rapm_lab/outputs/` directory.
