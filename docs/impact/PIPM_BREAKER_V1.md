# PIPM breaker v1

## Decision

Keep the 15-feature CourtSignal-targeted ridge prior as the research control.
None of the added context, larger feature banks, alternative RAPM labels,
correlation pruning, residual correction, or nonlinear learners improved the
downstream AIO on reused 2022--24 future games.

This does not promote the control to production. All three scored seasons were
already used during development, and Season 2027 remains untouched.

## Question and fixed comparison

The experiment separates four possible reasons a PIPM-like prior can work:

1. a small box-score feature bank;
2. the five-year RAPM labels used to train it;
3. the learner and regularization;
4. context such as minutes, starts, on/off, rebounding role and spacing.

Every candidate is trained only on windows ending before its rating season.
The rating seasons are 2021--23. Each prior receives the same one-season
terminal-lineup RAPM update with penalties `3000 / 3000 / 300` and prior scale
one. The resulting AIO predicts the same 3,687 games in 2022--24. The primary
comparison is equal-season mean whole-game margin MSE; RMSE is reported for
readable units. Uncertainty uses 5,000 paired whole-game bootstrap draws.

## Inputs tested

The small bank contains 15 per-100 box fields: points, assists, turnovers,
steals, blocks, offensive rebounds, defensive rebounds, personal fouls, fouls
drawn, free throws attempted/made, two-pointers attempted/made, and
three-pointers attempted/made.

The full banks contain 126 offense and 50 defense fields. They add tracking and
play-type volume, accuracy and frequency, empirical-Bayes rates, era-relative
rates, shot quality, Box Creation, Offensive Load, passing composites, and the
existing stabilized spacing field.

The explicit context challengers add:

- `minutes_5y`: minutes summed over the five-season window;
- `log_minutes_5y`: `log(1 + minutes_5y)`;
- `starter_share_squared_5y`: `(sum games started / sum games played)^2`;
- `position_adjusted_oreb_5y`: season-stabilized OREB% minus the
  possession-weighted guard, forward or center mean, then pooled over five
  seasons;
- `spacing_value_above_average_p100`: `3PA/100 * (1.5 * stabilized 3P% -
  league eFG%)`;
- `raw_onoff_offense_5y` and `raw_onoff_defense_5y`: ordinary five-year on/off.

The annual OREB value is empirical-Bayes shrunk with 500 offensive possessions
toward the same-season league mean before position adjustment. No later season
enters an earlier stabilized season or five-year window. Minutes and starts are
availability/role signals rather than basketball skills. On/off is deliberately
marked impure because it reuses lineup-outcome information that also enters the
AIO likelihood.

## Results

Lower future-game RMSE is better.

| Candidate | Mean RMSE | Margin correlation | MSE delta vs control |
|---|---:|---:|---:|
| Box, CourtSignal target, ridge | **13.8644** | **.3635** | -- |
| Box, Ryan target, ridge | 13.8668 | .3625 | +0.073 |
| Box, Ryan residual correction | 13.8967 | .3638 | +0.897 |
| Box + minutes + starter share | 13.9029 | .3624 | +1.089 |
| Box, CourtSignal target, tuned | 13.9050 | .3618 | +1.122 |
| Full, CourtSignal target, tuned | 13.9198 | .3615 | +1.557 |
| Full, CourtSignal target, ridge | 13.9233 | .3578 | +1.657 |
| Full, Ryan target, tuned | 13.9290 | .3591 | +1.818 |
| Box, Ryan target, tuned | 13.9297 | .3584 | +1.810 |
| Box + raw minutes | 13.9298 | .3588 | +1.886 |
| Full, Ryan target, ridge | 13.9359 | .3552 | +2.019 |
| Box + adjusted OREB + spacing | 13.9387 | .3572 | +2.067 |
| Box + ordinary on/off | 13.9424 | .3578 | +2.278 |
| Correlation-pruned full model | 13.9457 | .3568 | +2.308 |
| Box + all context | 13.9693 | .3569 | +3.041 |

The deltas are points-squared per game. Box/Ryan ridge was statistically tied
with the control: paired MSE delta `+0.073`, bootstrap 95% interval
`[-0.030, +0.179]`. Every larger fixed or tuned feature bank lost decisively.
The minutes/starts and residual arms also failed to win a season, although
their intervals slightly crossed zero. Ordinary on/off won one season but lost
overall, and its interval was wholly worse: `[+1.045, +3.547]`.

The learner search covered ridge, elastic net and histogram gradient boosting.
The nested selector chose the same histogram GBM for every tuned fold and side.
It fit the intermediate RAPM labels much better but predicted future games
worse. The fixed ridge control therefore remains the better learner for this
small historical sample.

## Why RAPM fit was misleading

The exact target/feature 2-by-2 comparison shows that more features reconstruct
five-year offense RAPM much better. For the CourtSignal label, offense target
RMSE falls from `1.282` to `1.010` and target correlation rises from `.555` to
`.703`. That gain does not transfer: downstream game RMSE worsens from
`13.864` to `13.923`.

The all-context Ryan model is more extreme. Its mean player-target RMSE is about
`.60` offense and `.69` defense in the first scored fold, but it has the worst
future-game MSE. Intermediate RAPM reconstruction is therefore diagnostic only;
it cannot select the AIO prior.

## Correlation audit

Across the feature banks, 19 pairs have absolute correlation at least `.95`.
The largest duplications are raw and era-relative versions of at-rim frequency,
FTA, AST, TOV, PTS, true shooting, shot quality, arc-three frequency and 3PA.
Defense also duplicates assists with assist points created and average seconds
per touch with average dribbles per touch.

The `.98` pruning challenger removed nine offense fields and two defense fields
per fold. It did not improve future games. Correlation is not itself a reason to
delete a feature from a regularized or tree model, so the full bank remains
unchanged and the pruning result stays a separate rejected arm.

## External PIPM agreement

The user-supplied PIPM database matches 99.12% of source minutes after identity
resolution. It combines regular season and playoffs, so it is never used as a
clean training target. On the 262 matched 2021 players with at least 250 PIPM
minutes, the final AIO net correlations with published PIPM are approximately
`.757` for the winning control and `.758` for Box/Ryan ridge. Agreement is a
sanity check, not predictive validation.

Automated access to `winsadded.com` returned Cloudflare HTTP 403 in this run.
The supplied CSV is sufficient for the comparison, but the site cannot be
treated as a reproducible programmatic source until access or a documented
download endpoint is available.

## Possession-source QA

The one-season AIO likelihood uses Gabriel Michael's `poss_data` repository,
not the older split pipeline. Ninety team-season CSVs for 2021--23 were hashed
into source manifests and converted into reproducible terminal-lineup caches.

| Season | Retained games | Retained possessions | Games quarantined | Unresolved lineup rows |
|---:|---:|---:|---:|---:|
| 2021 | 1,047 | 207,965 | 29 | 806 |
| 2022 | 1,187 | 233,279 | 39 | 812 |
| 2023 | 1,208 | 240,149 | 22 | 0 |

Games whose total reconstructed score did not match the official game log were
quarantined. The source's possession convention assigns some technical and
free-throw points to the opposite side of a possession boundary; pre-quarantine
side-specific score agreement is 80.9--83.8%, while total-score agreement is
96.8--98.2%. This is acceptable for the current total-points likelihood only.
It is not sufficient for factor RAPM or possession-type claims without a new
event-allocation contract.

## Reproduce

```bash
PYTHONPATH=src:research OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 .venv/bin/python research/run_pipm_breaker.py
```

Run artifact: `pipm_breaker_v1_d154ebea55`. The manifest pins feature, target,
Ryan RAPM, PIPM, Basketball-Reference, player-sheet, possession-cache and
source-manifest hashes. Season 2027 has zero rows.

## Next decision

Do not add minutes, starts, ordinary on/off, position-adjusted rebounding,
spacing or the residual correction to the current prior. The next useful test
is not another broad feature dump. It is one predeclared feature-family arm
with a basketball mechanism and complete coverage, judged by the same paired
future-game gate. Shot-quality passing or teammate-spacing change are plausible
choices once their data contracts are complete.
