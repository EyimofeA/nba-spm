# Four-factor Box15 residual test

## Decision

Keep Box15 as the research AIO prior. Shooting, shot volume, turnover, and
opponent offensive-rebound prevention all become easier to estimate with
specialist features. Their combined correction does not improve the final AIO
enough to pass the frozen gate.

Season 2027 remains untouched.

## Targets

The target build uses regular-season events and observed lineups from 2014
through 2026. It produces annual targets and matched five-year targets for
window ends 2018 through 2026. Every factor uses the same offense, defense, and
home design with `3000 / 3000 / 300` ridge penalties.

The four targets are:

1. Shooting TS. Field goals receive weight one and free throws receive weight
   `0.44`. The weighted outcome reproduces true shooting percentage.
2. Shot volume. Each anchored possession receives
   `FGA + 0.44 * FTA`. Positive defense means shot-volume suppression.
3. Turnover avoidance. Each anchored possession receives a binary turnover
   outcome. Positive offense means avoidance. Positive defense means forcing.
4. Opponent offensive-rebound prevention. Each resolved field-goal miss
   receives a binary offensive-rebound outcome. Positive defense means
   prevention.

The possession builder groups consecutive same-offense field goals,
nontechnical free throws, and turnovers. Made field goals, turnovers, offense
changes, and period changes close groups. Technical free throws do not consume
a possession and do not enter the shot-volume target.

The v2 target artifact contains 6,935 annual rows and 8,620 five-year rows. Its
minimum valid-lineup fractions are `95.88%` for shots, `95.82%` for rebounds,
and `96.13%` for anchored possessions. Team-game score reconstruction matches
at least `99.31%` of totals. The 2024 through 2026 turnover targets correlate
from `0.930` to `0.958` with the separate canonical possession build by side
and season.

## Specialist inputs

Every specialist retains Box15, then adds only mechanism-related fields.

Shooting offense adds stabilized zTS, SelfORB-adjusted TS, shot quality,
shotmaking above expectation, location shares, assisted share, transition
share, and rim shotmaking. Shooting defense adds assigned-shot quality,
shotmaking points saved, rim quality conceded, scorer-adjusted rim deterrence,
and source availability.

Shot-volume offense adds attempt rates, usage, offensive load, shooting fouls
drawn, touches, drives, offensive rebounds, and location shares. Shot-volume
defense adds defended-attempt volume, matchup attempt suppression, contest
volume, and blocks.

Turnover offense adds turnover types, travels, offensive fouls, drive turnover
rate, live-ball share, and turnover-to-load. Turnover defense adds steals,
deflections, charges, recovered loose balls, and scorer-adjusted matchup
turnovers forced.

Opponent-OREB creation adds offensive rebound chances, contests, defers,
distance, conversion above expectation, offensive box-outs, and height
interactions. Prevention uses the matching defensive rebound and defensive
box-out fields. SelfORB-adjusted TS does not enter the rebound model.

## Model

For each rating season from 2021 through 2026, the model trains only on
five-year windows that end earlier. Chronological inner folds choose ridge,
elastic net, or one histogram boosted tree for every factor and side. The
boosted tree wins every selection in this run.

Each factor produces cross-fitted historical predictions. A fixed ridge maps
those predictions to the part of normal five-year RAPM that Box15 misses. The
combined arm uses all four factor predictions. It does not add factor ratings
with incompatible units.

Every prior then receives the same rating-season possession update:

\[
\hat\beta=(X^TX+P)^{-1}\left[X^T(y-b)+Pc\right].
\]

The evaluation scores identical next-season games for outcome seasons 2022
through 2026. Five thousand whole-game bootstrap draws estimate paired MSE
intervals.

## Factor results

The table reports mean out-of-time R-squared across rating seasons 2021 through
2026.

| Factor | Side | Box15 | Specialist |
| --- | --- | ---: | ---: |
| Shooting TS | Offense | 0.261 | 0.594 |
| Shooting TS | Defense | 0.119 | 0.419 |
| Shot volume | Offense | 0.319 | 0.549 |
| Shot volume | Defense | 0.304 | 0.531 |
| Turnover avoidance | Offense | 0.449 | 0.648 |
| Turnover forcing | Defense | 0.398 | 0.665 |
| Opponent OREB creation | Offense | 0.325 | 0.622 |
| Opponent OREB prevention | Defense | 0.293 | 0.581 |

The specialist beats Box15 in all 112 audited role, exposure, team-change, and
source-coverage groups with at least ten rows. Feature-era drift remains high.
An early-versus-late classifier reaches AUC `0.975`.

## AIO results

Lower RMSE is better.

| Prior after the same RAPM update | RMSE | Correlation | RMSE gain vs Box15 | Paired MSE 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Box15 | 14.374 | 0.362 | -- | -- |
| Box15 plus OREB | 14.361 | 0.364 | 0.014 | [0.187, 0.588] |
| Box15 plus shot volume | 14.361 | 0.365 | 0.013 | [0.122, 0.638] |
| Box15 plus turnover | 14.373 | 0.363 | 0.001 | [-0.150, 0.235] |
| Box15 plus all four | 14.370 | 0.370 | 0.006 | [-0.440, 0.752] |
| Box15 plus shooting TS | 14.383 | 0.367 | -0.008 | [-0.767, 0.297] |

OREB and shot volume each lower paired MSE, but both miss the required `0.05`
points-per-game RMSE gain. The combined model wins three of five season folds,
but its paired interval crosses zero. It does not pass promotion.
Positive paired MSE values in the table favor the challenger.

## Artifacts

- Targets: `historical_factor_targets_v2_6cd7e959eb`.
- Specialist source: `historical_specialist_features_v1_de35da67fb`.
- Tournament: `historical_factor_residual_tournament_v2_c06bdebcd5`.

The artifacts contain derived targets, factor predictions, priors, ratings,
identical-game predictions, folds, 5,000-draw intervals, subgroup errors,
source hashes, and coverage checks. They contain no raw NBA events.
