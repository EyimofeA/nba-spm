# Historical Shooting and Rebound Factor Specialists

## Decision

Keep Box15 as the research AIO prior. The opponent offensive-rebound-prevention
specialist is a useful research challenger, but its `0.014` points-per-game
RMSE gain does not meet the `0.05` promotion threshold. Keep the shooting and
rebound specialists as skill models. Do not combine them into the public AIO.

Season 2027 remains untouched.

## Targets

The target build uses regular-season event rows and observed five-player
offensive and defensive lineups from 2014 through 2026. It fits annual targets
for every season and matched five-year targets for window ends 2018 through
2026. Every factor uses offense and defense player columns plus home advantage.
The ridge penalties remain `3000 / 3000 / 300`.

The shooting target is true shooting percentage in regression form. A field
goal attempt has weight one. A free throw has weight `0.44`. Made two-pointers,
three-pointers, and free throws receive values `1`, `1.5`, and `1 / 0.88`.
The weighted mean therefore equals:

\[
\mathrm{TS}=\frac{\mathrm{PTS}}{2(\mathrm{FGA}+0.44\,\mathrm{FTA})}.
\]

The rebound target uses every resolved missed field goal:

\[
Y_j=\mathbf{1}\{\text{the offense recovered miss }j\}.
\]

Positive defensive impact means a player lowers the opponent's offensive
rebound probability. The project calls this factor **opponent offensive-rebound
prevention**.

The target artifact contains 6,934 annual player rows and 8,618 five-year
player-window rows. The minimum valid-lineup fraction is `95.82%`. Every season
after 2020 has effectively complete factor-lineup coverage. Team-game score
reconstruction matches at least `99.31%` of team-game totals.

## Specialist inputs

The shot model separates mechanisms instead of combining them into one gravity
field.

### Shooting offense

- leave-one-game-out expected points from location zone, assisted status,
  transition context, and final-four-seconds context;
- empirical-Bayes shotmaking above expectation;
- rim, midrange, and three-point attempt shares;
- rim shotmaking above expectation;
- assisted and transition attempt shares;
- SelfORB-adjusted true shooting;
- the frozen Box15 rates.

### Shooting defense

- shot-level NBA defender tags joined by game and event;
- expected points conceded on assigned shots;
- empirical-Bayes points saved above shot expectation;
- expected rim-shot quality conceded;
- scorer-adjusted rim deterrence per 100 matchup possessions;
- a defender-assignment availability field;
- the frozen Box15 rates.

A shot with multiple defender tags gives each tagged defender equal fractional
weight. The assignment source covers `92.55%` to `95.27%` of regular-season
shots from 2018 through 2026. The source does not contain defender distance or
shot clock. The model does not substitute game clock for shot clock.

### Rebound responsibility

- offensive and defensive rebound chances per 100;
- contest and defer shares;
- average rebound distance;
- height-conditioned rebound conversion above expectation;
- offensive and defensive box-outs per 100;
- team and player rebound conversion after a box-out;
- height interactions with box-outs and contest shares;
- source-availability fields;
- the frozen Box15 rates.

The source supplies observed player box-out totals. It does not identify one
responsible box-out player for each missed shot. The result estimates rebound
responsibility from season totals and context. It does not claim event-level
causal responsibility.

## Models and splits

Each rating season from 2021 through 2026 uses only five-year windows that end
before that rating season. The factor tournament compares ridge, elastic net,
and one histogram boosted tree. Chronological inner folds select one family.
The boosted tree wins every factor-side selection.

The frozen Box15 control selects its ridge penalty from
`10, 30, 100, 300, 1000, 3000` inside each past-only training panel. Past
training rows receive leave-one-window-out predictions before the residual
mapping is fit. The factor residual mapping uses one fixed ridge model.

The tournament produces four priors:

1. Box15;
2. Box15 plus the shooting residual;
3. Box15 plus the opponent-OREB-prevention residual;
4. Box15 plus both residuals.

Every prior receives the same season-specific possession update:

\[
\hat\beta=(X^TX+P)^{-1}\left[X^T(y-b)+Pc\right].
\]

`c` contains the statistical prior. `P` contains the fixed
`3000 / 3000 / 300` penalties. The test scores identical next-season games.
The five evaluated outcomes are 2022 through 2026. Five thousand whole-game
bootstrap draws estimate paired MSE intervals.

## Factor results

The table reports the mean out-of-time R-squared across rating seasons 2021
through 2026.

| Factor | Side | Box15 | Specialist |
| --- | --- | ---: | ---: |
| Shooting TS | Offense | 0.261 | **0.554** |
| Shooting TS | Defense | 0.119 | **0.419** |
| Opponent OREB prevention | Offense | 0.325 | **0.620** |
| Opponent OREB prevention | Defense | 0.293 | **0.585** |

The specialist reduces factor MSE in every audited role, exposure, team-change,
and source-coverage group with at least ten rows. The result supports separate
shooting and rebound skill outputs.

Feature-era drift remains severe. A held-out early-versus-late classifier
reaches AUC `0.975`. Tracking, matchup, and box-out availability make the late
era easy to distinguish. The factor gains are stable across 2021 through 2026,
but they are not evidence that the same model would transfer to a different
data regime.

## AIO results

Lower RMSE is better.

| Prior after the same RAPM update | Equal-season RMSE | Mean margin correlation | RMSE change versus Box15 |
| --- | ---: | ---: | ---: |
| Box15 | 14.374 | 0.362 | -- |
| Box15 plus OREB prevention | **14.360** | 0.364 | **-0.014** |
| Box15 plus both | 14.377 | **0.367** | +0.002 |
| Box15 plus shooting | 14.381 | 0.366 | +0.006 |

For Box15 versus the OREB challenger, the paired MSE difference is `+0.405`.
Its 95% interval is `[+0.200, +0.602]`, so the challenger has lower MSE. It wins
four of five seasons. The RMSE gain remains below the frozen `0.05`
points-per-game practical threshold. The candidate does not pass promotion.

Shooting improves the factor targets but does not improve the final AIO. The
factor decomposition captures a cleaner player skill while the one-season RAPM
update already absorbs the part that transfers to team game margins.

## Artifacts

- Historical targets: `historical_factor_targets_v1_f4894bf588`.
- Specialist features: `historical_specialist_features_v1_de35da67fb`.
- Tournament: `historical_factor_residual_tournament_v1_f0b772f6e1`.

The tournament stores factor predictions, priors, ratings, identical-game
predictions, fold metrics, 5,000-draw intervals, source drift, subgroup errors,
hashes, and coverage checks. It does not include raw NBA rows.
