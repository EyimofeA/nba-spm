# Predictive player skills

Status: research current-skill estimates. These are not RAPM, SPM, AIO, causal
credit, or playing-time forecasts.

## Estimand

For each player and skill, estimate current underlying ability after combining
past seasons with the observed part of the named season. The public impact
models remain separate. Parameter selection ends in 2024, centers and residual
age curves are refit through 2025, and 2026 is a reused diagnostic and display
season. Season 2027 is never loaded.

## Raw observations

The frozen run contains 34 skills from the Gabriel player sheets and pinned
playtype, tracking, matchup-defense, and statistical feature artifacts.

| Group | Skills |
| --- | --- |
| Shooting | free throws; rim, short-mid, long-mid, corner-three, above-break-three, total-three, catch-and-shoot-three, and pull-up-three accuracy; shot quality; shotmaking above expectation; zTS |
| Creation | assist points created; potential assists; rim assists; three-point assists; passing efficiency; turnover and live-ball-turnover rates; drive creation; rim pressure; free-throw pressure; Offensive Load |
| Rebounding | offensive and defensive rebound rates; contested-rebound conversion proxy |
| Defense | rim deterrence; rim points saved; non-rim shot suppression; matchup-adjusted points saved; turnovers forced; foul discipline; deflections; recovered blocks |

Every definition, numerator, denominator, scale, direction, source family, and
opportunity field is stored in `skill_definitions.parquet`. Tracking neutral
fills are not counted as observed: the builder joins the original DFG, rim-DFG,
and hustle keys as observation masks.

## Candidate estimators

Let `x_{p,t}` be a player's raw rate in season `t`, `n_{p,t}` the opportunities,
`m_t` the opportunity-weighted league center, `k` the empirical-Bayes prior
strength, and `h` the half-life. For target season `T`, the historical weight is

```text
w_t = 1                         career EB
w_t = 2^((t - cutoff) / h)      time-decayed EB
```

For rate and binomial skills, the estimate is

```text
(sum_t w_t n_p,t x_p,t + k m) / (sum_t w_t n_p,t + k).
```

Continuous skills use the same opportunity-weighted shrinkage. A minimum-
exposure arm replaces thinner histories with the league center. The age arm
fits a ridge regression to strictly forward residuals using `(age - 27)` and
`(age - 27)^2`; binomial residuals use log-odds. Age is selected only if it
beats the otherwise identical time-decayed estimator in at least four of six
development seasons.

The five registered arms are previous-season raw, career EB, time-decayed EB,
time-decayed EB plus age, and role-conditional support. Role conditioning was
not fit because no frozen pre-season role label covers every 2019–24 fold.

## Selection and scoring

Each development season from 2019 through 2024 is predicted using only earlier
seasons. The grid is:

- prior strength: 25, 100, 250, 500 opportunities;
- half-life: 1, 2, 3, 5 years;
- minimum exposure: 0, 25, 100;
- age-ridge penalty: 10, 100, 1000.

Shooting proportions use grouped-binomial log loss as the primary score and
Brier score as the secondary score. Other skills use opportunity-weighted RMSE
then MAE. Fold scores are averaged equally by season. The selected parameters
are then frozen. For each displayed season, the model first builds a preseason
estimate from earlier seasons. When the age arm wins, the selected age residual
is applied to that preseason estimate. The named season then updates it:

```text
preseason precision = selected prior strength + decayed historical exposure
posterior = (preseason estimate * preseason precision
             + current source estimate * current opportunities)
            / (preseason precision + current opportunities).
```

For the 2026 current estimate, age curves and league centers use data only
through 2025. The 2026 observations are the update and reused diagnostic.
Historical trajectories are post-hoc stabilized histories under parameters
selected through 2024; they are not forecasts that were issued at the time.

For 2026 free throws and total three-point shooting, the localhost page also
shows the same posterior after each reconciled regular-season game:

```text
(preseason precision * preseason estimate + cumulative makes)
/ (preseason precision + cumulative attempts).
```

The game series stops when cumulative makes and attempts exactly equal the
frozen annual source row. Playoffs are excluded. If the totals do not reconcile,
the game series is not rendered. Missed games are gaps, not zero-attempt failures.

SelfORB-adjusted TS is calculated directly from points, FGA, FTA, and the
source's `SelfOReb` count. It is not currently part of the 34-skill stabilized
run above; it enters the next skill run only after its source coverage and
future-season stability pass the same gate.

## Frozen result

Run `predictive_player_skills_2026_v1_a7eb0386fe` contains 235,212 player-skill-
season rows and 558 current players. It estimates 97.24% of current player-skill
rows; a row stays missing when the player has neither current nor historical
evidence. Of the current players, 303 have a 2026 source observation for all 34 skills.
Across skills, one selected career EB, 20 selected time-decayed EB, and 13
selected time-decayed EB plus age.

The localhost UI fetches a 34-skill index and one compact shard for the selected
player. Those files stay under `web/local-data/` and are excluded from the
production build.

## Limits

- 2026 is reused evidence, not untouched confirmation.
- Defensive tracking and matchup outcomes are observational, not isolated
  defender responsibility.
- Shotmaking and matchup fields already contain source-level shrinkage before
  this temporal stabilization.
- Offensive and defensive rebound opportunity fields are not valid bounded
  binomial trials, so those skills use rate RMSE/MAE rather than grouped-
  binomial likelihood.
- The reported binomial standard error is an empirical-Bayes approximation.
  Continuous-skill uncertainty is not yet calibrated.
