# CourtSignal Gravity v1

## Decision

CourtSignal Gravity predicts next-season shooting outcomes better than the
published LASER score on the exact public LASER player-season cohort. The
result holds for shooting percentage, attempt volume, contested-shot share,
and an equal-weight composite.

CourtSignal Gravity does not improve player-impact estimation. The feature
worsens Box15, the completed full SPM, and both RAPM-updated AIO models. Keep
the metric as a shooting-skill output. Do not add it to an impact prior.

All results use reused historical diagnostics. Season 2027 remains untouched.

## Metric

The model uses annual three-point shooting splits from 2014 through 2026. It
estimates three player components for every origin season `t`.

For an observation from season `s <= t`, the model assigns this weight:

\[
w_{s,t} = 2^{-(t-s)/3}.
\]

The model selected the three-season half-life from a frozen grid of 1, 2, 3,
and 4 seasons. It selected a 50-attempt empirical-Bayes prior from a frozen
grid of 50, 100, and 200 attempts.

The first component estimates context-adjusted three-point ability:

\[
q_{i,t} =
\frac{\sum_s w_{s,t}M_{i,s} + 50\,p^{context}_{i,t}}
     {\sum_s w_{s,t}A_{i,s} + 50}.
\]

`p_context` uses the player's catch-and-shoot or pull-up mix, defender-distance
mix, and corner or above-the-break mix. Each context rate excludes the player
when it estimates the league comparison rate.

The second component measures time-decayed three-point attempts per 100
offensive possessions. The third component measures the time-decayed share of
three-point attempts with a tight or very tight nearest defender.

Players who appeared in the origin season define the mean and standard
deviation for all three components. The model then takes their arithmetic mean:

\[
Gravity_{i,t} = \frac{z(q_{i,t}) + z(3PA100_{i,t}) +
z(ContestedShare_{i,t})}{3}.
\]

The 2026 leaderboard requires a 2026 player row and at least 50 effective
time-decayed attempts. This reporting filter prevents retired and one-shot
players from dominating the table. The filter does not change model fitting or
benchmark scoring.

## LASER comparison

The benchmark uses the [public LASER table](https://datawrapper.dwcdn.net/FUVAa/1/dataset.csv).
The source contains 2,953 rows from 2014 through 2025. All names resolve to a
canonical player ID. The next-season join produces 2,784 common rows.

The development block uses origin seasons 2014 through 2020. The selection
block uses 2021 and 2022. The diagnostic block uses 2023 through 2025 and
therefore scores outcomes from 2024 through 2026. Both scalar metrics receive
the same weighted ridge calibration before each scored block.

The benchmark weights shooting percentage by next-season three-point
attempts. It weights volume by next-season offensive possessions. It weights
contested share by next-season classified defender-distance attempts. It
weights the composite by next-season offensive possessions. Each outcome uses
a within-season standardized scale.

| Next-season outcome | CourtSignal MSE | LASER MSE | CourtSignal correlation | LASER correlation | MSE difference, CourtSignal - LASER | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3P% | 0.425 | 0.472 | 0.341 | 0.211 | -0.048 | [-0.067, -0.028] |
| 3PA per 100 | 0.498 | 0.574 | 0.687 | 0.595 | -0.076 | [-0.140, -0.012] |
| Contested 3PA share | 0.599 | 0.906 | 0.675 | 0.460 | -0.307 | [-0.403, -0.217] |
| Equal-weight composite | 0.163 | 0.296 | 0.750 | 0.554 | -0.133 | [-0.162, -0.103] |

The 5,000-draw paired bootstrap resamples players within each outcome season.
All four point estimates favor CourtSignal. All four intervals exclude zero.

The comparison tests the published LASER scalar. It does not reproduce the
private LASER model. It does not prove causal spacing impact. It tests
next-season stability on the public LASER cohort.

## Impact-model test

The impact test adds CourtSignal Gravity to the offense side only. It compares
four statistical models:

1. Box15.
2. Box15 plus Gravity.
3. The completed full SPM with 128 offense and 72 defense inputs.
4. The completed full SPM plus Gravity.

Each rating season trains on earlier five-year windows only. Inner folds select
the ridge penalty. Each AIO uses the statistical estimate as the prior mean and
uses the rating season's one-season possession likelihood. The AIO penalties
remain 3000 offense, 3000 defense, and 300 home.

The evaluation predicts next-season game margins for 2022 through 2026. Every
candidate scores the same games. The comparison uses equal-season mean MSE.
The paired 5,000-draw bootstrap resamples whole games within each test season.

| Base model | Base mean RMSE | Plus-Gravity mean RMSE | Base - Plus-Gravity MSE | Paired 95% interval | Base fold wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Box15 | 14.676 | 14.687 | -0.335 | [-0.633, -0.040] | 3 of 5 |
| Full SPM | 14.616 | 14.618 | -0.072 | [-0.133, -0.012] | 3 of 5 |
| Box15 AIO | 14.374 | 14.383 | -0.256 | [-0.409, -0.104] | 5 of 5 |
| Full SPM AIO | 14.410 | 14.411 | -0.040 | [-0.075, -0.005] | 4 of 5 |

A negative MSE difference means the base model performs better. Gravity loses
all four comparisons. All four intervals favor the base model.

The full SPM assigns a small negative standardized coefficient to Gravity in
every rating-season fit. Box15 assigns a positive coefficient because Box15
lacks several correlated shooting fields. Neither pattern transfers to better
game prediction.

## Interpretation

Gravity measures a stable shooting-threat profile. It does not measure total
player impact. The impact prior already contains shooting, scoring, spacing,
and role signals. Gravity mainly repackages that information. The extra feature
adds correlated noise and changes ridge shrinkage.

The negative downstream result provides the useful boundary. CourtSignal can
publish Gravity as a player-skill metric. CourtSignal should keep Box15 and the
full SPM unchanged. Season 2027 can test the already frozen impact candidates;
it should not reopen this feature search.

## Reproducible artifacts

- LASER benchmark: `laser_breaker_v1_75d8ef37c1`
- Impact test: `gravity_spm_challenger_v1_23e07083d6`
- Model code: `src/nba_impact/models/shot_model_suite.py`
- Benchmark runner: `research/run_laser_breaker.py`
- Impact runner: `research/run_gravity_spm_challenger.py`
