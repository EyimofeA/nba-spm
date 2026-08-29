# Shot model suite v1

## Decision

Keep the possession-context expected-shot model as a research skill model. Keep
the new shooting-threat and teammate-spacing fields as descriptive outputs.
Do not add shooting threat to Box15 or the AIO prior. The controlled challenger
worsened next-season game prediction after the same RAPM update.

Season 2027 was not loaded. The 2026 tests reuse development evidence. They do
not promote a public model.

## Sources restored and verified

The build uses the pinned regular-season event, possession, lineup, and player
sheet sources.

- 654,376 non-heave shots from 2024 through 2026;
- 787,579 possessions from 3,907 games;
- 946,768 possession-lineup segments;
- 6,942 unique player-seasons from 2014 through 2026;
- complete event-context joins for the shot panel;
- no Season 2027 rows.

The 2018 player sheet contained 64 repeated Corey Brewer rows with identical
values for every field used here. The builder verifies equality and keeps one
row. It rejects conflicting duplicate totals.

## Expected shot quality

The location control uses the existing identity-free logistic model. Its inputs
are continuous court coordinates, distance, angle, discrete zone, two-versus-
three-point status, period, game clock, score margin, and home status.

The context challenger adds only information known when the shot occurs:

- elapsed time since the possession began;
- transition;
- putback;
- second chance;
- possession after a turnover;
- observable finish class: transition, putback, cut, drive, pull-up, post,
  spot-up, or other.

The model excludes player, team, defender, and lineup identity. It excludes
make or miss, points, assist attribution, and every post-shot event. An assist
flag would leak the result because the current event feed records assists only
on makes.

The chronological split trains on 2024, calibrates on 2025, and scores 2026.
Both models use logistic regression and later-season isotonic calibration.

| 2026 split | Location Brier | Context Brier | Difference | Location log loss | Context log loss | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All shots | 0.230752 | **0.228662** | **-0.002090** | 0.654134 | **0.648918** | **-0.005216** |
| Rim | 0.214987 | **0.211286** | **-0.003701** | 0.620475 | **0.611148** | **-0.009327** |
| Non-rim | 0.237007 | **0.235557** | **-0.001450** | 0.667490 | **0.663906** | **-0.003584** |

Whole-game bootstrap comparisons cover 1,228 games and 5,000 draws.

| Loss | Context minus location | 95% interval | Probability context is better |
| --- | ---: | ---: | ---: |
| Brier | -0.002079 | [-0.002325, -0.001831] | 100% |
| Log loss | -0.005040 | [-0.005671, -0.004401] | 100% |

Possession context clearly improves shot-outcome estimation. This result does
not establish player impact or causal shot creation.

## CourtSignal shooting threat

The public LASER description identifies release type, defender distance, shot
clock, location, touch time, correlation-aware context weighting, small-sample
smoothing, and an era-relative volume curve. It does not publish the joint-bin
data or exact numeric weights. CourtSignal does not label this metric LASER.

The available 2014–26 player sheets supply three complete three-point context
families:

1. catch-and-shoot versus pull-up;
2. very tight, tight, open, and wide-open defender-distance bins;
3. corner versus above-the-break location.

For player `i`, context family `d`, and bin `b`, the builder first calculates a
same-season leave-one-player-out league rate:

```text
p[-i,s,d,b] = (league makes - player makes) /
              (league attempts - player attempts)
```

It applies those rates to the player's attempt mix. It combines the three
family expectations with equal weights on the log-odds scale. Equal weights
are explicit because marginal public totals cannot identify the correlations
among joint shot contexts.

```text
logit(context_expected_3P%) = logit(league_3P%)
  + mean_d(logit(context_expected_3P%_d) - logit(league_3P%))
```

The ability estimate uses 100 context-matched prior attempts:

```text
ability_3P% = (3PM + 100 * context_expected_3P%) / (3PA + 100)
```

The final threat field has points-per-100 units. It does not use an unpublished
S-curve.

```text
shooting_threat_p100 = 3 * 3PA_per_100 * (ability_3P% - league_3P%)
```

The annual metric's mean year-to-year correlation is 0.425 among players with
at least 100 attempts in both seasons. Its 2025 leaders include Malik Beasley,
Zach LaVine, Stephen Curry, and Norman Powell. Those names provide a qualitative
sanity check against the public LASER article. The scale and formula remain
different.

## Teammate spacing

For every exact lineup segment, the builder averages the other four players'
annual shooting-threat values. It then takes a segment-duration-weighted mean
for each player-season.

```text
teammate_spacing[i] =
  sum_segments(seconds * mean(shooting threat of other four)) /
  sum_segments(seconds)
```

Missing low-exposure shooters receive zero, which is the season-relative
neutral point on this threat scale. The output describes recorded teammate
shooting. It is not a causal spacing effect or a reproduction of BOOST.

## Box15 challenger

The preregistered challenger adds only `shooting_threat_p100` to the offensive
side of Box15. Defense remains unchanged. Both priors use chronological outer
folds, training windows that end before the rating season, inner alpha
selection, the same player-possession weights, and the same one-season
`3000 / 3000 / 300` RAPM update. They score identical future games.

| Candidate | Equal-season MSE | RMSE | Mean margin correlation |
| --- | ---: | ---: | ---: |
| Box15 AIO | **207.421** | **14.402** | **0.3616** |
| Box15 plus shooting threat AIO | 208.088 | 14.425 | 0.3591 |

Box15 minus the challenger equals `-0.667` MSE. The 5,000-draw paired whole-game
interval is `[-1.101, -0.226]`. Box15 wins four of five folds. Reject the
challenger as an AIO input.

The standardized shooting-threat coefficient is positive in every fitted
offensive prior, from 0.491 to 0.541. The feature contains basketball signal.
The RAPM-updated model already captures enough of that signal through Box15,
so adding it worsens transfer.

## What remains unavailable

- exact nearest-defender distance on each modern shot;
- shot-clock state for the 2014–26 aggregate history;
- joint release, contest, location, touch-time, and clock bins;
- arena calibration;
- a valid shot-level scorer-defender assignment;
- an untouched season for promotion.

The next shot-model step should add one of these missing measurements. More
transformations of the same marginal totals are unlikely to improve the AIO.

## Artifacts

- `shot_model_suite_v1_2494cca535`
- `box15_shooting_threat_v1_029abd8c06`

Method references:

- [Beyond the Box Score: How Good Are You Really At Making Shots](https://hardscreenherald.substack.com/p/beyond-the-box-score-how-good-are)
- [LASER: A Metric for Capturing Shooting Threat and Spacing](https://hardscreenherald.substack.com/p/laser-a-metric-for-capturing-shooting)
- [Shot Quality in High Definition: The Play-by-Play Upgrade](https://hardscreenherald.substack.com/p/shot-quality-in-high-definition-the)
