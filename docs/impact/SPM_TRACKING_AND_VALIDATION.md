# SPM tracking, shot context, and validation contract

## Decisions

1. Use next-season team wins as the primary downstream SPM feature-selection
   gate. Keep next-season RAPM reconstruction as a mechanism diagnostic.
2. Keep single-season RAPM at `3000 / 3000 / 300` as the current reference, not
   as a claim that its alpha is optimal for next-season team wins.
3. Keep `rim_points_saved_p100`; expose its definition clearly.
4. Add Gabriel-style event stops as `event_stops_p100`, not Stop%.
5. Do not claim a modern shot-level nearest-defender model until a permitted
   row-level source exists.

## What shot data exists

The local exact event panel has shot result, point value, coordinates/shot
distance, zone, game clock, score context, assisted/transition indicators, and
the ten players on court. It does **not** have a primary or nearest defender on
each modern shot.

The public aggregate tracking panel has separate player-season tables for:

- nearest-defender distance buckets;
- catch-and-shoot and pull-up attempts;
- dribble and touch summaries;
- shot zones and exact PBP location;
- transition and other playtype shares.

Those are marginal aggregates. They do not reveal whether the same attempt was,
for example, a corner three after one dribble with a defender 2.5 feet away.
Joining the marginals as if they were shot rows would invent information.

A public 2014-15 NBA shot-log snapshot does contain `SHOT_DIST`,
`CLOSE_DEF_DIST`, closest-defender identity, shot clock, dribbles, touch time,
shot result, and two/three-point value. It is useful for a historical prototype
and functional-form checks, not as the production source for 2026 players.

The current historical prototype is
`historical_shot_quality_2015_v1_c5258e797c`. It matched 99.40% of 128,069
tracking shots to local play-by-play location/fast-break context within five
seconds. It also matched shooter and nearest-defender height on 99.83% of rows.
The first 675 games train the model and the later 229 games test it:

| Inputs | Test log loss | AUC | Log-loss gain vs location only |
|---|---:|---:|---:|
| Exact location + shot distance/value | 0.6450 | 0.6337 | 0.0000 |
| + nearest-defender distance | 0.6381 | 0.6504 | 0.0069 |
| + shot clock, dribbles, touch time, period, home, fast break | 0.6340 | 0.6576 | 0.0111 |
| + period clock and shooter-minus-defender height | **0.6336** | **0.6591** | **0.0115** |

The last arm follows the inputs described in
[Krishna Narsu's KOBE article](https://fansided.com/2015/09/28/introducing-kobe-a-measure-of-shot-quality/).
It is not a literal reproduction. KOBE used separate close- and long-shot
logistic models, while this prototype uses one histogram GBM and a later
within-season test split.

An earlier prototype included the PBP `assisted` flag and produced an absurd
0.91 AUC. That run is marked invalid: an assist is recorded only after a make,
so it leaks the target. The valid model uses zero dribbles and short touch time
as the pre-shot pass proxy and never matches rows on make/miss.

## Required shotmaking model

When a permitted modern row source is available, fit a cross-fitted make model
on shot rows:

```text
P(make) = model(
    exact location / shot distance,
    2P or 3P,
    nearest-defender distance,
    shot clock,
    dribbles and touch time,
    catch-and-shoot / pull-up,
    assisted / after-pass,
    transition / fast break,
    score and period context
)

shotmaking_i = 100 / offensive_possessions_i
               * sum_shots point_value * (made - P(make))
```

Train the expectation without player identity first. Evaluate later seasons and
team changers, calibrate by zone and defender-distance bucket, and shrink the
player residual by shot opportunities. Add shooter random effects only in a
separate predictive model. For defense, estimate contest responsibility and
shot suppression separately from make/miss residuals.

This distinction matters. Nylon Calculus found that non-rim nearest-defender
FG% is mostly unstable, while rim contest volume and rim results have more
year-to-year signal. Nearest defender is also not always the responsible
defender in a help scheme.

## Rim points saved and event stops

The raw basketball quantity is:

```text
rim_points_saved = 2 * rim_DFGA * (normal_rim_FG% - defended_rim_FG%)
rim_points_saved_p100_raw = 100 * rim_points_saved / defensive_possessions
```

For example, 300 attempts at 55% against a 60% normal expectation save 30
points. The current frozen SPM uses the stabilized version:

```text
reliability = rim_DFGA / (rim_DFGA + 100)
stabilized_diff = reliability * (defended_rim_FG% - expected_rim_FG%)
rim_points_saved_p100 = -2 * rim_DFGA_p100 * stabilized_diff
```

`stabilized_diff` is represented as a proportion in the final line; the code's
percentage-point representation divides by 100. Positive values are better.
The NBA defense dashboard's comparison FG% is the shooters' normal percentage
for that defended shot-distance category. It is not a row-level location and
contest xFG estimate.

## Self-offensive-rebound-adjusted true shooting

Gabriel's annual sheets contain the observed `SelfOReb` count. CourtSignal now
calculates:

```text
self_oreb_adjusted_true_shooting_pct =
    points / (2 * (FGA + 0.44 * FTA - SelfOReb))
```

This gives back the possession cost of misses the shooter recovered. It does
not claim that every self-rebound creates an identical follow-up opportunity.
The field is a research candidate and does not silently alter the frozen SPM.

Gabriel's source calls this event sum `Stops`:

```text
steals + recovered blocks + charges drawn + offensive fouls drawn
```

CourtSignal exposes the possession rate as `event_stops_p100`. It is not Dean
Oliver Stop%, because it does not allocate team defensive rebounds or forced
misses. Its four components already exist separately, so promotion requires a
downstream ablation rather than assuming that the sum adds information.

## What the FanSided work adds

The FanSided/Nylon Calculus metric is Andrew Johnson's Player Tracking
Plus-Minus (PT-PM), not a Kostya Medvedovsky metric. PT-PM regressed RAPM on
box and SportVU inputs. The reusable defensive ideas are rim attempt/contest
volume, stabilized opponent rim accuracy, steals, fouls, and contested
rebounding. CourtSignal already covers those families.

Kostya's directly relevant work is different: DARKO is a forward-looking daily
time-series model, and his stabilization/padding method shrinks same-season
rates toward a league prior based on opportunity. That supports our same-season
empirical-Bayes feature stabilization; it does not justify leaking previous
season outcomes into a season-Y descriptive feature.

Useful sources:

- [PT-PM at midseason](https://fansided.com/2017/01/11/nylon-calculus-pt-pm-halfway-nba-season/)
- [Projecting rim protection](https://fansided.com/2015/02/04/made-miss-projecting-rim-protection-metrics/)
- [Shot defense: metrics versus actions](https://fansided.com/2017/01/12/nylon-calculus-shot-defense-metrics-actions/)
- [Kostya Medvedovsky's stabilization/padding method](https://kmedved.com/)
- [Example public NBA shot-log schema](https://github.com/chineseballer06/nba_data_analysis)

## RAPM alpha evidence

The direct current-RAPM nested test selected `4500 / 4500 / 1000` on the
2024-to-2025 game-margin fold, then that candidate lost to `3000 / 3000 / 300`
when trained on 2024--25 and checked on 2026. Broader symmetric/asymmetric
searches also failed their future-game gates. A separate 300/1000/3000/10000
test concerns the defense **SPM ridge**, not RAPM alpha, and must not be cited as
RAPM tuning evidence.

So the candid answer is: **no, single-season RAPM alpha has not been optimized
for next-season team wins.** The work supports 3000 as a stable retrospective
reference under game-margin tests. A downstream label bakeoff would need to
refit annual RAPM labels and the SPM for each frozen alpha candidate, then apply
the resulting season-Y SPM to Y+1 team minutes and wins. That is a distinct
predictive experiment; it should not silently redefine the public
retrospective RAPM estimand.

For SPM feature and role selection, use:

```text
team_strength_(Y+1) = 5 * weighted_average(
    player_rating_Y,
    observed_player_team_minutes_(Y+1)
)
```

and correlate with team win percentage in Y+1. The observed minutes make the
test an oracle-minutes retrodiction. Projected minutes are still required for a
true preseason forecast.
