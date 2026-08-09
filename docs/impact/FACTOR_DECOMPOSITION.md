# Factor-Decomposed All-in-One Design

This document defines the next statistical all-in-one challenger. It separates
player inputs, factor impact targets, and the final points-per-100 rating. Read
it with [`ROADMAP.md`](ROADMAP.md) and
[`../modeling/PLAYBOOK.md`](../modeling/PLAYBOOK.md).

## Decision

Build and compare two decompositions:

1. **Eight-head model:** eFG, turnover rate, offensive rebound rate, and free
   throw rate on offense and defense.
2. **Six-head ablation:** true shooting, turnover rate, and rebound rate on
   offense and defense.

The eight-head model is the primary research specification. True shooting
already includes free throws. A model that uses both true shooting and a
separate free-throw head counts part of foul value twice. The six-head version
is valid only when true shooting replaces both eFG and free throws.

Do not replace the direct offensive and defensive statistical models before a
factor model wins on untouched chronological data. Four factors are useful
accounting outcomes, but they do not assign all individual value. Spacing,
screening, off-ball movement, help defense, matchup difficulty, and scheme can
remain in a residual.

## Target contract

Each target is a lineup-adjusted player effect. It is not the player's own box
rate. Positive values always mean better play.

| Side | Head | Raw team outcome | Positive player effect means |
|---|---|---|---|
| Offense | `off_efg` | `(FGM + 0.5 * 3PM) / FGA` | team eFG increases |
| Offense | `off_tov` | `TOV / (FGA + 0.44 * FTA + TOV)` | team turnover rate decreases |
| Offense | `off_oreb` | `OREB / (OREB + opponent DREB)` | team offensive rebound rate increases |
| Offense | `off_ftr` | `FTM / FGA` for scoring value; retain `FTA/FGA` for diagnosis | team free-throw scoring rate increases |
| Defense | `def_efg` | opponent eFG | opponent eFG decreases |
| Defense | `def_tov` | opponent turnover rate | forced turnover rate increases |
| Defense | `def_dreb` | `1 - opponent OREB%` | defensive rebound completion increases |
| Defense | `def_ftr` | opponent `FTM / FGA` | opponent free-throw scoring rate decreases |

Use separate opportunity denominators and reliability weights for each head.
Do not fit unweighted regressions to raw percentages. Preserve the underlying
counts so that the builder can compare weighted ridge, binomial models, and
possession-level encodings without changing the public target names.

For the six-head ablation, replace `off_efg` plus `off_ftr` with `off_ts` and
replace `def_efg` plus `def_ftr` with `def_ts`. Keep turnover and rebound heads.

## Model path

```text
ordered events + ordinal lineups
  -> opportunity counts by lineup segment
  -> 8 factor RAPM targets
  -> player-window feature table
  -> 8 statistical factor models
  -> time-safe factor-to-points calibration
  -> offensive and defensive explained impact
  -> residual model
  -> final offensive, defensive, and net AIO
```

The factor-to-points calibration must use training windows only. Start with a
linear map from the factor RAPM heads to normal offensive or defensive RAPM.
Allow a residual target equal to normal RAPM minus the cross-fitted factor sum.
The residual must be cross-fitted. An in-sample residual would leak the target.

Keep the direct offense and defense models as the production baseline. Compare:

1. direct offense plus defense;
2. six-head factor sum;
3. eight-head factor sum;
4. eight-head factor sum plus residual;
5. a multi-task factor model only after the separate ridge heads pass.

## Input feature contract

All rates use player or team possessions as appropriate. Raw opportunity counts
can set reliability weights. They cannot enter as predictive features. Exclude
age, experience, height, listed position, minutes, games, raw plus-minus,
on/off, and target-derived team ratings from the independent model.

### Offensive eFG

- 2-point and 3-point accuracy and frequency;
- rim, short-midrange, long-midrange, corner-three, and above-break-three rates;
- catch-and-shoot and pull-up accuracy and frequency;
- assisted and self-created scoring shares;
- defender-distance shot splits and expected shot value;
- three-point proficiency, rim-and-three frequency, and shot-profile entropy;
- playtype frequency and points above expectation, with transition separated;
- behavioral creation and spacing interactions.

### Offensive turnovers

- bad-pass, lost-ball, live-ball, travel, and offensive-foul rates;
- turnovers per touch, pass, drive, and creation event;
- assists, potential assists, passes, touches, drives, and time of possession;
- creation load and nonlinear creation-load by turnover-type interactions.

### Offensive rebounds

- offensive rebounds, rebound chances, contested chances, and putbacks;
- rim misses and shot-zone mix, which change rebound opportunity quality;
- paint touches, post touches, cuts, rolls, and interior role load;
- crash-versus-transition indicators when the event data supports them.

### Offensive free throws

- shooting fouls drawn, free-throw attempts, and free throws made;
- drives, rim attempts, paint and post touches;
- isolation, transition, cut, roll, and post-up frequency;
- foul-drawing efficiency conditional on creation load.

### Defensive eFG

- defended field-goal accuracy and volume by zone;
- expected-versus-actual opponent shot value;
- rim deterrence, rim contests, blocks, and recovered blocks;
- closeout and defender-distance splits;
- matchup difficulty and shot-profile mix when IDs are reliable.

### Defensive turnovers

- steals, deflections, loose balls recovered, charges drawn, and recovered blocks;
- forced live-ball turnovers and offensive fouls;
- disruption per foul and role-conditioned defensive activity.

### Defensive rebounds

- defensive rebounds, chances, contests, contested share, and box outs;
- opponent offensive-rebound opportunities and second-chance possessions;
- rim and interior matchup load.

### Defensive free throws

- personal, shooting, transition, clear-path, and offensive fouls separated;
- drives and rim attempts defended;
- contest activity per foul and matchup difficulty.

Use stat-specific empirical-Bayes stabilization and decay. Three-point accuracy
must not use the same history weight as turnovers, drives, or rebound chances.
Learn role from behavior. Do not use listed position. Any team calibration or
on/off input belongs in a separately labeled impact-assisted challenger.

## Versioned benchmark features

Implement public formulas only when the complete formula is available. Give the
formula a source year. Do not imply that an old public formula is the author's
current private version.

Ben Taylor's public 2017 Box Creation formula is:

```text
three_point_proficiency = (2 / (1 + exp(-3PA)) - 1) * 3P%
box_creation_2017 =
    0.1843 * AST
  + 0.0969 * (PTS + TOV)
  - 2.3021 * three_point_proficiency
  + 0.0582 * AST * (PTS + TOV) * three_point_proficiency
  - 1.1942
```

All inputs are per 100 possessions. Taylor reported mean absolute error of 0.90
against hand-tracked opportunities created for players with at least 500 tracked
possessions and 0.77 above 1,000. He also warned that the three-point term may
not transfer to the 1980s. A later Box Creation update is not public, so this
feature must remain `box_creation_2017`.

Taylor's public Offensive Load and creation-adjusted turnover formulas are:

```text
offensive_load_2017 =
    0.75 * (AST - 0.38 * box_creation_2017)
  + FGA + 0.44 * FTA + box_creation_2017 + TOV

creation_adjusted_tov_pct_2017 = TOV_per_100 / offensive_load_2017
```

Offensive Load estimates responsibility, not value. Its assist adjustment is
explicitly ad hoc. Use these as named challenger inputs, not ground truth.

Do not implement a claimed exact Ben Taylor Passer Rating or ScoreVal formula.
The authoritative public pages describe their inputs and intent but do not
publish a complete current equation.

## Lessons from public methods

| Method | Useful design lesson | Do not copy unchanged |
|---|---|---|
| [MAMBA](https://www.teemohoop.com/mamba/Blog%20Post%20Title%20One-mm8gk-cy9wh) | playtype POE, transition value, assist points created, rim points saved, time decay | self-reported win tests use actual minutes and weak rookie handling |
| [MAMBA Four Factor RAPM](https://www.teemohoop.com/mamba/four-factor-rapm) | six factor RAPM heads and a learned scale back to offensive and defensive RAPM | two-year samples are noisy; the reported near-additivity is not independent validation |
| [MAMBA rework](https://www.teemohoop.com/mamba/mamba-reworked-updated) | defense needs rim protection, missed shots against, charges, and constrained interactions | arbitrary caps and smell-test edits need preregistered validation |
| [Ben Taylor: Box Creation](https://fansided.com/2017/08/11/nylon-calculus-measuring-creation-box-score/) | assists, scoring pressure, turnovers, three-point proficiency, and their interaction estimate creation better than assists alone | the published formula was fit to a small hand-tracked sample and has an era warning |
| [Ben Taylor: Offensive Load](https://thinkingbasketball.net/2017/10/16/offensive-load-and-adjusted-tov/) | separate offensive responsibility from efficiency and scale turnovers by creation load | its non-creation assist adjustment is explicitly ad hoc |
| [Ben Taylor: Passer Rating](https://thinkingbasketball.net/2018/07/15/nba-passer-ratings-since-1978/) | passing production and passing ability are different; role changes observed volume | the canonical current equation is not public, so do not claim a replication |
| [Thinking Basketball archive](https://thinkingbasketball.net/archives-2/) | Box Creation, Offensive Load, Passer Rating, spacing, WOWY, and playoff stability are separate concepts | do not collapse talent, role, scalability, and observed impact into one target |
| [BPM 2.0](https://www.basketball-reference.com/about/bpm2.html) | long RAPM targets, per-100 inputs, learned role, and role-dependent coefficients | team adjustment and estimated position make it unsuitable as our independent prior contract |
| [RAPTOR](https://fivethirtyeight.com/features/how-our-raptor-metric-works/) | expected shot value, assisted shots, enhanced assists, contested rebounds, spacing, and box/on-off separation | descriptive and predictive versions use different inputs; historical approximation is a different model |
| [LEBRON](https://www.bball-index.com/lebron-introduction/) | role-based stabilization, a box prior, luck-adjusted RAPM, and explicit impact-versus-talent distinction | proprietary role/stabilization choices are not an independently reproducible benchmark |
| [EPM](https://dunksandthrees.com/about/epm) | stat-specific estimated skills, long RAPM targets, league-relative features, and decayed prior-informed RAPM | current EPM uses age and other context that the user excluded from this model |
| [DARKO](https://www.darko.app/about) | daily stat-specific decay, Kalman-style updating, interactions, and forward prediction | this is the later dynamic layer, not the first static factor prior |
| [PIPM](https://www.bball-index.com/player-impact-plus-minus/) | box interactions plus luck-adjusted on/off and separate wins conversion | team-quality and on/off terms belong outside the independent prior |
| [APBR RAPM discussion](https://apbr.org/metrics/viewtopic.php?t=8859&start=15) | possession-weighted stint regression and priors for low-information players | forum implementations are engineering references, not validation evidence |

## Validation gates

1. Verify event ordering, lineup assignment, score conservation, and factor
   opportunity conservation before any player coefficients.
2. Compare factor coefficients across adjacent three-year windows. A head that
   changes sign or rank erratically is not ready for interpretation.
3. Freeze factor definitions before tuning penalties.
4. Select penalties and factor-to-points coefficients only in older windows.
5. Score identical player rows on chronological outer windows.
6. Report offense, defense, net, and every factor head. Report direct-versus-
   decomposed deltas in RMSE and correlation.
7. Resample whole games for lineup-based target uncertainty. Random model seeds
   do not count as independent runs.
8. Require the eight-head model to beat the direct baseline in at least two
   chronological outer windows before promotion. If only the decomposition is
   useful, publish it as an explanation layer and keep the direct model.

## Current data constraint

The current rich canonical events cover only 2023–24 through 2025–26. That is
enough to build and test the target builder, but not enough to establish a
stable eight-head production model. The 1997–2024 legacy possession cache has
lineups and points but not the event fields needed for eFG, turnover, rebound,
and free-throw outcomes. Historical event ingestion, ideally 2014 onward, is a
promotion dependency. Do not manufacture historical factor labels from player
box totals.
