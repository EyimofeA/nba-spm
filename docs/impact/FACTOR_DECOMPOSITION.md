# Factor-Structured All-in-One Design

This document defines the conceptual and feature structure for the statistical
all-in-one. It does not require a separate RAPM target for every basketball
factor. Read it with [`ROADMAP.md`](ROADMAP.md) and
[`../modeling/PLAYBOOK.md`](../modeling/PLAYBOOK.md).

## Decision

Keep the primary supervised targets simple:

- three-year offensive RAPM;
- three-year defensive RAPM;
- net RAPM is the sum of the two predictions.

Use the basketball factors as:

1. feature families;
2. grouped model diagnostics;
3. the public explanation layer;
4. optional auxiliary targets or Factor RAPM research.

Do not require eight independently trained target models for the first AIO.
Factor RAPM remains useful research, but it is not a dependency for the direct
statistical model or prior-informed RAPM.

## Mental model

The Dean Oliver accounting factors remain the top-level team outcomes:

| Side | Shooting | Ball security | Rebounding | Free throws |
|---|---|---|---|---|
| Offense | create and convert valuable shots | avoid turnovers | extend possessions | draw and convert free throws |
| Defense | suppress shot quality and conversion | force turnovers | finish possessions | avoid valuable fouls |

This is an ontology, not a claim that the categories are independent. Creation,
spacing, role, transition, and playtype connect several outcomes. True shooting
can summarize shooting plus free throws. If free throws are shown separately,
use eFG for the shooting lane so that the explanation does not double count.

## Direct model path

```text
player-window box + tracking + playtype data
  -> stat-specific stabilization and era adjustment
  -> basketball feature families
  -> direct offensive RAPM model
  -> direct defensive RAPM model
  -> offensive + defensive = net AIO
  -> grouped contribution and counterfactual diagnostics
```

For nonlinear models, grouped feature contributions are explanations of the
model prediction. They are not causal player credit. Correlated variables can
move contribution between groups. Report grouped permutation results alongside
tree attribution when the explanation is important.

## Feature families

### Shooting and spacing

- 2-point and 3-point accuracy and frequency;
- rim, short-midrange, long-midrange, corner-three, and above-break-three rates;
- catch-and-shoot, pull-up, assisted, and self-created shooting;
- defender-distance and shot-quality splits;
- expected shot value and actual-minus-expected shot making;
- shooting proficiency, spacing, rim-and-three frequency, and profile entropy.

### Creation and passing

- Box Creation, Offensive Load, assists per Load, and creation per Load;
- potential assists, assist points created, secondary assists, and passes;
- touches, time of possession, dribbles, drives, and drive passes;
- playtype frequency, transition creation, and points above expectation;
- behavioral role interactions.

### Turnovers and ball security

- turnovers per Load, touch, pass, drive, and creation event;
- bad-pass, lost-ball, live-ball, travel, and offensive-foul rates;
- turnover-type by creation-load interactions.

### Offensive rebounding

- offensive rebounds, chances, contests, and putbacks;
- miss location and shot profile, which change rebound opportunity quality;
- paint touches, post touches, cuts, rolls, and interior role load;
- crash-versus-transition behavior when event data supports it.

### Free-throw pressure

- free-throw attempts and shooting fouls drawn;
- drives, rim attempts, paint touches, and post touches;
- isolation, transition, cut, roll, and post-up frequency;
- foul drawing conditional on creation load.

### Defensive shot suppression

- defended accuracy and volume by zone;
- expected-versus-actual opponent shot value;
- rim attempts defended, deterrence, contests, blocks, and recovered blocks;
- matchup difficulty, closeout, and defender-distance splits.

### Defensive disruption

- steals, deflections, loose balls recovered, and charges drawn;
- forced live-ball turnovers and offensive fouls;
- disruption per foul and role-conditioned activity.

### Defensive possession finishing

- defensive rebounds, chances, contests, contested share, and box outs;
- opponent offensive-rebound opportunities and second-chance possessions;
- foul type, drives defended, rim attempts defended, and contest-to-foul tradeoff.

## Versioned public benchmark features

Implement public formulas only when the full formula and scaling population are
known. Give each formula a source and version. Public formulas are candidate
features, not ground truth.

### Ben Taylor formulas published by CraftedNBA

The public 2017 Box Creation formula is:

```text
three_point_proficiency = (2 / (1 + exp(-3PA_per_100)) - 1) * 3P%

box_creation_2017 =
    0.1843 * AST_per_100
  + 0.0969 * (PTS_per_100 + TOV_per_100)
  - 2.3021 * three_point_proficiency
  + 0.0582 * AST_per_100 * (PTS_per_100 + TOV_per_100)
    * three_point_proficiency
  - 1.1942
```

Taylor reported mean absolute error of 0.90 against hand-tracked opportunities
created for players with at least 500 tracked possessions and 0.77 above 1,000.
He warned that the three-point term may not transfer to the 1980s. A later
update is not public, so retain the source year in the feature name.

The public Offensive Load formula is:

```text
offensive_load_2017 =
    0.75 * (AST_per_100 - 0.38 * box_creation_2017)
  + FGA_per_100 + 0.44 * FTA_per_100
  + box_creation_2017 + TOV_per_100
```

Offensive Load estimates responsibility, not value. Taylor described part of
the assist adjustment as ad hoc.

### CraftedNBA Passer Rating

[CraftedNBA](https://craftednba.com/glossary) publishes:

```text
z(Load)
+ 3.00 * z(AST / Load)
+ z_positional(AST / Load) / 1.75
- 2.00 * z(TOV / Load)
+ z(Creation / Load) / 2.00
+ z(Height) / 5.00
```

The result is converted to a 1–10 scale. CraftedNBA describes this formula as
**inspired by** Ben Taylor's Passer Rating. It is not documented as Taylor's
exact current formula.

The equation is not reproducible until the implementation freezes:

- the standardization population and season scope;
- whether standard deviations are minutes- or possession-weighted;
- the minimum sample and padding policy;
- the position definitions and positional comparison population;
- the final 1–10 transformation.

Keep two explicit variants:

1. `crafted_passer_rating_v1`: exact documented terms after the missing scaling
   rules are defined.
2. `behavioral_passer_rating_v1`: no height and no listed position. Replace the
   positional term with an older-window behavioral-role cluster or omit it.

Both variants are eligible as distinct challengers. Do not also expose raw height
or listed position as general model columns. The exact variant remains blocked
until canonical height and position metadata and a frozen standardization
contract exist. Never silently call the behavioral variant CraftedNBA Passer
Rating.

### Other CraftedNBA candidates

| Metric | Use |
|---|---|
| Shooting Proficiency | implement as a versioned volume-and-accuracy interaction |
| Spacing | benchmark the published formula; compare with defended 3PA and shot-location features |
| PTS_GAINED / PTS_SAVED | recreate from zone-relative expected points after the exact opportunity contract is frozen |
| AstUsg | retain as a simple passing-role ratio |
| RAD/g, DFG%, NDFG%, FGDiff%, deflections | prioritize for defensive feature canonicalization |
| Shooting Quality | use our own transparent padding and career blend; the glossary does not fully specify theirs |
| Portability | research output, not a primary input; it includes external impact and position terms |
| CraftedOPM / CraftedDPM | external ensemble benchmarks only; never train on them as independent inputs |

## Input rules

- Use possession or natural-opportunity rates.
- Use raw counts only for reliability weights and stabilization.
- Exclude age, experience, height, listed position, minutes, and games from the
  general predictive columns. A predeclared composite such as
  `crafted_passer_rating_v1` can include height and positional normalization as
  part of its published formula, but must be tested as its own challenger.
- Exclude raw plus-minus, on/off, RAPM, LEBRON, DARKO, BPM, and other target-like
  impact metrics from the independent prior.
- Label a separate impact-assisted challenger if target-like inputs are tested.
- Fit standardization, padding, role clusters, and league baselines inside the
  chronological training period only.
- Give efficiency and rate statistics their own stabilization/decay schedules.

## Evaluation

Compare feature blocks against the frozen direct offense and defense models.
Do not judge a feature by whether individual rankings look familiar.

1. State the expected basketball mechanism before the run.
2. Add one predeclared family at a time.
3. Use identical rows and chronological outer windows.
4. Tune only inside older data.
5. Require repeated out-of-sample improvement, not multiple random seeds.
6. Report RMSE, correlation, calibration, and role/volume subgroups.
7. Retain null results and feature provenance.

The existing 2022–24 tracking-era folds have already been inspected repeatedly.
New CraftedNBA-derived features can be implemented and explored there, but they
cannot earn a strong production claim without new target seasons or a frozen
historical backtest that was not used during feature design.

## Matchup-defense factor implementation

Run `matchup_defense_features_v1_09829b48c8` implements six positive-good,
scorer-adjusted defensive factors from 2018–25 primary-defender assignments:

- field-goal attempts suppressed per 100 matchup exposures;
- two-point-equivalent shot-making points saved per 100;
- three-point attempts suppressed per 100;
- turnovers forced above scorer expectation per 100;
- assists suppressed per 100;
- shooting fouls prevented per 100.

Each expected rate excludes the defender being scored. The builder centers each
factor within season. It then shrinks exposure rates with 500 matchup exposures
and shot-making with 200 field-goal attempts. The fields do not use minutes,
games, age, height, position, team rating, plus-minus, or on/off.

Same-season descriptive correlation with defensive RAPM is strongest for
shot-making points saved: 0.407–0.457 in every 2018–24 season. Turnovers forced
is 0.198–0.234. These are inspected associations, not held-out model gains.
Attempt suppression has a small negative association, which warns that scheme
and matchup role remain. Do not combine these fields into the frozen SPM until
new confirmation data exists.

## Optional Factor RAPM branch

Factor RAPM can still estimate lineup-adjusted effects on team eFG/TS,
turnovers, rebounds, and free throws. Use it for research and explanation after
historical event data is available. It is not the required label for the first
all-in-one.

## Exact current annual offense implementation

The annual builder uses one season. Main per-100 rates are stabilized as
`r*player_rate + (1-r)*league_rate`, where `r = exposure/(exposure+500)`.
Generic event/opportunity rates use
`(player_numerator + strength*league_rate)/(player_denominator+strength)`;
strength is 500 for touches, 150 for drives and turnovers, and 100 otherwise.
Era-relative fields subtract the possession-weighted season mean.

The active public-inspired block contains
`shooting_proficiency_2017_eb`, `box_creation_2017_eb_p100`,
`offensive_load_2017_eb_p100`, `assist_to_load_2017_eb`,
`turnover_to_load_2017_eb`, `creation_to_load_2017_eb`,
`behavioral_passer_score_v1`, and `crafted_spacing_stable_v1`.

The passer score is
`z(Load) + 3*z(AST/Load) - 2*z(TOV/Load) + 0.5*z(Creation/Load)`, with
possession-weighted z scores clipped to [-4, 4]. It intentionally excludes
positional and height terms. Spacing is stabilized 3PA per 100 times 1.5 times
stabilized 3P%, minus league eFG.

Project zTS uses
`TS = PTS/(2*(FGA+0.44*FTA))`, estimated playtype FTA equal to
`FT_frequency*playtype_possessions*2`, expected TS equal to the sum of playtype
share times league playtype TS, and `zTS = player_TS - expected_TS`. It is stored
in percentage points as `zts_pct_points`.

## Public-source boundary and missing tracking

- MAMBA supplied design candidates: playtype POE, rim points saved, charges,
  unassisted makes, and assist points created. Its published draft says offense
  was stronger than defense; it does not specify a complete defensive set.
- RAPTOR supplied comparison ideas including enhanced assists/rebounds,
  contested threes, turnover/foul context, fast-break starts, defended shots,
  positional opponent outcomes, and defensive distance traveled. Only a subset
  has clean analogues in the current table. RAPTOR on/off is deliberately
  excluded from the independent prior.
- LEBRON supplied the role/archetype stabilization idea, not transition POE or a
  tracking list. Published LEBRON explicitly did not use tracking. Our current
  league-level empirical Bayes is not yet LEBRON-style archetype stabilization.

Separate local DFG, rim-defense, and hustle tables contain defended-shot volume
and differential, rim points saved inputs, deflections, charges, contested
shots, and loose-ball activity. They are not yet canonicalized or used by the
clean annual SPM. The exact active columns are in each annual SPM `run.json`.

The first canonical defensive block now implements ten source-level fields:
defended attempts per 100, stabilized overall DFG differential, rim DFGA per
100, stabilized rim differential, rim points saved per 100, deflections,
charges, contested 2-point shots, contested 3-point shots, and defensive loose
balls recovered. These are local reproducible measurements. They are not
claimed as exact proprietary BBall Index grades.

The public BBall Index glossary page currently contains no metric definitions
or formulas. Do not reverse-engineer or invent exact BBall Index labels. Add a
BBall Index-named feature only when its public definition is reproducible. Use
independently named source statistics and cite their formulas otherwise.
