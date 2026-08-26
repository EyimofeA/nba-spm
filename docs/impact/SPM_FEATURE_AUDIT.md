# Five-year SPM feature audit

Status: diagnostic research. No public model or site data changed.

## Scope and decision

This numerical audit covers the exact 126-offense/50-defense base matrix saved
with `spm_target_horizon_full_v1_f0777db1d4`. The later feature-research model
declares 130 offense and 72 defense fields, but its extended training matrix was
not persisted in the consolidated repository. Its four added offense/defense
families therefore appear in the source map and prior family tests, but not in
the individual permutation table below. That is a reproducibility defect to fix
before another feature-research fit.

Do not select features from in-sample fit or a single error score. Keep the
current five-year SPM frozen while we replace redundant encodings through
chronological family ablations.

The saved five-year offense specification has genuine duplication. Across the exact
2014--23 five-year panel, 16 selected pairs have absolute correlation at least
0.95, eight exceed 0.98, and five exceed 0.995. The worst cases are raw and
era-relative versions of the same statistic:

| Raw | Re-encoding | Correlation |
| --- | --- | ---: |
| at-rim frequency | era-relative at-rim frequency | 0.9995 |
| assists per 100 | era-relative assists per 100 | 0.9991 |
| free-throw attempts per 100 | era-relative FTA per 100 | 0.9976 |
| turnovers per 100 | era-relative turnovers per 100 | 0.9973 |
| points per 100 | era-relative points per 100 | 0.9972 |
| true shooting | era-relative true shooting | 0.9939 |
| shot quality | era-relative shot quality | 0.9893 |

Dropping one side of those pairs did not produce a clean winner over five
chronological next-season folds:

| Offense specification | Features | Weighted MAE | Weighted RMSE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 126 | **1.0784** | 1.4000 | 0.4628 | 0.3516 |
| Drop redundant raw fields | 117 | 1.0797 | **1.3954** | 0.4599 | 0.3494 |
| Drop redundant era-relative fields | 117 | 1.0795 | 1.4054 | **0.4678** | **0.3553** |

The raw-drop version improved RMSE in all five folds but reduced Pearson and
Spearman in all five. That is exactly why RMSE cannot be the sole gate: the
variant became slightly less extreme without ranking players better. The
relative-drop version improved Pearson in four of five folds, but worsened MAE
in three and RMSE in three. Neither replaces the frozen model.

## What the public models actually use

This section separates exact public specifications from broad author
descriptions. It does not infer hidden coefficients.

### BPM 2.0

Basketball-Reference publishes the full structure. BPM uses adjusted per-100
box statistics: points, threes made, assists, turnovers, offensive and
defensive rebounds, steals, blocks, personal fouls, field-goal attempts, and
free-throw attempts. Coefficients vary with an estimated position and an
estimated offensive role. Team adjusted efficiency is reconciled back into
player values after the player regression. It was trained against four
five-year Bayesian RAPM samples.

Useful for CourtSignal:

- interactions between skill and offensive role;
- explicit team reconciliation;
- a small, auditable base feature set;
- multi-year RAPM labels.

Not useful as-is: BPM's position estimate uses listed position and team shares.
Our current contract excludes position, height, and age as general predictors.

Source: [Basketball-Reference BPM 2.0 methodology](https://www.basketball-reference.com/about/bpm2.html).

### RAPTOR

FiveThirtyEight described feature families rather than publishing a complete
coefficient table. Its box/player-tracking side included traditional stats,
assisted field goals, value-weighted assists, offensive-rebound types, time of
possession, spacing proxies such as contested threes, nearest-defender shot
frequency and results, opponent points and rebounds by guarded position, and
induced offensive fouls. A separate on/off component adjusted courtmates and
opponents.

The important method is its rejection test. Candidate variables had to predict
long-term RAPM in two non-overlapping three-year samples, with special attention
to players who changed teams. Nearest-defender opponent three-point percentage
looked good in-sample and failed out-of-sample, so it was excluded.

Useful for CourtSignal:

- test mechanisms on later seasons and team changers;
- prefer shot frequency/contest responsibility to noisy opponent shot making;
- keep box/tracking and on/off evidence as separate components before blending.

Sources: [archived RAPTOR introduction](https://web.archive.org/web/20191015125623/https:/fivethirtyeight.com/features/introducing-raptor-our-new-metric-for-the-modern-nba/) and [FiveThirtyEight RAPTOR data](https://github.com/fivethirtyeight/data/tree/master/nba-raptor).

### xRAPM

The public explainer identifies standard box statistics, play-by-play-derived
steal and block types, shot defense, and deflections as inputs to its statistical
prior. That prior then centers lineup RAPM. The author does not publish a full
feature/coefficient table on the explainer, so a more exact list would be
speculation.

Useful for CourtSignal:

- recovered block and steal-type features;
- defended-shot responsibility and deflections;
- statistical prior followed by a lineup likelihood, which matches our AIO
  architecture.

Source: [xRAPM explainer](https://xrapm.com/short_desc/xRAPM_explainer.html).

### PIPM

PIPM is not just a box model. Its published construction combines:

1. a box prior;
2. luck-adjusted on/off;
3. luck-adjusted on-court team rating.

The box portion uses pace-adjusted per-36 points, offensive and defensive
rebounds, assists, steals, blocks, turnovers, fouls, free-throw attempts,
two-point attempts, and three-point attempts, plus an average-team component
and a games/starts role term. Offense and defense have separate coefficients.

Useful for CourtSignal:

- separate offense and defense mappings;
- treat on/off and on-court team strength as distinct evidence;
- luck-adjust shooting before using team or on/off results.

Sources: [original PIPM description](https://fansided.com/2018/01/11/nylon-calculus-introducing-player-impact-plus-minus/) and [updated BBall Index PIPM page](https://www.bball-index.com/player-impact-plus-minus/).

## What the tutorial adds

Andrew Patton's [SPM tutorial](https://github.com/anpatton/basic-nba-tutorials/blob/main/spm/how_to_make_spm_R.md)
is useful mainly as a failure demonstration. Its first OLS design is singular
because it includes algebraically redundant fields such as total rebounds with
offensive and defensive rebounds, plus overlapping shooting totals. The revised
design removes some of those identities.

Keep these ideas:

- inspect the correlation matrix before fitting;
- remove exact identities;
- inspect coefficient stability for linear models;
- show out-of-sample predictions, not just fitted values.

Do not copy these parts:

- player-level leave-one-out validation on one pooled cross-section;
- unregularized OLS on correlated box statistics;
- treating leave-one-out coefficient variation as calibrated uncertainty;
- using minutes weights without tying them to the precision of the target.

LOOCV is a split rule, not an evaluation metric. For this project, chronological
next-season folds are the correct outer split because deployment moves forward
in time. MAE, RMSE, Pearson, Spearman, calibration, and downstream game/team
prediction answer different questions inside those folds.

## Current five-year SPM importance

The audit refit the frozen offense GBM and defense ridge on every five-year
window ending before the rating season. It predicted the next season's
one-year, zero-prior RAPM for 2019--23. Season 2027 was not accessed.

Equal-fold baseline means:

| Side | Weighted MAE | Weighted RMSE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Offense | 1.078 | 1.400 | 0.463 | 0.352 |
| Defense | 0.933 | 1.181 | 0.283 | 0.227 |

Grouped permutation results are more trustworthy than individual results when
features overlap:

| Side | Family | MAE increase when shuffled | Pearson drop | Positive MAE result by fold |
| --- | --- | ---: | ---: | ---: |
| Offense | shooting/scoring/spacing | 0.110 | 0.239 | 5/5 |
| Offense | rebounding/screening | 0.020 | 0.009 | 5/5 |
| Offense | public composites | 0.002 | 0.109 | 3/5 |
| Defense | disruption | 0.076 | 0.138 | 5/5 |
| Defense | creation/passing/role | 0.066 | 0.050 | 5/5 |
| Defense | foul pressure | 0.053 | 0.062 | 5/5 |
| Defense | rebounding | 0.047 | 0.071 | 5/5 |

Offensive creation/passing and ball-security families had negative average MAE
importance despite near-zero rank effects. That is a pruning candidate, not
proof that passing or ball security has no basketball value. The individual
GBM importances are heavily split among substitute shot variables.

The strongest individual defense dependencies were fouls drawn, touches,
free-throw attempts, assist points created, recovered blocks, steals, and
defensive rebound chances. These are predictive role proxies inside the model;
they are not causal defensive credit. The strange prominence of offensive
variables in the defense ridge is a reason to test a constrained defense feature
set rather than narrate those coefficients as defense skill.

## Next experiment

Run frozen, family-level refits under the same five chronological folds:

1. compact offense: one representation per raw/EB/relative concept;
2. offense without creation/passing and ball-security families;
3. compact defense with direct defensive activity, rebounding, matchup, and
   role-control fields separated;
4. compare all variants on player next-season scores, team changers, and
   downstream game/team prediction.

Require directionally consistent results across MAE, rank correlation, and the
downstream AIO/game test. Do not promote a feature set because it wins RMSE by
shrinking variance.

Exact outputs: [`run.json`](../../artifacts/research/five_year_spm_feature_audit/five_year_spm_feature_audit_v1_4172cf5408/run.json),
[`correlated_pairs.parquet`](../../artifacts/research/five_year_spm_feature_audit/five_year_spm_feature_audit_v1_4172cf5408/correlated_pairs.parquet),
[`group_permutation_importance.parquet`](../../artifacts/research/five_year_spm_feature_audit/five_year_spm_feature_audit_v1_4172cf5408/group_permutation_importance.parquet), and
[`redundancy_pruning_ablation.parquet`](../../artifacts/research/five_year_spm_feature_audit/five_year_spm_feature_audit_v1_4172cf5408/redundancy_pruning_ablation.parquet).
