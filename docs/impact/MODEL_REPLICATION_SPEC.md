# NBA Impact Model Replication Specification

Status: audit of the checked-in code and pinned artifacts on 2026-08-18.

This document is a replication map. It describes what the repository actually
fits. It does not turn a research artifact into a production claim.

## 1. Model map

There are three different objects that have been called “SPM” or “AIO” in the
repository. They must not be conflated.

| Object | Code path | Output | Status |
|---|---|---|---|
| Normal RAPM | `src/nba_impact/models/rapm.py` | lineup-adjusted offense, defense, net points per 100 | production reference for current 2024–26 scope; descriptive |
| Annual SPM | `src/nba_impact/models/single_season_spm.py` | box/tracking prediction of annual RAPM offense and defense | public reference through 2024; 2025–26 refresh is research null |
| Annual AIO | `src/nba_impact/models/annual_aio_ratings.py` plus `prior_informed_rapm.py` | SPM-centered RAPM posterior | research leaderboard, 2017–24 only |
| Rolling statistical AIO | `statistical_feature_v2` plus `statistical_aio` artifacts | three-season feature model predicting RAPM components | research challenger; not the annual centered-RAPM posterior |

The pinned annual AIO artifact is
`annual_aio_ratings_v1_23c4895f8f` (the later
`annual_aio_ratings_v1_b52b5aecd9` uses the same 2017–24 rating scope with
leave-one-season-out priors). The pinned annual SPM artifact audited here is
`single_season_spm_v1_47b3bd9b17`.

## 2. Normal RAPM: exact input and equation

### Row contract

One row is one possession-level outcome. The model input is not a player box
feature vector. It is:

- `pts`: points scored by the offense on that possession;
- five offensive player IDs;
- five defensive player IDs;
- `home_poss`: `+1` when the home team is the offense and `-1` otherwise;
- `gameid`, `season`, period and possession identifiers for grouping and QA.

Current canonical data are read by `load_current_possessions()` in
`src/nba_impact/models/rapm.py`. The production current policy selects the
terminal ordinal lineup segment for a possession. Historical legacy rows use
the terminal lineup available in the legacy cache. A possession with multiple
lineup segments is not split in the production baseline. Fractional exposure is
a parked sensitivity only.

### Design matrix

For `N` players, the sparse design matrix has `2N + 1` columns:

```text
X[row, player_offense_j] = 1 for each of five offensive players
X[row, N + player_defense_j] = 1 for each of five defensive players
X[row, 2N] = +1 if home team is on offense, else -1
y[row] = points scored on the possession
```

The zero-prior ridge fit in `fit_coefficient_center_path()` starts with
`b = mean(y)` and solves:

```text
P = diag(lambda_off for offense players,
         lambda_def for defense players,
         lambda_home for the home column)

beta = (X'X + P)^-1 X'(y - b)
```

The frozen penalties are `lambda_off = 3000`, `lambda_def = 3000`, and
`lambda_home = 300`. The clean current baseline is therefore a zero-prior,
terminal-lineup ridge, not a prior blend. After fitting, the code recenters the
offense and points-allowed-defense blocks by their possession-weighted means and
adds the removed level back to the intercept. This resolves the separate
offense/defense location ambiguity without changing fitted row predictions.

The output convention in `ratings_table()` is:

```text
offense_per_100 = 100 * beta_offense
defense_per_100 = -100 * beta_points_allowed_defense
net_per_100 = offense_per_100 + defense_per_100
```

The minus sign makes a player who reduces opponent scoring positive on defense.
The normal RAPM design contains no SPM, role, age, height, position, minutes,
games, on/off, BPM, xRAPM, or other external rating feature.

### RAPM target and scope

Annual SPM labels use one-season regular-season Normal RAPM targets. The target
builder fits each season independently with the same `3000/3000/300` penalties.
The current reference run `rapm_v0_01b5084f0a` covers the validated current
2024–26 scope. The historical legacy cache supports research-only annual
targets through 2024. The historical V3 program is not yet a complete 2017--23
RAPM input. A strict 2023 pilot now has ordinal lineups attached for 1,085 of
1,230 regular-season games, while the remaining games and historical seasons
still require the same season-level build and QA. The 2017 possession candidate
also retains eight rejected games.

## 3. Annual SPM: exact target, windows, learners, and features

### Target and split

The target is the matching player-season annual Normal RAPM component:

```text
target_offense = annual RAPM offense_per_100
target_defense = annual RAPM defense_per_100  # positive-good, after sign conversion
target_net = target_offense + target_defense
```

The panel uses `sample_weight = sqrt(min(Poss_Off, Poss_Def))`. This is a
training reliability weight. It is not a model feature. The final audited run
has 6,942 panel rows and 10 leave-one-season-out folds for output seasons
2017–26. For the held-out season `T`, the model trains on every other available
season in the panel. This is a descriptive reconstruction test, not a
next-season forecast. The final leaderboard refit uses every available labeled
season.

The public reference remains the 2017–24 SPM/AIO. The 2014–26 refresh exists as
an artifact, but it did not improve the matched 2017–24 result and had weak
2025–26 defense folds. It must not be described as a promoted 2025/26 model
until a new predeclared confirmation passes.

### Learners

The frozen model factory in `statistical_feature_ablation.py` selects:

- offense: `HistGradientBoostingRegressor(learning_rate=0.03,
  max_iter=250, max_leaf_nodes=7, min_samples_leaf=30,
  l2_regularization=1.0, early_stopping=False, random_state=20260808)`;
- defense: `Ridge(alpha=3000)`;
- both are preceded by median imputation with missing indicators; offense is
  not standardized in the histogram pipeline, while the ridge pipeline uses
  `StandardScaler` after imputation;
- fitting passes the square-root exposure weight described above.

The audited SPM artifact contains 127 offense columns and 68 defense columns.
The exact lists are in Appendix A.

### Stabilization and feature construction

The three-season statistical feature builder aggregates natural counts first,
then computes rates. The expanded feature builder uses empirical-Bayes rate
shrinkage:

```text
rate_EB = (player_numerator + k * league_rate)
           / (player_denominator + k)
```

The default `k` is 500 possessions for the main rate block, 500 touches for
touch denominators, 150 for drives or turnovers, and 100 for other natural
opportunity denominators. Weighted season-relative values use possession
weights. The 2017 Box Creation, Offensive Load, passer, and spacing features
are explicit derived columns. Age, experience, height, listed position,
minutes, games, on/off and target-like ratings are excluded as general inputs.

## 4. Roles: what happened and what did not happen

Roles were added to feature-table candidates, not to the selected SPM/AIO
models.

The behavior-role artifact fits eight clusters from 2014–18 behavior inputs,
then applies the fixed map through 2024. The separate role artifact fits six
offense clusters on 2014–18 and five defense clusters on 2018–21. Roles use
behavior and deployment, not impact. They exclude age, height, position,
minutes, games, efficiency, on/off, RAPM, SPM, and team outcome.

The role output has continuous PCA axes and soft affinities. Hard labels are
descriptive. The stabilized role artifact uses a selected current-season weight
of 0.70 and improves later adjacent hard-role persistence, but it remains a
descriptive display. It is not an impact input.

The role-enabled V2 feature tables contain role axes/affinities and, in one
candidate build, role-by-defense-skill interactions. The selected model run
does not contain them. The role challenger was rejected on the fixed older
comparison; defense roles alone were worse in the 2020 selection fold and
almost neutral in 2021. Therefore the answer to “why do roles not go into SPM
or AIO?” is: they are available as research candidates, but the frozen model
selection did not select them and the evidence did not justify promotion.

The current role maps stop at 2024 because their frozen application artifact
was built and validated through 2024. They must be refit or explicitly extended
through 2025/26 with complete source features, then checked on an untouched
later period. Extending labels mechanically is not enough to claim current-role
validity.

## 5. Annual AIO: exact posterior update

Annual AIO is a centered RAPM posterior. It is not `SPM + RAPM` arithmetic.

`build_leave_one_season_out_annual_spm_priors()` converts held-out SPM
predictions to positive-good offense and defense priors. `build_prior_center()`
maps them into RAPM coefficient signs:

```text
c_off_j = prior_offense_j / 100
c_def_j = -prior_defense_j / 100
```

Each side is then centered by the possession-weighted mean over players in the
RAPM design. Missing player priors are zero before centering. The AIO fit uses
the same design, response, and penalties as Normal RAPM, but its coefficient
center scale is `s = 1`:

```text
beta_AIO = (X'X + P)^-1 [ X'(y - b) + s P c ]
```

The code evaluates `s = 0` and `s = 1` in
`fit_annual_aio_season()`. The former is Normal RAPM. The latter is the AIO
posterior. It returns `aio_offense`, `aio_defense`, and `aio_net`; the exact
identity `aio_net = aio_offense + aio_defense` is checked in the artifact.

The audited annual AIO uses complete-season features and possession outcomes.
It is retrospective, not a preseason forecast. Its 2017–24 scope, legacy
terminal-lineup policy, and absence of calibrated uncertainty are documented in
the run manifest. It is research-only.

The separate `statistical_aio` artifact is a feature model: it predicts RAPM
components from player-season features. It uses the same offense GBM and
defense ridge families but does not itself perform the centered possession-level
posterior update. Its selected feature drops are offense `creation_role` and
defense `shot_profile` plus `turnover_detail` in the parent ablation. Do not
use its name as evidence that a centered AIO posterior was fitted.

## 6. Feature importance and what it means

The repository has no causal coefficient importance for the GBM. The available
importance artifact is a diagnostic grouped and individual permutation audit:
`artifacts/models/statistical_interpretability/statistical_interpretability_v1_94d3f2c24b`.
It refits the frozen model on training windows through 2021 and evaluates the
reused 2024 fold. It is not promotion evidence.

Grouped weighted-RMSE increase when the family is jointly permuted:

| Side | Family | Features | RMSE increase |
|---|---|---:|---:|
| Offense | shooting/scoring/spacing | 90 | 0.2719 |
| Offense | public composites | 8 | 0.1081 |
| Offense | rebounding/screening | 6 | 0.0106 |
| Defense | defensive disruption | 3 | 0.1167 |
| Defense | creation/passing/role | 24 | 0.0816 |
| Defense | rebounding/screening | 6 | 0.0793 |

The largest individual diagnostic features were behavioral passer score,
creation-to-load, era-relative TS, era-relative points, and latest points on
offense; defensive-rebound chances, steals, fouls drawn, recovered blocks, and
average dribbles per touch on defense. Because correlated columns share signal,
individual permutation ranks are not causal feature importance.

## 7. zTS and shot-quality-adjusted passing

zTS is part of the audited annual SPM offense feature list as
`zts_pct_points`. Its implementation is the independent `zts/` project:

```text
zTS = player_TS% - expected_TS_from_player_playtype_mix
```

Playtype expected TS uses league playtype rates and estimated free throws from
Synergy FTFreq. It is available from 2014 onward, uses a 20-possession league
minimum, and excludes final players below 250 minutes. It is a feature, not an
impact metric or an AIO posterior component by itself.

The repository does not currently have a shot-quality-adjusted passing metric
in the production SPM/AIO. The validated assist-quality block is a separate
candidate. It contains `ft_assists_p100_eb`,
`adjusted_assist_points_p100_eb`, `adjusted_potential_assists_p100_eb`, and
`assist_points_per_potential_eb`. It adjusts for teammate free throws and
shrinks rates. It does not estimate the expected shot value of a pass or the
pass's actual-minus-expected shot outcome. The candidate features are not in the
selected SPM lists above.

## 8. What is needed to publish 2025/26 SPM, AIO, and roles

The blocker is not code availability. It is evidence and source completeness.

1. Finish the complete official player-game and source-feature panels for
   2025/26, including every selected offense and defense field. Do not neutral-
   fill missing current defense as if it were measured.
2. Extend the frozen role maps only after the current role descriptors pass
   key, coverage, and stability QA.
3. Build 2025/26 annual Normal RAPM targets from possession and ordinal lineup
   inputs that pass the unchanged QA contract.
4. Fit SPM with the predeclared feature list and compare against the pinned
   2017–24 reference on identical rows and a predeclared untouched confirmation.
5. If an AIO is tested, generate the SPM center without the rated season's RAPM
   labels, fit the centered ridge, and compare it with zero-prior RAPM on the
   same games. Do not tune a center scale after seeing the confirmation.
6. Keep the older 2027 untouched confirmation reserved for promotion. Until
   that passes, expose 2025/26 candidates as research, not public production.

## 9. Reproduction commands and source anchors

The central code paths are:

- RAPM design and fit: `src/nba_impact/models/rapm.py`;
- annual SPM target and held-out fit: `src/nba_impact/models/single_season_spm.py`;
- feature construction and stabilization:
  `src/nba_impact/data/statistical_features.py` and
  `src/nba_impact/data/statistical_features_v2.py`;
- annual SPM prior generation: `src/nba_impact/models/annual_spm_priors.py`;
- centered AIO: `src/nba_impact/models/annual_aio_ratings.py` and
  `src/nba_impact/models/prior_informed_rapm.py`;
- model-family selection: `src/nba_impact/models/statistical_model_comparison.py`;
- importance audit: `src/nba_impact/models/statistical_interpretability.py`;
- role maps: `src/nba_impact/data/behavior_roles.py` and
  `src/nba_impact/data/side_roles.py`;
- zTS: `zts/src/zts.py` and `zts/compute_zts.ipynb`.

The exact pinned feature names below come from
`single_season_spm_v1_47b3bd9b17/run.json`. If code and a stale narrative
disagree, the run manifest and its hashes are the replication source.

## Appendix A — exact audited annual SPM feature lists

### Offense (127)

```text
PTS_p100, AST_p100, TOV_p100, STL_p100, BLK_p100, OREB_p100, DREB_p100, PF_p100,
PFD_p100, FTA_p100, FTM_p100, FG2A_p100, FG2M_p100, FG3A_p100, FG3M_p100, drive_turnovers_p100,
catch_shoot_fga_p100, pull_up_fga_p100, at_rim_fga_p100, short_mid_fga_p100, long_mid_fga_p100,
corner3_fga_p100, arc3_fga_p100, open_fga_p100, wide_open_fga_p100, tight_fga_p100,
very_tight_fga_p100, live_ball_turnovers_p100, bad_pass_turnovers_p100, lost_ball_turnovers_p100,
travels_p100, offensive_fouls_p100, shooting_fouls_drawn_p100, rebound_contests_p100,
rebound_chances_p100, dreb_contests_p100, dreb_chances_p100, recovered_blocks_p100,
fg2_pct, fg3_pct, ft_pct, at_rim_accuracy, short_mid_accuracy, long_mid_accuracy,
corner3_accuracy, arc3_accuracy, catch_shoot_accuracy, catch_shoot_3_accuracy,
pull_up_accuracy, pull_up_3_accuracy, open_accuracy, open_3_accuracy, wide_open_accuracy,
wide_open_3_accuracy, tight_accuracy, tight_3_accuracy, very_tight_accuracy,
very_tight_3_accuracy, at_rim_frequency, short_mid_frequency, long_mid_frequency,
corner3_frequency, arc3_frequency, drive_turnovers_per_drive, live_ball_turnover_share,
bad_pass_turnover_share, lost_ball_turnover_share, usage_events_p100, true_shooting_pct,
shot_quality_average, arc3_accuracy_eb, arc3_frequency_eb, at_rim_accuracy_eb,
at_rim_frequency_eb, bad_pass_turnover_share_eb, catch_shoot_3_accuracy_eb,
catch_shoot_accuracy_eb, corner3_accuracy_eb, corner3_frequency_eb, drive_assists_per_drive_eb,
drive_fta_per_drive_eb, drive_points_per_drive_eb, drive_turnovers_per_drive_eb,
elbow_points_per_touch_eb, fg2_pct_eb, fg3_pct_eb, ft_pct_eb, live_ball_turnover_share_eb,
long_mid_accuracy_eb, long_mid_frequency_eb, lost_ball_turnover_share_eb, open_3_accuracy_eb,
open_accuracy_eb, paint_points_per_touch_eb, passes_per_touch_eb, post_points_per_touch_eb,
potential_assists_per_touch_eb, pull_up_3_accuracy_eb, pull_up_accuracy_eb,
short_mid_accuracy_eb, short_mid_frequency_eb, tight_3_accuracy_eb, tight_accuracy_eb,
very_tight_3_accuracy_eb, very_tight_accuracy_eb, wide_open_3_accuracy_eb,
wide_open_accuracy_eb, AST_p100_relative, FG3A_p100_relative, FTA_p100_relative,
PTS_p100_relative, TOV_p100_relative, arc3_frequency_relative, at_rim_frequency_relative,
drives_p100_relative, potential_assists_p100_relative, shot_quality_average_relative,
true_shooting_pct_relative, shooting_proficiency_2017_eb, box_creation_2017_eb_p100,
offensive_load_2017_eb_p100, assist_to_load_2017_eb, turnover_to_load_2017_eb,
creation_to_load_2017_eb, behavioral_passer_score_v1, crafted_spacing_stable_v1,
zts_pct_points
```

### Defense (68)

```text
PTS_p100, AST_p100, TOV_p100, STL_p100, BLK_p100, OREB_p100, DREB_p100, PF_p100,
PFD_p100, FTA_p100, FTM_p100, FG2A_p100, FG2M_p100, FG3A_p100, FG3M_p100, drives_p100,
drive_points_p100, drive_assists_p100, drive_fta_p100, touches_p100, front_court_touches_p100,
paint_touches_p100, post_touches_p100, elbow_touches_p100, time_of_possession_p100,
passes_made_p100, passes_received_p100, potential_assists_p100, secondary_assists_p100,
assist_points_created_p100, rebound_contests_p100, rebound_chances_p100, dreb_contests_p100,
dreb_chances_p100, recovered_blocks_p100, fg2_pct, fg3_pct, ft_pct, drive_points_per_drive,
drive_assists_per_drive, drive_fta_per_drive, passes_per_touch, potential_assists_per_touch,
paint_points_per_touch, post_points_per_touch, elbow_points_per_touch, usage_events_p100,
true_shooting_pct, avg_seconds_per_touch, avg_dribbles_per_touch, dfg_attempts_p100,
dfg_diff_pct_eb, rim_dfga_p100, rim_diff_pct_eb, rim_points_saved_p100, deflections_p100,
charges_drawn_p100, contested_2pt_p100, contested_3pt_p100, def_loose_balls_recovered_p100,
matchup_opponent_adjusted_points_saved_p100_eb, matchup_fga_suppressed_vs_scorer_p100_eb,
matchup_shotmaking_points_saved_vs_scorer_p100_eb, matchup_three_pa_suppressed_vs_scorer_p100_eb,
matchup_turnovers_forced_vs_scorer_p100_eb, matchup_assists_suppressed_vs_scorer_p100_eb,
matchup_shooting_fouls_prevented_vs_scorer_p100_eb, matchup_blocks_p100
```

### Provenance and caveat

The source run is `artifacts/models/single_season_spm/single_season_spm_v1_47b3bd9b17/run.json`.
The exact code and input hashes are stored there. The feature lists describe
the audited artifact. They do not mean every source is complete in every season,
and they do not by themselves justify publishing 2025/26.
