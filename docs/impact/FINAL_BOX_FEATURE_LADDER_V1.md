# Final BoxPIPM Feature Ladder

## Decision

Keep the 15-feature BoxPIPM-style prior as the research AIO prior. No cumulative
feature addition reduced equal-season next-season game-margin MSE after the same
single-season RAPM update.

This result closes the current SPM feature search. It does not promote the model
to production. Season 2027 remains the untouched confirmation season.

## Model contract

Each statistical prior uses chronological five-year RAPM labels. A rating for
season `t` trains only on windows that end before `t`. Offense and defense use
separate ridge regressions. Each training fold selects its ridge penalty from
`10, 30, 100, 300, 1000, 3000` using earlier windows inside that fold.

Every prior then enters the same season-`t` RAPM likelihood:

\[
\hat\beta=(X^TX+P)^{-1}\left[X^T(y-b)+Pc\right].
\]

`c` contains the offense and defense statistical priors in RAPM coefficient
units. `P` contains penalties `3000 / 3000 / 300` for offense, defense, and home
advantage. The likelihood uses only season-`t` possession outcomes and lineups.
The game test uses the following season. All candidates score identical games.

The experiment scores five reused folds, with rating seasons 2021 through 2025
and outcome seasons 2022 through 2026. The primary score is the equal-season
mean of whole-game squared margin errors. The paired intervals use 5,000
whole-game bootstrap draws within outcome season.

## Frozen cumulative ladder

The feature ladder added each family to all earlier families. Its order was
frozen before the run:

1. Box15 control;
2. shooting efficiency;
3. shot profile and spacing;
4. creation and passing;
5. load and security;
6. defensive events;
7. shot defense;
8. contesting and rebounding;
9. matchup defense.

The complete 127-offense and 68-defense ridge model was a ceiling control. It
was not eligible for selection. Cumulative results depend on this frozen order.
The later matchup result does not isolate the marginal value of matchup fields.

## Results

Lower MSE and RMSE are better.

| AIO prior | Equal-season MSE | RMSE | Mean margin correlation | Probability best |
| --- | ---: | ---: | ---: | ---: |
| Box15 | **207.421** | **14.402** | 0.362 | **57.08%** |
| Through matchup defense | 207.537 | 14.406 | **0.369** | 38.80% |
| Through shot profile and spacing | 207.719 | 14.412 | 0.361 | 0.94% |
| Through shot defense | 207.719 | 14.412 | 0.363 | 3.14% |
| Through shooting efficiency | 207.768 | 14.414 | 0.360 | 0.00% |
| Through defensive events | 208.105 | 14.426 | 0.361 | 0.00% |
| Through contesting and rebounding | 208.109 | 14.426 | 0.361 | 0.00% |
| Full completed ridge ceiling | 208.443 | 14.438 | 0.366 | 0.04% |
| Through creation and passing | 208.541 | 14.441 | 0.358 | 0.00% |
| Through load and security | 208.602 | 14.443 | 0.358 | 0.00% |

Box15 minus each challenger has a negative paired MSE difference when Box15 is
better. It beats shooting efficiency in all five folds with interval
`[-0.507, -0.185]`. It beats the full ridge ceiling in four of five folds with
interval `[-2.007, -0.022]`. The matchup difference is unresolved at `-0.116`,
with interval `[-0.944, +0.737]`. No challenger has a lower point MSE, so none
passes the frozen selection rule.

## Interpretation

A post-selection audit keeps the fitted Box15 models fixed. It permutes one
feature family at a time within each rating season, reruns only the RAPM update,
and measures the increase in next-season game-margin MSE.

| Box15 family | Feature slots across both sides | Mean MSE increase | Mean correlation drop |
| --- | ---: | ---: | ---: |
| Disruption and fouls | 8 | 7.063 | 0.0359 |
| Shooting and scoring | 14 | 6.076 | 0.0335 |
| Creation and security | 4 | 3.059 | 0.0159 |
| Rebounding | 4 | 1.361 | 0.0052 |

All four groups increase MSE in all five permutation repeats. This measures
model dependence. It does not identify causal value. Correlated features divide
and substitute for one another.

The largest stable standardized offense coefficients are `FG2A_p100` (-1.319),
`PTS_p100` (+1.080), `FG2M_p100` (+0.957), `OREB_p100` (+0.852), and
`FG3M_p100` (+0.691). The largest defense coefficients are `PFD_p100` (+0.745),
`STL_p100` (+0.728), `FTA_p100` (-0.699), `DREB_p100` (+0.494), and
`BLK_p100` (+0.431). Defense signs mix basketball skill, role, and target
correlation. They are not clean defensive mechanisms.

## Leaderboard correction

The original run emitted ratings for every player in the five-year matrix. That
table included 894 rows with zero 2026 offensive or defensive possessions. It
must not serve as a current-season leaderboard. The interpretation artifact
adds `active_2026_leaderboard.parquet`, requires positive exposure on both
sides, and has 100% player-name coverage. This reporting fix does not change any
fit, game prediction, or model-selection result.

## Artifacts

- Model comparison: `final_box_feature_ladder_v1_8bb26f12e7`.
- Interpretation and corrected leaderboard:
  `final_box_interpretability_v1_652799efb6`.
- Frozen contract: `research/experiments/final_box_feature_ladder_v1.yml`.

The artifacts contain fitted fold models, priors, ratings, identical-game
predictions, fold metrics, bootstrap draws, coefficients, permutation results,
hashes, and QA metadata. Actual future lineups act as exposure weights in this
diagnostic. The results do not form a deployable pregame forecast.
