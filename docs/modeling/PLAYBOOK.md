# NBA Statistical Modeling Playbook

This is the default procedure for RAPM, all-in-one, win probability, draft,
injury, contract, and related NBA models. It adapts Andrej Karpathy's debugging
recipe to statistical sports modeling. It does not replace model-specific theory.

Source: https://karpathy.github.io/2019/04/25/recipe/

## 1. Write the model contract first

State these items before code:

- estimand: what quantity the model estimates;
- grain: possession, stint, game, player-game, or player-season;
- prediction time: what information is available when the estimate is made;
- target and units;
- population: regular season, playoffs, or both;
- intended use: description, retrodiction, forecast, ranking, or decision support;
- forbidden information and known leakage paths;
- primary metric, uncertainty method, and promotion rule.

Do not call retrodiction a forecast. Do not use observed future lineups, minutes,
availability, or results in a pregame model.

## 2. Understand and validate the data

Inspect real rows before model code. Verify:

- primary-key uniqueness and exact duplicate rows;
- game, team, player, season, and source-ID joins;
- source coverage and incomplete recent seasons;
- score, points, minutes, possession, and lineup conservation;
- player changes, trades, multi-team rows, and replacement IDs;
- regular-season, play-in, and playoff labels;
- missingness and schema changes by season;
- feature availability at the stated prediction time;
- outcome and feature distributions by season and role.

Keep immutable raw data, source hashes, retrieval times, schema versions, and a
quarantine table. Never relax a quality gate to make a model run.

## 3. Build the full evaluation path with simple baselines

Run data → features → fit → predictions → metrics → artifact before complex work.
Use baselines that can expose broken logic:

- constant or league-average prediction;
- home-court or team-only model;
- minutes or possession-only reliability baseline;
- zero-prior ridge RAPM;
- previous-season rating;
- simple linear or logistic model.

A complex model must beat the appropriate simple baseline on identical rows.

## 4. Test model correctness

For all models:

- recover known coefficients from synthetic data;
- test home/away sign reversal and player-ID permutation invariance;
- test that each possession has five offensive and five defensive exposures;
- use negative controls and shuffled targets;
- verify that future information cannot change an earlier prediction;
- compare a clear loop implementation with optimized vectorized code when logic
  is difficult;
- inspect the largest errors and influential rows.

For neural models only:

- verify the initial loss;
- overfit a tiny batch;
- inspect the exact tensor and label immediately before the model;
- record train/validation curves, gradients, convergence, and prediction dynamics;
- train on cloud compute, not the local Mac.

## 5. Use NBA-safe validation

- Split by time. Never use random possession or player-season folds as the main
  evidence.
- Keep outer test seasons frozen. Tune only inside earlier seasons.
- Resample whole games for possession and win-probability uncertainty.
- Cluster player-season evaluation when repeated rows belong to one player.
- Score every candidate on identical eligible rows.
- Report regular season and playoffs separately when sample size permits.
- Report Brier/log loss/calibration for probabilities; game-margin error for RAPM;
  and minute-weighted next-season prediction for player ratings.
- Report coverage, uncertainty, coefficient stability, and important subgroups.
- Seeds measure optimizer variance. Seeds are not independent seasons.

## 6. Add one change at a time

Before each run, write:

- the hypothesis;
- the exact changed feature, loss, prior, or architecture;
- the expected direction and metric;
- the failure condition.

Freeze all other choices. Save the full configuration and code/data hashes. Log
null and negative results. Do not tune after reading an outer test result.

## 7. Handle regularization and priors carefully

- Select ridge penalties inside chronological training data.
- Compare common and separate offensive/defensive penalties.
- Make prior strength depend on reliability only when the rule is predeclared.
- Build priors from information available before the target window.
- Exclude on/off, plus-minus, team rating, or other target-derived features from
  an independent statistical prior.
- Evaluate the prior alone and its downstream RAPM blend.
- Check low-minute players, rookies, traded players, and missing-feature eras.

## 8. Promote the smallest model that passes

A production candidate must:

1. improve the primary metric across the required chronological folds;
2. have uncertainty consistent with a real gain;
3. not materially regress calibration or a high-value subgroup;
4. pass leakage, invariance, conservation, and coverage tests;
5. have versioned inputs, configuration, outputs, and rollback artifacts;
6. justify every additional feature and operational dependency.

Research models can remain available after they fail promotion. Production stays
simple until additional complexity earns its cost.
