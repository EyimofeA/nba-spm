# Win Probability Architecture Ladder

Last updated 2026-08-08. This is the preregistered nonlinear research queue,
not a claim that a larger model is better.

## Research question

Does recent causal event history, nonlinear state interaction, or a learned
lineup representation improve calibrated NBA win probability beyond the current
logistic model on unseen seasons?

The first test is whether the current state is already close to sufficient. A
transformer that only sees score, time, and possession repeated over a sequence
has little reason to beat a well-specified nonlinear tabular model. Sequence
models become scientifically interesting when they receive strictly historical
information such as foul/bonus state, timeouts, lineup changes, possession
outcomes, event type, and substitution context.

## Evidence reviewed

- Maddox, Sides, and Harvill build an NBA dynamic Bayesian estimator over time
  and score-differential cells, blend pregame information over time, and compare
  against ESPN: https://arxiv.org/abs/2207.05114
- A 2024 UCLA thesis compares logistic, random forest, neural, recurrent, and
  recurrent mixture-density NBA WP models. Neural models were slightly better
  and its recurrent mixture-density model scored best, but preprocessing data
  were proprietary: https://escholarship.org/uc/item/1n78759j
- Generic TCN evidence supports causal dilated convolutions as a strong, efficient
  sequence baseline before assuming recurrence is necessary:
  https://arxiv.org/abs/1803.01271
- Temporal Fusion Transformers combine static covariates, observed time-varying
  inputs, recurrent local processing, gating, and attention. That input taxonomy
  is relevant, although multi-horizon forecasting is not identical to NBA WP:
  https://arxiv.org/abs/1912.09363
- NBA2Vec learns player embeddings by predicting possession outcomes from the ten
  players on court: https://arxiv.org/abs/2302.13386
- Set Transformer supplies a permutation-invariant attention architecture for
  interacting player sets: https://proceedings.mlr.press/v97/lee19d.html
- Jenkins proposes distributional WP temporal-difference learning plus neural
  Shapley attribution for player value. It belongs after WP calibration is stable:
  https://www.sloansportsconference.com/research-papers/deep-reinforcement-learning-for-nba-player-valuation-a-temporal-difference-approach-with-shapley-attribution
- XGBoost and CatBoost are high-value tabular challengers; ordered boosting is
  especially relevant when categorical player/team features could leak targets:
  https://arxiv.org/abs/1603.02754 and https://arxiv.org/abs/1706.09516

## Fixed model ladder

| Stage | Model | Input | Purpose | Initial parameter budget |
|---|---|---|---|---:|
| 0 | Logistic | current state + pregame context + possession | frozen baseline | existing |
| 1 | Dynamic Bayesian / GAM | same tabular state | smooth interpretable nonlinearity | rejected: worse both folds |
| 2 | HistGBM / XGBoost / CatBoost | same tabular state | strongest cheap interaction test | HistGBM rejected: worse both folds |
| 3 | Residual MLP | same tabular state | test generic nonlinear function | feed-forward proxy rejected; residual untested |
| 4 | Causal TCN | last 32 possession/event tokens + static context | local and multi-scale history | 4x32 channels |
| 5 | GRU | identical sequence and parameter budget to TCN | recurrence control | hidden 64 |
| 6 | Causal transformer | identical sequence | attention only if history adds value | 2 layers, width 64, 4 heads |
| 7 | Set/lineup encoder | unordered home and away player embeddings | interactions and lineup strength | embeddings 16-32 |
| 8 | Recurrent mixture-density / distributional TD | state sequence and final margin/value distribution | uncertainty and later credit assignment | research only |

Do not jump directly to Stage 6. If Stage 2 cannot beat logistic on identical
states, a transformer win is more likely to come from leakage, tuning freedom, or
different inputs than from architecture.

## Causal input tiers

1. **State parity:** exact current logistic inputs. This isolates architecture.
2. **Rich current state:** add team fouls/bonus, timeouts, on-court lineups, and
   availability only when each is known at prediction time.
3. **Sequence:** last 32 completed possession/event tokens. A token may contain
   duration, points, outcome type, foul/timeout/substitution flags, offense side,
   and lineup change. The current possession outcome is always masked.
4. **Player representation:** frozen prior-season embeddings first; jointly
   learned embeddings only in a separately labeled research run.

## Evaluation contract

- Use identical prediction states and labels across candidates.
- Chronological folds on the now-available three-season silver data:
  - train/tune within 2023-24, test 2024-25;
  - freeze choices, train through 2024-25, test 2025-26.
- Report Brier, log loss, AUC, calibration intercept/slope, and reliability bins.
- Resample whole games. Neural candidates run five fixed seeds; seeds measure
  optimizer noise and never count as independent evidence.
- Report tipoff, halftime, last six minutes, close last two minutes, overtime,
  regular season, and playoffs separately where sample size permits.
- Enforce prefix invariance, home/away symmetry, bounded probabilities, causal
  masking, and identical source-game coverage.

## Promotion gate

A challenger advances only if:

1. whole-game Brier delta is below zero in both outer folds with 95% intervals
   excluding zero in the pooled result;
2. log loss is non-inferior and calibration slope is between 0.9 and 1.1 before
   any shared post-hoc calibration;
3. no predeclared high-leverage subgroup materially regresses;
4. the gain survives a state-parity comparison, or any extra inputs are clearly
   named as the reason for the gain;
5. inference cost is acceptable for the eventual API.

Production remains the smallest model that passes. Larger models stay available
as research and ensemble candidates even when they fail promotion.

The two-fold gate is cleared for starter-free rolling margin plus rest and causal
possession-start control. Adding prior-season starter RAPM is unresolved, so the
smaller model remains the frozen Stage 0 control for every later comparison.

Run `wp_stage1_v1_7e6c77d51a` rejects the fixed five-knot spline GAM and bounded
HistGBM on both folds. Their pooled candidate-minus-logistic Brier deltas are
+0.00214 and +0.01303, respectively, with intervals entirely above zero. Do not
retune them on an outer test season. Proceed to the fixed residual MLP; a later
sequence-model gain must be attributed to causal history unless state-parity also
improves.

Run `wp_mlp_v1_7a7825bf09` rejects the available 64×64 feed-forward proxy across
five fixed seeds. The pooled Brier penalty is +0.03171 [0.02619, 0.03745], and
nine of ten fits reach the frozen epoch cap. PyTorch is unavailable, so this does
not test residual connections. Preserve the null rather than installing/tuning
after seeing the outer folds. Build and validate causal history tokens next.

This ladder is paused. The frozen logistic model is good enough for the current
platform, playoff evidence is sample-limited, and neural training should not run
on the local Mac. Revisit sequence architectures only on cloud compute after a
causal token contract exists and higher-priority impact work is complete.
