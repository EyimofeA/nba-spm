# Win Probability Model Card

Last updated: 2026-08-08. This is the human-readable source of truth for the
current NBA win-probability lane.

Nonlinear and sequence candidates are preregistered in `WP_ARCHITECTURES.md`.

## Estimand and split

The model estimates the home team's probability of winning after a recorded
play. Frozen logistic comparisons use two chronological folds: train on 2023–24
and test on 2024–25, then train on 2024–25 and test on 2025–26. Training states
are sampled every 30 seconds; terminal states are excluded. Model selection uses
Brier score, log loss, AUC, calibration, checkpoint metrics, and paired
whole-game bootstraps.

## Implemented features

| Family | Inputs | Timing rule |
|---|---|---|
| Game state | home score differential, regulation/OT seconds remaining, elapsed fraction, OT flag | post-action only |
| Interactions | score / sqrt(minutes + 1), score × elapsed fraction | current state only |
| Elo | pregame Elo difference, Elo × sqrt(time fraction) | same-date games batch-update afterward |
| Starter strength | sum of frozen prior-season RAPM for each official starting five; home minus away | 2024–25 uses ratings through 2023–24; 2025–26 through 2024–25 |
| Team context | exponentially updated point-differential difference, rest-day advantage, and time interactions | computed before any result from the date; rest capped at seven days |
| Possession start | home or away control, control / sqrt(time + 1), control × elapsed fraction | score uses only completed prior possessions; the current possession outcome is excluded |

Unrated starters receive the centered RAPM mean of zero. Actual minutes, final
box statistics, current-game results, and future availability are not features.

## Models and verdicts

All current variants are `StandardScaler` + logistic regression (`C=1`, LBFGS).

| Variant | 2024–25 Brier | 2025–26 Brier | Verdict |
|---|---:|---:|---|
| State + Elo | 0.15502 | 0.14961 | retained comparison |
| + prior starter RAPM | 0.15496 | 0.14908 | inconclusive in both folds |
| + starter-free rolling margin and rest | **0.15302** | **0.14777** | frozen Stage 0 |
| + rolling context and starter RAPM | 0.15378 | 0.14724 | larger research variant |

Starter-free context beats Elo in both outer folds. Its paired whole-game Brier
deltas are -0.00201, 95% interval [-0.00365, -0.00042], on 2024–25 and -0.00189,
[-0.00377, -0.00008], on 2025–26. Adding starter RAPM to that model is unresolved:
the smaller model is better by 0.00077 on the first fold and worse by 0.00053 on
the second, with both paired intervals crossing zero. The smaller model is frozen
because the extra player-data dependency has not earned promotion.

On the later fold, the combined model's tipoff difference from ESPN is not
identified: local-minus-ESPN interval [-0.00093, 0.00765].
ESPN remains an external 2025–26 benchmark, not a label: Brier is 0.14759 on
matched plays and 0.20210 at tipoff.

### Possession-start result

Runs `wp_possession_start_v2_1db472e450` and
`wp_possession_start_v2_0a5d626234` use the frozen starter-free context and
confirm time-interacted possession control on both folds. On 2024–25, it lowers
Brier from 0.15270 to 0.15253; the whole-game delta is -0.000177 with 95% interval
[-0.000206, -0.000147]. On 2025–26, it lowers Brier from 0.14689 to 0.14671; the
delta is -0.000175 with interval [-0.000205, -0.000146]. All 5,000 bootstrap
draws favor possession in each fold.

The fitted home-possession swing is about 2.1 percentage points overall. It rises
to 11.6–12.4 points in close last-two-minute states and 19.8–21.8 points when tied
inside ten seconds. Close-last-two-minute Brier improves from 0.17933 to 0.17626
on 2024–25 and from 0.17019 to 0.16594 on 2025–26. This matches Inpredictable's
qualitative late-game behavior and clears the repeated-fold research gate, but
possession is valid only at causally constructed possession starts.

### Architecture parity result

Run `wp_stage1_v1_7e6c77d51a` compares the frozen logistic model with an additive
five-knot spline logistic model and a 200-tree histogram GBM on identical rows and
the same 12 inputs. Neither nonlinear model passes:

| Model | 2024–25 Brier | 2025–26 Brier | Pooled delta vs logistic |
|---|---:|---:|---:|
| Logistic | **0.15302** | **0.14777** | control |
| Spline GAM | 0.15560 | 0.14943 | +0.00214 [0.00088, 0.00340] |
| Histogram GBM | 0.17030 | 0.15661 | +0.01303 [0.00970, 0.01639] |

Both challengers also lose AUC in both folds. The GAM misses the calibration gate
on 2024–25 (slope 0.876); the GBM misses it in both folds. They are rejected, not
retuned against the outer seasons.

Run `wp_mlp_v1_7a7825bf09` tests a fixed 64×64 feed-forward MLP across five seeds.
It is not the preregistered residual network because PyTorch is unavailable; that
deviation is explicit in the artifact. The seed ensemble is decisively worse:
Brier is 0.18965 versus 0.15302 on 2024–25 and 0.17448 versus 0.14777 on 2025–26.
The pooled delta is +0.03171 [0.02619, 0.03745], both AUCs fall, and calibration
slopes collapse to 0.41 and 0.49. Nine of ten seed fits reach the frozen 100-epoch
cap. Reject this implementation; do not infer that all neural sequence models fail.

## Inpredictable reference-surface validation

Run `wp_inpredictable_surface_v1_56696b0386` queries the public calculator at
eight checkpoints, eleven margins (-15 to +15), and both possession states.
Because the calculator assumes a zero point spread and the local model is a
home-team model, the comparison removes the local home intercept by averaging
mirrored team orientations. Inpredictable possession/no-possession predictions
are averaged because possession is not yet a local feature.

- 88 neutral score/clock states
- correlation: 0.9983
- mean absolute probability difference: 1.64 percentage points
- RMSE difference: 2.32 points
- maximum difference: 5.97 points, concentrated around ±3 late
- mean absolute Inpredictable possession swing: 2.87 points
- tied with 10 seconds left: possession swing 23.4 points

This validates state-surface shape, not outcome accuracy. It identifies
possession/control as the next in-game feature.

## Research basis

The implementation was derived independently as the smallest leakage-safe
baseline. It was not copied from a paper. The design is consistent with:

- Stern's score-lead/time sports model: https://doi.org/10.1080/01621459.1994.10476851
- Inpredictable's locally weighted logistic model using time, margin,
  possession, and point spread: https://www.inpredictable.com/2015/02/updated-nba-win-probability-calculator.html
- McFarlane's NBA late-game logistic model using time, score, possession, and
  point spread: https://doi.org/10.3233/JSA-180231
- Deshpande and Jensen's win-probability player-impact framing:
  https://arxiv.org/abs/1604.03186

ESPN is used only as a play-aligned external forecast benchmark because its
feature set and training procedure are proprietary.

## Rejected or deferred

- Actual game minutes: rejected as outcome leakage.
- ESPN Net Points/WPA as training labels: rejected; benchmark-only and upstream
  redistribution rights are unresolved.
- Raw CDN `possession` on arbitrary action rows: rejected; rebounds and made shots
  make naïve joins leak. The retained implementation collapses ordered control
  runs and scores each start only from completed prior possessions.
- Terminal-state scoring: excluded because probabilities are mechanically 0/1.
- Spline GAM and histogram GBM: rejected on both folds with identical inputs.
- Fixed 64×64 feed-forward MLP: rejected in all five seeds and both folds. It is
  not evidence about a residual architecture or causal sequence history.
- Sequence/RL models: require validated causal history tokens and are a
  different-input test, not evidence that attention itself improves WP.
- Starter RAPM alone: retained as a logged negative/inconclusive result.

## Reproduce

```bash
uv run python -m nba_impact.cli compare-wp-lineup-strength
uv run python -m nba_impact.cli compare-wp-lineup-strength \
  --train-season 2023-24 --test-season 2024-25 --skip-espn
uv run python -m nba_impact.cli compare-wp-possession
uv run python -m nba_impact.cli compare-wp-possession \
  --train-season 2023-24 --test-season 2024-25
uv run python -m nba_impact.cli compare-wp-stage1
uv run python -m nba_impact.cli compare-wp-mlp
uv run python -m nba_impact.cli benchmark-inpredictable \
  --model-run artifacts/models/win_probability_lineup/wp_pregame_ablation_v3_cdbcea84ee
```
