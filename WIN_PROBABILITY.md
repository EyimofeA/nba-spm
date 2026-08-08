# Win Probability Model Card

Last updated: 2026-08-08. This is the human-readable source of truth for the
current NBA win-probability lane.

Nonlinear and sequence candidates are preregistered in `WP_ARCHITECTURES.md`.

## Estimand and split

The model estimates the home team's probability of winning after a recorded
play. Logistic models train on 2024–25 and test once on the full 2025–26 season.
Training states are sampled every 30 seconds; terminal states are excluded.
Model selection uses Brier score, log loss, AUC, calibration, checkpoint metrics,
and paired whole-game bootstraps. This remains one outer fold.

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

| Variant | 2025–26 overall Brier | Tipoff Brier | Verdict |
|---|---:|---:|---|
| Constant/state-free | 0.24708 at tipoff | 0.24708 | rejected |
| State only | 0.16385 | 0.24708 | rejected; no team strength |
| State + Elo | 0.14987 | 0.21181 | retained baseline |
| + prior starter RAPM | 0.14922 | 0.21057 | inconclusive; do not promote |
| + rolling margin and rest | **0.14731** | **0.20592** | research challenger |
| ESPN | 0.14759 on matched plays | 0.20210 | external benchmark, not a label |

The rolling-context improvement over starter RAPM has a whole-game Brier delta
of -0.00195 with 95% interval [-0.00373, -0.00020]. At tipoff it beats the
starter model in every bootstrap draw. Its tipoff difference from ESPN is not
identified: local-minus-ESPN interval [-0.00063, 0.00839].

### Possession-start result

Run `wp_possession_start_v1_9af34729ef` evaluates 261,222 possession starts in
1,288 held-out 2025–26 games. On identical rows, possession plus time interactions
reduces Brier from 0.14651 to 0.14632. The whole-game delta is -0.000189 with 95%
interval [-0.000214, -0.000163]; all 5,000 bootstrap draws favor possession.

The average fitted home-possession swing is 2.04 percentage points overall,
11.57 points when the margin is at most three in the last two minutes, and 19.67
points when tied inside ten seconds. In the close-last-two-minute subset, Brier
improves from 0.17122 to 0.16719. This closely matches Inpredictable's qualitative
late-game behavior, but it remains a one-fold research candidate. The 2023–24
canonical data is now available; the independent 2023–24 → 2024–25 fold has not
yet been run.

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
- GBM/neural/RL models: ordered in `WP_ARCHITECTURES.md` and deferred until the
  second chronological fold is scored. Complexity before repeated validation
  is low-value.
- Starter RAPM alone: retained as a logged negative/inconclusive result.

## Reproduce

```bash
uv run python -m nba_impact.cli compare-wp-lineup-strength
uv run python -m nba_impact.cli compare-wp-possession
uv run python -m nba_impact.cli benchmark-inpredictable \
  --model-run artifacts/models/win_probability_lineup/wp_pregame_ablation_v2_522e1a36f2
```
