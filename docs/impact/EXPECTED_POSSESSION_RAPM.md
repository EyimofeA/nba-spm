# Expected-Possession Residual RAPM

Status: **expected-points baseline complete; residual RAPM deferred**.

## Goal

This is the defensible version of “expected possession outcome versus player.”
It is not a primary-defender model and it is not an EPV system with optical
tracking.

1. Predict expected points for a possession using only information known when
   that possession begins.
2. Calculate residual points: `actual possession points - expected points`.
3. Fit normal terminal-lineup RAPM to those residuals on exactly the same games
   and lineup policy as the normal-RAPM baseline.
4. Keep it only if it improves held-out equal-game team-margin prediction and
   does not create component or low-exposure failures.

This is a context-residual RAPM challenger. It can reduce state-related noise;
it cannot by itself identify individual shot defense or causal player value.

## Possession-start data contract

Build command:

```bash
uv run python -m nba_impact.cli build-possession-start-context
```

Snapshot `possession_start_context_ebcae214e662d404` contains 787,579 canonical
possessions across 3,907 games from the validated 2023--25 current-event lake.
Its one-row-per-possession start context is reconstructed from **prior canonical
possession outcomes**, not from a potentially mismatched event feed. This gives
complete start-state coverage and makes the timing rule explicit.

Allowed expected-points inputs:

| Feature | Known when? |
|---|---|
| period, overtime flag, period/regulation time remaining | possession start |
| offense score differential and home/away side | possession start |
| previous possession points and first-possession flag | immediately before start |

The target is the current possession's points. Player IDs, team IDs, lineup IDs,
current-possession actions, duration, end state, and points are forbidden model
inputs. Team and lineup identifiers remain join keys only for the later RAPM
assignment.

## Why not call this EPV yet?

A genuine basketball EPV model needs a rich within-possession state—ideally
player and ball locations, shot/drive context, and learned transition dynamics.
The available public events do not provide that state. Including the actions
that occur after the possession begins would leak its outcome. Including
player/team identity in the expectation model would absorb the effect RAPM is
supposed to estimate.

Expected shot outcome remains useful for a separate shot-quality decomposition.
It does not replace possession residuals because it omits turnovers, offensive
rebounds, free throws, and many transition outcomes.

## Expected-points baseline result

Run `expected_possession_points_v1_c9581a23b1` cross-fits a player-neutral
Poisson baseline on regular-season possessions. The 2024 fold trains on 2023;
the 2025 fold trains on 2023--24. It uses exactly the allowed start-state inputs
above and makes 497,177 out-of-fold predictions across 2,454 games.

| Test season | RMSE, constant | RMSE, context | Poisson deviance, constant | Poisson deviance, context |
|---|---:|---:|---:|---:|
| 2024 | 1.19587 | 1.19556 | 1.63493 | 1.63425 |
| 2025 | 1.19306 | 1.19258 | 1.62078 | 1.61978 |

The context direction is favorable in both folds, and mean bias stays below one
hundredth of a point per possession. But the average gain is only 0.00039 RMSE
and 0.00084 Poisson deviance—roughly five hundredths of one percent. This is far
too small to justify a residual-RAPM refit: it would mostly subtract a near
constant from the normal-RAPM target.

## Frozen next experiment

Do not fit residual RAPM from this baseline. Reopen the lane only with a new,
causal context family—such as validated transition/halfcourt state or richer
event context—and predeclare this prospective gate before looking at results:

- at least 0.25% mean Poisson-deviance improvement over the constant baseline
  in both chronological folds;
- small mean calibration bias in both folds;
- player-neutral, out-of-fold predictions for every possession sent to RAPM.

Only then fit residual RAPM and compare it with normal RAPM on identical games,
using equal-season margin RMSE, correlation, calibration, and a paired
whole-game interval. Do not add nonlinear or player-aware features until that
baseline passes.
