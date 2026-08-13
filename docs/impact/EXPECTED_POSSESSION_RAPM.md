# Expected-Possession Residual RAPM

Status: **data contract ready; player model not yet fit**.

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

## Frozen next experiment

Use cross-fitted, player-neutral expected points from a simple multinomial or
Poisson baseline trained chronologically. Fit the residual RAPM only after every
prediction is out of fold. Compare it with normal RAPM on identical held-out
games, using equal-season margin RMSE, correlation, calibration, and a paired
whole-game interval. Do not add nonlinear or player-aware features until that
baseline passes.
