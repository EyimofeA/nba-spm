# Single-season expected-outcome RAPM

## Decision

Keep ordinary realized-points RAPM. Replacing field-goal and free-throw results
with player-neutral expected conversion makes the rating materially worse at
predicting next-season game margins. The richer shot-context model predicts
individual makes better than the location-only model, but it does not rescue
the RAPM target.

This is reused 2025 and 2026 diagnostic evidence. It is not independent
confirmation and it does not authorize a public rating change. Season 2027 was
not loaded.

## Frozen comparison

Each fold fits a zero-prior, terminal-lineup, one-season RAPM with penalties
`3000 offense / 4500 defense / 300 home` and scores the next season's games.
Every arm uses the same possessions, players, game outcomes, and lineup policy.

The two expected-conversion responses are:

1. Location: shot location, distance, angle, shot value, period, game clock,
   score state, and home status.
2. Context: all location inputs plus time since possession start, transition,
   putback, second chance, after-turnover status, and observable finish type.

Both models exclude shooter, defender, team, lineup, and shot result. They use
three whole-game folds inside the rating season, so the shot model never sees
the game whose conversion it replaces. The target is

`observed possession points - observed conversion points + expected conversion points`.

Turnovers, offensive-rebound sequences, foul drawing, and other nonconversion
parts of the possession stay observed. Free throws use a player-neutral rate
estimated only from earlier seasons, with one-year decay and 50 prior attempts.
The conversion ledger maps 535,321
events and accounts for 99.013% of realized rating-season points.

## Next-season game-margin result

| Rating season | Test season | Arm | RMSE | r | Calibration slope | Predicted margin SD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2024 | 2025 | Ordinary points | 15.0602 | 0.3262 | 0.8183 | 6.3299 |
| 2024 | 2025 | Location expected conversion | 15.7839 | 0.1321 | 0.6394 | 3.2815 |
| 2024 | 2025 | Context expected conversion | 15.7115 | 0.1556 | 0.7316 | 3.3772 |
| 2025 | 2026 | Ordinary points | 15.6797 | 0.2960 | 0.7929 | 6.1094 |
| 2025 | 2026 | Location expected conversion | 15.9100 | 0.2339 | 1.0682 | 3.5828 |
| 2025 | 2026 | Context expected conversion | 15.8729 | 0.2437 | 1.0986 | 3.6291 |

Equal-season mean RMSE is 15.3700 for ordinary points, 15.8470 for the location
arm, and 15.7922 for the context arm. The context arm loses by 0.6513 RMSE in
2025, with a paired whole-game 95% interval of `[0.4185, 0.8716]`. It loses by
0.1931 in 2026, with interval `[-0.0360, 0.4312]`.

The expected-conversion ratings are too compressed. Their average predicted
game-margin standard deviation is 3.50 points, versus 6.22 for ordinary RAPM.
That is the main failure, not a weak shot model.

## Shot-model check

The context additions lower whole-game-out Brier score from 0.233208 to
0.230043 in 2024 and from 0.232561 to 0.229542 in 2025. The mean predicted make
rates match the observed rates within 0.00005 in both seasons. Better expected
shot conversion still removes repeatable player shotmaking and some lineup
shot-creation signal from the player target.

This model is only qSQ-style. It lacks optical tracking, defender distance,
release pressure, player geometry, and the proprietary qSQ specification.

## Possession-start expected points

The separate player-neutral possession-start model was reproduced from the
current lake as `expected_possession_points_v1_c9581a23b1`. It uses only period,
clock, score differential, home side, previous-possession points, and a
first-possession flag. The 2024 fold trains on 2023; the 2025 fold trains on
2023 through 2024.

Across 497,177 out-of-fold possessions, mean RMSE improves from 1.194461 for a
constant baseline to 1.194070. Mean Poisson deviance improves from 1.627856 to
1.627015. The gain is about 0.05%, below the frozen 0.25% gate in both folds.
A residual RAPM refit remains deferred because the model would mostly subtract
a near-constant from every possession.

## Reproducibility

- Contract: `research/experiments/single_season_expected_outcome_rapm_v1.yml`
- Runner: `research/rapm_lab/run_single_season_expected_outcome_rapm.py`
- Run: `single_season_expected_outcome_rapm_v1_12a37167cf`
- Focused tests: 11 passed
