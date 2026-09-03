# Reliability-weighted conversion RAPM

## Decision

Keep ordinary realized-points RAPM. Partial shooting-luck adjustment reverses
across the two reused next-season folds. The history-reliability arm also loses
2025 and improves 2026. Neither clears the frozen requirement to improve both
folds. Do not publish or promote any challenger from this test.

This is reused diagnostic evidence. Season 2027 was not loaded.

## Question

The earlier player-neutral expected-conversion RAPM removed too much useful
variation. This follow-up asks whether retaining a reliable share of observed
conversion fixes that compression.

The experiment uses the pinned conversion ledger from
`luck_adjusted_rapm_v1_8580bb30e9`. Its field-goal expectation is
player-neutral and uses location, clock, score, home side, and period. The 2024
expectations are cross-fitted by whole game. The 2025 expectations train only
on earlier seasons. Preseason shooter skill uses empirical-Bayes rates chosen
on shooting seasons through 2024.

Every arm fits a one-season terminal-lineup RAPM with penalties
`3000 offense / 4500 defense / 300 home`, then scores the next season's games.
Every comparison uses identical possessions, lineups, players, and scored
games.

## Targets

The three fixed neutral blends retain 25%, 50%, or 75% of the observed
conversion residual:

`neutral expected points + weight * (actual points - neutral expected points)`

The history-reliability arm starts from the preseason player-skill expectation.
For each player, season, and shot category, its weight is:

`current-season attempts / (current-season attempts + frozen prior attempts)`

It then uses:

`preseason skill expected points + reliability * (actual points - preseason skill expected points)`

The frozen prior strengths range from 50 attempts for free throws to 200 for
three-point categories. Mean event reliability ranges from about 0.27 for
corner threes to 0.75 for free throws. All nonconversion possession points stay
observed.

## Next-season game-margin results

| Rating season | Test season | Arm | RMSE | r | Calibration | Predicted margin SD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2024 | 2025 | Ordinary points | **15.0602** | **0.3262** | 0.8183 | 6.3299 |
| 2024 | 2025 | Neutral residual, 25% retained | 15.4666 | 0.2266 | 0.9691 | 3.7134 |
| 2024 | 2025 | Neutral residual, 50% retained | 15.2378 | 0.2817 | 1.0083 | 4.4357 |
| 2024 | 2025 | Neutral residual, 75% retained | 15.1015 | 0.3108 | 0.9254 | 5.3325 |
| 2024 | 2025 | Preseason skill expectation | 15.7190 | 0.1622 | 0.6715 | 3.8358 |
| 2024 | 2025 | History-reliability hybrid | 15.1760 | 0.2950 | 0.9744 | 4.8071 |
| 2025 | 2026 | Ordinary points | 15.6797 | 0.2960 | 0.7929 | 6.1094 |
| 2025 | 2026 | Neutral residual, 25% retained | 15.7248 | 0.2797 | 1.1756 | 3.8925 |
| 2025 | 2026 | Neutral residual, 50% retained | 15.6256 | 0.2978 | 1.0919 | 4.4619 |
| 2025 | 2026 | Neutral residual, 75% retained | **15.6105** | 0.3002 | 0.9402 | 5.2249 |
| 2025 | 2026 | Preseason skill expectation | 15.7812 | 0.2649 | 1.0637 | 4.0741 |
| 2025 | 2026 | History-reliability hybrid | **15.5881** | **0.3042** | 0.9965 | 4.9958 |

The 75% arm is the best fixed blend on the 2025 selection fold. It loses to
normal by 0.0413 RMSE, with a paired whole-game 95% interval of
`[-0.0206, 0.1008]`. It improves reused 2026 by 0.0693, with interval
`[-0.1304, -0.0085]`.

The history-reliability arm loses 2025 by 0.1158 RMSE, with interval
`[0.0070, 0.2229]`. It improves reused 2026 by 0.0917, with interval
`[-0.1929, 0.0113]`.

The equal-season mean favors the 75% arm by only 0.0140 RMSE, while normal has
the better mean correlation, 0.3111 versus 0.3055. Averaging the reversal does
not satisfy the frozen both-fold decision rule.

## Interpretation

The monotone 2025 pattern confirms that the earlier expected-outcome target
over-adjusted shooting. Retaining more realized conversion steadily restores
prediction spread and accuracy. The 2026 result suggests that a small shooting
variance correction may sometimes help, but the season reversal prevents a
stable model claim.

The player-history formulation does not solve the instability. It uses a
reasonable empirical-Bayes weight, but its category-level attempt reliability
does not measure how much of observed shotmaking is portable player skill. A
future version would need richer shot state and a reliability model selected
strictly before all scored seasons.

## Validation

- Run: `reliability_weighted_conversion_rapm_v1_cff32e3e85`
- Conversion rows: 535,321
- Rating seasons: 2024 and 2025
- Test games: 1,226 in 2025 and 1,228 in 2026
- Whole-game bootstrap draws: 5,000 per reported comparison
- Focused tests: 5 passed
- Artifact audit: identical games and outcomes across arms; saved RMSE values
  independently recomputed from game predictions
- Season 2027 loaded: false
