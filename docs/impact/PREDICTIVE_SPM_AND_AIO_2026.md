# Predictive SPM and current-strength AIO, 2026

## Decision

The research current-strength model uses five trailing seasons of possession
data, a two-year half-life, and the raw predictive SPM as a ridge center. It
improves future-game margin prediction in development, but it is not a public
or confirmed rating. Season 2027 is the untouched confirmation season.

## Three separate outputs

| Output | Question | 2026 information |
| --- | --- | --- |
| Annual SPM | What did the player's box and tracking profile imply about 2026 retrospective RAPM? | 2026 statistics; model trained on 2014-25 labels |
| Predictive SPM | What did information through 2025 predict for the player's 2026 RAPM? | Statistical features through 2025 only |
| Current-strength AIO | What current player coefficients best combine recent lineup evidence and the predictive prior? | 2021-25 possessions with exponential decay; predictive SPM prior through 2025 |

These are not interchangeable. The annual SPM is retrospective. The predictive
SPM is a next-season forecast. The AIO is a filtered possession model evaluated
by how its player coefficients predict held-out game margins.

## Predictive SPM

The predictor keeps the frozen basketball-information feature set and fits
offense and defense separately. The trajectory ablation tested four outputs:

1. the raw frozen predictive SPM;
2. a shared smooth age residual;
3. separate offense and defense age residuals;
4. separate age residuals plus prior-season minutes and games.

Each test season trains only on earlier target seasons. The selection sample is
2020-24. Age is prior-season age plus one; opportunity is prior-season minutes
and games. Opportunity is a forecast input, never a retrospective skill.

| Method | Mean weighted net RMSE, 2020-24 |
| --- | ---: |
| Raw predictive SPM | 1.604696 |
| Side-specific age | 1.612733 |
| Shared age | 1.612834 |
| Side age plus opportunity | 1.621506 |

The raw model wins. Age, minutes, and games do not enter the chosen prior.

## Current-strength AIO

For a test season `t`, the model uses the five completed seasons `t-5` through
`t-1`. A possession from season `s` receives weight

`w(s,t,h) = 2^-((t - 1 - s) / h)`,

where `h` is the half-life in seasons. The ridge design contains five offensive
player indicators, five defensive player indicators, and one home indicator.
The outcome is points on the possession. The fixed penalties are 3000 for each
player side and 300 for home.

The coefficient prior is the raw predictive SPM mapped to RAPM coefficient
units:

- offense center = predicted offense / 100;
- points-allowed defense center = -predicted defense / 100;
- home center = 0.

The fit solves

`beta = (X'WX + P)^-1 [X'W(y - mean(y)) + P c]`.

Offense and points-allowed defense are possession-weight centered after the fit,
and the intercept is adjusted so predictions do not change. Published defense
is sign-flipped so positive defense is good. Net is exactly offense plus
defense.

Half-life selection compared `0.5`, `1`, `2`, `3`, `5`, and no decay using
equal-season mean held-out game-margin RMSE from 2020-24. The two-year half-life
won at `13.7429` for zero-prior RAPM.

| Arm | Mean game-margin RMSE, 2020-24 |
| --- | ---: |
| Two-year decay plus SPM prior | 13.7122 |
| Two-year decay, zero prior | 13.7429 |
| Five years plus SPM prior, no decay | 13.7550 |
| Five years, zero prior, no decay | 13.7681 |

The chosen AIO wins four of five development folds. In 10,000 paired whole-game
bootstrap draws, its mean squared-error difference favors it over all three
frozen comparators. Those intervals are conditional on the already-selected
candidate grid; they are not selection-aware confirmation intervals. Reused
diagnostics also favor it in 2025 and 2026, but those seasons were already
inspected and do not confirm the model.

| Reused season | Chosen AIO RMSE | Five-year zero-prior RMSE |
| ---: | ---: | ---: |
| 2025 | 14.7719 | 14.8551 |
| 2026 | 15.1817 | 15.2962 |

## 2026 bundle and limits

The local bundle contains 582 active 2026 players. Predictive SPM priors cover
79.73% of them. Missing priors are set to the neutral ridge center, which is
transparent but not yet a satisfactory rookie forecast.

The evaluation observes held-out game lineups. It tests whether the estimated
player coefficients predict game margins; it is not a roster-only season
forecast. The model is research-only until the population policy, uncertainty,
and untouched Season 2027 confirmation are complete.

## Reproduction pointers

- Contract: `research/experiments/predictive_current_aio_2026_v1.yml`
- Runner: `research/run_predictive_current_aio.py`
- AIO artifact: `artifacts/models/predictive_current_aio/predictive_current_aio_2026_v1_c18e2472ec`
- Audit: `research/audits/predictive_current_aio_2026_v1`
- Bundle: `artifacts/models/courtsignal_2026_research_bundle/courtsignal_2026_research_bundle_v1_3913f9efd6`
