# Annual SPM learner screen

## Question

Which small tabular learner and audited feature family best reconstructs
single-season offense and defense RAPM across unseen seasons?

This experiment evaluates a retrospective annual SPM. Each row contains one
player's statistics and RAPM label from the same season. The model never uses a
later season's RAPM target to fit an earlier fold.

## Design

Run `annual_spm_learner_screen_v1_74808a8ae2` uses the completed 2014--26
annual feature panel and canonical annual RAPM targets.

- Each test season trains on earlier seasons.
- The latest training season selects hyperparameters.
- The 2018--21 folds select learners and feature arms.
- The selected choices remain fixed for the 2022--26 diagnostic folds.
- Predictor-only pruning removes constants and correlations of at least `.95`
  inside each training fold. Box15 fields win a tie against a redundant field.
- The target-free feature atlas excludes known source shifts and circular or
  predictive-only fields.
- Player exposure supplies `sqrt(min(Poss_Off, Poss_Def))` sample weights. It
  does not enter as a feature.

The learner screen compares ridge, elastic net, histogram boosting, Extra
Trees, and additive quadratic splines with ridge regularization. The feature
screen compares Box15, Box15 plus each audited family, and the full audited
pool.

## Selection results

| Side | Learner | Weighted RMSE | Correlation | Calibration slope |
| --- | --- | ---: | ---: | ---: |
| Offense | Elastic net | **0.9784** | **0.6566** | 0.9696 |
| Offense | Ridge | 0.9811 | 0.6555 | 0.9604 |
| Offense | Additive spline ridge | 0.9869 | 0.6550 | 0.9731 |
| Offense | Histogram boosting | 0.9956 | 0.6441 | 0.9683 |
| Offense | Extra Trees | 0.9958 | 0.6453 | 1.0497 |
| Defense | Ridge | **0.9518** | **0.5515** | 0.8708 |
| Defense | Additive spline ridge | 0.9582 | 0.5344 | 0.9404 |
| Defense | Elastic net | 0.9609 | 0.5379 | 0.9290 |
| Defense | Histogram boosting | 0.9701 | 0.5186 | 0.9414 |
| Defense | Extra Trees | 0.9721 | 0.5173 | 1.1411 |

The full audited pool wins both feature screens. No single added family matches
the full pool. Shooting and creation are the strongest offense additions.
Matchup shot defense is the strongest isolated defense addition.

Extra Trees wins the separate Box15 learner screen on both sides. It lowers
development offense RMSE from `1.1041` to `1.0779` and defense RMSE from
`1.0746` to `1.0592` relative to ridge.

The frozen model uses elastic net on 112 offense fields and ridge on 66 or 67
defense fields after fold-specific pruning. Every later offense fold selects
elastic-net `alpha=0.003` and `l1_ratio=0.1`. Every later defense fold selects
ridge `alpha=3000`.

## Later diagnostic

| Component | Model | Weighted RMSE | Correlation | Calibration slope |
| --- | --- | ---: | ---: | ---: |
| Offense | Rich winner | **1.0314** | **0.6577** | 1.0029 |
| Offense | Box15 ridge | 1.1559 | 0.5792 | 1.0688 |
| Defense | Rich winner | **0.9663** | **0.6021** | 1.0059 |
| Defense | Box15 ridge | 1.1334 | 0.3513 | 1.1314 |
| Net | Rich winner | **1.4176** | **0.6596** | 1.0219 |
| Net | Box15 ridge | 1.6590 | 0.5013 | 1.0962 |
| Net | Box15 selected learner | 1.6739 | 0.4736 | 1.2412 |

The rich model lowers net RMSE in every diagnostic season.

| Season | Rich net RMSE | Box15 net RMSE |
| ---: | ---: | ---: |
| 2022 | **1.4023** | 1.6170 |
| 2023 | **1.3418** | 1.5688 |
| 2024 | **1.4371** | 1.6823 |
| 2025 | **1.4365** | 1.6817 |
| 2026 | **1.4703** | 1.7455 |

## Decision

Use elastic net for offense and ridge for defense in the annual rich-SPM
challenger. Linear regularization beats the nonlinear learners. This result
also confirms that the rich fields contain real annual RAPM information.

Do not replace Box15 ridge with the development-selected Extra Trees model.
Its development gain reverses on the later folds and its calibration worsens.

Do not replace Box15 as the AIO prior from this result. Box15 has previously
lost standalone SPM comparisons and still performed better after the RAPM
update. The next experiment must compare the new annual rich prior with Box15
under the same precision-aware RAPM update and identical games.

The target-free atlas inspected predictor distributions through 2026 before
this run. No RAPM target entered that audit, but the later folds remain
diagnostics rather than untouched confirmation. Annual RAPM is also a noisy
label. This run does not prove next-season game accuracy or causal skill value.
