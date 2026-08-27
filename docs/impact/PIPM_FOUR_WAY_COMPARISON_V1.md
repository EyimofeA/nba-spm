# PIPM four-way comparison

## Answer

The BoxPIPM-style AIO won this reused historical test. The standalone BoxPIPM
and PIPM priors were tied within sampling error.

The comparison does not use an original full-season PIPM release. The attached
PIPM file stops about 20 games into 2020-21. The test uses the full regular-season
PIPM tables in the existing third-party reference artifact. This source limit
prevents a definitive claim about original PIPM.

## Models

The run compares four ratings.

1. `box_prior` uses the 15-feature CourtSignal BoxPIPM-style prior.
2. `pipm_reference` uses the third-party PIPM table.
3. `box_prior_plus_rapm` updates the BoxPIPM prior with one-season RAPM.
4. `pipm_reference_plus_rapm` updates the PIPM reference with the same RAPM.

The run replaces each PIPM row below 250 minutes with -1.0 offense and -1.0
defense. This gives a -2.0 net replacement value and prevents tiny-sample PIPM
values from dominating future games. The rule was fixed before scoring.

Both AIO models use the same terminal-lineup possession rows. They use ridge
penalties of 3000 for offense, 3000 for defense, and 300 for home advantage.
The prior center scale equals 1.0.

## Folds

A fold is one chronological rating and test pair. Fold 2021 to 2022 builds a
rating with information through 2021 and scores 2022 games. The run repeats the
same process for 2022 to 2023 and 2023 to 2024.

The test uses 1,230 games in 2022, 1,230 in 2023, and 1,227 in 2024. Every model
scores the same games and outcomes inside each fold.

| Model | Mean fold RMSE | Mean correlation |
| --- | ---: | ---: |
| BoxPIPM plus RAPM | 13.8644 | 0.3635 |
| PIPM reference plus RAPM | 14.0578 | 0.3450 |
| BoxPIPM | 14.0595 | 0.3222 |
| PIPM reference | 14.0835 | 0.3309 |

RMSE uses points of final game margin. The paired tests use MSE because squared
errors add cleanly game by game.

## Paired results

The table reports `candidate MSE - reference MSE`. Negative values favor the
candidate.

| Candidate | Reference | MSE difference | 95% interval | Fold wins |
| --- | --- | ---: | ---: | ---: |
| BoxPIPM plus RAPM | PIPM reference plus RAPM | -5.473 | [-7.108, -3.776] | 3 of 3 |
| BoxPIPM | PIPM reference | -0.583 | [-4.373, 3.126] | 2 of 3 |
| PIPM reference | PIPM reference plus RAPM | +0.589 | [-1.484, 2.689] | 1 of 3 |
| BoxPIPM | BoxPIPM plus RAPM | +5.480 | [2.168, 8.756] | 1 of 3 |

The BoxPIPM RAPM update improved the equal-season MSE. The PIPM RAPM update did
not show a clear improvement. PIPM already contains an on-off component, so the
second RAPM update can reuse some of the same possession signal.

## The 5,000-draw interval

The bootstrap fits each model once. It does not fit 5,000 RAPMs.

For draw `b`, the procedure performs these steps.

1. Sample 2022 games with replacement until the draw has 1,230 games.
2. Repeat the sampling inside 2023 and 2024.
3. Calculate each model's MSE inside each season.
4. Average the three season MSE values with equal season weights.
5. Subtract the reference MSE from the candidate MSE.

The same sampled games feed all four models in each draw. This pairing removes
schedule noise that every model shares. The 2.5th and 97.5th percentiles of the
5,000 differences form the 95% interval.

The interval measures sensitivity to these historical game samples. It does
not measure stability across unseen NBA eras. Season 2027 remains the untouched
confirmation.

## Artifacts

Run `pipm_four_way_comparison_v1_0f1473b838` stores the fold metrics, player
ratings, game predictions, source audits, coverage, bootstrap draws summary,
and hashes under `artifacts/research/pipm_four_way_comparison/`.
