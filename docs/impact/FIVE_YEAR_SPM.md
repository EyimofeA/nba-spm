# Five-year SPM with a one-season RAPM update

Status: selected research SPM; not yet the public model.

## Model

For a window ending in season `t`, the SPM input is one player row pooled over
`t-4` through `t`. The label is that same window's zero-prior RAPM. Offense is
a histogram gradient-boosting model over 127 frozen fields. Defense is ridge
over 68 frozen fields. Net is offense plus defense. Target possessions set the
sample weight; they are not model features.

Historical ratings are forward chained. A rating for `t` trains only on
five-year windows ending before `t`. Complete windows begin in 2018 because the
statistical source begins in 2014. The 2026 model trains on window ends
2018--2025 and scores statistics pooled over 2022--2026.

The AIO remains one centered possession regression, not an arithmetic sum:

`AIO(t) = RAPM on possessions from t, with the five-year SPM(t) as its prior mean`.

The RAPM design contains five offensive players, five defensive players, and
home court. It uses the fixed `3000 / 3000 / 300` penalties and terminal
lineups. No five-year possession rows enter the AIO likelihood.

## Predictive test

Every rating season was applied to game lineups in the next season. All arms
scored identical games.

| Test season | Annual-target AIO RMSE | Five-year SPM AIO RMSE | Zero-prior RAPM RMSE |
| ---: | ---: | ---: | ---: |
| 2022 | 14.4915 | **14.3730** | 14.4623 |
| 2023 | 12.7513 | **12.7186** | 12.8325 |
| 2024 | 14.6300 | **14.6035** | 14.7669 |
| 2025 | 14.9481 | **14.8960** | 15.0541 |
| 2026 | 15.5318 | **15.4115** | 15.7326 |
| Mean | 14.4705 | **14.4005** | 14.5697 |

Mean game-margin correlation improves from `.3462` for the annual-target AIO
and `.3219` for zero-prior RAPM to `.3652`. The equal-season paired game
bootstrap gives a development MSE difference of `-1.6763` with a 95% interval
of `[-3.0445, -0.1203]`. Reused 2025--26 diagnostics give `-2.6385`, interval
`[-4.6215, -0.7399]`.

The standalone five-year SPM is not better at reproducing the magnitude of the
next season's noisy one-year RAPM: net RMSE is `1.9832`, versus `1.7543` for the
annual-target SPM. Its net correlation is slightly higher, `.4232` versus
`.4106`. The downstream AIO game test is the selection metric because that is
the intended use.

## Decision and limits

Run `five_year_target_spm_v1_65550acb79` replaces the one-year-target SPM as the
research default for the single-season AIO. It does not replace the public
2017--24 table yet. The full-feature correction was made after inspecting a
common-feature pilot, and 2025--26 have been reused repeatedly. Season 2027 is
the next untouched promotion test.

Reproduction:

- contract: `research/experiments/five_year_target_spm_v1.yml`;
- runner: `research/run_five_year_target_spm.py`;
- model: `src/nba_impact/models/five_year_target_spm.py`;
- selected artifact: `artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79`.
