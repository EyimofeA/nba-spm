# Full SPM history ablation

Run `full_spm_history_ablation_v1_34725a86aa` refits the five-year SPM on the
complete 2014-26 feature panel. It compares three priors on the same 2022-26
future games:

- full SPM with 127 offense and 68 defense fields;
- history-complete SPM with the same offense fields and 55 defense fields;
- BoxPIPM-style with 15 box fields.

The history-complete arm removes five hustle fields and eight matchup-defense
fields. Those source families start in 2018. The test removes the 13 fields as
one block. It does not search subsets.

## Result

The primary table reports the square root of equal-season mean MSE. This value
differs slightly from the arithmetic mean of the five fold RMSE values.

| Rating | Equal-season RMSE | Mean correlation |
| --- | ---: | ---: |
| BoxPIPM-style plus RAPM | 14.375 | .3658 |
| Full SPM plus RAPM | 14.429 | .3663 |
| History-complete SPM plus RAPM | 14.445 | .3596 |
| Zero-prior RAPM | 14.602 | .3219 |
| Full SPM | 14.609 | .3250 |
| BoxPIPM-style | 14.638 | .3097 |
| History-complete SPM | 14.705 | .2999 |

Removing the 13 defense fields hurts the standalone SPM. Full SPM lowers MSE
by 2.820 points squared per game. Its paired 95% interval is `[-4.095, -1.555]`.
Full SPM wins four of five seasons.

The RAPM update absorbs much of this difference. Full SPM plus RAPM lowers MSE
by 0.461 against history-complete SPM plus RAPM. Its interval is
`[-1.026, +0.109]`. Full SPM wins three of five seasons. The interval includes
zero.

Full SPM and BoxPIPM-style are tied as standalone ratings. Full SPM lowers MSE
by 0.843. Its interval is `[-2.800, +1.112]`.

BoxPIPM-style wins after the same RAPM update. Full SPM plus RAPM raises MSE by
1.550 against BoxPIPM-style plus RAPM. Its interval is `[+0.690, +2.399]`.
BoxPIPM-style wins four of five seasons. History-complete SPM plus RAPM loses
all five seasons to BoxPIPM-style plus RAPM.

## Interpretation

Keep the 13 late-start fields. They carry real defensive signal despite their
shorter history. Removing them makes the five-year target fit and future-game
prediction worse.

Keep BoxPIPM-style as the research AIO prior. Full SPM produces a similar
future-game ranking, but its posterior has too much spread. Its mean calibration
slope is .755. BoxPIPM-style plus RAPM reaches .808. The BoxPIPM-style win comes
from calibration and squared error, not a material correlation advantage.

The same-window target fit does not select the best downstream prior. Full SPM
fits five-year RAPM much better than BoxPIPM-style in every component. The extra
fit does not transfer through the one-season RAPM update.

## Validation

- Five evaluated rating folds end in 2021-25.
- The artifact also contains unscored current 2026 SPM and AIO ratings.
- Test seasons cover 2022-26.
- Every candidate scores the same 6,141 game-season rows.
- The bootstrap resamples whole games inside each season for 5,000 draws.
- Prior possession coverage is 100% for every candidate and fold.
- Offense plus defense equals net with zero numerical error.
- Recovered 2024 and 2025 annual matrices reproduce their rolling sufficient
  statistics within `1.14e-13`.
- Season 2027 is absent.

All test seasons contain reused evidence. This result does not promote a public
model.
