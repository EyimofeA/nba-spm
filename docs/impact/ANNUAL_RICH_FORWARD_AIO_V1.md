# Annual rich SPM forward AIO test

## Decision

Keep five-year Box15 as the research AIO prior. The frozen rich annual SPM is
the better standalone next-season rating, but it becomes worse after both
priors receive the same precision-aware one-season RAPM update.

This result separates two jobs:

- Use the rich annual SPM when the product needs a statistical rating alone.
- Use five-year Box15 when the statistical model supplies a prior mean for
  one-season RAPM.

## Frozen comparison

Run `annual_rich_forward_aio_v1_0f766b0ee4` scores rating seasons 2022--25 on
the next season's games. The test seasons are 2023--26. Every candidate scores
the same 4,911 games and actual margins.

The rich annual model uses elastic net for offense and ridge for defense. The
2018--21 screen fixed both learners and the audited feature arm before this
comparison. Each rating season trains on earlier annual seasons only.

The Box15 model is the canonical five-year ridge prior. It predicts five-year
RAPM from the fixed 15 box features. This run does not refit Box15.

Both AIO arms use the rated season's terminal-lineup possession likelihood.
Each arm selects offense and defense ridge penalties from `1500`, `3000`,
`4500`, and `6000` using only earlier game folds. The home penalty remains
`300`. The prior center scale remains one.

## Results

The primary score gives every test season equal weight.

| Model | MSE | RMSE | Mean correlation | Mean calibration slope |
| --- | ---: | ---: | ---: | ---: |
| Five-year Box15 AIO | 207.296 | 14.398 | .360 | .879 |
| Annual rich SPM AIO | 210.151 | 14.497 | .346 | .853 |
| Annual rich SPM | 214.413 | 14.643 | .312 | 1.085 |
| Five-year Box15 | 218.853 | 14.794 | .282 | 1.047 |

The standalone rich SPM beats standalone Box15 by `4.440` MSE points. The
paired whole-game 95% interval is `[-6.694, -2.246]` for rich minus Box15. The
rich model wins all four seasons.

The order reverses after the RAPM update. Rich AIO trails Box15 AIO by `2.855`
MSE points. The paired 95% interval is `[1.553, 4.185]` for rich AIO minus
Box15 AIO. Box15 AIO wins all four seasons.

## Why the order reverses

The annual rich prior resembles the one-season possession signal that updates
it. The five-year Box15 prior contributes a more distinct, stable signal.

Across the four folds:

| Diagnostic | Annual rich | Five-year Box15 |
| --- | ---: | ---: |
| Correlation between prior forecast and RAPM update | .507 | .322 |
| Correlation between RAPM update and prior residual | .149 | .229 |
| MSE gained from the RAPM update | 4.262 | 11.557 |

The rich prior and its update move together more often. The update also aligns
less strongly with what the rich prior missed. This evidence supports a
redundancy explanation. It does not prove that any individual rich feature is
harmful.

## Precision selections

Defense selects `6000` in every scored fold. Offense selects `4500` or `6000`.
The procedure chooses penalties separately for each prior from earlier game
folds. Box15 does not win because it received a weaker or stronger fixed update.

## Interpretation boundary

These seasons contain reused research evidence. The result chooses the current
research baseline. It does not promote a public model or establish true player
impact. The experiment also compares different target horizons by design:
annual rich SPM predicts one-season RAPM, while Box15 predicts five-year RAPM.
The downstream game test decides which prior works better for AIO.

## Reproduce

```bash
.venv/bin/python research/run_annual_rich_forward_aio.py
```

The run saves priors, player ratings, selected penalties, aligned game
predictions, fold metrics, paired bootstrap draws, coverage checks, matrix
reconstruction checks, and complementarity diagnostics under
`artifacts/research/annual_rich_forward_aio/`.
