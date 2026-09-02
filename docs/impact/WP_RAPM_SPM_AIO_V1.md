# WP-RAPM statistical-prior test

## Decision

Do not use Box15 or rich SPM as a prior for WP-RAPM. Both priors worsened the
same next-season games relative to zero-centered WP-RAPM.

## Estimand

The model assigns lineup-adjusted credit for changes in player-neutral win
probability. It does not estimate ordinary point impact.

For each possession, a past-only logistic surface estimates the home team's
win probability before the possession. Historical seasons use score, home
possession, overtime, and possession progress. The target is the next state's
probability minus the current state's probability. The last possession closes
to the observed game result. The possession changes therefore conserve the
game's final result minus its initial probability.

WP-RAPM regresses this conserved change on five offensive players, five
defensive players, and home possession. The frozen penalties are 3,000 on
offense, 10,000 on defense, and 300 on home.

## Statistical priors

The Box15 and rich statistical models train on earlier rolling five-year
WP-RAPM targets. Each prediction becomes the center of a one-season WP-RAPM
fit. The zero-centered control uses the same possession rows and penalties.

The run scores rating seasons 2021 through 2025 on outcome seasons 2022
through 2026. Every candidate uses the same games. The paired comparison uses
5,000 within-season whole-game bootstrap draws.

## Results

| Candidate | Folds | Mean MSE | Mean RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-centered WP-RAPM | 5 | 1.1245 | 0.9805 | 0.1450 | 0.2873 |
| Box15 WP-AIO | 5 | 1.4679 | 1.1305 | 0.1492 | 0.1146 |
| Rich WP-AIO | 5 | 1.4582 | 1.1190 | 0.1573 | 0.1334 |

Box15 minus zero-centered MSE is `+0.3430`. Its paired 95% interval is
`[+0.3176, +0.3699]`. Rich minus zero-centered MSE is `+0.3335`. Its interval
is `[+0.3082, +0.3596]`. Neither prior wins a bootstrap draw.

The rich model reconstructs the historical WP-RAPM target better than Box15,
but that advantage does not survive the one-season lineup update. This mirrors
the complementarity problem found in points-based AIO research.

## Limits

- Historical timing uses possession progress, not exact clock time.
- The rating measures leverage-weighted win-probability credit.
- The result is a reused historical diagnostic, not untouched confirmation.
- The statistical-prior leaderboard remains local research only.

Run: `wp_spm_aio_v1_5d7272a48f`.

## Full historical WP-RAPM panel

The separate full-history build uses 6,738,828 source possessions. It produces
25 rolling five-year windows from 1998–2002 through 2022–2026. The panel covers
34,344 rating games and conserves game WP to a maximum absolute error of
`1.11e-15`. The past-only progress surfaces have Brier scores from `0.1614` to
`0.1766` and AUC from `0.8027` to `0.8411`.

Run: `rolling_5y_wp_rapm_v1_0e6e0304f0`.
