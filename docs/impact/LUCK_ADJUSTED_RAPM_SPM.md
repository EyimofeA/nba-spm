# Luck-adjusted RAPM and SPM

## Decision

Keep normal realized-points RAPM. Removing broad shot-conversion variance makes
ratings smoother, but it worsens future-game prediction. The narrower FT/3P
adjustment nearly repeats its earlier 2026 gain, yet loses in 2025 and remains
inside sampling noise.

## What was tested

The possession model and `3000/3000/300` ridge penalties stay fixed. Only the
scoring target or the reported component blocks change.

| Arm | Offense | Defense |
| --- | --- | --- |
| Normal | Realized-points RAPM | Realized-points RAPM |
| Opponent luck adjusted | Normal offense | Player-neutral expected-conversion defense |
| Teammate and opponent adjusted | Expected-conversion offense plus repeatable shooter skill | Player-neutral expected-conversion defense |
| Full expected outcome | Player-neutral expected-conversion offense | Player-neutral expected-conversion defense |

The shooter-skill add-back is offense-only. For player `j`, it is the sum of
his pre-season empirical-Bayes expected points above player-neutral shot value,
divided by his offensive possessions. This prevents a player's repeatable
shooting from becoming a defensive effect or being assigned to all teammates.

## Expectation model

Field-goal expectation uses shot location, distance, angle, clock, score state,
home status, shot value, period, and zone. It never uses shooter, defender,
team, lineup, or the shot result. The logistic regularization grid is selected
on a chronological split inside 2024. The chosen `C` is `0.05`; all three
candidates are practically tied.

The 2024 expected shots are cross-fitted by whole game. The 2025 model trains
on 2024. The 2026 model trains on 2024-25. Current-game outcomes never enter
their own expectation.

Repeatable shooter skill uses annual histories before the adjusted season.
Prior sample size and half-life are selected on next-season binomial log loss
over 2019-24:

| Skill | Prior attempts | Half-life |
| --- | ---: | ---: |
| FT | 50 | 1 year |
| Rim | 100 | 1 year |
| Short midrange | 100 | 1 year |
| Long midrange | 100 | 2 years |
| Corner three | 200 | 5 years |
| Above-break three | 200 | 5 years |

The possession target replaces only mapped conversion points:

`expected points = actual possession points - mapped actual conversion + mapped expected conversion`.

All turnovers, offensive rebounds, shot selection, free-throw generation, and
unmapped events remain in the target. Mapped conversion events account for
99.00% of all possession points.

## Results

### Future-game margin RMSE

| Arm | 2025 | 2026 |
| --- | ---: | ---: |
| Normal | 15.0541 | 15.4732 |
| Opponent luck adjusted | 15.5132 | 15.5675 |
| Teammate and opponent adjusted | 15.6971 | 15.5693 |
| Full expected outcome | 15.7966 | 15.7941 |
| Narrow FT/3P diagnostic | 15.1197 | 15.3898 |

The full expected-outcome model is under-dispersed: its predicted game-margin
standard deviation falls to `3.46` in 2025, versus `6.63` for normal RAPM.
Removing conversion also removes repeatable signal that the lineup model needs.

The narrow FT/3P diagnostic changes RMSE by `+0.0656` in 2025 and `-0.0834` in
2026. Its paired whole-game 2026 interval is `[-0.2318, +0.0717]`. That closely
reproduces the earlier `-0.093` point estimate, including the failure to clear
uncertainty.

### Future normal-RAPM net prediction

| Arm | 2025 RMSE / corr | 2026 RMSE / corr |
| --- | ---: | ---: |
| Normal | 2.0925 / .3811 | 2.1134 / .4755 |
| Opponent adjusted | 1.9945 / .3327 | 2.0852 / .4116 |
| Teammate and opponent adjusted | 1.9458 / .3379 | 1.9507 / .4100 |
| Full expected outcome | 1.8591 / .3053 | 1.9181 / .3914 |
| Narrow FT/3P | 1.9010 / .4015 | 1.9588 / .4838 |

The broad targets reduce RMSE by shrinking player spread, not by preserving
rank signal. Their calibration slopes and correlations are worse. Easier RAPM
reconstruction is not sufficient when future-game margins get worse.

## Why no luck-adjusted SPM was fit

Complete shot-level expected-outcome labels start in 2024. A 2026 output would
have only two legal labeled seasons, 2024 and 2025. That cannot support the
required chronological feature selection and model comparison. A forced fit
would produce a leaderboard, not evidence.

The blocker is historical shot-event coverage, not compute. Until more seasons
have current-game-out expected outcomes, normal RAPM remains the SPM target and
the luck arms remain research diagnostics.

## Reproduction

- Contract: `research/experiments/luck_adjusted_rapm_spm_v1.yml`
- Runner: `research/run_luck_adjusted_rapm.py`
- Source artifact: `artifacts/models/luck_adjusted_rapm/luck_adjusted_rapm_v1_8580bb30e9`
- Audit: `research/audits/luck_adjusted_rapm_spm_v1`

Seasons 2025 and 2026 are reused diagnostics. Season 2027 was not loaded.
