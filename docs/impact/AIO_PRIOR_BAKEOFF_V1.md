# AIO prior bake-off

Status: reused development result. No production change.

## The question

The earlier cheating-ladder run compared standalone player ratings. It asked
whether five-year SPM or a BoxPIPM-style model better reproduced team wins and
five-year RAPM. That does not answer whether PIPM is the better prior inside
the all-in-one model.

The all-in-one estimate is a new ridge fit:

`posterior = one-season possession likelihood updated around a player prior`.

Changing the prior changes the ridge solution. The possession likelihood can
shrink or reverse a difference seen between the standalone ratings. The proper
test must replace only the prior and then score the resulting posteriors.

## Frozen comparison

Run `aio_prior_bakeoff_v1_0a3591a402` holds these terms fixed:

- one rated season of terminal-lineup possessions;
- five offensive and five defensive player columns plus home court;
- penalties `3000 / 3000 / 300`;
- prior center scale `1.0`;
- identical next-season games for every arm.

The five arms are zero-prior RAPM, the frozen five-year SPM, the selected
five-year SPM, a forward-chained BoxPIPM-style prior, and a PIPM-like prior that
adds derived raw on/off context to the box model. BoxPIPM-style is not full
historical PIPM. The original model also used luck-adjusted on/off and details
that are not fully public.

The box and PIPM-like priors train only on five-year target windows ending
before the rating season. Rating seasons are 2021 through 2023. Test seasons
are 2022 through 2024. Season 2027 is absent.

## Result

Game-margin RMSE, lower is better:

| Test season | Zero prior | Frozen 5Y SPM | Selected 5Y SPM | BoxPIPM prior | PIPM-like prior |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 14.4623 | 14.3692 | 14.3595 | **14.2915** | 14.2886 |
| 2023 | 12.8325 | 12.7184 | **12.7122** | 12.7423 | 12.7279 |
| 2024 | 14.7669 | 14.6032 | 14.5988 | **14.4743** | 14.4769 |
| Mean | 14.0206 | 13.8969 | 13.8902 | **13.8360** | 13.8312 |

The BoxPIPM prior lowers mean RMSE by `0.0541` points per game versus the
selected five-year SPM prior. It wins two of three seasons. Its paired
whole-game mean-squared-error difference is `-1.6008`, with a 95 percent
bootstrap interval of `[-2.6854, -0.5279]`.

The PIPM-like arm lowers mean RMSE by `0.0590` and also wins two seasons. It is
not the preferred result. It reuses same-season on/off information in both the
prior and the RAPM likelihood, and its on/off input is not luck adjusted.

The BoxPIPM result is the useful finding. A much smaller box feature set made a
better prior after the possession update, even though the standalone box model
lost badly to SPM in the prior experiment.

## Decision

Do not replace the current research SPM yet. These seasons have already been
used for development, and the selected SPM arm itself reflects feature choices
made with 2022 through 2024 evidence. The result earns a clean follow-up:

1. build the BoxPIPM prior through 2026 with the canonical possession source;
2. compare it with the five-year SPM prior on the reused 2025 and 2026 games;
3. freeze the winner and leave 2027 untouched for confirmation.

Reproduction:

- contract: `research/experiments/aio_prior_bakeoff_v1.yml`;
- runner: `research/run_aio_prior_bakeoff.py`;
- artifact: `artifacts/research/aio_prior_bakeoff/aio_prior_bakeoff_v1_0a3591a402`.
