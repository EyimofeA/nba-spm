# AIO prior canonical follow-up

Status: frozen research prior selected. No production or public-rating change.

## Answer

BoxPIPM-style is the better tested prior for the current AIO update. It beat the
selected five-year SPM prior on both reused follow-up seasons and passed the
predeclared paired-game gate. This answers the prior question; it does not say
that standalone BoxPIPM is the better player rating.

The distinction is:

`prior -> one-season possession RAPM update -> AIO -> next-season games`

The earlier standalone experiment stopped before the RAPM update. This run
holds the update fixed and changes only its center.

## Frozen design

Run `aio_prior_canonical_followup_v1_8c61405875` uses:

- rating seasons 2024 and 2025;
- test seasons 2025 and 2026;
- one rated season of terminal-lineup possession evidence;
- five offensive and five defensive player columns plus home court;
- ridge penalties `3000 / 3000 / 300`;
- prior-center scale `1.0`;
- identical next-season games for every arm.

No Season 2027 row is loaded. The two annual training matrices were recovered
from stored canonical five-year sufficient statistics and direct annual
matrices. Recombining each annual matrix reproduced its stored five-year source
to floating-point precision. The selected-SPM arm reproduced the prior frozen
AIO evaluation within `9.2e-11` RMSE and `2.9e-12` correlation.

## Result

Game-margin RMSE, lower is better:

| Test season | Zero prior | Frozen 5Y SPM | Selected 5Y SPM | BoxPIPM prior | PIPM-like prior |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 15.0541 | 14.8963 | 14.8944 | **14.8081** | 14.8338 |
| 2026 | 15.7326 | 15.4107 | 15.4233 | **15.3943** | 15.4128 |
| Mean | 15.3933 | 15.1535 | 15.1588 | **15.1012** | 15.1233 |

BoxPIPM lowers equal-season mean RMSE by `0.0577` points per game versus the
selected five-year SPM prior. Its paired whole-game mean-squared-error
difference is `-1.7293`; the season-stratified 95 percent bootstrap interval is
`[-3.0402, -0.4367]`, with `99.46%` of draws below zero.

Mean game-margin correlation is `0.3652` for BoxPIPM and `0.3636` for selected
five-year SPM. BoxPIPM covers 100 percent of rated-season offensive and
defensive possessions after missing priors are centered at league average.

The PIPM-like raw-on/off arm is not eligible. Its same-season on/off input
reuses lineup outcome evidence that is already present in the RAPM likelihood.

## Decision

Freeze BoxPIPM-style as the research AIO prior. Keep zero-prior RAPM as the
public reference and do not alter the production site. The BoxPIPM AIO still
needs the untouched 2027 confirmation before production promotion.

Reproduction:

- contract: `research/experiments/aio_prior_canonical_followup_v1.yml`;
- runner: `research/run_aio_prior_canonical_followup.py`;
- artifact: `artifacts/research/aio_prior_canonical_followup/aio_prior_canonical_followup_v1_8c61405875`.
