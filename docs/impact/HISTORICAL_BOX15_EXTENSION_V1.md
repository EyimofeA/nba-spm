# Historical Box15 Extension

## Decision

Keep Box15 unchanged. Extend the research history without changing the modern
model.

RAPM covers 1997--2026. A complete five-year Box15 SPM prior and its
single-season RAPM posterior cover 2001--2026. The 2001--03 priors are
descriptive full-fit backcasts. The 2004--26 priors train only on earlier
five-year windows.

Season 2027 was not loaded.

## Probability best

The earlier 57.08% value came from 5,000 paired whole-game bootstrap draws.
Each draw resampled games within each of five outcome seasons, calculated each
candidate's mean season-level MSE, and counted the candidate with the smallest
MSE. Box15 won 2,854 draws.

This is not a posterior probability that Box15 is the true best model. It is a
resampling stability measure over the same reused seasons and candidate set.

## Focused feature follow-up

Run `box15_top_feature_followup_v1_d9c274ca12` tested the strongest non-Box15
fields from the frozen chronological full-SPM permutation audit.

The offense addition used lost-ball turnovers, long-midrange frequency, drive
free throws, short-midrange attempts, and drive turnovers. The defense addition
used assist points created, touches, defensive-rebound chances, recovered
blocks, and paint touches. The defense fields include role proxies and do not
have a causal defensive interpretation.

| AIO prior | MSE | RMSE | Margin correlation |
| --- | ---: | ---: | ---: |
| Box15 plus defense fields | 207.355 | 14.400 | 0.3621 |
| Box15 | 207.421 | 14.402 | 0.3616 |
| Box15 plus both sides | 207.893 | 14.418 | 0.3594 |
| Box15 plus offense fields | 207.982 | 14.422 | 0.3589 |

The defense-only point gain is too small to adopt. Box15 minus defense-only MSE
is `+0.066`, with paired interval `[-0.160, +0.317]`. The offense and combined
arms are worse with intervals that exclude zero in Box15's favor.

## Historical inputs and targets

The extension imports the already-audited local terminal-lineup caches for
1997--2018. Hashes for the overlapping 2019--23 files match the active checkout.
No network scraper was required.

The 1997--2000 player sheets lack `OffPoss`. Their box rates use the observed
`POSS` field. From 2001 onward, box rates use `OffPoss`. RAPM targets and SPM
sample weights always use exact lineup-derived offensive and defensive
possessions.

The builder independently refits the 2014--18 five-year RAPM window. Its
offense, defense, and net values match the pinned 2018 target with zero maximum
absolute error. Annual RAPM overlap from 2014 through 2018 also matches exactly.

## Modern accuracy retention

Run `historical_box15_validation_v1_fa08210f64` changes only the amount of
earlier SPM training history. Both candidates use the same Box15 features, ridge
selection, one-season RAPM update, penalties, games, lineups, and outcomes.

| AIO history | Equal-season MSE | RMSE | Mean correlation |
| --- | ---: | ---: | ---: |
| Expanded 2001 onward | 207.416 | 14.4019 | 0.3632 |
| Original 2018 onward | 207.421 | 14.4021 | 0.3616 |

The RMSE difference is `-0.0002`. The paired MSE interval is
`[-0.363, +0.363]`. Expanded history preserves modern accuracy. It does not
provide evidence of a better modern model.

The prior-only expanded model changes more than the posterior, while the final
AIO barely changes. This is expected. The AIO uses the same season's possession
likelihood, so the RAPM update absorbs much of the prior difference.

## Rating and site artifacts

- `historical_box15_ratings_v1_d65c267829` contains 1997--2026 RAPM,
  2001--2026 SPM priors, and 2001--2026 AIO ratings.
- `historical_impact_web_bundle_v1_bfd251d751` contains 14,570 named
  player-seasons and one compact JSON file per season.

The bundle is local. It has not replaced the current web snapshot or been
deployed. The site integration must update the catalog descriptions, season
selectors, and release manifest together.
