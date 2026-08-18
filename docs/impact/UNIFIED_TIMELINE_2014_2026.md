# Unified annual RAPM, SPM, and AIO timeline: 2014--2026

Status: research timeline complete. It is not a public-model promotion.

## What is unified

One terminal-lineup adapter now fits every annual RAPM and AIO season through
the same interface:

| Seasons | Input contract | Rows in complete AIO run |
| --- | --- | ---: |
| 2014--2023 | score-conserved legacy possession cache with terminal lineups | 5,791 |
| 2024--2026 | canonical NBA event possessions with ordinal terminal lineups | 1,151 |
| 2014--2026 | explicit source-transition panel | 6,942 |

The transition is deliberate. The source format changes in 2024, but the
estimand does not: regular-season, terminal-lineup, zero-prior ridge RAPM with
3000 / 3000 / 300 offense / defense / home penalties. The independently
audited 2024 overlap passed before canonical rows replaced legacy rows.

The unified run is `unified_annual_aio_ratings_v1_72d524ff46`.

- 1,706 distinct players; 6,942 unique player-season rows.
- No missing names or duplicate player-season keys.
- 100% player-prior and lineup-slot prior coverage.
- `offense + defense = net` has maximum floating-point error
  `8.88e-16`.
- Its zero-prior RAPM columns exactly reproduce the unified annual target panel
  for every component and every player-season.

This does not assert that the two raw-event systems are identical. The source
is retained per season in the run manifest. The historical V3 reconstruction
is still research-only because it did not beat the legacy source on its frozen
held-out prediction gate.

## SPM across the whole timeline

Run `single_season_spm_v1_423ec3c88e` trains and scores every season from
2014 through 2026. Each rated season is held out from its own training labels.
The remaining twelve seasons train the retrospective SPM mapping. The final
model then fits all thirteen seasons for the descriptive leaderboard.

| Component | Held-out seasons | Mean weighted RMSE | Mean correlation |
| --- | ---: | ---: | ---: |
| Offense | 13 | 1.0109 | 0.6284 |
| Defense | 13 | 0.9730 | 0.5067 |
| Net | 13 | 1.4103 | 0.6016 |

The model uses the existing frozen 127 offense and 68 defense feature lists.
zTS is present from 2014 onward. 2026 playtype, tracking, and hustle data are
now present. 2026 DFG/rim-DFG and scorer-adjusted matchup fields remain
unobserved, so their neutral-filled current-season model results are research
only.

## Chronological SPM train-window comparison

All four variants use the same 2017--2026 test seasons, target rows, 127/68
feature contract, and exact current data. The training set is strictly earlier
seasons. “Expanding” means all earlier seasons; the other variants retain only
the most recent 1, 3, or 5 earlier seasons.

| Window | Offense RMSE / corr | Defense RMSE / corr | Net RMSE / corr |
| --- | --- | --- | --- |
| Expanding | **1.0253 / .6228** | **.9908 / .4949** | **1.4307 / .5860** |
| Five years | 1.0279 / .6222 | .9912 / **.4983** | 1.4346 / **.5889** |
| Three years | 1.0323 / .6185 | 1.0007 / .4951 | 1.4426 / .5860 |
| One year | 1.0770 / .5854 | 1.0072 / .4735 | 1.4791 / .5559 |

Decision: use expanding history as the default SPM training rule. It wins the
primary error metric for offense, defense, and net. Five years is a useful
stability sensitivity, not the selected default. This comparison does not
promote the 2025--26 public SPM/AIO: those current defense seasons remain
inspected research outputs.

## Why 2014--16 were previously absent from the site

They were not absent from the underlying RAPM target panel. The public site
began in 2017 because the former public SPM/AIO artifact was only validated and
exported for 2017--24. This unified research run supplies 2014--16 SPM and AIO
rows, but must remain visibly research-scoped until the site contract is
updated and the chosen public promotion rule is met.
