# RAPM Publication Diagnostics V1

## Decision

The five-year RAPM point estimates are reproducible and the active player graph
is connected in every 2018--26 window. Do not publish analytic intervals as
uncertainty about true player impact. They measure sampling variation around a
ridge estimator and become artificially narrow for heavily shrunk,
low-exposure players.

## Inputs

- Point estimates: `rolling_5y_rapm_2014_2026_a7754bfb77`.
- Fixed-window analytic intervals:
  `production_5y_rapm_intervals_v1_e86fb09750`.
- External comparison:
  `external_reproduction_benchmark_v1_0a95702214`.
- Publication audit: `rapm_publication_diagnostics_v1_f0c9c3f65b`.

Season 2027 was not loaded.

## Connectivity

The stored matrices include players who appear only in the next-season
evaluation set. The audit excludes those zero-training-exposure players before
building the lineup graph. Every remaining five-year player graph has one
connected component. Only the 2018 John Wall--Marcin Gortat pair exceeds the
predeclared 0.80 teammate-column cosine flag.

The release metadata includes each player's minimum side possessions, most
linked teammate on each side, maximum teammate cosine, distinct teammates,
graph component, analytic standard errors, and a separation status. These are
design diagnostics. They do not measure basketball quality.

## Scale audit

Against exact player-season Ryan Davis normal RAPM rows, CourtSignal has pooled
net correlation 0.967 and a CourtSignal-on-reference slope of 1.391. Exact
five-year windows have net correlation 0.957 and slope 1.350. The slope falls
with exposure. Annual net slopes are about 1.73 in the lowest exposure quartile
and 1.30 in the highest. Five-year net slopes are about 2.19 and 1.22.

This pattern rules out a single global display correction. It is consistent
with different low-exposure handling, penalty choices, centering populations,
or player-universe rules. The audit does not identify one cause because the
external model contract is not identical. Keep CourtSignal in its fitted units.

## Uncertainty boundary

The repository has two complete 1,000-draw whole-game bootstrap pilots: 2025
single-season and 2022--24 trailing RAPM. It also has analytic intervals for all
nine fixed five-year windows. Bootstrap agreement with analytic widths is close
for high-exposure players in the two pilots.

Neither method corrects ridge bias. Low-exposure analytic net standard errors
are smaller than standard-exposure errors because the penalty fixes those
coefficients near zero. Therefore:

- publish point estimates and exposure;
- label analytic intervals as fixed-estimator sampling diagnostics;
- do not publish exact-rank claims from those intervals;
- require a bias-aware simulation or posterior-coverage study before calling
  them player-impact error bars.
