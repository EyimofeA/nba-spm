# Impact validation v2

## Decision

CourtSignal will validate retrospective impact and current strength separately.
The two models answer different questions and cannot share one promotion score.

The frozen machine-readable contract is
`research/experiments/impact_validation_v2.yml`.

## Retrospective impact

The retrospective rating describes one completed regular season. It may use
that season's statistics and possessions. It does not use an aging trajectory
or a latent current-strength state.

Its primary test holds out whole games from the same season. Each fold must
remove the held-out games from both the RAPM likelihood and every feature
derived from that season. The SPM mapping and every tuning choice use earlier
seasons only. The model predicts the held-out games with their observed
lineups.

This test requires timestamped raw inputs. A season-total tracking feature is
ineligible when the pipeline cannot reconstruct it after removing the held-out
games.

The test measures reconstruction and reliability. It does not establish causal
player value.

## Current strength

The current-strength rating asks what was knowable at a historical cutoff and
how well it predicted the next 14 days.

The model creates a snapshot every Monday from November 1 through April 1. A
latent feature may combine the player's previous posterior, new observations,
age, time decay, and source availability. It may never use information after
the cutoff.

Two future tests remain separate.

1. The oracle-lineup test uses observed future lineups only as exposure. It
   isolates player-quality estimation but is not deployable.
2. The operational test uses only projected minutes, roster status, and injury
   information available before tipoff. It scores margin RMSE, log loss, Brier
   score, and calibration.

## Ordered decision rule

The primary comparison uses identical whole games and equal season weights.
The paired bootstrap resamples whole games within season for 5,000 draws.

A challenger must satisfy every condition:

- lower RMSE by at least 0.05 points per game;
- paired MSE interval entirely below zero;
- correlation decline no greater than 0.01;
- calibration-slope distance from one cannot worsen by more than 0.05;
- no exposure or source-era segment with at least 500 games can worsen RMSE by
  more than 0.10.

The evaluation reports the metrics in this order. It does not average them into
one composite rank.

## Evidence boundary

Box-only models may use 2004--17 for development and 2018--21 for selection.
Rich features have only 2018--20 development and 2021 selection evidence.
Seasons 2022--26 are reused diagnostics. Season 2027 remains the untouched
confirmation.

The short independent history for rich features limits any promotion claim.

## Data gate

Every feature must declare its grain, key, source, numerator, denominator, unit,
availability date, stabilization rule, and leakage class.

The pipeline stops before fitting when it finds an impossible physical value,
a unit mismatch, an unexplained one-sign distribution for a two-sided metric,
future information, or double-counted possession outcomes.

This gate caught two material defects. The first distribution audit found an
invalid zTS fallback and a unit mismatch in the stored rim-points-saved input.

The corrected pipeline rejects player TS outside `[0, 1.5]`, repairs invalid
source rows with the same-season valid median, and applies a same-season
100-attempt empirical-Bayes prior. zTS subtracts the best available expected
shot mix from that stabilized TS. The corrected
rim-defense builder normalizes `DFG%` and expected `FG%` to percentage points
before subtraction and aggregates repeated observations with defended-shot
weights. Run `spm_input_distribution_audit_v1_f54723b16e` reports no blocking
feature failures. Historical rich-SPM results that consumed the earlier inputs
remain provisional until refit. Box15 does not consume either field.

## Implementation status

Gate A implements same-season blocked-game reconstruction for Box15. Run
`impact_validation_v2_gate_a_090cb2d323` uses the 466 regulation games whose
cached possession points match the official home and away final scores. It
removes each held game from the current-season Box15 ledger and RAPM
likelihood. Box15 AIO improves RMSE by `.2449` points per game. Its paired MSE
interval is `[-12.3020, -2.1471]`, and calibration slope improves from `1.1325`
to `1.0023`.

The run passes the four scored Gate A conditions for further research. The
source-era segment gate remains unscored, so production promotion remains
false. An earlier 1,079-game run scored the legacy cache's internal target. It
included truncated overtime games and 556 regulation games with side-specific
score mismatches, so it is not official game-margin evidence.

The strict gate retains 466 of 1,079 cache games, or 43.2%. The retained games
have a larger mean absolute final margin than the excluded games, 13.24 versus
11.40 points, and eligibility changes over the season. The result therefore
applies only to the score-conserved regulation subset. It is not a
representative full-season estimate. A stronger gate requires a rebuilt
score-conserving possession source or replication across additional seasons.

The five held-game folds are date-sorted round-robin folds. They cover the
whole season and are not forward chronological folds. A future run should save
the full inner alpha-selection table in addition to the selected penalties.

The repository implements a current-strength oracle-lineup diagnostic with
chronological season folds, identical games, equal-season MSE, calibration,
and paired whole-game resampling. It does not yet implement weekly historical
cutoffs or projected pregame minutes. Raw timestamped tracking inputs still
limit blocked-game reconstruction for the full feature bank.

The BoxSPM versus TrackingSPM pilot uses one reused oracle-lineup fold. It
trains on five-year windows ending before 2025, forms a 2025 statistical prior,
applies the fixed 2025 one-season RAPM update, and scores identical 2026 games.
This pilot diagnoses the feature banks. It cannot promote either model.
