# RAPM research lab

This directory contains local RAPM challengers and historical-data utilities.
It is not a production package. Its ratings do not enter the public API or site
unless a separate promotion review approves them.

## Boundaries

- Canonical inputs are read from `data/lake/` and `rapm/data/` in the repo.
- Downloads live in `data/` below this directory and are ignored by Git.
- Results live in `outputs/` below this directory and are ignored by Git.
- Large external mirrors live in `external/external/` and are ignored by Git.
- Season 2027 remains untouched confirmation data.

## Entry points

- `rapm_ridge.py`: zero-prior ridge baseline.
- `apm.py`: unregularized diagnostic baseline.
- `tune_bakeoff.py`: sequential 2024-26 development bake-off.
- `validate_persistence.py`: previous-season persistence gate.
- `run_rolling_rapm.py`: unified 2014-26 rolling RAPM build and lambda matrices.
- `run_lambda_grid.py`: frozen 2019-23 selection and 2024-26 diagnostic grid.
- `run_lambda_frontier.py`: broad scalar, bivariate, adaptive-EB, and GCV search.
- `audit_lambda_frontier.py`: independent GCV-seed and paired-game audit.
- `run_context_adjustments.py`: frozen garbage-time, home, quarter-margin, and
  clock-state ablations on five-year rolling RAPM.
- `run_rubberband_adjustment.py`: actual-clock, whole-game cross-fitted estimate
  of the pre-possession score-margin scoring curve. This estimates the target
  adjustment before any adjusted player rating is fit.
- `run_rubberband_progress_rapm.py`: compares that clock curve with a fixed
  pre-possession-count proxy, refits both adjusted player RAPMs, scores the same
  2026 games, and saves complete player leaderboards.
- `run_rubberband_je_replication.py`: fits JE-style exact pre-possession
  score-margin indicators jointly with 2014-25 player RAPM and checks both
  neutral and observed-score-path predictions on reused 2026 games.
- `run_age_adjusted_rapm.py`: fits categorical offense and defense lineup-age
  controls jointly with 1997-2026 player RAPM, validates actual-age and same-age
  predictions separately, and saves the age-27 leaderboard.
- `run_factor_rapm_reconstruction.py`: fits annual true-shooting, turnover, and
  offensive-rebound RAPMs, then tests a chronologically learned reconstruction
  of annual points RAPM on reused 2026 player-seasons.
- `run_rubberband_5pt_lambda.py`: freezes the empirical five-point score curve,
  score-state controls, and differential offense/defense penalty test.
- `run_wp_rapm_lambda.py` and `run_rolling_5y_wp_rapm.py`: tune conserved
  win-probability credit and build the 2014-2026 rolling five-year panel.
- `run_full_coach_age_rapm.py`: fits the full-span player, age, and coach model
  and compares its descriptive coach ratings with xRAPM.
- `run_pair_exposure_bucketing.py`: tests low-exposure pair retention and stops
  higher-order bucketing when the pair gate fails.
- `run_production_5y_rapm_intervals.py`: builds fixed-window rolling five-year
  ratings with analytic ridge sampling intervals.
- `run_rapm_target_horizon_bakeoff.py`: compares one-, three-, five-, six-year,
  and full-span target panels on identical next-season games.
- `run_ts_factor_rapm.py`: builds six offense/defense TS, turnover, and rebound
  factor ratings.
- `run_luck_teammate_shooting_rapm.py`: tests cross-fitted FT/3P luck adjustment
  and a shooter-excluded teammate-eFG diagnostic.
- `run_ryan_davis_comparison.py`: matches 2019 normal RAPM to Ryan Davis's
  public tutorial output by NBA player ID.
- `run_teammate_play_channels.py`: fits focal-player effects on the other four
  teammates' outcomes plus observable rim, transition, shot, and finish
  channels for 2024-2026.
- `run_time_decay_actual_age_5y_rapm.py`: tunes a five-year half-life, actual
  lineup-age control, and shared player shrinkage on 2025, then applies the
  frozen choice to reused 2026 diagnostics.
- `run_lineup_interactions.py`: residual pair through lineup layers fitted after
  ordinary one-player RAPM.
- `run_standalone_unit_rapm.py`: pair-only, trio-only, four-man-only, and
  lineup-only RAPM. These models contain no individual player columns.
- `run_team_home_adjustments.py`: partially pooled franchise-specific home
  deviations against the frozen global-home baseline.
- `scrape_pbp.py`: guarded, resumable historical play-by-play scraper.
- `bbref_topup.py`: guarded, resumable historical box-score crawler.
- `overnight_pull.py`: local download orchestrator. Read its help before use.

Read `validation_spec_v1.md` before running a scored experiment. Full fits and
downloads are never part of repository smoke tests.

The rolling build stores sparse `X'X`, centered `X'y`, player order, exposure,
and next-season game aggregates. It does not store the possession-level design.
Season 2027 is not loaded.
