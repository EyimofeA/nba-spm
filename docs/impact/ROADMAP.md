# RAPM and All-in-One Roadmap

This file gives the detailed player-impact plan. `../../ROADMAP.md` remains the
short project queue. `../modeling/PLAYBOOK.md` gives the validation rules.

## Current facts

- Canonical rich possessions cover 2023–24 through 2025–26.
- Legacy possessions cover 1997–2024 but are stale and use a smaller schema.
- Current zero-prior RAPM is a baseline, not the final rating.
- Start-lineup attribution is rejected on one outer fold.
- Terminal lineup is the provisional simple policy.
- Fractional exposure is a project-created sensitivity analysis, not a published
  standard. Commit `db4cb02` introduced it on 2026-08-08.
- Existing SPM/all-in-one outputs are stale and contain known processed-data and
  prior problems. Do not promote them unchanged.

## Phase 1 — Freeze current RAPM

1. Generalize the lineup-policy comparison to explicit train/test seasons.
2. Compare terminal and fractional policies on:
   - train 2023–24 → test 2024–25;
   - train 2024–25 → test 2025–26.
3. Use identical regular-season possessions and whole-game bootstrap intervals.
4. Keep terminal unless fractional shows a reliable repeated gain.
5. Select offensive, defensive, and home-court ridge penalties using only older
   training seasons. Do not tune on 2025–26.
6. Refit the frozen specification and publish player offense, defense, net,
   possessions, standard errors or bootstrap intervals, and model provenance.

Production target: a simple current regular-season RAPM. Playoff RAPM remains a
separate low-sample product.

## Phase 2 — Historical and alternate RAPM products

After current RAPM is frozen:

- annual one-season RAPM;
- rolling three-year and five-year RAPM;
- playoff RAPM with regular-season shrinkage;
- time-decayed RAPM with predeclared decay and rolling validation;
- role/lineup-context research variants;
- WP-RAPM and Net Points credit as separate estimands.

Do not merge these into one unexplained number. Publish the estimand and window
with every rating.

## Phase 3 — Build independent statistical priors

Create time-safe player-season features:

- core box score and age;
- shooting zones and efficiency;
- tracking and hustle where available;
- playtype and role features where available;
- prior seasons and career history;
- availability only when the prediction contract permits it.

Create separate era models because tracking/playtype coverage begins much later
than box-score coverage. Exclude target-derived on/off, plus-minus, and team-rating
features.

Predict next-window offensive and defensive RAPM separately. Start with ridge or
elastic net. Compare bounded tree models only after the linear baseline passes.
Use minute-weighted chronological player-season evaluation and player-clustered
uncertainty.

## Phase 4 — Create the all-in-one rating

1. Evaluate the statistical prior by itself.
2. Use the prior as the center of the possession-level RAPM penalty.
3. Tune prior strength only inside chronological training data.
4. Compare zero-prior RAPM, prior-only SPM, and prior-informed RAPM on identical
   future games and player-seasons.
5. Publish offense, defense, net, uncertainty, minutes/possessions, prior value,
   RAPM evidence, and the amount of shrinkage.
6. Keep the simplest repeated winner as the production all-in-one.

The all-in-one must be decomposable. A user must be able to see why the final
rating differs from raw RAPM or the box/tracking prior.

## Phase 5 — Dynamic careers and peaks

- Produce one-, three-, and five-year peak tables.
- Fit a validated aging curve without using future seasons in past estimates.
- Compare fixed rolling windows with a time-decayed or state-space trajectory.
- Publish annual points with uncertainty. Interpolate only for display; do not
  present interpolation as new evidence.
- Add career and peak views in the style of NBA RAPM peak datasets.

## Phase 6 — Product contract

Freeze Parquet/DuckDB schemas first. Then add a thin read-only API for:

- leaderboards and filters;
- player rating decomposition;
- annual and peak trajectories;
- model version, coverage, and uncertainty;
- later WP and research-query endpoints.

Training remains offline. The API serves versioned artifacts. Build the frontend
only after these contracts are stable.

## Immediate next task

Generalize `rapm_lineup_policy_v1` to both chronological folds. Confirm terminal
versus fractional attribution. Do not start the all-in-one model until the RAPM
label and lineup policy are frozen.
