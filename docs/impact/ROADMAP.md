# RAPM and All-in-One Roadmap

This file gives the detailed player-impact plan. `../../ROADMAP.md` remains the
short project queue. `../modeling/PLAYBOOK.md` gives the validation rules.

## Current facts

- Canonical rich possessions cover 2023–24 through 2025–26.
- Legacy possessions cover 1997–2024 but are stale and use a smaller schema.
- Current zero-prior RAPM is a baseline, not the final rating.
- Normal RAPM uses terminal-lineup assignment. Fractional exposure remains a
  research sensitivity and is not part of the active production path.
- Fractional exposure is a project-created sensitivity analysis, not a published
  standard. Commit `db4cb02` introduced it on 2026-08-08.
- Existing SPM/all-in-one outputs are stale and contain known processed-data and
  prior problems. Do not promote them unchanged.

## Phase 1 — Freeze current RAPM

1. ~~Generalize the lineup-policy comparison to explicit train/test seasons.~~
2. ~~Compare terminal and fractional policies on:~~
   - train 2023–24 → test 2024–25;
   - train 2024–25 → test 2025–26.
3. ~~Use identical regular-season possessions and whole-game bootstrap intervals.~~
4. ~~Compare lineup assignment.~~ Fractional had the best RMSE in both
   folds. Its pooled squared-error gain was 1.18 versus start and 0.81 versus
   terminal; both 95% whole-game intervals excluded zero. The first fold was
   effectively tied versus start, so this is a small engineering decision rather
   than a broad production claim. The active normal RAPM still uses the simpler
   terminal assignment.
5. ~~Select ridge penalties inside older data and confirm once on 2025–26.~~ The
   selected 4500/4500/1000 candidate lost to 3000/3000/300 on confirmation.
6. ~~Refit the frozen specification and publish offense, defense, net,
   possessions, and provenance.~~ Uncertainty is deferred by user direction.

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

Create time-safe player-window features. The primary model excludes age,
experience, height, listed position, minutes, and games. Possession and attempt
counts can supply reliability weights but cannot enter as predictive features.

Candidate feature families include:

- core box rates;
- shooting zones and efficiency;
- tracking and hustle where available;
- playtype and role features where available;
- learned behavioral role features rather than listed positions.

Create separate era models because tracking/playtype coverage begins much later
than box-score coverage. Exclude target-derived on/off, plus-minus, and team-rating
features. Evaluate on/off in a separately labeled impact-assisted challenger.

Predict next-window offensive and defensive RAPM separately. Start with ridge or
elastic net. Compare bounded tree models only after the linear baseline passes.
Use purged chronological player-window evaluation and possession-based reliability
weights. Do not use minutes or games as input columns.

For the linear ridge baseline, fit offense and defense separately and add the
predictions for net impact. Run `statistical_impact_v2_48f6ad776f` found no
measurable advantage from fitting net RAPM directly on the advanced feature sets.
The separate models preserve the requested offense/defense decomposition.

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

Test direct nonlinear net RAPM against the component model selected by
`statistical_model_comparison_v1_dd31e7957d`: histogram GBM offense plus ridge
defense. Keep the same purged folds and inner-only tuning. Then run feature-family
ablations and compare the user's feature set under the same player-window contract.
