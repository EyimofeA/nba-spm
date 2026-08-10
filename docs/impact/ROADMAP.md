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

The v2 feature layer keeps the learners frozen while it tests basketball-domain
engineering. It adds stabilized percentages, era-relative rates, recent-season
levels, trends, volatility, scoring topology, creation quality, behavioral role,
defensive interactions, and versioned public basketball formulas. Run
`statistical_feature_v2_comparison_9b8d0555e0` selects stabilized ratios,
era-relative rates, recent levels, temporal dynamics, and the public-metric
block for offense on 2022–23. It selects no defensive block. On the reused 2024
check, net RMSE improves from 1.29843 to 1.26244 and correlation improves from
0.54462 to 0.57485. This is an exploratory research challenger because earlier
family comparisons already inspected 2024. More subset search on these folds is
not valid.

Run `statistical_priors_v1_2c81b23662` creates the historical handoff. For each
eligible prediction window `T`, its offense GBM and defense ridge train only on
target windows ending by `T-3`. It predicts every feature-covered player before
joining labels for evaluation. The output covers 4,656 player-windows from
2019–24 with no duplicate keys, missing values, non-finite values, or purge
violations. Six-fold prior-only net RMSE is 1.25131 and correlation is 0.51980.
These are same-window retrodictions. They are not forecasts, and 2022–24 are not
untouched promotion evidence.

External run `external_impact_benchmark_v1_bab43a4087` aggregates annual
Basketball Reference BPM and xRAPM with NBA minutes over the same three-season
windows. It matches at least 98.47% of SPM rows per window. On the 2,295 windows
with at least 3,000 offensive and defensive possessions, net SPM has Pearson
correlation 0.876 with BPM and 0.756 with xRAPM. The weakest external agreement
is defensive SPM versus defensive xRAPM at 0.630. This is a diagnostic for
feature work, not evidence that either external metric is truth.

Annual run `single_season_spm_v1_51adc53061` is separate from the rolling
model. It builds current-season-only features for 2014–24, excludes temporal
features, and learns one global mapping with each reported 2017–24 season held
out in turn. The final descriptive leaderboard refits on all 2014–24 labels.
Across eight held-out seasons, net weighted RMSE is 1.4611 and mean correlation
is 0.5314 against noisy one-season normal RAPM. On 2,860 matched player-seasons
with at least 1,000 possessions per side, net correlation is 0.897 with BPM and
0.762 with xRAPM. Annual xRAPM is still a multi-window, prior-informed metric;
it is not a one-season label. Defensive SPM versus xRAPM correlation is 0.590.
The saved disagreement tables use the same high-exposure rule.

Annual playtype run `single_season_spm_v1_fcdb9559f6` adds exact project zTS to
offense. Mean held-out offense RMSE improves from 1.0060 to 0.9972 and
correlation from 0.6178 to 0.6302. A larger playtype block is nearly identical
and is not promoted. The next feature task is a canonical annual DFG,
rim-defense, and hustle layer; those raw tables exist but are not in the clean
annual model.

That task is complete in `defensive_tracking_features_v1_9f66c664eb`. The ten
feature block improves defensive RMSE/correlation from 1.0578/0.3091 to
0.9595/0.4964 and wins both metrics in all eight annual folds. Keep the block
whole; do not subset-search the inspected folds. Contract
`configs/models/annual_spm_v1.json` freezes offense histogram GBM plus zTS and
defense ridge plus the full ten-feature tracking block. Cleaned assist-quality
features remain research inputs. Run `single_season_spm_v1_d6de68348c` loses
offense and net accuracy in both 2023 and 2024, so they are not in the contract.

The next feature challenger uses
[`FACTOR_DECOMPOSITION.md`](FACTOR_DECOMPOSITION.md). The basketball factors are
feature families and explanation groups. The supervised targets remain direct
offensive and defensive RAPM. Factor RAPM is a separate research branch, not a
dependency for the first AIO.

## Phase 4 — Create the all-in-one rating

1. Evaluate the statistical prior by itself.
2. Use the prior as the center of the possession-level RAPM penalty.
3. Tune prior strength only inside chronological training data.
4. Compare zero-prior RAPM, prior-only SPM, and prior-informed RAPM on identical
   future games and player-seasons.
5. Publish offense, defense, net, uncertainty, minutes/possessions, prior value,
   RAPM evidence, and the amount of shrinkage.
6. Keep the simplest repeated winner as the production all-in-one.

The first integration gate is complete in run
`prior_informed_rapm_v1_122ef63045`. Scale 1.0 won selection on 2020–22. On the
later 2023–24 check it improved mean game-margin RMSE by only 0.00327, won one of
two folds, and produced an equal-season paired-game MSE delta of -0.195 with a
95% bootstrap interval from -1.119 to +0.729. The prior-only model was worse
(13.7952 RMSE) than zero-prior RAPM (13.5296). This does not demonstrate a
repeatable prior benefit, so zero-prior remains the production RAPM and the
statistical rating remains a separately labeled research estimate.

Do not tune more prior scales on these seasons. A future integration test needs
genuinely new data or one predeclared sample-size-adaptive shrinkage rule.

The first annual integration is complete. Run `annual_spm_priors_v1_1107680642`
fits each SPM(T) mapping only on seasons before T, then predicts the complete
season-T feature table. Run `prior_informed_rapm_v1_e1239679c1` compares normal
one-season RAPM(T), SPM(T) alone, and RAPM(T) centered exactly on SPM(T), with no
amplitude search, on season T+1 game margins. The full center beats normal RAPM
in all three later 2022–24 tests: mean RMSE is 13.8118 versus 13.8892. The paired
equal-season game-MSE delta is -2.143 with a 95% bootstrap interval from -3.270
to -0.965. Prior-only SPM is worse at 14.0128 RMSE. This supports the combined
model as a research challenger. It is not clean production promotion evidence
because 2022–24 influenced earlier feature work.

The all-in-one must be decomposable. A user must be able to see why the final
rating differs from raw RAPM or the box/tracking prior.

Run `annual_aio_ratings_v1_23c4895f8f` is the first decomposed rating panel. It
contains 4,341 player-seasons for 2017–24 with complete names and prior coverage.
Each row exposes raw and possession-centered SPM, zero-prior normal RAPM, final
AIO offense/defense/net, the RAPM update from the centered SPM value, offensive
and defensive possessions, and annual rank. Component identities hold to
floating-point precision. The 2024 cache has 1,229 regular-season games, so that
season remains one game short.

## Phase 5 — Dynamic careers and peaks

- Produce one-, three-, and five-year peak tables.
- Fit a validated aging curve without using future seasons in past estimates.
- Compare fixed rolling windows with a time-decayed or state-space trajectory.
- Publish annual points with uncertainty. Interpolate only for display; do not
  present interpolation as new evidence.
- Add career and peak views in the style of NBA RAPM peak datasets.

The rolling normal-RAPM peak table is complete in
`rolling_rapm_peaks_v1_584adf4f3d`. It independently fits 26 three-year and 24
five-year regular-season windows over 1997–2024. The model uses zero-prior
3000/3000/300 ridge, terminal lineups, a home term, and season scoring-environment
normalization. Published peaks require at least 1,000 offensive and defensive
possessions per window season. The output contains 36,530 rolling ratings and
7,866 player/component peaks with no duplicate keys or missing peak names.
Three-year and five-year peak values have 0.963 correlation for 1,223 players
eligible in both. The two unresolved archive IDs, 471 and 775, occur only in
1997, are not peak eligible, and remain explicitly unnamed rather than guessed.

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

The read-only API, first player trajectory page, current possession quality gate,
and frozen 2024–26 normal RAPM are complete. The page now shows that current
normal RAPM beside annual AIO and historical rolling peaks. Do not expand it into
a full site yet. Next build current 2025 and 2026 statistical feature panels,
measure their schema/coverage drift against 2024, and only then extend annual AIO
beyond 2024. Use the saved annual defensive disagreements to define future
defensive feature families; do not tune them on the same 2017–24 table.
