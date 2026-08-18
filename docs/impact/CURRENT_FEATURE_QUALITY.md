# Current Statistical Feature Quality

Updated 2026-08-18. This note covers season labels 2025 and 2026, which mean the
2024–25 and 2025–26 NBA seasons.

## Decision

- Use the 2025 feature panel for diagnosis and research.
- Do not publish a 2025 annual AIO from the frozen SPM model.
- Do not build a 2026 annual AIO. The source season is incomplete.
- Do not tune the frozen model on the inspected 2025 result.

## Feature coverage

The base annual panel is
`statistical_features_v1_5db80fc1de`. It contains 6,918 player-seasons and 97
features for 2014–26. The 2025 schema renames `Offensive Fouls` to
`Offensive_Fouls`; the builder now maps this explicit alias.

The complete 2025 panel is `statistical_features_v2_9e7c27e281`. It contains
6,360 player-seasons and 265 features for 2014–25. It joins:

- playtype and zTS run `playtype_features_v1_b84ebdf73e`;
- DFG, rim-defense, and hustle run
  `defensive_tracking_features_v1_2148109d4a`.

All player-season keys are unique. There are no non-finite values or bounded
ratio violations. The defensive source joins cover 99.64% of DFG rows, 99.65%
of rim rows, and 100% of hustle rows.

The 2026 base sheet has 81.8% of the prior two-season median offensive/defensive
possession exposure. The feature builder marks it
`structurally_validated_partial_latest_season`. Structural validity does not
mean season completeness.

## Exact selected SPM coverage audit (2025--26)

The current research run
`single_season_spm_v1_47b3bd9b17` selects exactly 127 offense and 68 defense
columns. Its feature input is
`statistical_features_v2_b808fc1bf1`, created 2026-08-17. The input artifact
has all 195 column names because the builder median-neutral-fills missing
merged-family values. A column being present in that parquet is therefore not
evidence that the underlying source was observed for a player-season.

The following table measures source-row exposure among the 1,000-possession
eligible rows used by the selected SPM contract (`OffPoss >= 1000` and
`DefPoss >= 1000`). “Observed” means a matching `PLAYER_ID` exists in the
versioned source-family artifact for that season; it is not a claim that all
possible players were active or that every source statistic was non-zero.

| Selected source family | Selected fields | 2025 observed / eligible | 2026 observed / eligible | Current artifact | Status |
|---|---:|---:|---:|---|---|
| Box/player-sheet base | Core box and rate fields | 379 / 379 | 385 / 385 | `statistical_features_v1_5db80fc1de` | present, but 2026 exposure is partial |
| Playtype/zTS | 1 selected field (`zts_pct_points`) | 378 / 379 | 0 / 385 | `playtype_features_v1_b84ebdf73e` | 2026 blocker; 2025 one unmatched ID |
| DFG/rim/hustle tracking | 10 selected defense fields | 379 / 379 | 0 / 385 | `defensive_tracking_features_v1_b0bf4ef279` | 2026 blocker |
| Scorer-adjusted matchup defense | 8 selected defense fields | 0 / 379 | 0 / 385 | `matchup_defense_features_v1_b265e245c4` | 2025--26 blocker; artifact stops before 2025 |

At the complete feature-panel level, only 97 of 569 2025 rows and 105 of 582
2026 rows are finite across all 127 offense columns; the corresponding counts
for all 68 defense columns are 290 and 294. These are diagnostics, not an
alternative eligibility rule. The selected model can run because missing
values are neutral-filled, but a current SPM/AIO rebuild should not be called
defensible while entire selected source families are absent for a season.

The current feature run records this explicitly: `new_features=203`,
`new_feature_missing_values_after_neutral_fill=0`, and a maximum pre-fill
missing fraction of `0.4469893402477672`. The zero after-fill count is a
pipeline invariant, not a coverage guarantee.

## Smallest rights-aware refresh plan

Do not silently replace the pinned source revision. The local ingest manifests
pin Gabriel1200 `site_Data` at `bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`,
while the repository currently reports a newer HEAD (`782ec8b4c09fdfb023f06dbcd3e601123cf6d698`)
and a reorganized per-season layout. Both the existing manifest and the
upstream repositories declare no license; keep the data research-only and do
not put raw rows in the website bundle.

The narrowest reproducible refresh is:

1. Pin and QA Gabriel `player_sheets` 2025 and 2026 from a specific revision;
   do not fit until row counts, required identifiers, exposure, and hashes are
   recorded.
2. Add a versioned adapter/manifest for the current per-season Gabriel
   `site_Data` layout and fetch only the 2025--26 playtype, shooting, and
   tracking files needed to reconstruct the normalized family artifacts.
3. Obtain 2026 DFG/rim/hustle from a source with explicit redistribution
   permission, or leave those fields unavailable. Gabriel's current tree does
   not expose a 2026 DFG/rim pair and does not provide a reproducible 2026
   hustle family in the audited layout.
4. Extend the matchup source only after rights and provenance are documented.
   Gabriel does not supply the eight scorer-adjusted matchup fields; the
   selected matchup artifact is not current through 2025. If it cannot be
   legally and reproducibly extended, remove the matchup block in a separately
   versioned research challenger rather than neutral-filling it and calling
   the result a refresh.
5. Rebuild each family into a new hash-addressed artifact, rerun the exact
   127/68 coverage table, and only then refit SPM/AIO. No existing public table
   should be overwritten in place.

This plan requires a small, auditable current download, not a bulk NBA archive.
The two hard blockers are current 2026 tracking coverage and current
scorer-adjusted matchup coverage; possession data alone does not resolve them.

## Frozen 2025 confirmation

Run `current_spm_confirmation_v1_9b4cca0b12` applies the saved 2014–24 models
from `single_season_spm_v1_bff6060df6` to 2025 features. It does not refit or
tune the models. The target is one-season terminal-lineup normal RAPM with
zero-prior penalties 3000/3000/300.

The run matches all 569 feature players to the RAPM target. It covers 1,226
regular-season games and 247,630 possessions. Names and predictions are
complete.

| Component | 2025 RMSE | 2017–24 worst RMSE | 2025 correlation | 2017–24 worst correlation |
|---|---:|---:|---:|---:|
| Offense | 1.102 | 1.041 | 0.619 | 0.583 |
| Defense | 1.154 | 1.019 | 0.331 | 0.467 |
| Net | 1.610 | 1.453 | 0.500 | 0.534 |

All three components are outside the historical RMSE range. Defense and net
are also outside the historical correlation range. This range check was added
after the 2025 result, so it is transparent diagnostic evidence, not a
predeclared promotion gate. The decision is still clear: do not promote.

## Pipeline checks

The rebuilt 2024 panel matches the frozen 2024 panel exactly for every selected
feature and every saved-model prediction. This rules out the schema update as
the cause of the lower 2025 score.

Legacy and canonical one-season RAPM targets on the shared 2024 season correlate
0.974 for offense, 0.964 for defense, and 0.975 for net. Scoring the same 2024
out-of-fold predictions against the canonical target adds about 0.03 RMSE per
component. That target-pipeline difference is real but is too small to explain
the 2025 regression.

The largest 2025 feature shifts are in contested-shot and hustle rates. Some are
continuations of multi-season league trends. Treat them as diagnostic leads, not
as proof of a source break.

## Next experiment

Run `current_spm_diagnostics_v1_59632783de` completes the no-tuning failure
audit. Low exposure does not explain the miss: the 276 players above 2,000
possessions have defensive RMSE 1.232 and correlation 0.433. Their target
standard deviation is 1.346, while the frozen predictions have standard
deviation 0.649. The model is under-dispersed for this group.

Target instability also does not explain the high-exposure regression. First-
half versus second-half defensive RAPM correlation at 1,000 possessions in each
half is 0.329 in 2024 and 0.331 in 2025. This is weak absolute reliability, but
it did not deteriorate in 2025.

The defensive tracking block still adds signal. Neutralizing all ten DFG, rim,
and hustle features worsens 2025 defense RMSE from 1.154 to 1.182 and correlation
from 0.331 to 0.230. Neutralizing only DFG/rim is worse than neutralizing only
hustle. Do not remove the block.

Develop the next defensive challenger with nested or forward-only validation
inside 2014–24. Run `annual_defense_ridge_nested_v1_5b06407982` already tests
whether the fixed ridge penalty causes the compressed spread. It selects among
300, 1000, 3000, and 10000 using only the two prior seasons inside each 2020–24
outer fold. Adaptive selection beats fixed 3000 in only one of five folds and
slightly worsens mean RMSE (+0.0015) and correlation (-0.0004). It fails the
predeclared gate. Do not change ridge strength.

Focus next on additional stable defensive signal and calibration, not on deleting
tracking, filtering low-minute players, or retuning alpha. Reserve the next
complete season for one untouched confirmation.

The tracking builder now also emits three interpretable interactions:

- overall DFG two-point-equivalent points saved per 100;
- rim matchup-attempt share;
- contested-three share.

Run `annual_defense_features_nested_v1_22b677e1ef` compares the frozen baseline,
these matchup interactions, seven existing defensive interactions, four
era-relative rates, and their union. Each outer 2020–24 fold selects a block
using only its two prior validation seasons. The selected block wins two of five
outer folds, worsens mean RMSE by 0.00024, and worsens mean correlation by
0.00274. It fails the predeclared gate. Keep the derived fields for research,
but do not add them to the frozen SPM.

An earlier run, `annual_defense_features_nested_v1_d913f807c5`, used a panel
built through 2025. Global fallback medians could therefore see 2025. Treat that
run as invalid for the pre-2025 contract. The final run rebuilds all defensive
features strictly through 2024 and reaches the same non-promotion conclusion.

The next defense lane needs genuinely new information, not another recombination
of the same annual aggregates. The first matchup-assignment lane is complete in
`matchup_defense_features_v1_86d13d7357`. It covers 2018–25 and passes identity,
point-conservation, exposure, and finite-value gates.

Run `annual_defense_features_nested_v1_eaeca704eb` tests the new data without
using 2025. Each 2022–24 outer fold selects among four predeclared matchup blocks
on its two prior seasons. The event-context block is selected in all three folds.
It lowers weighted RMSE in all three folds and by 0.0131 on average. However, it
lowers mean correlation by 0.0202. The predeclared gate fails, so do not add the
block to the frozen SPM.

The matchup block increases prediction spread toward the target spread, but its
player ranking is worse. Raw leaderboards also show strong role structure. The
next matchup experiment must control shot location, offensive role, and lineup
context. Do not subset-search these inspected 2022–24 folds.
