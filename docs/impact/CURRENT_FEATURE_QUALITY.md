# Current Statistical Feature Quality

Updated 2026-08-10. This note covers season labels 2025 and 2026, which mean the
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
inside 2014–24. Focus on additional stable defensive signal and calibration,
not on deleting tracking or filtering low-minute players. Reserve the next
complete season for one untouched confirmation.
