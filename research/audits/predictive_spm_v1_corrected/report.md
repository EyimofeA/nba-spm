# Corrected predictive SPM decision

## Decision

The frozen next-season SPM is useful but does not pass its promotion contract.
Keep it as a research comparator. Do not use it as the production predictive
SPM or AIO prior.

## What changed

The corrected runner parses the YAML contract, enforces exact artifact IDs and
folds before data access, rejects Season 2027, scores all arms on identical
player-seasons, computes weighted correlation, reports held-out weighted
calibration, records row inclusion, and checkpoints every fold. The fit itself,
features, learners, seasons, and targets did not change.

## Results

| Scope | Persistence net RMSE | Raw SPM net RMSE | Calibrated SPM net RMSE |
|---|---:|---:|---:|
| Development, 2019-24 | 1.9651 | 1.6094 | 1.6083 |
| Reused diagnostics, 2025-26 | 2.0826 | 1.7546 | 1.7563 |

The model beats persistence in every scored season. That is not enough for
promotion. The preregistered defense dispersion gate requires a prediction to
target spread ratio from 0.85 to 1.15 in both reused diagnostic seasons. Raw
SPM reaches only 0.353 in 2025 and 0.334 in 2026. Calibration compresses it
further, to 0.322 and 0.309.

The apparent net RMSE gain therefore comes partly from conservative shrinkage.
The model ranks future defense moderately, but it does not reproduce the
cross-player spread required by the frozen contract.

## Data policy

- Development and selection used 2019-24 only.
- Seasons 2025-26 were rescored as already-inspected diagnostics; they did not
  alter the features, learners, calibration rule, or decision gate.
- Season 2027 was rejected before any parquet read and was not loaded.

Source artifact: `predictive_spm_v1_9392b98d58`.
