# SPM feature selection

## Current compact contract

`compact_spm_correlated_v1` starts from the corrected 128-offense and
72-defense feature contract. It removes at least one member of every
within-side feature pair with absolute correlation of at least `.95`.

The compact contract has 105 offense inputs and 58 defense inputs. It keeps
`crafted_spacing_stable_v1` as the three-point-volume representative. It also
keeps empirical-Bayes shot-zone frequencies, season-relative scoring and
turnover rates, and expected rather than completed passing volume.

The pruning is unsupervised. It does not use a RAPM target, a game outcome, or
a fitted model. It therefore cannot establish that the compact contract
predicts better. It only removes obvious duplicate encodings and makes later
importance analysis easier to read.

## Why correlation matters

High correlation does not automatically damage prediction. Ridge can share
weight across correlated inputs, and boosted trees can choose either input.
The main problems are interpretation and stability. Permutation importance
splits credit across substitutes. A small data change can make a tree switch
from one substitute to another. Duplicate inputs also spend model capacity on
the same basketball fact.

Correlation becomes more serious when a model uses unregularized linear
coefficients, when the sample is small, or when the feature source changes by
season. It matters less when the goal is prediction, the learner regularizes
well, and all sources remain stable.

## Selection methods

Use these methods in order. Later methods do not repair a bad source or a
leaking feature.

1. **Lineage and basketball meaning.** Reject mislabeled fields, future
   information, unstable units, and variables that do not match the model
   side. This step removed the stale relative true-shooting field before any
   selection.
2. **Coverage and range checks.** Inspect missingness by season, impossible
   values, source-era breaks, low variance, and extreme tails. Keep an
   availability flag when a real data source starts late.
3. **Correlation pruning.** Group features within offense and defense. Select
   a stable and interpretable representative from each near-duplicate group.
   Do not prune across sides because offense and defense have separate models.
4. **Grouped chronological ablation.** Add or remove a basketball family, such
   as spacing or rim defense, using earlier seasons only. Score the family on
   later seasons. This tests transferable signal instead of one noisy column.
5. **Regularized linear selection.** Ridge keeps correlated groups. Elastic
   net can keep groups while shrinking weak inputs. Lasso selects one member
   aggressively and often changes its choice across folds.
6. **Stability selection.** Repeat elastic-net or permutation selection across
   chronological folds and bootstrap samples. Retain features or families that
   recur. Report the selection frequency.
7. **Conditional importance.** Permute a feature within a similar role or
   condition on its correlated group. Ordinary permutation importance can
   understate every member of a redundant group.
8. **Wrapper selection.** Sequential selection and recursive elimination can
   work after pruning. They cost more and can overfit a short season history.
   Use them only inside nested chronological validation.

## Recommended model-selection rule

Freeze the feature families and target before a scored run. Select features on
earlier chronological folds. Compare the compact model with the full model on
identical later games. Report standalone SPM error, calibration, and the final
AIO game-margin difference. Promote a feature set only when the downstream AIO
gain repeats across seasons and does not come from one source era or exposure
group.

The existing full SPM remains immutable. Run
`compact_spm_comparison_v1_2a0f8a6f31` fits the compact contract. Compact SPM
improves standalone next-season MSE, but its RAPM posterior does not improve on
Full SPM. Keep it as a research and interpretation arm.

## Exhaustive registry and mechanism screen

Registry `exhaustive_spm_feature_registry_v1_1ea059390e` inventories 337 unique
fields: the 295-field research panel, 34 predictive-skill definitions, and eight
new mechanism features. It assigns every field to a modeling lane before target
selection. The clean retrospective lane excludes `OnOffRtg` and `OnDefRtg` as
lineup-derived circular inputs. It also keeps 91 trajectory and predictive-skill
fields outside the retrospective model.

Panel `mechanism_feature_panel_v1_9224606a01` adds these same-season features:

| Side | Feature | Definition |
| --- | --- | --- |
| Offense | pass value | Empirical-Bayes assist points created per potential assist |
| Offense | load-adjusted shot quality | Leave-one-player-out shot-quality residual after load and shot-location mix |
| Offense | load-adjusted creation | Leave-one-player-out Box Creation residual after load, touches, and potential assists |
| Offense | spacing-creation interaction | Stabilized spacing multiplied by stabilized Box Creation |
| Defense | rebound conversion above expected | Exposure-shrunk defensive-rebound conversion residual after chance volume and contest share |
| Defense | foul-adjusted activity | Exposure-shrunk contest, deflection, and recovered-block activity residual after fouls |
| Defense | workload-adjusted suppression | Exposure-shrunk matchup points-saved residual after defended-shot workload |
| Defense | rim workload value | Stabilized rim points saved multiplied by the square root of rim attempts defended |

The residual features use analytic weighted leave-one-player-out regression
within each season. They do not use RAPM, future seasons, player identity, or an
external impact metric. Five-year values possession-weight the already frozen
annual values.

Run `mechanism_feature_challenger_v1_5f3e0bad98` rejects the offensive block.
Its AIO RMSE is `14.4266`, compared with `14.3792` for Box15. The defensive
block lowers AIO RMSE to `14.3516`, improves four of five folds, and has a paired
Box15-minus-challenger MSE interval of `[0.2557, 1.3193]`.

Run `defense_mechanism_screen_v1_181a68516f` isolates the defensive inputs.
Every feature improves the AIO point estimate. Workload-adjusted suppression
and rim workload value produce the largest individual gains. The four-feature
block performs best. Its RMSE gain is `.0275` points per game. That result falls
below the frozen `.05` practical threshold and loses narrowly in the reused
2026 fold. Keep the block as a research challenger. Do not replace Box15.

## Independent model audit

An isolated external audit used Pi `0.84.2`, Cursor, and
`cursor/fable-5@1m` at high reasoning. The model had no repository context or
tools during its independent design pass.

The audit recommends these learner roles:

- Keep ridge as the permanent baseline and default defense learner.
- Use histogram gradient boosting as the primary nonlinear offense challenger.
- Use elastic net for fold-internal stability selection, not as the presumed
  production model.
- Use a low-degree GAM as the first interpretable nonlinear diagnostic after
  pruning.
- Use ExtraTrees only as a bounded robustness and importance check.
- Fit one nonnegative out-of-fold blend only after two base learners pass on
  their own.

The audit also recommended source-era panels, repeated noise and
target-permutation controls, and a consensus feature rule learned only inside
each chronological training fold.

Follow-up run `spm_consensus_complementarity_v1_8f49b7448f` implements those
controls. Each fold uses 40 player-cluster bootstrap fits, 20 Gaussian noise
inputs, within-season target permutations, and a recurrence cutoff no lower
than `.60` or the largest observed null recurrence. Box15 remains mandatory.

The selected additions improve standalone stable-RAPM reconstruction. They
worsen the AIO by `1.588` MSE against target-excluded Box15, with a paired 95%
interval of `[1.232, 1.947]`, and lose nine of ten seasons. This closes broad
consensus selection as a null. Recurrence identifies stable RAPM signal. It
does not identify signal that complements the possession likelihood.
