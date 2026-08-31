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

The existing full SPM remains immutable. The compact contract is a new
research input until that comparison runs.
