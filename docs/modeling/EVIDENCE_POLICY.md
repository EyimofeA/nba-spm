# Model Evidence Policy

## The short answer

Do not rerun a deterministic RAPM on the same split. That produces the same model
and adds no evidence.

For one new idea against one baseline:

1. **Screen:** four chronological test seasons, each using the prior three seasons
   for training. This is eight fits total: baseline + candidate in each fold.
2. **Research claim:** preferably six to eight chronological test seasons, paired
   game-level comparison, and no important era or minutes subgroup regression.
3. **Production claim:** pass the research gate, freeze the design, then confirm
   once on a season that was untouched during development.
4. **Stochastic models only:** use at least three seeds per fold and report both
   between-season and between-seed variation.

Three reruns on one season are weaker than one run on three genuinely different
seasons. Seasons—not random seeds—are the main replication unit for deterministic
RAPM research.

## Current automatic labels

The walk-forward harness uses the first candidate as the baseline.

- `insufficient_folds`: fewer than three chronological outer folds.
- `promising_research_challenger`: at least 90% paired-bootstrap probability of
  lower game-margin loss and wins at least half the folds.
- `candidate_requires_untouched_confirmation`: at least 1% mean RMSE improvement,
  at least 95% bootstrap probability, and wins at least 70% of folds.
- `improvement_not_demonstrated`: the candidate did not clear those thresholds.

These labels control research attention. They do not turn lineup-conditioned
retrodiction into a deployable forecast.

## Multiple ideas

If there are `k` deterministic candidates and `f` folds, the basic cost is
`f × (k + 1)` fits because the baseline is reusable within each fold. Hyperparameters
must be chosen inside earlier data; repeatedly choosing ideas from the same outer
fold eventually overfits that fold. Keep the newest complete season sealed for final
confirmation.
