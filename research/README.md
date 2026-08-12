# Research control plane

This directory contains the small machine-readable layer that controls model
claims. It is authoritative for estimand names, season exposure, and
preregistered experiments. Model artifacts remain under `artifacts/`.

- `estimands.yml` separates retrospective impact, latent strength, forecasts,
  playoffs, peaks, and win probability.
- `season_exposure.yml` records which modern seasons have already influenced
  development or selection.
- `pinned_artifact_audit.csv` maps each Ratings API field to its current artifact
  and unresolved lineage work.
- `experiments/` contains frozen experiment plans. A plan does not imply that a
  model is approved or has run.

The provisional public flagship is retrospective single-season impact. The
production reference method is terminal-lineup, zero-prior normal RAPM. Annual
SPM and prior-informed AIO remain research challengers.

Season 2027 is reserved as the next untouched annual confirmation. Do not use
its outcome for feature selection, hyperparameter selection, or debugging.
