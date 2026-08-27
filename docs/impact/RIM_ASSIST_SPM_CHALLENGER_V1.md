# Rim-assist SPM challenger v1

## Decision

Keep rim assists in the player-skill layer. Do not add rim assists to the
five-year SPM or AIO prior.

## Feature

The source counts assists on rim attempts. The model converts the count into a
same-season empirical-Bayes rate:

```text
100 * (rim assists + 500 * season league rate) / (offensive possessions + 500)
```

The estimator uses no prior-season or future-season center. The five-year
feature pools five frozen annual rates with offensive-possession weights.

The source observes 89.23% of matched player-window rows and 98.79% of matched
possessions. Missing rows mainly represent fringe players. The pipeline fills
those rows with the same-window possession-weighted league center. The model
does not receive a missingness indicator.

## Test

The control uses the 15 BoxPIPM-style fields and CourtSignal five-year RAPM
labels. The challenger adds only stabilized rim assists to the offense model.
The defense prior stays fixed. Both priors receive the same one-season
terminal-lineup RAPM update with `3000 / 3000 / 300` penalties. Both models
score the same 3,687 games in 2022--24.

| Prior | Mean margin RMSE | Mean margin correlation | Fold wins |
|---|---:|---:|---:|
| Control | **13.8644** | **.3635** | 3 |
| Control plus rim assists | 13.8764 | .3620 | 0 |

The challenger increases paired whole-game MSE by `0.337` points squared per
game. The 5,000-draw 95% interval is `[+0.172, +0.508]`.

## Interpretation

The 15-field control already contains assists and scoring volume. Rim assists
do not add enough independent signal to improve future-game predictions in
this sample. Rim assists still describe passing style and creation. The
predictive player-skill model should continue to display the statistic.

## Reproduce

```bash
PYTHONPATH=src:research OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 .venv/bin/python research/run_rim_assist_spm_challenger.py
```

Artifact: `rim_assist_spm_challenger_v1_23e599d812`. The manifest records all
source hashes. Season 2027 remains absent.
