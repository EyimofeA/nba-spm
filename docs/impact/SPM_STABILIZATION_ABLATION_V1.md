# SPM stabilization ablation v1

## Answer

Same-season empirical-Bayes stabilization improves the standalone five-year
SPM. It does not improve the final one-season AIO after possession evidence
updates the prior.

Keep stabilization in the research SPM feature layer. Do not cite this run as
evidence that stabilization improves AIO.

## Test

Run `spm_stabilization_ablation_v1_db618f06e8` compares two feature arms.

- The raw and stabilized arms use the same basketball concepts.
- Each arm has 98 offense and 68 defense inputs.
- The offense comparison contains 37 paired concepts. The defense comparison
  contains 10 paired concepts.
- The stabilized arm removes each raw duplicate. No arm contains two values for
  one paired concept.
- Every annual estimate uses only data from that season. The five-year input
  pools five finished annual estimates.
- Both arms use the frozen offense histogram GBM, defense ridge, five-year RAPM
  target, chronological folds, and square-root exposure weights.
- Both priors receive the same one-season terminal-lineup RAPM update with
  `3000 / 3000 / 300` penalties and center scale 1.
- Five scored seasons cover 2022 through 2026. The 2026 rating remains unscored.
  Season 2027 does not enter the run.
- The paired interval resamples whole games within season 5,000 times and gives
  each season equal weight.

The run derives raw defended-shot fields by reversing their documented
same-season reliability weights. It derives the raw passer composite from raw
load, assist-to-load, turnover-to-load, and creation-to-load values. Missing
values remain missing until each training fold fits its own median imputer.

## Results

### Five-year RAPM target fit

Mean weighted metrics across the six chronological rating folds:

| Component | Raw RMSE | Stabilized RMSE | Raw correlation | Stabilized correlation |
| --- | ---: | ---: | ---: | ---: |
| Offense | 1.0258 | 1.0230 | 0.7426 | 0.7467 |
| Defense | 1.1039 | 1.0484 | 0.5178 | 0.5961 |
| Net | 1.5281 | 1.4803 | 0.6607 | 0.6982 |

Stabilization improves every component as a five-year RAPM emulator. Defense
shows the largest gain.

### Next-season games

| Rating | Equal-season RMSE | MSE difference, raw minus stabilized | 95% interval | Probability stabilized has lower MSE | Fold wins, stabilized |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone SPM | 14.6251 raw / 14.6105 stabilized | +0.4252 | [-0.2914, +1.1672] | 88.70% | 4 of 5 |
| SPM prior plus RAPM | 14.4300 raw / 14.4254 stabilized | +0.1338 | [-0.2335, +0.5050] | 76.82% | 3 of 5 |

The standalone result points toward a small stabilization gain, but its 95%
interval crosses zero. The final AIO result is a precise practical tie at the
point estimate and remains uncertain across resamples.

## Data quality

- Both AIO arms score identical games in every fold.
- Prior possession coverage is 100% for offense and defense.
- Offense plus defense equals net exactly.
- Annual matrix reconstruction error stays below `1.14e-13`.
- The artifact contains no Season 2027 row.
- All recorded artifact hashes reproduce.

Stabilization reduces missing values because low-opportunity observations can
borrow their own season's league center. Across the paired feature ledger, mean
missing fractions fall from 14.71% to 9.61% on offense and from 17.28% to 9.61%
on defense. This behavior belongs to the stabilization treatment. It does not
come from another season.

## Decision

Retain same-season stabilization for standalone SPM research. The current test
does not justify preferring the stabilized prior for AIO. The one-season RAPM
likelihood absorbs the small standalone difference.

This run uses reused historical seasons. It cannot promote a public model.

## Reproduction

```bash
PYTHONPATH=src:research .venv/bin/python \
  research/run_spm_stabilization_ablation.py
```

The immutable artifact lives at
`artifacts/research/spm_stabilization_ablation/spm_stabilization_ablation_v1_db618f06e8`.
