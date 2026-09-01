# SPM consensus selection and shared-error diagnostic

## Decision

Box15 remains the research AIO prior. A fold-local consensus selector improves
standalone stable-RAPM reconstruction, but it makes the downstream AIO worse.
The result closes the broad stability-selection proposal as a null.

A second diagnostic removes the shared-reference defect from the earlier error
correlation test. Rich SPM has higher error correlation on both sides, but
neither difference clears the material threshold. A held-out likelihood test
does show that rich SPM complements RAPM worse. It does not prove which rich
inputs cause the difference.

Run: `spm_consensus_complementarity_v1_8f49b7448f`

## Selector

Every rating season from 2016 through 2025 runs a separate expanding fold. The
training target is nine-year RAPM ending before the feature season. The selector
never sees the scored rating season or its next-season games.

The selector applies these steps separately to offense and defense:

1. Remove constants and one member of every training-fold correlation pair at
   or above `.95`.
2. Run 40 player-cluster bootstrap fits. A sampled player contributes every
   historical row in the training fold.
3. Fit an elastic net with alpha `1.0` and L1 ratio `.1` in each resample.
4. Add 20 deterministic Gaussian noise inputs.
5. Repeat each fit after permuting the target within season.
6. Set the recurrence cutoff to the larger of `.60` and the maximum observed
   null recurrence.
7. Keep Box15 as a mandatory base and add only rich features that meet the
   fold-local cutoff.

The cutoff ranges from `.60` to `.80`. The selector retains 31 to 42 offense
features and 24 to 35 defense features, including the mandatory Box15 fields.

The following offense additions survive all ten folds:

- behavioral passer score;
- Box Creation;
- Offensive Load;
- zTS;
- spacing and creation interaction;
- usage events;
- open shot attempts;
- bad-pass turnovers;
- stabilized bad-pass turnover share;
- stabilized long-midrange accuracy.

The following defense additions survive all ten folds:

- stabilized rim field-goal residual;
- stabilized defended field-goal residual;
- defensive rebound chances;
- passes per touch;
- usage events;
- defensive rebound contests.

Selection recurrence does not establish downstream value. Several recurrent
defense fields measure realized opponent shooting outcomes. Those fields can
reconstruct RAPM while adding little information after the possession update.

## Standalone result

The table reports the equal-season mean of possession-weighted player-level
RMSE against the past-only nine-year RAPM target. Lower values are better.

| Prior | Offense RMSE | Defense RMSE | Net RMSE |
| --- | ---: | ---: | ---: |
| Box15 | 1.589 | 1.406 | 2.242 |
| Box15 plus consensus features | 1.459 | 1.345 | 2.090 |
| Full rich SPM | **1.444** | **1.345** | **2.082** |

The consensus model removes most of the standalone gap between Box15 and the
full rich model. This confirms that the selected additions contain stable RAPM
signal.

## AIO result

Every candidate receives the same one-season terminal-lineup RAPM update with
penalties `3000 / 4500 / 300`. All candidates score the same 11,969 next-season
games.

| Candidate | Ten-fold MSE | RMSE | Wins versus target-excluded Box15 |
| --- | ---: | ---: | ---: |
| Current-control Box15 | 191.038 | 13.822 | 3 |
| Target-excluded Box15 | **190.988** | **13.820** | reference |
| Box15 plus consensus features | 192.576 | 13.877 | 1 |
| Full rich SPM | 193.130 | 13.897 | 0 |

Consensus minus target-excluded Box15 MSE is `+1.588`. Its paired 95% whole-game
interval is `[+1.232, +1.947]`. The consensus prior loses nine of ten folds.

On the five later diagnostic seasons, consensus minus current-control Box15 MSE
is `+2.192`. The corresponding RMSE loss is `.076` points per game, and the
consensus model wins no season.

Stability selection solves neither the reversal nor the prior-design problem.
The rich signal remains useful as a standalone impact estimate and harmful as
an AIO center.

## Disjoint-reference error test

The earlier diagnostic subtracted the same future RAPM reference from both
errors. That construction creates positive correlation even after target
permutation.

The replacement fits two future RAPMs on disjoint game halves. It calculates
the two cross-reference correlations and averages them:

```text
correlation(prior - future_A, one_year_RAPM - future_B)
correlation(prior - future_B, one_year_RAPM - future_A)
```

No game appears in both future references. Eight rating seasons have three
complete future seasons. Each half uses one-half of the full target penalty so
the penalty-to-possession ratio remains fixed.

| Side | Mean rich-minus-Box15 error correlation | 80% interval | 95% interval | Positive seasons |
| --- | ---: | ---: | ---: | ---: |
| Offense | +0.025 | [+0.018, +0.032] | [+0.014, +0.036] | 8 of 8 |
| Defense | +0.040 | [+0.032, +0.047] | [+0.028, +0.051] | 8 of 8 |

Neither side clears the preregistered `.05` material-difference rule. The
consistent positive direction is suggestive, but the test remains unresolved.
The result no longer depends on a common subtracted reference.

## Held-out likelihood check

The second check uses fully lagged priors. It splits each rating season by game,
fits the RAPM update on one half, and scores the other half. It then swaps the
halves. Statistical inputs, possession fitting rows, and scored games are
disjoint in time or by game. The half-season fit uses one-half of the full AIO
penalties so the penalty-to-possession ratio remains fixed.

| Prior | Equal-season MSE | RMSE | Season wins |
| --- | ---: | ---: | ---: |
| Box15 | **175.175** | **13.235** | 8 |
| Rich SPM | 175.666 | 13.254 | 2 |

Rich minus Box15 MSE is `+0.490`. Its paired 95% interval is
`[+0.111, +0.875]`. This check establishes that the fully lagged rich prior
works worse with an independently fitted RAPM update. It does not identify the
source of that failure.

## Interpretation

The evidence now supports a narrower claim. Rich SPM adds less complementary
information to a partial-season RAPM update than Box15 does. Cross-reference
error correlations point toward overlap, but their differences are too small
for the frozen material rule. The available data cannot prove that one feature
family causes the failure.

The recurrent defense additions explain the likely mechanism. Rim and defended
field-goal residuals encode realized possession outcomes. They help recreate a
long RAPM target. The one-season possession likelihood already observes related
outcomes. Removing those fields made the rich model worse in the earlier
outcome-censoring test, so deletion does not produce a better prior.

The dual-head decision remains:

- `spm_impact`: full rich SPM;
- `spm_prior`: Box15;
- AIO: one-season RAPM centered on `spm_prior`.

This run does not change the public model, site, or API.
