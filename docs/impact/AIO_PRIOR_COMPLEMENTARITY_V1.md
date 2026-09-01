# AIO Prior Complementarity Experiment

## Technical summary

Box15 remains the research AIO prior. The best rich challenger adds a
target-excluded defensive residual to Box15. It improves the ten-fold paired
game MSE by `0.624` and wins eight of ten seasons. Its later-period RMSE gain is
only `0.022` points per game. The frozen promotion gate requires `0.050`.

The experiment supports a dual-head design:

- `spm_impact` uses the current rich SPM because it best reconstructs stable
  RAPM as a standalone statistical rating.
- `spm_prior` uses current-control Box15 because it best complements the
  one-season RAPM likelihood under the frozen gate.

The target-excluded defense residual is the closest challenger. It is not a
selected model. The public model, website, and API remain unchanged.

## The defense residual helps, but the gain is too small

The primary score is the equal-season mean of next-season whole-game margin
MSE. RMSE is the square root of that aggregate MSE. Lower values are better.
Every candidate uses the same games and common prior-covered players.

| Candidate | Period | Folds | MSE | RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current-control Box15 AIO | 2017--26 | 10 | 190.598 | 13.806 | 0.361 | 0.926 |
| Target-excluded defense residual AIO | 2017--26 | 10 | **189.974** | **13.783** | **0.364** | **0.940** |
| Current-control Box15 AIO | 2022--26 | 5 | 206.608 | 14.374 | 0.356 | 0.950 |
| Target-excluded defense residual AIO | 2022--26 | 5 | **205.975** | **14.352** | **0.360** | **0.970** |

Across all ten folds, the challenger-minus-Box15 MSE difference is `-0.624`.
The paired 95% whole-game bootstrap interval is `[-0.904, -0.354]`. The
challenger wins eight folds. Across the five later diagnostics, it wins four
folds and loses in 2023.

The later-period correlation improves by `0.0036`, so the candidate passes the
correlation guardrail. It also has no source-era reversal. It fails the result
gate because its RMSE gain is less than half of the required `0.050` points per
game.

| Outcome season | Challenger minus Box15 MSE | Challenger minus Box15 RMSE |
| ---: | ---: | ---: |
| 2022 | -2.181 | -0.077 |
| 2023 | +0.680 | +0.027 |
| 2024 | -0.109 | -0.004 |
| 2025 | -0.775 | -0.026 |
| 2026 | -0.777 | -0.025 |

## Scope and model definitions

The suite scores rating seasons 2016--25 against games in 2017--26. It uses
2017--21 outcomes for design selection and treats 2022--26 as reused later
diagnostics. These results cannot promote a public model.

The common one-season possession model uses terminal lineups. Its design has
five offensive player columns, five defensive player columns, and one home
column. The baseline AIO penalties are `3000` for offense, `4500` for defense,
and `300` for home.

Let `prior` be the statistical center, `lambda_total` be the ridge penalty,
and `trust` be the fraction of that penalty assigned to the prior. The AIO
solves:

```text
beta_hat = inverse(X'X + P(lambda_total))
           * [X'(y - intercept) + trust * P(lambda_total) * prior]
```

This form separates general ridge shrinkage from trust in the statistical
center. It is equivalent to combining a zero-centered penalty and a
prior-centered penalty:

```text
lambda_zero * ||beta||^2 + lambda_prior * ||beta - prior||^2
    = lambda_total * ||beta - trust * prior||^2 + constant

trust = lambda_prior / lambda_total
```

Box15 uses 15 box-score rates for each side and ridge regression. Rich SPM uses
the full annual statistical feature panel. Offense uses elastic net. Defense
uses ridge. Both models use square-root minimum-side possession weights for
their RAPM labels. Possessions are weights, not input features.

## Three information designs test overlap

The experiment compares three chronological definitions.

| Design | Training pair | Input at rating season `t` | Interpretation |
| --- | --- | --- | --- |
| Current control | `features[s] -> current 9-year RAPM[s]` | `features[t]` | Target and input include season `t` evidence |
| Target-excluded | `features[s] -> past-only 9-year RAPM[s]` | `features[t]` | Stable RAPM label excludes season `t` |
| Fully lagged | `features[s] -> current 9-year RAPM[s]` | `features[t-1]` | Statistical input and target precede season `t` |

Target exclusion helps Box15 by only `0.0035` aggregate RMSE across all ten
folds. Its paired MSE interval crosses zero. Full lagging worsens Box15 by
`0.0145` RMSE. Label overlap alone does not explain the reversal.

Rich SPM still loses after the possession update under every information
design. Removing direct defended-shot outcome fields makes the rich prior and
its AIO worse. The available outcome fields carry useful predictive signal;
simple censoring does not create complementarity.

## Complementary signal is concentrated on defense

The direct blend takes Box15 offense and defense and adds separate fractions
of the rich-minus-Box difference:

```text
offense_prior = box_offense + offense_weight * (rich_offense - box_offense)
defense_prior = box_defense + defense_weight * (rich_defense - box_defense)
```

Later walk-forward selections keep `offense_weight = 0` and use nonzero defensive
weight. Rich offense does not add transferable signal after the one-season
RAPM update.

The best candidate keeps Box15 offense and predicts a defensive residual
against cross-fitted past-only RAPM. Its activity pool includes defended-shot
workload, rim workload, contests, deflections, recovered blocks, charges,
loose-ball recoveries, rebound opportunities, rebound conversion, foul-adjusted
activity, and source-availability indicators. The outcome-augmented version
also includes stabilized defended-shot and rim suppression outcomes.

The outcome-augmented residual beats the activity-only residual by `0.153` MSE
across all ten folds. The improvement remains too small for selection. This
result identifies defense as the only live rich-feature lane.

## Prior trust and heterogeneous precision do not rescue rich SPM

Each fold selects offense penalty from `{1500, 3000, 4500, 6000}`, defense
penalty from `{3000, 4500, 6000, 9000}`, and side-specific prior trust from
`{0, .25, .5, .75, 1}` using earlier game folds only.

The later defense-residual folds select total penalties `6000 / 9000`, full
trust on both sides, and heterogeneous defensive precision. Rich SPM therefore
does not lose because the experiment forces excessive prior trust. The search
can weaken the rich center or increase general shrinkage. It still cannot
match Box15 after the update.

Player-specific precision uses log exposure, absolute Box-versus-rich
disagreement, and unavailable source-family count. The selected later mode
changes defensive precision only. It does not create a material new gain. The
model has already extracted the useful reliability adjustment from the tested
inputs.

## The shared-error diagnostic is unresolved

Both prior error and one-season RAPM error subtract the same future RAPM
reference:

```text
prior_error = prior - future_3_year_RAPM
rapm_error  = current_1_year_RAPM - future_3_year_RAPM
```

The observed defense correlation is `0.683` for Box15 and `0.740` for rich
SPM. The rich-minus-Box difference is about `0.057`. Offense differs by only
`0.021`.

These raw correlations do not establish shared basketball error. A
within-season permutation of the future reference still produces correlations
of `0.755` for Box15 defense and `0.801` for rich defense. The common subtracted
reference mechanically creates much of the correlation. Gaussian noise
features also reach maximum absolute correlations of `0.121` on defense and
`0.148` on offense across the small season panel.

The suite therefore treats shared error as unresolved. The downstream blocked
game results support a defense-only residual. They do not prove why it works.

## The RAPM update helps Box15 more

For prior prediction error `e` and RAPM update `u`, the scored identity is:

```text
MSE(AIO) = MSE(prior) + mean(u^2) - 2 * mean(e * u)
```

The identity holds to a maximum numerical error of `8.53e-14`. Every candidate
benefits from the one-season possession update. Box15 leaves more useful signal
for that update. Rich SPM reconstructs stable RAPM better, but its errors and
scale do not combine as well with the new possession evidence.

This explains the dual-head decision. Standalone target reconstruction and
downstream prior complementarity are different objectives. One model does not
need to win both.

## Robustness and quality checks

- All candidates score identical games and common prior-covered players.
- The ten annual matrices contain 212,549 to 247,630 possession rows and 634 to
  711 player columns.
- Every point estimate preserves offense plus defense equals net.
- The source-era cuts show no challenger reversal.
- Leave-one-season-out aggregate sensitivity keeps the defense residual ahead
  of Box15 for every omitted season.
- High- and low-exposure cuts show no isolated defensive failure.
- Matrix hashes, feature selections, parameter selections, and checkpoints are
  deterministic and stored with the run.
- The future three-year reference enters diagnostics only. It never enters
  training or model selection.

The experiment uses reused historical outcomes. Its bootstrap interval measures
sampling variation within this fixed design. It does not convert reused model
selection evidence into a new confirmation.

## Decision and next steps

The final classification is **Box15 retained**.

1. Keep current rich SPM as `spm_impact` for standalone statistical impact.
2. Keep current-control Box15 as `spm_prior` for the research AIO.
3. Preserve the target-excluded outcome-augmented defense residual as a frozen
   research challenger.
4. Stop broad rich-feature and target-window searches on the same outcomes.
5. Run the external all-in-one benchmark on identical metric coverage and
   next-season games before any product change.

The run emits `dual_head_spm.parquet`, which contains both statistical outputs.
The full result is stored under
`artifacts/research/aio_prior_complementarity/aio_prior_complementarity_v1_4d83e381af`.

Reproduce the suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/run_aio_prior_complementarity.py \
  --config research/experiments/aio_prior_complementarity_v1.yml
```
