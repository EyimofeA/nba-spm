# CourtSignal frontier research plan

## Decision

Keep Box15 as the frozen research prior. Do not promote height or the partial
CARUSO feature pack. Neither candidate produced a decisive 2026 factor-RAPM
gain. The next SPM work should improve the target and evidence harness before
it adds more columns.

Season 2026 remains reused diagnostic evidence. Season 2027 remains untouched.

## The 2026 height and defense-mechanism test

The test used the same ridge learner, players, targets, exposure weights, and
chronological split for every candidate:

- train on 2024;
- select ridge strength on 2025;
- refit on 2024–25;
- diagnose on 387 players in 2026 with at least 1,000 possessions on each side.

The control used the 15 BoxPIPM-style box features. The height candidate added
listed player height. The partial CARUSO candidate added four observable
mechanisms, their within-position percentiles, and five position indicators:

- stabilized rim points saved per 100;
- event stops per 100;
- contested defensive rebounds above a season-level expectation based on
  defended-shot volume and defended-rim frequency;
- deflections above the rate predicted by steals.

The source does not contain CARUSO's on/off rim-attempt deterrence component.
The result is a partial feature test, not a CARUSO reproduction.

| 2026 defensive factor | Box15 R² | + height R² | + partial CARUSO R² | + both R² |
| --- | ---: | ---: | ---: | ---: |
| Shooting true shooting | .129 | .131 | .113 | .113 |
| Turnovers | .379 | .375 | .396 | .396 |
| Offensive rebounds | .120 | .132 | .113 | .119 |

Height added `.012` R² for defensive rebounding and `.002` for shot defense,
but it reduced turnover-defense R² by `.004`. The partial CARUSO pack added
about `.017` R² for turnover defense and lost `.016` for shot defense. Every
paired 95% MSE interval crossed zero. The observed changes remain unresolved.

## Work in order

### 1. Repair the evidence harness

- Freeze one executable SPM experiment contract.
- Enforce identical scoring rows for every comparator.
- Enforce chronological folds and hard-reject Season 2027.
- Record code, configuration, data, row-set, and output hashes.
- Report weighted correlation, calibration slope, spread ratio, and paired
  whole-game MSE in addition to RMSE.

### 2. Finish source lineage

- Preserve observed, structural-zero, empirical-Bayes, median, and unavailable
  states in the feature matrix.
- Reconcile the public 2017–24 source, the 2014–26 refresh, and the repaired
  defense sources.
- Test the defense-source transition only when observed 2020–24 values change.
- Keep 2025–26 as diagnostics when no earlier observed source comparison exists.

### 3. Freeze the SPM target experiment

Compare one-year, three-year, five-year, latent-state, and uncertainty-weighted
RAPM labels with identical features and learners. Judge each target by the
downstream prior-informed RAPM on later identical games. Do not select the
label from in-sample player R².

### 4. Replace possession weights with a separate precision test

Keep the current square-root possession weight as the control. Test a bounded
inverse-variance weight from game-cluster RAPM covariance. Do not change the
target, source, and weighting rule in one experiment.

### 5. Test a small residual SPM

Use Box15 as the stable base. Train a cross-fitted residual model on
`RAPM - Box15`. Add one mechanism family at a time. Tune the residual blend on
earlier seasons. Allow the residual scale to shrink to zero.

### 6. Build the defense mechanism pack from new information

- Obtain true on/off rim-attempt deterrence.
- Obtain defender assignment and shot-context data.
- Replace contested-rebound activity with lineup-based team rebound effect or
  player rebound probability above expected.
- Separate steals, recovered blocks, charges, deflections, and foul discipline.
- Retest height only as a pooling or interaction variable, not a generic value
  feature.

### 7. Build the feature registry and negative controls

Record the definition, unit, denominator, source, availability, stabilization,
and causal timing for every feature. Select feature families inside historical
folds. Include shuffled and future-impossible negative controls so the harness
can expose leakage.

### 8. Control same-possession label sharing

Cross-fit shot quality and other play-by-play features by game where practical.
Use future-season and downstream AIO tests to detect features that only explain
noise shared with the RAPM label.

### 9. Finish the RAPM publication layer

- Release a reproducible frozen normal-RAPM bundle.
- Add 80% and 95% intervals, offense-defense covariance, exposure, and lineup
  connectivity.
- Audit the approximately `1.391×` CourtSignal-to-Ryan-Davis scale difference.
- Finish the unified play-by-play and lineup-policy source audit.

### 10. Test the dynamic model

Compare raw annual RAPM, shrunk annual observations, independent offense and
defense AR(1), and a bivariate state-space model. Use rolling origins and later
game outcomes. Keep retrospective annual ratings separate from current latent
strength.

### 11. Expand only after the measurement core passes

- shared-covariance multi-target factor RAPM;
- low-rank teammate and opponent synergy;
- role-relative skills and supported role counterfactuals;
- playoff portability;
- Event Points and conserved win-probability credit;
- injuries, contracts, draft, and roster projections.

## Stop rules

- Do not add broad feature banks after the full bank lost five of six factor
  heads to Box15.
- Do not reopen a model zoo or defensive ridge search without new evidence.
- Do not put score margin into total player impact. Keep it in a live context
  estimand.
- Do not use external all-in-one metrics as labels or ground truth.
- Do not promote any result from reused 2025–26 diagnostics.
- Do not inspect Season 2027 before the frozen confirmation run.

## Evidence

- Experiment: `caruso_factor_feature_audit_v1_c6942f1fd8`
- Runner: `research/run_caruso_factor_feature_audit.py`
- Control factor audit: `spm_factor_failure_audit_v1_23a5d60bca`
- External method: [CARUSO: A Transparent Way to Measure NBA Defense](https://hardscreenherald.substack.com/p/caruso-a-transparent-way-to-measure)
