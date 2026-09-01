# Nine-year Box15 defense extension

## Decision

Keep nine-year Box15 as the selected retrospective AIO prior. Retain a
three-feature defensive residual as a research challenger.

The residual improves the point estimate, its paired game interval excludes
zero, and it wins nine of ten next-season folds. Its RMSE gain is only `.0111`
points per game. The frozen practical threshold is `.05`. The challenger also
loses the 2026 fold.

## Question

Can a small defensive feature set improve the nine-year Box15 prior without
discarding Box15's longer historical training record?

## Model

The control is `box15_9y_normal` from run
`target_window_spm_aio_v1_8e028133cb`.

For each historical player-season, the runner calculates:

```text
defense residual = nine-year defensive RAPM - chronological Box15 defense
```

The Box15 value is an out-of-time prediction. The residual model therefore
does not train against an in-sample Box15 error.

Each rating season fits a standardized defensive ridge model on earlier
seasons. Leave-one-season-out weighted MSE selects alpha from `10`, `30`,
`100`, `300`, `1000`, and `3000`. The label weight is the square root of the
smaller offensive or defensive possession count.

The residual changes only the defensive prior. Offense remains the saved
nine-year Box15 prediction. Every AIO then applies the same one-season RAPM
update with penalties `3000 / 4500 / 300` for offense, defense, and home.

## Feature screen

The first screen tested four individual fields and four fixed families.

| Candidate | Added defensive information |
| --- | --- |
| Rebound chances | Defensive rebound chances per 100 |
| Recovered blocks | Recovered blocks per 100 |
| Workload suppression | Matchup points-saved residual after defended-shot workload |
| Rim workload | Stabilized rim points saved multiplied by square-root rim workload |
| Top five | Five fields from the prior chronological importance screen |
| Mechanism four | Rebound conversion, foul-adjusted activity, workload suppression, rim workload |
| Core four | Rebound chances, recovered blocks, workload suppression, rim workload |
| All nine | Top five plus all four mechanisms |

The mechanism-four family won the first screen. A post-hoc isolation run then
tested both missing single features, the natural activity and shot pairs, and
each leave-one-feature-out family.

## Validation

- Rating seasons: 2016 through 2025.
- Outcome seasons: 2017 through 2026.
- Primary metric: equal-season mean next-season whole-game margin MSE.
- Secondary metrics: RMSE, correlation, and calibration slope.
- Uncertainty: 2,000 paired whole-game bootstrap draws within season.
- Coverage: identical players, possessions, and games for every candidate.
- Limitation: actual next-season lineups supply exposure weights. This is a
  rating test, not a deployable minutes or availability forecast.

All outcome seasons have already informed research. The run cannot promote a
public model.

## Results

| AIO prior | MSE | RMSE | Correlation | Fold wins | MSE difference vs Box15 | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Box15 | 190.713 | 13.8099 | .3638 | -- | -- | -- |
| Mechanism four | 190.433 | 13.7997 | .3662 | 8/10 | -.280 | [-.499, -.065] |
| Mechanism without foul activity | 190.407 | 13.7988 | .3663 | 9/10 | -.306 | [-.525, -.086] |
| Workload suppression only | 190.672 | 13.8084 | .3651 | 4/10 | -.041 | [-.236, .170] |
| Rim workload only | 190.728 | 13.8104 | .3636 | 5/10 | +.014 | [-.040, .071] |
| Rebound chances only | 190.799 | 13.8130 | .3631 | 1/10 | +.085 | [.063, .110] |
| Recovered blocks only | 190.867 | 13.8155 | .3626 | 2/10 | +.154 | [.095, .214] |
| Top five | 191.017 | 13.8209 | .3623 | 2/10 | +.304 | [.205, .403] |

The selected three-feature diagnostic contains:

- stabilized defensive-rebound conversion above expected;
- workload-adjusted shot suppression;
- rim-protection workload value.

It improves the defense prior's weighted target RMSE from `1.3901` to `1.3552`.
Mean target correlation rises from `.3078` to `.3620`. The downstream gain
appears in both halves of the sample. Candidate-minus-control MSE is `-.270`
for 2017 through 2021 and `-.342` for 2022 through 2026.

The 2026 fold reverses. Candidate MSE is `236.378`, compared with `235.544`
for Box15. That reversal and the small aggregate RMSE gain block selection.

## Interpretation

The defense additions contain real long-window RAPM signal. Most of that signal
does not add much after the one-season possession likelihood updates the prior.
The residual model improves correlation more than calibration. The result fits
the broader finding that rich defensive data reconstructs RAPM but overlaps
with the evidence RAPM already uses.

The foul-adjusted activity residual weakens this family. The post-hoc
three-feature result may guide the current-state model. It must not replace the
pre-run mechanism-four result in formal evidence.

## Artifacts

- Family screen: `box15_defense_extension_9y_v1_346afdf3a2`.
- Isolation follow-up: `box15_defense_mechanism_followup_9y_v1_57b6011cd8`.
- Contracts:
  `research/experiments/box15_defense_extension_9y_v1.yml` and
  `research/experiments/box15_defense_mechanism_followup_9y_v1.yml`.
- Runner: `research/run_box15_defense_extension_9y.py`.
