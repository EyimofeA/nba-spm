# Final retrospective SPM prior stack

## Decision

Box15 remains the retrospective AIO prior. Broad retrospective SPM feature
research stops after this run. Work moves to the current-state AIO.

Run: `spm_final_prior_stack_v1_3424be9c7d`

## What this run combined

The previous experiments left one positive candidate and one unresolved idea:

- a target-excluded, outcome-augmented defensive residual improved the AIO but
  missed the practical threshold;
- fold-local consensus features improved standalone SPM but worsened the AIO.

This run adds both increments to Box15:

```text
offense prior
    = Box15 offense
    + offense consensus weight * (consensus offense - Box15 offense)

defense prior
    = Box15 defense
    + defense consensus weight * (consensus defense - Box15 defense)
    + residual weight * (defense residual - Box15 defense)
```

Each weight comes from `0, .25, .50, .75, 1`. Each rating fold selects weights
using earlier design outcomes only. The selected prior then receives the same
fold-local search over total offense and defense penalties, prior trust, and
player-specific precision used in the complementarity suite.

The run scores rating seasons 2016--25 on 11,969 next-season games from
2017--26. Every candidate scores the same games and player coverage.

## Result

| Candidate | Ten-fold MSE | RMSE | Correlation |
| --- | ---: | ---: | ---: |
| Current-control Box15 | 190.598 | 13.806 | .361 |
| Target-excluded Box15 | 190.500 | 13.802 | .361 |
| Final stack | **190.157** | **13.790** | **.363** |
| Full rich SPM | 193.103 | 13.896 | .346 |

The final stack beats current-control Box15 by `.441` MSE across all ten folds.
The paired 95% whole-game interval is `[-.728, -.160]`. It wins seven folds.

On the five later diagnostics, Box15 has RMSE `14.374` and the final stack has
RMSE `14.352`. The gain is `.0216` points per game. The stack wins four of five
seasons and improves mean correlation by `.0035`.

## Promotion gate

| Requirement | Observed | Result |
| --- | ---: | --- |
| RMSE improvement at least `.05` | `.0216` | Fail |
| Paired MSE 95% interval below zero | `[-.728, -.160]` | Pass |
| Correlation loss no more than `.01` | `-.0035` | Pass |
| At least three of five later wins | `4` | Pass |

The practical-effect requirement is conjunctive with the other requirements.
Passing three checks does not offset failing it.

## What the combination selected

The later five folds select:

- consensus offense weight `0`;
- consensus defense weight `0`;
- defensive residual weight `1`;
- total penalties `6000 / 9000`;
- full prior trust on both sides;
- heterogeneous defensive precision.

The combination therefore collapses to the earlier defensive-residual model.
The consensus additions do not become useful when combined. This answers the
last live combination question without another broad feature search.

## Independent Fable review

Cursor-backed Claude Fable 5 reviewed the evidence before seeing this result.
It recommended one final frozen test centered on the defensive residual. It
recommended stopping broad retrospective SPM feature research if the model
failed the `.05` practical gate.

Fable rejected more offense feature work. It identified four defensible
defensive families for a future new-data test:

- the target-excluded defensive residual;
- rim workload and workload-adjusted suppression;
- expected-versus-actual defended-shot residuals;
- rebound chances and contests.

The existing residual already includes those families. This run gives them
the final retrospective combination test. New features should reopen this lane
only when a new data source measures player responsibility better or new outcome
evidence becomes available.

## Final model split

- `spm_impact`: full rich SPM for standalone retrospective statistical impact;
- `spm_prior`: Box15 for the retrospective RAPM update;
- research challenger: target-excluded defensive residual;
- next research lane: current-state AIO.

This run does not change the public model, website, or API.
