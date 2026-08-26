# Five-year SPM teammate context

Status: research candidate. No public model or site data changed.

Run `five_year_spm_teammate_context_v1_13d270986a` tests whether teammate
context adds information to the actual five-year SPM. It starts from the exact
stored predictions in `five_year_target_spm_v1_65550acb79`. A second-stage
ridge predicts each side's remaining five-year RAPM error from six pooled
leave-one-player-out teammate fields.

The offense fields are spacing, creation, rim pressure, turnover burden,
offensive load and offensive rebounding. The defense fields are defensive
rebounding, rebound contests, event stops, deflections, rim points saved and
shot contests. Each annual context value excludes the focal player, then the
model weights the five annual values by that player's possessions.

For a rating ending in year `t`, the correction trains only on stored SPM
predictions ending before `t`. Penalties are selected on the 2024 rating and
frozen at 10,000 offense and 1 defense. The model then refits through each
available prior window before scoring 2025 and 2026. Season 2027 is absent.

## Results

| Five-year target | Baseline RMSE | With context | Change |
| --- | ---: | ---: | ---: |
| 2024 selection | 1.446 | 1.437 | -.009 |
| 2025 diagnostic | 1.517 | 1.505 | -.012 |
| 2026 diagnostic | 1.661 | 1.647 | -.014 |

| Next annual RAPM | Baseline RMSE | With context | Change |
| --- | ---: | ---: | ---: |
| 2024 rating to 2025 | 1.471 | 1.469 | -.002 |
| 2025 rating to 2026 | 1.520 | 1.508 | -.012 |

The correction improves offense, defense and net on both next-season folds.
The gain is small. Average absolute net corrections fall from `.162` points
per 100 in 2024 to `.135` in 2026. This is enough to retain the feature family
for a future joint refit, not enough to change the current research SPM.

The evidence is not independent. Season 2025 is already inspected and 2026 is
heavily reused. Complete pooled-context coverage is 87.5 percent because the
stored five-year rating panel contains players without every annual context
source; the training-fold imputer handles those rows. DFG and rim-DFG stop in
2025. Annual `TEAM_ID` also makes traded-player context approximate.

The preferred next-season team-win test was not rerun. This checkout lacks the
exact 2025--26 player-team minute-stint table needed for traded-player
allocation, and assigning all minutes to one annual `TEAM_ID` would weaken the
benchmark. The saved next-season test therefore uses matched annual RAPM only.

Exact metrics and predictions are under
`artifacts/research/five_year_spm_context/five_year_spm_teammate_context_v1_13d270986a`.
