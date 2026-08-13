# Filtered Time-Decayed Normal-RAPM Trajectories

Status: **research baseline**. This is not a public annual-impact rating or a
fully specified latent-strength model.

## What it answers

Annual normal RAPM describes observed impact during one completed season. It is
noisy. This panel instead answers a narrower current-strength question at the
end of each named season: what does the player's history through that season
suggest after giving more weight to recent annual normal-RAPM observations?

For player `i`, component `c`, and season `T`, the filtered value is:

```text
sum(t <= T) decay^(T - t) * Poss(i,t,c)^p * annual_normal_RAPM(i,t,c)
----------------------------------------------------------------------
sum(t <= T) decay^(T - t) * Poss(i,t,c)^p
```

The filter is run separately for offense and defense. Net is their exact sum.
There is one row only for an observed player-season; missing seasons are not
interpolated. A gap between seasons receives the corresponding multi-year
decay, so an old observation does not regain weight when a player returns.

## First frozen evaluation

Command:

```bash
uv run python -m nba_impact.cli build-time-decayed-trajectories \
  --targets artifacts/models/single_season_rapm_targets/<run-id>/targets.parquet
```

Run `time_decayed_trajectory_v1_4706719bfb` used annual zero-prior normal-RAPM
targets from 2014--24. It tested twelve fixed filters:

- decay: 0.50, 0.65, 0.80, 0.90;
- possession exponent `p`: 0.0, 0.5, 1.0;
- selection origins: 2018--21, predicting the next annual normal-RAPM target;
- later diagnostics: 2022--23, predicting 2023--24;
- scoring: equal-season player net-RMSE among players with at least 1,000
  offensive and defensive possessions in the target season.

The selection winner uses decay `0.80` and `p = 0.0`: each observed annual
rating has equal initial weight, then loses 20% per elapsed season. Its implied
half-life is about 3.1 seasons. Possession weighting did not win the frozen
selection grid; possession remains an eligibility/reliability field, not a
general player feature.

| Evaluation | Latest annual normal RAPM | Filtered trajectory | Change |
|---|---:|---:|---:|
| Selection, 2018--21 origins | 1.9481 | 1.7166 | -11.9% |
| Later diagnostic, 2022--23 origins | 2.0285 | 1.7758 | -12.5% |

The same direction appears in both scopes. The filter also retains a positive
net correlation with the next annual target (0.419 selection; 0.435 diagnostic).
One source row with zero defensive possessions was explicitly excluded; the
final panel contains 5,790 player-seasons and 1,499 players.

## Boundaries

- The next annual RAPM target is a noisy proxy, not ground truth and not a team
  margin prediction.
- The source archive is legacy and ends in 2024. It is not current NBA data.
- There are no trajectory confidence intervals, age curve, injury state, or
  uncertainty-aware transition process in this version.
- This does not make the existing `current_latent_strength_v1` contract
  implemented. That later model must explicitly separate observation error and
  player evolution, then beat this baseline on forward tests.
- Do not present fitted curves between observed seasons as evidence.

## Relation to expected outcomes

Expected shot outcome is useful for a shot-quality decomposition. Expected
possession outcome would be a stronger RAPM challenger because it includes
turnovers, rebounds, and free throws. The current public event data supports a
minimal possession-start context model, but not a rich player-neutral EPV model:
adding post-start actions would leak the outcome, while player/team identity in
the expectation model would absorb the effects that RAPM is meant to estimate.
Keep expected-possession residual RAPM as a separate preregistered challenger
after its state contract is frozen; do not substitute it for this trajectory
baseline.
