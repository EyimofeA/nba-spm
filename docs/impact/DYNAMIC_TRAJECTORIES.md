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

## Canonical current-data extension

The source panel now extends through the 2025--26 regular season. The annual
season label is the ending year: 2024 means 2023--24.

`current_single_season_rapm_targets_v1_9c0cdda919` fits the frozen normal-RAPM
contract separately for 2024--26: zero prior, 3000/3000/300 ridge penalties,
regular season only, and terminal ordinal lineups. It contains 1,723
player-seasons over 1,227, 1,226, and 1,228 eligible games, respectively.

Before joining it to the older panel, source-transition run
`canonical_annual_target_panel_v1_2d9ff74ca3` compared the shared 2024 season.
After excluding the known legacy row with zero defensive possessions, 571
players matched. Canonical versus legacy Pearson correlation was 0.974 offense,
0.964 defense, and 0.975 net. The net scale ratio was 1.019 and the mean net
difference was 0.010 points per 100. These pass an explicit compatibility
gate; they do not prove that either source is more accurate.

Run `time_decayed_trajectory_v1_8ed684a8aa` then applies the already-selected
0.80-decay, equal-initial-weight filter to the combined 2014--26 panel. It does
not retune the filter. Its historical selection result is unchanged (1.9481 to
1.7166 net RMSE). The later 2022--23 diagnostic, with canonical 2024 as the
last target, remains directionally positive (2.0549 to 1.8086). The 2025--26
extension is descriptive only: Season 2027 remains untouched, so it is not a
new annual confirmation.

## Boundaries

- The next annual RAPM target is a noisy proxy, not ground truth and not a team
  margin prediction.
- The panel uses a legacy source through 2022--23 and canonical current data
  from 2023--24 onward. The transition passed a compatibility audit, but it can
  still create small source-specific differences.
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
