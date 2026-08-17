# Behavior-Only NBA Player Roles

Date: 2026-08-17

## Decision

The first behavior-only role foundation passes its frozen coverage and
stability gates. Continuous role axes and soft role affinities are eligible as
research AIO inputs. The hard cluster label remains descriptive and is not a
player-value feature.

Run: `behavior_roles_v1_e0fb51c026`

## Estimand boundary

The role model describes **how a player was used** during a season. It does not
estimate how well the player performed, why the role occurred, or what the
player would do in a different system.

Excluded inputs:

- age, experience, height, and listed position;
- minutes, games, and future availability;
- points, shooting accuracy, expected-versus-actual skill, and efficiency;
- on/off, plus-minus, RAPM, SPM, and team outcome;
- contracts, salary, draft status, and reputation.

## Input families

The model uses 32 season-relative behavior measurements:

| Family | Inputs |
|---|---|
| Shot allocation | 2PA, 3PA, FTA, rim, short-midrange, long-midrange, corner-three, and arc-three frequency |
| Creation behavior | drives, pull-up share, self-created scoring share, creation load, time of possession, seconds and dribbles per touch |
| Passing behavior | passes, potential assists, drive-pass rate, and front-court touches |
| Interior behavior | paint, post, and elbow touches; interior load; offensive rebounding and rebound contests |
| Dribble context | zero-dribble and three-plus-dribble shares for all shots and jump shots |

Every input is centered and robustly scaled within its season. This removes
most league-era level shifts before role fitting. Missing values use the same-
season median. A player-season needs at least 80% observed descriptor coverage.

## Frozen model

1. Fit the role map only on 2014–18.
2. Reduce the 32 inputs to six deterministic principal-component axes.
3. Fit eight K-means clusters with 50 initializations and a fixed seed.
4. Apply the same scaler, axes, and cluster centers unchanged through 2024.
5. Produce six continuous axes, eight soft affinities, one hard descriptive
   cluster, distance, and confidence.

Only the six axes and the first seven affinities are eligible model candidates.
The eighth affinity is omitted to avoid an exact sum-to-one dependency. The hard
cluster is never an AIO input.

## Stability results

| Check | Frozen gate | Result | Decision |
|---|---:|---:|---|
| Eligible row coverage | at least 90% | 93.71% | pass |
| Median seed adjusted Rand | at least 0.90 | 0.9845 | pass |
| Out-of-sample adjacent exact role persistence | at least 50% | 61.66% | pass |
| Out-of-sample adjacent median axis cosine | at least 0.75 | 0.9149 | pass |
| Smallest later-period role share | at least 2% | 6.27% | pass |
| Largest later-period role share | at most 30% | 18.93% | pass |

The six axes explain 83.18% of development-period descriptor variance. There
are 5,427 eligible rows, including 2,427 development rows and 3,000 fixed-map
out-of-sample rows. Adjacent-season checks contain 2,397 matched later-period
player pairs.

Continuous role is more stable than exact cluster membership. This supports
soft role inputs and argues against treating the eight clusters as natural,
permanent player types.

## Provisional cluster interpretation

These labels are documentation aliases only. The artifact retains `role_0`
through `role_7` so a subjective name cannot silently become model truth.

| Cluster | Provisional behavior description | Representative 2024 players near the centroid |
|---|---|---|
| `role_0` | Assisted spacing shooter | Georges Niang, Julian Champagnie, Simone Fontecchio |
| `role_1` | Rim-running and rebounding interior | Daniel Theis, Jabari Walker, Tari Eason |
| `role_2` | Post and elbow interior | Kenneth Lofton Jr., Evan Mobley, Karl-Anthony Towns |
| `role_3` | Low-creation corner wing | Troy Brown Jr., Chris Duarte, Matisse Thybulle |
| `role_4` | Balanced connecting wing | Caleb Martin, Patrick Williams, Bruce Brown |
| `role_5` | Pass-first secondary handler | Tre Jones, Dennis Smith Jr., Tyus Jones |
| `role_6` | Scoring-oriented handler | Austin Reaves, D'Angelo Russell, Jordan Poole |
| `role_7` | High-load primary creator | James Harden, LaMelo Ball, Donovan Mitchell |

Representative players explain the centroid. They are not evidence that every
player in the cluster has the same skills or value.

## AIO integration

The validated role candidates are integrated without model fitting:

- rolling table: `statistical_features_v2_2bb78bc737`, 6,689 player-windows;
- annual table: `statistical_features_v2_d8dd1d8dc2`, 5,791 player-seasons.

Each table has 270 feature columns, unique player-period keys, no infinite or
bounded-value failures, and no missing new-feature values after season-neutral
imputation. No role subset was selected. No impact score changed.

## What this unlocks

The next research steps must remain separate:

1. **Role-relative skill:** estimate whether a player performs better than
   players with similar behavior. This remains retrospective.
2. **Role interaction:** test whether an impact model benefits from interactions
   between skill measurements and soft role affinities.
3. **Role-fit counterfactual:** predict a player under another feasible role.
   This needs overlap checks and stronger causal assumptions. A role cluster
   alone cannot identify it.
4. **Lineup composition:** summarize the role and skill mix of five-player
   units, then validate future lineup outcomes without using actual future
   minutes or availability.

The role-context source is pinned to `gabriel1200/site_Data` revision
`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`. Upstream declares no license.
Derived role artifacts remain research-only.
