# Separate Player Roles and Defense Challenger

Date: 2026-08-17

## Decision

The role layer is now split into offense and defense.

- Keep six offensive deployment clusters and five defensive deployment clusters
  as descriptive outputs.
- Keep continuous axes and drop-one soft affinities as research inputs.
- Add the eight-feature scorer-adjusted matchup block to the defense research
  challenger.
- Do not add role features or role-by-skill interactions to the defense model.
  They did not improve the fixed older-season comparison.
- Do not promote the defense challenger. The feature family and 2022--24
  seasons are already inspected.

Validated artifacts:

- split roles: `side_roles_v1_2c228f4b9e`;
- matchup features: `matchup_defense_features_v1_b265e245c4`;
- integrated annual features: `statistical_features_v2_907e4ee4b9`;
- defense comparison: `defense_role_challenger_v1_4dcd557af4`.

## Role contract

Roles describe observed deployment. They do not measure player quality. The
role model excludes RAPM, SPM, on/off, team outcome, age, height, position,
minutes, games, and efficiency outcomes.

The offense map uses 41 behavior inputs. They cover shot allocation, creation,
passing, touches, dribble context, and 11 playtype shares. It fits only on
2014--18, then applies the frozen map through 2024.

The defense map uses 17 deployment inputs. They cover rebound contest shares,
shot and rim contest volume, matchup activity, and the offensive-role mix of
the assigned scorers. It fits only on 2018--21, then applies the frozen map
through 2024.

Cluster count selection is target-free. Each candidate count is evaluated on
development-period silhouette, seed stability, and later-period cluster share.
RAPM is not used to select the count.

| Side | Candidate counts | Selected | PCA axes | Explained variance | Seed ARI | Adjacent exact role | Adjacent axis cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| Offense | 6, 8, 10, 12 | 6 | 8 | 84.37% | 0.9809 | 71.30% | 0.9103 |
| Defense | 5, 6, 7, 8, 9 | 5 | 6 | 92.01% | 0.9910 | 70.44% | 0.8451 |

The local result does not reproduce BBall Index's proprietary assignments.
BBall Index publishes 12 offensive archetype names and seven defensive role
names. Its defensive system uses matchup distributions and rim-defense time.
Those public descriptions are a semantic benchmark, not labels forced onto
this model. See [Offensive Archetypes](https://www.bball-index.com/offensive-archetypes/)
and [Defensive Roles](https://www.bball-index.com/defensive-roles/).

## Provisional role descriptions

The artifacts retain anonymous labels. These descriptions are for reading the
centroids. They are not permanent player types.

| Offense cluster | Centroid description |
|---|---|
| `off_role_0` | High-load, self-created primary handler |
| `off_role_1` | Secondary or combined handler |
| `off_role_2` | Spot-up, off-screen, or corner shooter |
| `off_role_3` | Versatile roll, cut, and elbow big |
| `off_role_4` | Post and elbow big |
| `off_role_5` | Rim, putback, and cut finisher |

| Defense cluster | Centroid description |
|---|---|
| `def_role_0` | Low-involvement wing or spacer assignment |
| `def_role_1` | Creator and handler assignment |
| `def_role_2` | Rebound and contest specialist; low-sample warning |
| `def_role_3` | Active, versatile forward contest role |
| `def_role_4` | Interior and rim assignment big |

## New defense information

The selected block has eight features:

1. opponent-adjusted matchup points saved;
2. field-goal attempts suppressed versus each scorer's own baseline;
3. shot-making points saved versus each scorer's own baseline;
4. three-point attempts suppressed versus each scorer's own baseline;
5. turnovers forced versus each scorer's own baseline;
6. assists suppressed versus each scorer's own baseline;
7. shooting fouls prevented versus each scorer's own baseline;
8. matchup blocks per 100 assigned possessions.

The first seven estimates use empirical-Bayes shrinkage. The scorer baselines
leave the current defender out. This reduces direct opponent-quality bias. It
does not control the full lineup, scheme, help defense, or assignment process.
The features remain observational.

Five additional source fields are present but constant zero in every 2018--24
archive: potential assists, switches, help blocks, help field-goal attempts,
and help field-goal percentage. The artifact reports them as zero-variance.
They are excluded from model candidates.

## Fixed chronological comparison

The defense target is one-season normal RAPM defense. The model uses ridge with
alpha 3000. The baseline is the frozen 60-feature annual defense SPM. Training
starts in 2018. The model selects one predeclared addition on 2020 and 2021.
Only after selection does it report 2022--24 diagnostics.

| Season | Baseline RMSE | Matchup RMSE | Baseline correlation | Matchup correlation |
|---:|---:|---:|---:|---:|
| 2020 selection | 1.0016 | 0.9459 | 0.4581 | 0.5423 |
| 2021 selection | 0.9343 | 0.8758 | 0.4796 | 0.5692 |
| 2022 diagnostic | 0.9656 | 0.9027 | 0.5011 | 0.5843 |
| 2023 diagnostic | 0.9403 | 0.9067 | 0.5002 | 0.5634 |
| 2024 diagnostic | 0.9678 | 0.9467 | 0.5467 | 0.5808 |

The matchup block wins all three diagnostic seasons. Mean diagnostic RMSE
changes by -0.0392 and correlation by +0.0601. Defense roles alone are worse in
2020 and almost neutral in 2021. Adding roles or role interactions to the
matchup block is also worse on both selection seasons.

This is strong research evidence that scorer-adjusted matchup factors contain
useful defense information. It is not promotion evidence because later seasons
informed this research program.

## Next model step

Carry the eight selected matchup features into the predeclared rolling
three-season AIO contract. Aggregate source counts before recomputing rates and
shrinkage. Do not average annual rate estimates. Keep offense and defense role
outputs for description and subgroup diagnostics. Reopen role inputs only after
a new hypothesis predicts an older-fold improvement.

## Descriptive role stabilization

Run `role_stabilization_v1_f5b426dd5d` selects a forward filter without RAPM,
SPM, age, size, position, minutes, or games. Candidate current-season weights
are 0.40, 0.55, 0.70, 0.85, and 1.00. The selection target is next-season raw
role affinity on the original development seasons. Both sides select 0.70.

On later adjacent seasons, exact hard-role persistence changes from 71.30% to
79.89% for offense and from 70.44% to 76.97% for defense. The stable and raw
labels disagree in 7.55% and 6.53% of later player-seasons. Reset the filter
after a missing season. Show raw membership as an option because the stable
state can lag a real role change. Do not use the stabilized hard label as an
impact-model input.
