# Roles and zone-adjusted shotmaking in the five-year SPM

## Decision

Keep both soft role context and hard role experts as research challengers. Do
not replace the five-year SPM yet. Do not add zone-adjusted shotmaking to SPM
from this result.

The frozen run is
`five_year_spm_role_research_v1_3edacae610`. It trains the persisted
126-offense/50-defense five-year SPM on window ends before the rating season,
then predicts the following season's one-year zero-prior RAPM. The three test
seasons are 2021, 2022, and 2023. Every candidate scores the same players.

The original player-level score against next-season RAPM is now secondary. The
primary selection score is downstream next-season team-win association: apply
each season-Y rating to the player's observed team minutes in Y+1, form a
five-player team rating, then correlate it with Y+1 win percentage. This is an
oracle-minutes retrodiction, not a preseason forecast.

## Primary downstream result

Artifact: `spm_role_team_win_benchmark_v1_21bdb974c8`. It covers rating seasons
2020--22 and outcome seasons 2021--23, with 30 teams in each fold. Players
without a qualifying season-Y rating receive -2.0. Basketball-Reference
player-team minutes resolve to NBA IDs at at least 99.9875% minute coverage.

| Candidate | Mean Y+1 win R² | Change vs baseline | Pooled R² | Decision |
|---|---:|---:|---:|---|
| Hard role experts | **0.5839** | **+0.0266** | **0.5841** | retain challenger |
| Soft role context | 0.5660 | +0.0088 | 0.5580 | retain challenger |
| Roles + zone shotmaking | 0.5612 | +0.0039 | 0.5538 | no shotmaking promotion |
| Baseline | 0.5573 | 0.0000 | 0.5510 | reference |
| Zone shotmaking | 0.5489 | -0.0083 | 0.5435 | reject as SPM input |

Hard experts lose in the 2020-to-2021 fold, then win in the next two folds. In
the only fold strictly after defense-role map development, 2022-to-2023, hard
experts improve R² from 0.4509 to 0.5197 and soft roles improve it to 0.4585.
That is enough to continue the role lane, not enough to promote it: there are
only three folds, the benchmark uses known future minutes, and only 83.1--85.9%
of outcome minutes have a qualifying role-cohort rating.

## Secondary next-season RAPM result

Change versus the same-row baseline:

| Candidate | Side | Pearson | Spearman | MAE | RMSE | Decision |
|---|---:|---:|---:|---:|---:|---|
| Soft role context | Net | +0.0045 | +0.0024 | -0.0100 | -0.0172 | retain as challenger |
| Soft role context | Offense | -0.0004 | +0.0003 | -0.0027 | -0.0003 | essentially tied |
| Soft role context | Defense | +0.0015 | -0.0047 | +0.0181 | +0.0164 | mixed |
| Hard role experts | Net | -0.0032 | +0.0114 | +0.0418 | +0.0445 | loses this secondary gate |
| Zone shotmaking | Offense | -0.0030 | -0.0032 | +0.0070 | +0.0046 | reject as SPM input |
| Roles + zone shotmaking | Net | +0.0005 | -0.0012 | +0.0001 | -0.0077 | reject |

Soft roles use the annual offense or defense role coordinates, role indicators,
and a known-role flag. The offense histogram GBM can learn nonlinear role
interactions. The defense ridge receives role offsets and coordinates. Hard
role experts fit a separate frozen model within each role, with a global-model
fallback below 100 training rows. That split is too costly in sample size.

The defense role map was developed on 2018-21 data. Only the 2022 rating to
2023 target fold is strictly after the map-development period. On that fold,
soft role context changes net Pearson by +0.0069, net Spearman by +0.0080, net
MAE by -0.0056, and net RMSE by -0.0193. One clean fold is not enough to
promote a model. The disagreement between player-RAPM reconstruction and team
wins is exactly why the estimand must be named before selecting features.

## Zone-adjusted shotmaking

The existing shotmaking feature conditions on defender-distance buckets and
two-versus-three-point shot type. It does not distinguish rim attempts from
midrange attempts. This can reward players whose two-point attempts are mostly
at the rim even when they merely finish an easy mix at the expected rate.

The new metric uses five zones: rim, short midrange, long midrange, corner
three, and above-break three. For player `i`, window `t`, and zone `z`, let
`A_izt` be attempts, `M_izt` makes, and `v_z` the point value. The
leave-one-player-out league expectation is

```text
p_minus_i_zt = (league_makes_zt - M_izt)
               / (league_attempts_zt - A_izt)
```

Raw zone-adjusted shotmaking is

```text
100 / offensive_possessions_it
    * sum_z v_z * (M_izt - A_izt * p_minus_i_zt)
```

It is shrunk toward zero:

```text
zone_shotmaking_EB = attempts / (attempts + 200) * raw_shotmaking
```

This answers: how many points per 100 possessions did the player make above a
same-window league shooter with the same five-zone attempt mix? It does not
jointly adjust for shot location and defender distance because the available
sources are separate aggregates. A shot-level model is required for that.

The metric produces credible descriptive leaders, but it does not add
next-season SPM signal beyond the existing shooting features. It lowers offense
Pearson in all three folds. Keep it for player skill display and shotmaking
research, not the current impact model.

## Public-model feature coverage

We do not have every relevant input from BPM, RAPTOR, xRAPM, PIPM, and
Basketball Index. Some specifications are not public, and several plus-minus
inputs should not be copied into an SPM.

| Reference | Covered now | Missing or intentionally separate |
|---|---|---|
| Patton SPM tutorial | All traditional box fields; exposure weighting; regularized/nonlinear learners | Its player-level LOOCV is replaced by chronological season folds |
| BPM 2.0 | Traditional per-100 box families and creation/load composites | Exact BPM position and offensive-role coefficient interactions; team reconciliation |
| RAPTOR box | Shot zones, pull-up/catch-and-shoot, creation, turnover detail, spacing, zTS, rebound and defensive activity families | Full coefficient list is not public; adjusted on/off belongs downstream; several assisted-shot and positional-rebound constructions are not exact reproductions |
| xRAPM statistical prior | Box, play-by-play detail, tracking defense, matchup defense, and SPM-to-RAPM prior architecture | Full prior feature list and coefficients are not public |
| PIPM | Traditional box prior families and a separate box-PIPM recreation | Luck-adjusted on/off, on-court team strength, and team reconciliation are not SPM features; they are separate impact-model challengers |
| Basketball Index | Passing quality, spacing, shooting context, screening, hustle, shot-defense, and matchup candidates exist | Only passing and selected shot-defense/matchup groups survived the earlier gate; many proprietary glossary metrics cannot be reproduced exactly |

This inventory means the next feature experiment should not add everything at
once. Re-score each candidate family on chronological next-season prediction,
with the same rows and a team-changer slice. The prior feature-family run used
held-out RAPM reconstruction RMSE as its main selection score, which is no
longer the preferred gate.

## Reproducible outputs

- `artifacts/research/five_year_spm_role_research/five_year_spm_role_research_v1_3edacae610/run.json`
- `fold_metrics.parquet`: fold-level next-season scores
- `deltas_vs_baseline.parquet`: candidate changes on identical rows
- `predictions.parquet`: player-level predictions and targets
- `expert_coverage.parquet`: hard-role sample sizes and fallbacks
- `zone_shotmaking.parquet`: raw and shrunk shotmaking estimates
- `artifacts/research/spm_role_team_win_benchmark/spm_role_team_win_benchmark_v1_21bdb974c8/summary.parquet`:
  primary downstream team-win comparison
