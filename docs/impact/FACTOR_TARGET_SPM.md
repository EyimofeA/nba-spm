# Sparse factor-target SPM

Status: research diagnostic. No public rating or site data changed.

Run `factor_target_sparse_spm_v1_5b120e918f` asks two questions. First, can
small statistical models estimate annual shooting-TS, turnover and
offensive-rebound RAPMs? Second, do teammate-context fields help?

The split is chronological. Models train on 2024, select ridge penalties on
2025, refit on 2024-25 and diagnose on reused 2026. The panel contains 365,
381 and 387 qualified players. Season 2027 remains untouched.

## Individual features

| Target | Offense | Defense |
| --- | --- | --- |
| Shooting TS | zTS, playtype difficulty, stabilized shooting proficiency, spacing, rim attempts/100 | overall points saved, rim points saved, contested 2s/100, contested 3s/100 |
| Turnovers | stabilized cTOV, live-ball TOV/100, bad-pass TOV/100, lost-ball TOV/100, offensive fouls/100 | steals/100, deflections/100, charges/100, direct event stops/100 |
| Offensive rebounds | OREB/100, self-OREB/100, OREB contests/100, OREB chances/100, offensive boxouts/100 | DREB/100, DREB contests/100, DREB chances/100, contested DREB share, defensive boxouts/100 |

Every factor-side model uses only its row. It does not receive the other
factor families, player demographics, roles, minutes, games, on/off, external
ratings or team outcomes.

## Teammate context

Each context field is the possession-weighted team-season average after
subtracting the focal player's own contribution. The candidate adds only the
contexts tied to that factor:

| Target | Added offense context | Added defense context |
| --- | --- | --- |
| Shooting TS | teammate spacing, creation, rim pressure | teammate rim points saved, shot contests, event stops |
| Turnovers | teammate spacing, creation, cTOV | teammate event stops, deflections |
| Offensive rebounds | teammate OREB, spacing, offensive load | teammate DREB, DREB contests, rim points saved |

This is context, not skill. The annual source assigns one `TEAM_ID` to each
player. Players traded during a season therefore receive approximate context.

## Factor-target results

| 2026 target | Individual R² | With teammate context R² |
| --- | ---: | ---: |
| Shooting offense | .218 | .217 |
| Shooting defense | .037 | .035 |
| Turnover offense | .266 | .295 |
| Turnover defense | .367 | .413 |
| Offensive-rebound offense | .279 | .432 |
| Offensive-rebound defense | .126 | .159 |

Context improved all six factor-target RMSEs on 2025 and four of six on 2026.
It does almost nothing for shooting in 2026. The useful gains come from
turnovers and rebounding, especially offensive rebounding.

Teammate spacing is not the main driver. Its final standardized coefficient is
near zero for shooting offense and small elsewhere. Teammate creation,
teammate turnover burden, teammate OREB, teammate DREB and teammate event
stops carry more conditional weight. Coefficients are dependence diagnostics,
not causal effects.

## Reconstruction of normal RAPM

The three factor ratings use different opportunity scales, so raw addition is
invalid. Separate offense and defense ridge maps convert the three ratings to
points per 100. Net always equals offense plus defense.

| 2026 model | RMSE | Correlation | R² |
| --- | ---: | ---: | ---: |
| Oracle factor ratings | .470 | .974 | .948 |
| Direct related features plus teammate context | 1.710 | .563 | .305 |
| Predicted factors plus teammate context | 1.781 | .505 | .247 |
| Direct related features only | 1.794 | .486 | .236 |
| Predicted factors only | 1.835 | .457 | .200 |
| Mean baseline | 2.052 | .000 | .000 |

The decomposition itself works when the true factor RAPMs are known. The
current limitation is estimating those factor ratings from sparse statistics.
Forcing the factor route costs `.070` RMSE against the direct context model,
but makes the output easier to explain.

The result is descriptive and uses same-season inputs. Factor RAPM and normal
RAPM share lineups and outcomes, so the oracle result is not independent
validation. Defended-shot data end in 2025; 2026 shooting-defense inputs use
neutral fallbacks for the missing source family.

Exact metrics, selected penalties, coefficients, predictions and context rows
are under
`artifacts/research/factor_target_spm/factor_target_sparse_spm_v1_5b120e918f`.
