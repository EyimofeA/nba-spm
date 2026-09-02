# CourtSignal PULSE release

## Result

PULSE is the public retrospective player-impact rating. It combines a statistical prior with one season of lineup evidence.

The canonical walk-forward test covers 12 rating seasons. Each fold trains the prior on earlier rating seasons, rates season `t`, and scores season `t+1` games with the observed lineups from those games.

| Model | RMSE | Correlation | Calibration slope |
|---|---:|---:|---:|
| PULSE | 13.614 | 0.362 | 0.871 |
| RAPM | 13.723 | 0.340 | 0.934 |
| PULSE prior | 14.096 | 0.314 | 1.544 |

PULSE improves RMSE by 0.109 points per game against RAPM. Its paired MSE difference is -2.970. The 95% whole-game bootstrap interval is [-3.697, -2.216]. PULSE wins 11 of 12 outcome seasons.

## PULSE prior

The prior is a ridge model with separate offense and defense fits. It predicts nine-year normal RAPM from one season of 15 per-100 box rates:

1. Points
2. Assists
3. Turnovers
4. Steals
5. Blocks
6. Offensive rebounds
7. Defensive rebounds
8. Personal fouls
9. Fouls drawn
10. Free throws attempted
11. Free throws made
12. Two-point attempts
13. Two-point makes
14. Three-point attempts
15. Three-point makes

The offense ridge penalty is 300. The defense ridge penalty is 1,000. Training labels receive the square root of the smaller offensive or defensive possession count as a reliability weight.

## Lineup update

The canonical source covers 1997–2026. It groups possessions with the same ten-player lineup into score-conserving stints. Aggregated possession weights preserve the possession-level ridge cross-products.

The fit uses penalties of 3,000 on offense, 4,500 on defense, and 300 on home court. Positive defense means points prevented. The prior and lineup evidence enter one joint fit.

PULSE prior + lineup update = PULSE.

The source reconciles at least 99.916% of official game scores in every season and has valid ten-player lineups for 100% of published stints. It excludes 37,735 identified made technical free throws. No identified technical free throw remains unmatched.

## Decomposition

The public factor ledger reports true-shooting value, turnover value, offensive-rebound value, and opponent offensive-rebound prevention. Native factor units are calibrated into points per 100. A balancing residual preserves each total exactly. The residual is statistical remainder, not a basketball skill.

## Limits

- PULSE describes the completed season. It is not a preseason forecast.
- The validation uses observed next-season lineups as exposure weights.
- Player effects are associations after lineup adjustment. They are not causal credit.
- Early descriptive ratings use the final fitted prior mapping. They are not chronological validation rows.
- Matchup ratings remain local research. The public bundle contains no matchup payload.
- Replications are labeled as exact, methodology-aligned, or proxy. Similar names do not imply identical private inputs.
