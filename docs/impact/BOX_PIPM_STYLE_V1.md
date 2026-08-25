# BoxPIPM-style v1

## Status

**Research baseline.** This is a transparent box-score comparator. It is not a
replication of Jacob Goldstein's full PIPM, and it is not an SPM or AIO input.

## Why the name is qualified

Public descriptions of PIPM state that it combined a box-score prior with
luck-adjusted on/off information. The historical implementation and its full
luck adjustment are not reproducible from a complete public specification.
Calling a box-only reproduction "PIPM" would overstate what it is.

This baseline isolates the reproducible design idea: a linear box-score prior
for RAPM. It deliberately excludes the on/off component so it remains a clean
comparison with SPM and BPM.

## Contract

- **Target:** annual, zero-prior, terminal-lineup RAPM offense and defense.
- **Population:** 2017--24 annual player seasons.
- **Training:** leave one season out. Ridge strength is selected only within
  the remaining seasons.
- **Reliability:** square root of the smaller offensive/defensive RAPM exposure.
  It is a fit weight, not a feature.
- **Net:** offensive prediction plus defensive prediction.

The only 15 predictors are traditional per-100 box rates:

`PTS, AST, TOV, STL, BLK, OREB, DREB, PF, PFD, FTA, FTM, FG2A, FG2M, FG3A, FG3M`.

It excludes age, experience, height, listed position, minutes, games,
possessions, on/off, team ratings, tracking, playtype, and external ratings.

## First run

`box_pipm_style_v1_1768252352` used the same 5,791-row 2014--24 panel and the
same 4,341 2017--24 held-out rows as the pinned annual SPM.

| Target | Mean weighted RMSE | Mean correlation |
|---|---:|---:|
| Offense RAPM | 1.0635 | .5350 |
| Defense RAPM | 1.0657 | .2988 |
| Net RAPM | 1.5242 | .4595 |

It is weaker than the current SPM. That is useful: it quantifies the value of
the SPM's shot, tracking, playtype, and matchup feature families beyond a
traditional-box baseline.

## Next decision

Do not blend this into AIO. A full PIPM/LEBRON-style challenger requires a
separate, audited luck-adjustment contract. That work must define its inputs,
event-time availability, and relation to the independent RAPM target first.
