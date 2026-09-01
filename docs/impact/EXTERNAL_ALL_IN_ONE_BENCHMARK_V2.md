# External All-in-One Benchmark V2

## Result

The RAPM update matters more than the choice between the tested statistical
priors. The defense-residual AIO has the lowest internal-model error, but its
advantage over Box15 AIO remains small. Box15 AIO clearly beats rich-SPM AIO.

MAMBA has the best standalone result across 2017--24 and on the 2025 holdout.
The defense-residual AIO has the best result on the strict 2017--20 panel. Its
MSE advantage over MAMBA has a 95% paired interval that crosses zero.

These reused outcomes do not justify a production change. They support the
existing decision: keep rich SPM as a standalone statistical impact estimate,
keep Box15 as the simple AIO prior, and retain the defensive residual as the
only live prior challenger.

## Strict common-coverage result

This is the cleanest multi-metric comparison. Every candidate uses the same
player intersection and the same next-season games. Rating seasons 2017--20
predict games in 2018--21.

| Candidate | Folds | MSE | RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Defense residual + RAPM | 4 | **178.878** | **13.375** | 0.359 | 0.852 |
| Box15 + RAPM | 4 | 179.062 | 13.381 | 0.357 | 0.855 |
| Box15 2014+ + RAPM | 4 | 179.514 | 13.398 | 0.354 | 0.859 |
| MAMBA | 4 | 180.973 | 13.453 | **0.367** | 0.838 |
| Rich elastic SPM + RAPM | 4 | 181.088 | 13.457 | 0.346 | 0.816 |
| DARKO DPM | 4 | 184.186 | 13.572 | 0.340 | 0.950 |
| EPM | 4 | 184.188 | 13.572 | 0.343 | 0.856 |
| xRAPM | 4 | 184.662 | 13.589 | 0.343 | 0.831 |
| LEBRON | 4 | 185.819 | 13.632 | 0.328 | 1.054 |
| PIPM | 4 | 186.909 | 13.671 | 0.322 | 0.879 |
| Defense residual prior | 4 | 188.179 | 13.718 | 0.333 | 1.609 |
| BPM 2.0 | 4 | 189.066 | 13.750 | 0.324 | 0.733 |
| Box15 prior | 4 | 189.291 | 13.758 | 0.320 | 1.550 |
| RAPTOR | 4 | 189.697 | 13.773 | 0.332 | 0.690 |
| Rich elastic SPM prior | 4 | 189.874 | 13.779 | 0.300 | 1.236 |
| Box15 2014+ prior | 4 | 190.829 | 13.814 | 0.308 | 1.576 |

The common player set contains 367--388 players per rating season. It covers
73.3--74.8% of the next-season lineup slots. The missing slots belong mainly to
players who lack one of the public metrics. All candidates receive the same
zero value for those unmatched slots, so the comparison is paired but does not
represent the entire league.

Key paired MSE differences use 5,000 whole-game bootstrap draws within season:

| Comparison | MSE difference | 95% interval |
| --- | ---: | ---: |
| Defense residual AIO minus Box15 AIO | -0.184 | [-0.523, 0.161] |
| Defense residual AIO minus MAMBA | -2.095 | [-4.540, 0.268] |
| Box15 AIO minus MAMBA | -1.912 | [-4.339, 0.439] |
| Box15 AIO minus rich-SPM AIO | **-2.027** | **[-2.854, -1.224]** |
| Box15 2014+ AIO minus long-history Box15 AIO | +0.453 | [0.157, 0.755] |
| DARKO DPM minus EPM | -0.002 | [-1.741, 1.782] |

The first three differences do not establish a winner. Box15 AIO does
establish a lower error than rich-SPM AIO on this panel. Restricting Box15 to
2014 onward makes it worse here, so the long historical box panel is useful.

## Train-through-2023 holdout

The 2024 rating predicts 2025 games. Rich SPM uses 2014--23 training rows. The
restricted Box15 arm uses the same 2014--23 range. The long-history Box15 arm
uses 2005--23.

| Candidate | MSE | RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: | ---: |
| MAMBA | **219.047** | **14.800** | **0.389** | 0.814 |
| Defense residual + RAPM | 219.905 | 14.829 | 0.369 | 0.806 |
| Box15 + RAPM | 220.884 | 14.862 | 0.363 | 0.807 |
| Box15 2014+ + RAPM | 221.637 | 14.887 | 0.359 | 0.806 |
| Rich elastic SPM + RAPM | 224.581 | 14.986 | 0.350 | 0.756 |
| xRAPM | 228.072 | 15.102 | 0.342 | 0.776 |
| EPM | 229.855 | 15.161 | 0.328 | 0.796 |
| DARKO DPM | 232.242 | 15.239 | 0.304 | 0.857 |
| Defense residual prior | 233.362 | 15.276 | 0.301 | 1.300 |
| BPM 2.0 | 233.377 | 15.277 | 0.327 | 0.694 |
| Rich elastic SPM prior | 234.972 | 15.329 | 0.282 | 0.983 |
| Box15 prior | 239.600 | 15.479 | 0.249 | 1.146 |
| Box15 2014+ prior | 240.515 | 15.509 | 0.242 | 1.140 |
| LEBRON | 241.845 | 15.551 | 0.238 | 0.772 |

MAMBA's MSE advantage over the defense-residual AIO is `0.858`, with a paired
95% interval of `[-4.651, 6.290]` for defense residual minus MAMBA. One season
cannot separate them. The defense residual beats Box15 AIO by `0.979` MSE on
this fold, with interval
`[-1.851, -0.128]`. This is reused evidence and does not satisfy the frozen
multi-season promotion contract.

Box15 AIO beats rich-SPM AIO by `3.696` MSE, with a paired 95% interval of
`[-5.461, -2.005]`. The restricted 2014+ Box15 AIO loses `0.753` MSE to the
long-history Box15 AIO, with interval `[0.344, 1.166]`. DARKO trails EPM by
`2.387` MSE, but its interval `[-1.840, 6.692]` does not separate them.

This strict holdout intersection contains 415 rated players and covers 77.7%
of next-season lineup slots.

## All available coverage

The broad table uses every available rating season from 2017--24 and each
metric's own player coverage. It is a coverage audit, not a head-to-head
ranking. PIPM has five folds. RAPTOR has six. The other listed modern metrics
have eight.

| Candidate | Folds | MSE | RMSE |
| --- | ---: | ---: | ---: |
| MAMBA | 8 | 187.918 | 13.708 |
| Defense residual + RAPM | 8 | 189.217 | 13.756 |
| Box15 2014+ + RAPM | 8 | 189.587 | 13.769 |
| Box15 + RAPM | 8 | 189.614 | 13.770 |
| Rich elastic SPM + RAPM | 8 | 191.688 | 13.845 |
| xRAPM | 8 | 192.578 | 13.877 |
| DARKO DPM | 8 | 193.301 | 13.903 |
| EPM | 8 | 193.650 | 13.916 |
| PIPM | 5 | 194.587 | 13.949 |
| LEBRON | 8 | 195.415 | 13.979 |
| RAPTOR | 6 | 198.953 | 14.105 |
| BPM 2.0 | 8 | 200.300 | 14.153 |

Public metrics cover about 85--89% of lineup slots in their available folds.
The internal priors cover about 75--77%. This difference prevents a causal
interpretation of the broad ordering.

## DARKO history and timing sensitivity

The supplied DARKO workbook contains 13,726 player-season rows from 1997
through 2024. It has 2,703 unique NBA player IDs and no duplicate player-season
keys. Every row has offensive, defensive, and net DPM. The component identity
`o_dpm + d_dpm = dpm` holds exactly. The benchmark uses its 2017--24 seasons.

DARKO and EPM are indistinguishable on the strict 2017--20 panel: DARKO's MSE
is `184.186` and EPM's is `184.188`. The DARKO-minus-EPM interval is
`[-1.741, 1.782]`. DARKO's broad 2017--24 MSE is `193.301`. Its strict 2025
holdout MSE is `232.242`.

The benchmark also keeps one dated preseason 2018--19 snapshot from the
[Andrew Patton team-ratings tutorial](https://github.com/anpatton/basic-nba-tutorials/blob/main/team_ratings/how_to_make_team_ratings.md).
The snapshot predicts 2018--19 games on one strict common-player panel.

DARKO scores `183.493` MSE and `13.546` RMSE. MAMBA scores `182.826` MSE on the
same panel. The defense-residual AIO scores `178.453`. This is one fold with a
different timestamp. It checks information timing; the historical workbook
provides the general season-end comparison.

## Model and evaluation contract

Each statistical rating for season (t) predicts game margins in season
(t+1). The evaluator uses the actual players who appeared in each future game
only as exposure weights. It does not use their future box statistics or
ratings. This measures rating quality conditional on observed participation;
it is not a deployable preseason forecast because projected availability and
minutes are not modeled.

The internal posteriors use the same one-season terminal-lineup RAPM update:

$$
\hat\beta=(X^TX+P)^{-1}\left[X^T(y-b)+P\mu\right],
$$

with offense, defense, and home penalties of `3000`, `4500`, and `300`.
Box15, Box15 2014+, rich SPM, and the defense residual differ only in prior
center $\mu$. External metrics remain standalone. EPM, xRAPM, PIPM, RAPTOR,
LEBRON, MAMBA, and DARKO already contain different amounts of on/off or impact
information. Applying another RAPM update would not be a controlled comparison.

The primary score is the equal-season mean next-season whole-game margin MSE.
RMSE is its square root in points per game. Correlation measures ordering.
Calibration slope measures scale: values below one indicate that predictions
vary too much relative to outcomes; values above one indicate that they vary
too little.

## Data quality and limitations

- Exact NBA IDs identify DARKO, EPM, and LEBRON rows. Season-specific normalized
  names identify the other imported metrics. Modern identity match rates are
  97.7--99.7%.
- MAMBA comes from the supplied historical file. Its internal offense plus
  defense identity passes, but this run does not independently reproduce its
  methodology. Treat its leaderboard position as a benchmark result, not a
  verified model claim.
- BPM and xRAPM use the repository's pinned external annual table. PIPM ends in
  2021. The public RAPTOR file ends in 2022.
- The strict panel fixes player and game coverage across candidates. The broad
  panel does not.
- Every outcome has informed prior research. These are research diagnostics,
  not untouched promotion evidence.
- Artifact file hashes, runner hash, finite predictions, candidate-game key
  uniqueness, identical strict games, and identical strict coverage pass.

## Decision

Keep Box15 as `spm_prior`. Keep rich elastic SPM as `spm_impact`. Keep the
target-excluded defense residual as a frozen challenger. Do not promote it from
these reused comparisons.

The next product step can package the existing Box15 AIO, rich SPM, and RAPM
as separate views. The next modeling step should wait for new outcome evidence
or materially better defensive information rather than another broad feature
search.
