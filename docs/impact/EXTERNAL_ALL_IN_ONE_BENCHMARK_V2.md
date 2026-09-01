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
| Defense residual AIO minus Box15 AIO | -0.184 | [-0.532, 0.171] |
| Defense residual AIO minus MAMBA | -2.095 | [-4.483, 0.316] |
| Box15 AIO minus MAMBA | -1.912 | [-4.343, 0.491] |
| Box15 AIO minus rich-SPM AIO | **-2.027** | **[-2.826, -1.195]** |
| Box15 2014+ AIO minus long-history Box15 AIO | +0.453 | [0.154, 0.750] |

The first three differences do not establish a winner. Box15 AIO does
establish a lower error than rich-SPM AIO on this panel. Restricting Box15 to
2014 onward makes it worse here, so the long historical box panel is useful.

## Train-through-2023 holdout

The 2024 rating predicts 2025 games. Rich SPM uses 2014--23 training rows. The
restricted Box15 arm uses the same 2014--23 range. The long-history Box15 arm
uses 2005--23.

| Candidate | MSE | RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: | ---: |
| MAMBA | **219.691** | **14.822** | **0.386** | 0.809 |
| Defense residual + RAPM | 220.370 | 14.845 | 0.367 | 0.802 |
| Box15 + RAPM | 221.583 | 14.886 | 0.360 | 0.802 |
| Box15 2014+ + RAPM | 222.247 | 14.908 | 0.356 | 0.801 |
| Rich elastic SPM + RAPM | 224.619 | 14.987 | 0.350 | 0.756 |
| xRAPM | 226.052 | 15.035 | 0.352 | 0.790 |
| EPM | 229.699 | 15.156 | 0.329 | 0.795 |
| BPM 2.0 | 234.050 | 15.299 | 0.325 | 0.686 |
| Defense residual prior | 234.060 | 15.299 | 0.296 | 1.280 |
| Rich elastic SPM prior | 235.033 | 15.331 | 0.282 | 0.981 |
| Box15 prior | 240.662 | 15.513 | 0.240 | 1.109 |
| Box15 2014+ prior | 241.405 | 15.537 | 0.234 | 1.105 |
| LEBRON | 241.722 | 15.547 | 0.239 | 0.774 |

MAMBA's MSE advantage over the defense-residual AIO is `0.678`, with a paired
95% interval of `[-4.522, 5.983]` for defense residual minus MAMBA. One season
cannot separate them. The defense residual beats Box15 AIO by `1.214` MSE on
this fold, but this is reused evidence and does not satisfy the frozen
multi-season promotion contract.

Box15 AIO beats rich-SPM AIO by `3.036` MSE, with a paired 95% interval of
`[-4.741, -1.317]`. The restricted 2014+ Box15 AIO loses `0.664` MSE to the
long-history Box15 AIO, with interval `[0.263, 1.073]`.

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
| EPM | 8 | 193.650 | 13.916 |
| PIPM | 5 | 194.587 | 13.949 |
| LEBRON | 8 | 195.415 | 13.979 |
| RAPTOR | 6 | 198.953 | 14.105 |
| BPM 2.0 | 8 | 200.300 | 14.153 |

Public metrics cover about 85--89% of lineup slots in their available folds.
The internal priors cover about 75--77%. This difference prevents a causal
interpretation of the broad ordering.

## DARKO sensitivity

No historical DARKO panel was available. The benchmark uses one dated,
preseason 2018--19 snapshot from the
[Andrew Patton team-ratings tutorial](https://github.com/anpatton/basic-nba-tutorials/blob/main/team_ratings/how_to_make_team_ratings.md).
The snapshot predicts 2018--19 games on one strict common-player panel.

DARKO scores `183.493` MSE and `13.546` RMSE. MAMBA scores `182.826` MSE on the
same panel. The defense-residual AIO scores `178.453`. This is one fold with a
different timestamp. It cannot establish a general DARKO comparison.

## Model and evaluation contract

Each statistical rating for season (t) predicts game margins in season
(t+1). The evaluator uses the actual players who appeared in each future game
only as exposure weights. It does not use their future box statistics or
ratings. This measures rating quality conditional on observed participation;
it is not a deployable preseason forecast because projected availability and
minutes are not modeled.

The internal posteriors use the same one-season terminal-lineup RAPM update:

\[
\hat\beta=(X^TX+P)^{-1}\left[X^T(y-b)+P\mu\right],
\]

with offense, defense, and home penalties of `3000`, `4500`, and `300`.
Box15, Box15 2014+, rich SPM, and the defense residual differ only in prior
center \(\mu\). External metrics remain standalone. EPM, xRAPM, PIPM, RAPTOR,
LEBRON, MAMBA, and DARKO already contain different amounts of on/off or impact
information. Applying another RAPM update would not be a controlled comparison.

The primary score is the equal-season mean next-season whole-game margin MSE.
RMSE is its square root in points per game. Correlation measures ordering.
Calibration slope measures scale: values below one indicate that predictions
vary too much relative to outcomes; values above one indicate that they vary
too little.

## Data quality and limitations

- Exact NBA IDs identify EPM and LEBRON rows. Season-specific normalized names
  identify the other imported metrics. Modern identity match rates are
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
