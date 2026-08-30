# Walk-forward SPM pilot

Run `walk_forward_spm_pilot_v1_afcc388e8d` checks one chronological cutoff.
It builds 2024 ratings, freezes them, and scores 1,226 games from 2025.

The SPM mapping trains on five-year windows ending before 2024. The 2024
player inputs pool 2020 through 2024. Each AIO uses the corresponding SPM as
the prior for a 2024 one-season possession RAPM update. Every arm then scores
the same observed 2025 lineups.

## Results

| Rating | Future-game RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: |
| Full SPM | 15.103 | .3237 | .776 |
| BoxSPM | 15.263 | .2772 | .986 |
| Full SPM plus RAPM | 14.894 | .3753 | .725 |
| BoxSPM plus RAPM | 14.864 | .3704 | .764 |

Full SPM lowers standalone MSE by `4.856` points squared per game against
BoxSPM. The 1,000-draw paired whole-game interval is `[-9.543, -0.160]`.

After the same RAPM update, Full SPM raises MSE by `0.884` against BoxSPM. The
paired interval is `[-0.897, +2.724]`. The posterior comparison is unresolved.
Full SPM has slightly higher correlation, while BoxSPM has slightly lower RMSE
and better calibration.

## Data checks and limits

- Every arm scores the same 1,226 games and outcomes.
- Game predictions contain no duplicate keys, missing margins, or missing
  predictions.
- Both priors cover 100% of 2024 offensive and defensive possession exposure.
- Offense plus defense equals net exactly in every saved rating.
- The future-game design contains 242,663 unknown player-possession slots out
  of 2,476,300 total slots, or 9.80%. The scorer assigns zero impact to these
  unseen players for every arm.
- Actual future lineups supply exposure. This isolates rating quality but does
  not represent a deployable pregame forecast.
- The feature contract and frozen Full SPM learner were developed using other
  inspected seasons. This run enforces the numerical cutoff, but it is not a
  clean nested outer-fold model-selection result.
- One reused cutoff can verify the implementation. It cannot select features
  or settle which prior is better.

The next validation review must decide how to handle unseen players, projected
minutes, nested model selection, and the balance between future-game accuracy
and retrospective same-season impact before the full historical run.
