# BoxSPM versus TrackingSPM pilot

## Decision

TrackingSPM beats BoxSPM in the reused 2026 oracle-lineup diagnostic. The result
justifies a frozen historical comparison. It does not justify promotion.

## Model contract

Both models use separate offense and defense ridge regressions. Both train on
five-year windows ending from 2018 through 2024. Inner leave-one-window-out
validation selects ridge strength from `10, 30, 100, 300, 1000, 3000`. The
training label is zero-prior terminal-lineup RAPM over the same five seasons.
The row weight is the square root of the smaller offensive or defensive
possession count.

BoxSPM uses the 15 traditional per-100 box rates on each side. TrackingSPM uses
53 offense and 27 defense fields. It excludes every Box15 field. The offense
bank contains shot-location volume, drive and turnover events, stabilized shot
accuracy and frequency, touch ratios, and shot quality. The defense bank
contains rebound contests, DFG and rim defense, hustle events, matchup defense,
and source-availability flags. zTS is not in this pilot because zTS is a
playtype metric rather than a tracking metric.

The statistical prior then receives the same 2025 one-season RAPM update with
penalties `3000 / 3000 / 300`. The evaluation supplies observed 2026 lineups as
exposure and scores 1,228 identical games.

## Results

| Model | 2026 margin RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: |
| TrackingSPM plus RAPM | **15.320** | **.369** | .765 |
| TrackingSPM | 15.352 | .346 | **.949** |
| BoxSPM plus RAPM | 15.415 | .349 | .783 |
| Zero-prior RAPM | 15.733 | .294 | .738 |
| BoxSPM | 15.774 | .267 | 1.114 |

BoxSPM minus TrackingSPM MSE is `+13.155`, with a 5,000-draw paired whole-game
interval of `[+7.784, +18.583]`. After the same RAPM update, BoxSPM minus
TrackingSPM MSE is `+2.934`, with interval `[+0.438, +5.376]`. Tracking lowers
AIO RMSE by `.095` points per game.

The tracking prior also fits the 2025 five-year RAPM label better. Net weighted
RMSE is `1.680` for TrackingSPM and `1.894` for BoxSPM. Net correlation is
`.624` versus `.362`.

## Limits

This test has one reused outcome season. It uses actual future lineups, so it is
not a deployable forecast. At least one unknown player slot appears in 1,122 of
1,228 games. Tracking wins that large segment, but loses slightly in the 106
fully known-lineup games. The smaller segment cannot satisfy the 500-game
segment guard. The model definition must remain frozen before testing earlier
outcome seasons.

Season 2027 was not loaded.

## Artifacts

Run `box_vs_tracking_spm_pilot_v1_efe4736254` stores the exact feature lists,
selected penalties, priors, game predictions, coverage, metrics, 5,000 paired
draws, source hashes, and QA results.
