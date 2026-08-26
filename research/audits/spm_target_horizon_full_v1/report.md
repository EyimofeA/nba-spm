# Full matched-window SPM target-horizon decision

## Decision

Keep one-year RAPM as the retrospective SPM target. Use five-year zero-prior
RAPM as the predictive history backbone. Do not center the five-year fit on the
current fixed SPM at full strength.

## Future-game result

| Rank | Horizon | Model | Mean RMSE | Mean correlation |
|---:|---|---|---:|---:|
| 1 | 5y | Zero-prior RAPM | **13.7681** | 0.3681 |
| 2 | 6y | Zero-prior RAPM | 13.8010 | 0.3658 |
| 3 | 3y | Zero-prior RAPM | 13.8189 | 0.3567 |
| 4 | 5y | SPM-centered AIO | 13.8269 | **0.3693** |
| 5 | 3y | SPM-centered AIO | 13.8321 | 0.3648 |
| 6 | expanding | Zero-prior RAPM | 13.8547 | 0.3620 |
| 7 | 6y | SPM-centered AIO | 13.8559 | 0.3670 |
| 8 | 1y | SPM-centered AIO | 13.8736 | 0.3427 |
| 9 | expanding | SPM-centered AIO | 13.9034 | 0.3622 |
| 10 | 1y | Zero-prior RAPM | 13.9698 | 0.3204 |

The comparison uses the same held-out games in 2020, 2021, 2022, 2023, and
2024. Five-year zero-prior RAPM beats six-year by 0.0329 RMSE with paired 95%
interval [0.0134, 0.0527]. It beats three-year by 0.0508, interval [0.0058,
0.0951]. Every other candidate's paired interval is also above zero.

## What the SPM target did

Longer-window SPMs correlate more strongly with their smoother labels. That is
not the operational goal. The downstream posterior result is worse:

| Horizon | Centered AIO minus matching zero-prior RMSE |
|---|---:|
| 1y | -0.0962 |
| 3y | +0.0132 |
| 5y | +0.0588 |
| 6y | +0.0549 |
| expanding | +0.0487 |

Only the one-year center helps. The five-year center loses in all five folds.
This rejects the assumption that a more stable RAPM label automatically makes
a better statistical prior.

## Controls

- Fifty complete RAPM target windows were rebuilt from 2009-23 possession data.
- Rolling statistical features cover the exact same source seasons as their
  labels; expanding history begins in 2014.
- Every fold trains SPM only on earlier window ends.
- All preprocessing remains inside the frozen learner pipelines.
- All horizons use 126 offense and 50 defense feature names, terminal lineups,
  and 3,000 / 3,000 / 300 RAPM penalties.
- All 25 model cells and 50 target fits are checkpointed.
- Paired uncertainty uses 10,000 whole-game resamples within season.
- Maximum loaded season is 2024. Seasons 2025, 2026, and 2027 were not used.

The five-year winner is now frozen for a reused 2025-26 diagnostic. It is not
production evidence and does not change the public retrospective estimand.
