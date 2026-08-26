# Matched-window SPM target-horizon pilot

## Decision

Run the full five-horizon comparison. Do not promote the five-year SPM arm.

The pilot separates two claims that had previously been conflated. Five-year
zero-prior RAPM carries more stable future-game signal than one-year RAPM, but
the fixed five-year SPM center hurts the posterior in both folds.

| Horizon | Candidate | 2023 RMSE | 2024 RMSE | Mean RMSE |
|---|---|---:|---:|---:|
| 1y | Zero-prior RAPM | 12.8325 | 14.7669 | 13.7997 |
| 1y | SPM-centered AIO | 12.7822 | 14.6090 | **13.6956** |
| 5y | Zero-prior RAPM | **12.7332** | **14.6645** | 13.6989 |
| 5y | SPM-centered AIO | 12.8588 | 14.6849 | 13.7718 |

The one-year SPM center improves its zero-prior comparator by 0.050 and 0.158
RMSE. The five-year center worsens its comparator by 0.126 and 0.020.

## Controls

- Features and RAPM targets cover the same trailing window.
- Both arms use the same 126 offense and 50 defense feature names.
- The frozen offense histogram-GBM and defense ridge learners are unchanged.
- Each statistical model trains only on earlier window ends.
- Zero-prior and centered RAPM use the same 3,000 / 3,000 / 300 penalties.
- Game row-set hashes are identical by test season across horizons.
- All four cells are checkpointed and resumable.
- Seasons 2025, 2026, and 2027 were not used.

This is a two-fold pilot. It authorizes the full comparison; it does not select
a production horizon.
