# Luck-adjusted RAPM and SPM decision

Normal realized-points RAPM remains the reference. None of the broad conversion-adjusted arms improves future-game prediction in either reused diagnostic season.

## Future-game margin RMSE

| Arm | 2025 | 2026 |
| --- | ---: | ---: |
| normal_realized_points | 15.0541 | 15.4732 |
| opponent_luck_adjusted | 15.5132 | 15.5675 |
| teammate_and_opponent_luck_adjusted | 15.6971 | 15.5693 |
| full_expected_outcome | 15.7966 | 15.7941 |
| ft3p_player_skill_adjusted_joint | 15.1197 | 15.3898 |

## Earlier FT/3P result

The corrected pre-season player-skill FT/3P diagnostic changes RMSE by +0.0656 in 2025 and -0.0834 in 2026 versus normal RAPM. The paired whole-game intervals are stored in `paired_game_bootstrap.parquet`.

It nearly reproduces the earlier 2026 gain, but the 95% interval crosses zero and the same frozen arm loses in 2025. That is a null, not a promotion.

## Prediction of future normal RAPM

| Arm | 2025 RMSE | 2025 corr | 2026 RMSE | 2026 corr |
| --- | ---: | ---: | ---: | ---: |
| normal_realized_points | 2.0925 | 0.3811 | 2.1134 | 0.4755 |
| opponent_luck_adjusted | 1.9945 | 0.3327 | 2.0852 | 0.4116 |
| teammate_and_opponent_luck_adjusted | 1.9458 | 0.3379 | 1.9507 | 0.4100 |
| full_expected_outcome | 1.8591 | 0.3053 | 1.9181 | 0.3914 |
| ft3p_player_skill_adjusted_joint | 1.9010 | 0.4015 | 1.9588 | 0.4838 |

Expected-outcome ratings often lower future-RAPM RMSE by compressing the rating spread, but they lose net correlation to normal RAPM in both seasons. The calibration slopes and low/high-exposure slices are stored in `future_normal_rapm_metrics.parquet`. Smoother labels are not automatically better AIO priors.

## SPM stop

A luck-adjusted SPM was not fit. Complete shot-level expected-outcome labels begin in 2024, leaving only 2024 and 2025 as legal training labels for a 2026 output. Two seasons cannot support the required chronological feature and learner selection. Forcing that model would be less defensible than recording the data limit.

2025 and 2026 are reused diagnostics. Season 2027 was not loaded.
