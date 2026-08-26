# Predictive current AIO decision

The two-year half-life plus raw predictive-SPM prior is the research champion.
It is not a public or confirmed model.

## Development result

| Arm | Mean game-margin RMSE |
| --- | ---: |
| selected_decay_spm_prior_aio | 13.7122 |
| selected_decay_zero_prior | 13.7429 |
| five_year_spm_prior_aio | 13.7550 |
| five_year_zero_prior | 13.7681 |

The selected AIO beat five-year zero-prior RAPM in 4 of 5 folds.
The paired whole-game MSE interval favored it against every frozen comparator.

## Reused diagnostics

| Season | Selected AIO | Five-year zero prior |
| ---: | ---: | ---: |
| 2025 | 14.7719 | 14.8551 |
| 2026 | 15.1817 | 15.2962 |

The diagnostics support the development result but do not confirm it.
Season 2027 stays untouched.
