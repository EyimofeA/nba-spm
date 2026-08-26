# Hand-selected sparse five-year SPM

Status: research null. No public model or site data changed.

Run `hand_selected_sparse_spm_v1_f04379a684` freezes the first twelve metrics
named by the principal before scoring. Separate offense and defense ridge
models use alpha 3000, complete rolling five-year rows, same-window z-scores,
training-fold median imputation and square-root possession exposure weights.
Historical predictions train only on earlier complete windows.

| Side | Function | Input |
| --- | --- | --- |
| Offense | Scoring | `PTS_p100` |
| Offense | Efficiency | `zts_pct_points` |
| Offense | Turnovers | `turnover_to_load_2017_eb` |
| Offense | Passing | `box_creation_2017_eb_p100` |
| Offense | Offensive rebounding | `OREB_p100` |
| Offense | Spacing | `crafted_spacing_stable_v1` |
| Offense | Offensive load | `offensive_load_2017_eb_p100` |
| Offense | Rim pressure | `at_rim_fga_p100` |
| Defense | Event stops | `event_stops_p100` |
| Defense | Rim protection | `rim_points_saved_p100` |
| Defense | Contested rebounding | `dreb_contests_p100` |
| Defense | Foul discipline | `shooting_fouls_committed_p100` |

`event_stops_p100` is the direct rate of steals, recovered blocks, charges and
offensive fouls drawn. It is intentionally not called Dean Oliver Stop%.
Oliver's broader estimate allocates forced misses and non-steal turnovers and
uses an estimated individual defensive-possession denominator.

## Result

| Evaluation | Hand-selected | Full five-year SPM | Winner |
| --- | ---: | ---: | --- |
| Mean next-season team-win R², two folds | .4720 | .5446 | Full |
| Next-season one-year RAPM net Pearson, five folds | .3986 | .4241 | Full |
| Next-season one-year RAPM net Spearman, five folds | .3108 | .3308 | Full |
| Next-season one-year RAPM net RMSE, five folds | 1.8913 | 1.9813 | Hand-selected |

The hand-selected model improves RMSE but loses both team-win folds: `.4483`
versus `.5321` for 2021-to-2022 and `.4958` versus `.5571` for 2022-to-2023.
It also loses player ordering on offense, defense and net. Keep the full model
as the research baseline; do not run this challenger's AIO update.

The defended-rim source ends in 2025. The 2026 five-year row therefore pools
observed 2022-25 rim data. zTS is available through 2026. Exact artifacts,
predictions, coefficients and feature coverage live under
`artifacts/research/hand_selected_sparse_spm/hand_selected_sparse_spm_v1_f04379a684`.
