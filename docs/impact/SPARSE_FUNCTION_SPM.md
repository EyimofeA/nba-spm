# Sparse function-first SPM

Status: invalid feature lineage. No public model or site data changed.

The stored run is reproducible, but its declared foul-pressure feature is
wrong. Gabriel's `ShootingFouls` field records shooting fouls committed. The
run labeled it shooting fouls drawn, divided it by offensive possessions and
placed it in the offense model. Separate two-point and three-point shooting
fouls drawn columns exist. Therefore the numerical comparison below is kept
only as an audit record and cannot support model selection.

## Model

Run `sparse_function_spm_v1_4f1ecaa353` tests whether a small, auditable SPM
can replace the 195-field five-year model. It uses separate ridge regressions
for offense and defense. Each feature is standardized against players in the
same five-year window. Possession exposure supplies the training weight and is
not a feature.

| Side | Function | Input |
| --- | --- | --- |
| Offense | Scoring load | `offensive_load_2017_eb_p100` |
| Offense | Shotmaking | `effective_fg_pct` |
| Offense | Passing creation | `potential_assists_p100_relative` |
| Offense | Ball security | `turnover_to_load_2017_eb` |
| Offense | Spacing | `crafted_spacing_stable_v1` |
| Offense | Foul pressure | `shooting_fouls_drawn_p100` |
| Offense | Offensive rebounding | `OREB_p100` |
| Defense | Rim-protection proxy | `BLK_p100_relative` |
| Defense | Disruption | `STL_p100_relative` |
| Defense | Contest involvement | `rebound_contest_share` |
| Defense | Defensive rebounding | `DREB_p100_relative` |
| Defense | Foul discipline | `PF_p100_relative` |

Blocks are not rim deterrence or defended-shot quality. The full historical
panel does not contain an honest modern rim-points-saved or matchup-defense
field, so the model does not pretend otherwise.

## Result

| Evaluation | Sparse | Full five-year SPM | Winner |
| --- | ---: | ---: | --- |
| Mean next-season team-win R², two folds | .4547 | .5446 | Full |
| Next-season one-year RAPM net Pearson, five folds | .3519 | .4241 | Full |
| Next-season one-year RAPM net Spearman, five folds | .2727 | .3308 | Full |
| Next-season one-year RAPM net RMSE, five folds | 1.9027 | 1.9813 | Sparse |

Even without the lineage defect, the lower RMSE would not be enough: the model
compresses estimates, loses player ordering and loses both team-win folds. It
is not retained for AIO.

The run, predictions, coefficients and exact feature registry are under
`artifacts/research/sparse_function_spm/sparse_function_spm_v1_4f1ecaa353`.

The corrected, principal-selected follow-up is documented in
`docs/impact/HAND_SELECTED_SPARSE_SPM.md`.
