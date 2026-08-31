# Compact SPM comparison

Run `compact_spm_comparison_v1_2a0f8a6f31` fits the corrected Full SPM,
correlation-pruned Compact SPM, and BoxPIPM-style model on identical rolling
five-year targets. Every rating season trains only on earlier window ends.
Each fitted prior then receives the same one-season terminal-lineup RAPM update
with penalties `3000 / 3000 / 300`.

The Full SPM uses 128 offense and 72 defense inputs. Compact SPM uses 105
offense and 58 defense inputs. Both rich models keep the frozen histogram
gradient-boosting offense learner and ridge defense learner.

## Five-year RAPM target fit

These figures average the five scored rating seasons. Lower RMSE is better.

| Model | Side | RMSE | Correlation |
| --- | --- | ---: | ---: |
| Full SPM | Offense | 1.0224 | .7461 |
| Compact SPM | Offense | 1.0227 | .7457 |
| Full SPM | Defense | 1.0445 | .5267 |
| Compact SPM | Defense | 1.0496 | .5269 |
| Full SPM | Net | 1.4811 | .6597 |
| Compact SPM | Net | 1.4855 | .6614 |

Full SPM fits the five-year RAPM labels slightly better. Compact SPM has a
slightly higher net correlation. Neither difference establishes better game
prediction.

## Next-season game prediction

The table averages each test season equally. It uses identical games and actual
future lineups only as exposure weights.

| Model | Mean RMSE | Correlation | Calibration slope |
| --- | ---: | ---: | ---: |
| BoxPIPM-style AIO | 14.3792 | .3607 | .8075 |
| Full SPM AIO | 14.4063 | .3654 | .7535 |
| Compact SPM AIO | 14.4096 | .3648 | .7543 |
| Zero-prior RAPM | 14.5697 | .3219 | .8444 |
| Compact SPM | 14.5896 | .3212 | .8202 |
| Full SPM | 14.6004 | .3202 | .8080 |
| BoxPIPM-style | 14.6877 | .2936 | 1.1243 |

Compact SPM improves standalone next-season MSE over Full SPM. Full minus
Compact MSE is `+0.2878`, with a paired whole-game 95% interval of `[+0.0450,
+0.5400]`. Compact wins three of five folds.

The RAPM update reverses that small advantage. Full AIO minus Compact AIO MSE
is `-0.1112`, with a 95% interval of `[-0.2627, +0.0374]`. Full AIO has the
better point estimate, but the interval crosses zero.

BoxPIPM-style AIO has the lowest mean error. Compact AIO minus BoxPIPM-style
AIO MSE is `+0.9030`, with a 95% interval of `[-0.0519, +1.8802]`. This reused
comparison does not independently establish a winner.

## Decision

Correlation pruning improves the standalone prior's future-game result. It
does not improve the posterior after possession evidence enters the model.
Keep the compact contract for interpretation and later stability-selection
work. Do not replace Full SPM or the frozen BoxPIPM-style research prior from
this run.

The result supports a narrow conclusion. Some rich features are redundant for
future game prediction, while the RAPM likelihood already supplies much of the
signal that remains. A supervised selector must use nested chronological folds
and must score the final AIO rather than the standalone SPM alone.
