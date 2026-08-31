# AIO prior calibration and feature atlas v1

## Decision

Keep Box15 as the research prior. Use stronger side-specific prior precision as
a research challenger. Do not apply an affine rescaling to player priors.

The richer Full and Compact SPMs contain useful ranking signal. Their main
downstream weakness is overconfident AIO magnitude. A post-model game forecast
calibration recovers that signal, but it does not define a retrospective player
rating and cannot replace the AIO estimator.

## Prior calibration audit

Run `aio_prior_calibration_precision_v1_6f9b7ce1e9` uses saved Box15, Full, and
Compact priors. It does not refit SPM or change features, games, lineups, or the
possession likelihood. Each scored rating season selects parameters on strictly
earlier next-season game folds. The four scored outcome seasons are 2023--26.

| Prior and update | Equal-season RMSE | Mean correlation | Calibration slope |
| --- | ---: | ---: | ---: |
| Full, game-affine forecast diagnostic | 14.3296 | .3689 | .9897 |
| Compact, game-affine forecast diagnostic | 14.3368 | .3679 | .9875 |
| Box15, game-affine forecast diagnostic | 14.3660 | .3632 | .9796 |
| Box15, selected side precision | 14.3978 | .3601 | .8794 |
| Full, selected side precision | 14.4095 | .3682 | .7870 |
| Compact, selected side precision | 14.4152 | .3672 | .7881 |
| Box15, fixed 3000 / 3000 | 14.4295 | .3632 | .8019 |
| Full, fixed 3000 / 3000 | 14.4515 | .3689 | .7508 |
| Compact, fixed 3000 / 3000 | 14.4578 | .3679 | .7512 |

Selected prior precision improves every candidate. Relative to its fixed
`3000 / 3000` update, Full improves MSE by `1.2123`, with a paired 95% interval
of `[0.4923, 1.9433]`. Compact improves by `1.2322`, with interval
`[0.4988, 1.9424]`. Box15 improves by `0.9147`, with interval
`[0.1146, 1.7588]`.

The nested selections usually choose offense penalties of `3000` or `4500`
and a defense penalty of `6000`. The defense choice reaches the grid boundary,
so the audit identifies a precision direction rather than a final penalty.

Player-level affine calibration uses earlier out-of-fold five-year RAPM labels:

```text
target_side = intercept_side + slope_side * prior_side
```

It worsens every candidate. Full MSE increases by `0.7068`; Compact increases
by `0.7326`; Box15 increases by `1.0200`. The intercept disappears when the
RAPM system recenters each side, while the fitted slope changes the prior
spread. The label-calibrated spread does not improve game prediction.

The game-affine diagnostic fits actual margin from predicted margin on earlier
games. It makes Full the best forecast arm. This establishes that Full has
useful ordering information and poor finished-prediction calibration. It does
not produce a player rating and remains outside the retrospective AIO.

## Target-free feature atlas

Run `spm_feature_atlas_v1_6949ad7b60` evaluates 178 completed five-year fields
from 2018--26 before a learner sees RAPM. It measures ranges, adjacent-season
stability, early-versus-late drift, and within-era redundancy.

| Result | Count |
| --- | ---: |
| Completed finite fields | 178 |
| Eligible for fold-internal screening | 138 |
| Source-shift flags | 14 |
| Stable correlation pairs at absolute correlation at least .95 | 13 |

The new pass-value feature has strong source drift. Its early-versus-late AUC
is `.878` and its standardized mean shift is `1.665`. Exclude it from the next
retrospective SPM screen until the source definition is reconciled.

Three new defensive mechanisms pass the measurement screen without a source
shift flag: defensive-rebound conversion above expected, foul-adjusted
activity, and workload-adjusted shot suppression. Rim-protection workload is
stable, but it correlates above `.98` with rim points saved in both eras. Keep
one representation inside each training fold.

The atlas also confirms stable duplicate pairs among raw and season-relative
PTS, AST, TOV, FTA, at-rim frequency, potential assists, and drives. The next
selector should keep the declared Box15 control intact and prune redundant
alternatives rather than remove Box15 fields after seeing a target.

## Learner decision

Keep histogram gradient boosting as the offense challenger and ridge as the
defense default. Elastic net remains a stability selector. A low-degree GAM and
ExtraTrees remain bounded diagnostics.

A clean pre-2022 learner tournament is not identifiable under the current rich
five-year panel. The completed panel starts at window end 2018. Five-year RAPM
labels from adjacent window ends overlap by four seasons. Purging that overlap
leaves no earlier rich-feature training window for a pre-2022 outer test.

Running the requested tournament anyway would either share target seasons
between train and test or fit a flexible learner from one window end. Both
choices violate the chronological contract. The next valid options are:

1. build a longer historical feature panel with consistent definitions;
2. specify a separate annual target and keep its result separate from the
   five-year AIO prior; or
3. retain the current learner contract until more nonoverlapping five-year
   windows exist.

The third option is the current decision. The repository already has 20 seeded
noise controls, target-sharing diagnostics, correlation pruning, and isolated
family tests. Another reused learner tournament would add selection pressure
without new evidence.
