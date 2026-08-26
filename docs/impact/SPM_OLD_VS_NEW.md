# Old and new five-year SPM

## The short version

The new SPM is not a different target, training window, or learner. It is the
old five-year SPM plus seven feature inputs that passed the frozen feature-group
screen. Both models predict five-year RAPM. Both train only on five-year windows
ending before the window being scored.

| Contract | Old SPM | New SPM |
|---|---|---|
| Training row | Player and five-year window end | Same |
| Feature window | Seasons `t-4` through `t` | Same |
| Target | Zero-prior RAPM over seasons `t-4` through `t` | Same |
| Historical training | Window ends strictly before `t` | Same |
| Offense learner | Histogram GBM | Same |
| Defense learner | Ridge | Same |
| Sample weight | Square root of smaller offensive/defensive possession count | Same |
| Base inputs | 127 offense, 68 defense | Same |
| Added inputs | None | 3 offense, 4 defense |
| Final inputs | 127 offense, 68 defense | 130 offense, 72 defense |

## Matched five-year inputs and targets

For player `i` and window end `t`, the statistical row pools seasons `t-4`
through `t`. The target is offense or defense from zero-prior RAPM fit over the
same five seasons.

```text
x[i,t] = aggregate player statistics from seasons t-4 ... t
y_off[i,t] = five-year zero-prior ORAPM over seasons t-4 ... t
y_def[i,t] = five-year zero-prior DRAPM over seasons t-4 ... t
w[i,t] = sqrt(min(offensive possessions, defensive possessions))
```

Possessions determine aggregation and sample weight. Minutes, games,
possessions, age, height, position, on/off, BPM, and xRAPM are not fitted
features.

The implementation enforces unique player-window rows and builds the weight in
`five_year_target_spm.py`:

```python
panel = feature.merge(
    target[target_columns],
    on=["PLAYER_ID", "Window_End"],
    how="inner",
    validate="one_to_one",
)
panel["sample_weight"] = np.sqrt(
    np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
)
```

## Chronological fitting

To score the window ending in season `t`, the model trains on every available
five-year label ending before `t`. It never trains on the scored window or a
later window.

```python
train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
scored = inference.loc[inference["Window_End"].eq(rating_season)].copy()

for side in ("offense", "defense"):
    model = _fit_model(
        _frozen_model(side), train, selected[side], f"target_{side}"
    )
    output[f"prior_{side}_per_100"] = model.predict(
        scored.loc[:, selected[side]]
    )

output["prior_net_per_100"] = (
    output["prior_offense_per_100"] + output["prior_defense_per_100"]
)
```

Offense uses a histogram gradient-boosted tree with learning rate `0.03`, 250
iterations, seven maximum leaves, minimum leaf size 30, and L2 regularization
`1.0`. Defense uses median imputation, missing indicators, standardization, and
ridge with alpha `3000`.

## What changed in the new SPM

Every candidate begins as an annual statistic stabilized only against that
season's league distribution and that player-season's opportunities. The code
then possession-weights those already frozen values across the five-year
window:

```python
window = annual.loc[annual["Window_End"].between(end - 4, end)]
values = pd.to_numeric(window[feature], errors="coerce")
weights = pd.to_numeric(window[weight_field], errors="coerce").clip(lower=0)
pooled = (
    (values.where(valid, 0.0) * weights.where(valid, 0.0))
    .groupby(window["PLAYER_ID"]).sum()
    / weights.where(valid, 0.0).groupby(window["PLAYER_ID"]).sum()
)
```

The feature screen used future annual RAPM in 2022, 2023, and 2024. A group had
to improve mean RMSE, win at least two folds, avoid more than `0.01` RMSE loss
among team changers, and avoid more than `0.01` correlation loss among team
changers. Opponent shooting outcome was a forced falsification group and could
not pass.

Selected offense additions:

```text
pass_creation_points_per_potential_assist_eb
high_value_assist_share_eb
bad_pass_turnovers_per_100_passes_eb
```

Selected defense additions:

```text
dfg_two_point_equivalent_saved_p100
rim_matchup_attempt_share
contested_3pt_share
matchup_3pa_share
```

The new model refits the exact same offense and defense learners after adding
those fields. No role label enters either model.

## How SPM becomes AIO

The SPM prediction is converted from points per 100 into a prior center for the
player coefficients. The RAPM solver then uses only possessions from season `t`
as likelihood evidence.

```text
c_off = SPM offense / 100
c_def = -SPM defense / 100
P = diag(3000 offense, 3000 defense, 300 home)

beta = (X'X + P)^-1 [X'(y - intercept) + P c]
```

The possession design still contains only five offensive players, five
defensive players, and home. The seven added statistics affect AIO only through
the SPM prior center.

In the feature-research implementation:

```python
center, _ = build_prior_center(
    design,
    prior,
    prior_window_end=rating_season,
    train_mask=train_mask,
    test_mask=test_mask,
)
beta, intercept = fit_coefficient_center_path(
    design, config, center, center_scales=(1.0,), row_mask=train_mask
)[1.0]
```

## Current decision

The new features produced small AIO game-RMSE gains through 2025 and lost by
`0.0126` in reused 2026. Across 2021 through 2024 player-seasons, New AIO and
Old AIO net ratings correlate `0.9994`. In the oracle-minutes team test, mean
R-squared is `0.6405` for New AIO and `0.6407` for Old AIO. The new model remains
a localhost research challenger. It has not replaced the old five-year SPM
reference.
