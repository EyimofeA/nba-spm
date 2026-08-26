# Published annual SPM and five-year research SPM

## The short version

There are two different models. The website SPM predicts one-season RAPM from
one player-season of statistics. The research challenger pools five seasons of
statistics and predicts five-season RAPM. Calling the latter the “old AIO” was
wrong: it was never the exact website baseline.

| Contract | Website annual SPM | Five-year research SPM |
|---|---|---|
| Training row | Player-season | Player and five-year window end |
| Feature window | Season `t` only | Seasons `t-4` through `t` |
| Target | Zero-prior RAPM in season `t` | Zero-prior RAPM over `t-4...t` |
| Training split | Leave one season out | Window ends strictly before scored window |
| Offense learner | Histogram GBM | Histogram GBM |
| Defense learner | Ridge | Ridge |
| Sample weight | Square root of smaller offensive/defensive possession count | Same |
| Website inputs | 127 offense, 68 defense | Starts from the same contract |
| Selected additions | None | 3 offense, 4 defense |

## Website annual inputs and targets

For the published model, each row is one player-season. The statistical inputs
and the RAPM label come from the same season. Training holds out the complete
season being scored; it does not use a rolling five-year row.

```text
x[i,t] = player statistics from season t
y_off[i,t] = one-season zero-prior ORAPM in season t
y_def[i,t] = one-season zero-prior DRAPM in season t
w[i,t] = sqrt(min(offensive possessions, defensive possessions))
```

The exact selected feature list is stored in the SPM Lab and in
`artifacts/research/spm_weight_ablation/spm_weight_ablation_v1_9a4136a6d7/feature_catalog.parquet`.
It contains 170 unique fields: 127 offense inputs and 68 defense inputs, with
25 fields shared by both sides.

## Five-year research inputs and targets

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

## What changed in the five-year challenger

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

The selected five-year features produced small AIO game-RMSE gains through 2025
and lost by `0.0126` in reused 2026. Across 2021 through 2024 player-seasons,
the five-year research AIO and Website AIO net ratings correlate `0.960`. In
the oracle-minutes team test, mean R-squared is `0.640` for the research AIO and
`0.618` for Website AIO. The five-year model remains a localhost research
challenger until the untouched confirmation.
