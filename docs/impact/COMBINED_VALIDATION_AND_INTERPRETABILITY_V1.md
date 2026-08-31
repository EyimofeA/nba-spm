# Combined validation and interpretable Box15 AIO v1

## Decision

Keep Box15 as the research AIO prior. The richer Full SPM is materially better
at reproducing its five-year RAPM training target, but that advantage does not
survive the fixed one-season RAPM update on next-season games. This is a
calibration and prior-likelihood interaction result, not evidence that rich
features contain no signal.

The exact report-only run is
`combined_validation_interpretability_v1_0f84dac95b`. It completed in `3.212`
seconds, refit no model, used no Season 2027 data, and generated row-aligned
validation, 2026 player component ledgers, factor-skill scores, and 2021--26
player trajectories.

There is no SVM or SVR result in the active repository. The relevant historical
comparison is **SPM**: Full SPM versus the simpler Box15 statistical prior. If
"SVM" meant a different external model, it cannot be diagnosed from the pinned
evidence here.

## The offense/defense asymmetry

Offense and defense do not use different evaluation standards. The richer SPM
compared ridge, elastic net, and a small histogram gradient-boosted tree on the
same purged chronological folds, row sets, possession weights, and primary
RMSE loss. The selected learners differ because the held-out evidence differs:

| Target | Histogram GBM mean RMSE | Ridge mean RMSE | Fold result | Frozen rich-SPM learner |
|---|---:|---:|---|---|
| Offense | 0.9349 | 0.9725 | GBM won 3 of 3 | Histogram GBM |
| Defense | 0.9083 | 0.9028 | Ridge won 3 of 3 | Ridge |

Offense has a stronger nonlinear mapping from role, shooting, creation, and
load to RAPM. Defense is a noisier target, has thinner direct measurement, and
the tested tree added variance without lowering RMSE. The correct conclusion
is therefore "same standard, different validated learner," not "different
standards for offense and defense."

Box15 is a separate low-variance baseline. It uses the same 15 box features and
ridge for both sides, with side-specific alpha selected chronologically. The
RAPM update itself remains symmetric at `lambda_off = 3000` and
`lambda_def = 3000`; the learner result does not justify changing those
penalties.

## Why the simpler model wins after the RAPM update

For a prior center `mu`, the AIO solves the fixed penalized possession model

```text
beta_hat = argmin_beta ||W^(1/2)(y - X beta)||^2
           + lambda_off ||beta_off - mu_off||^2
           + lambda_def ||beta_def - mu_def||^2
           + lambda_home h^2.
```

The statistical prior and the possession likelihood are therefore one system.
A prior can fit the five-year RAPM target better in isolation and still produce
worse magnitude predictions after the current-season update.

The combined evidence shows that exact reversal:

| Evidence panel | Full SPM | Box15 | Result |
|---|---:|---:|---|
| Five-year target, net weighted RMSE, six equal seasons | 1.4875 | 1.8465 | Full SPM is much better standalone |
| Next-season AIO, equal-season MSE, 6,141 identical games | 208.2290 | 207.5483 | Box15 is better after the same update |
| Next-season AIO, square root of equal-season MSE | 14.4301 | 14.4065 | Box15 advantage is 0.0236 points/game |
| Next-season AIO, mean correlation | 0.3660 | 0.3610 | Full SPM ranks outcomes slightly better |
| Next-season AIO, mean calibration slope | 0.7551 | 0.8044 | Box15 is closer to the ideal slope of 1 |

For Full SPM AIO minus Box15 AIO, the paired whole-game MSE difference is
`+0.6806`, with a 5,000-draw 95% interval of `[-0.2604, +1.6136]`. Only 7.44%
of draws favor Full SPM. The interval crosses zero, so this is not proof of a
nonzero Box15 advantage; it is proof that Full SPM did not establish the
required improvement.

The most plausible mechanism is variance and calibration:

1. Full SPM has 127 offensive and 68 defensive inputs, shorter independent
   history, and large source-era changes. A source-era classifier previously
   separated early from late rich-feature rows with AUC `0.975`.
2. Five-year normal RAPM is itself noisy. Flexible models can learn stable
   signal and target-specific noise at the same time.
3. The one-season possession likelihood then re-observes related basketball
   outcomes. Signal that helps the standalone prior can be redundant or
   over-dispersed after the RAPM update.
4. Full SPM's slightly better correlation but worse MSE and calibration slope
   is the empirical signature of useful ranking signal with less accurate
   magnitude calibration.
5. Box15 supplies a more conservative center. The current-season RAPM update
   can add lineup evidence without first undoing as much prior amplitude.

This is a diagnosis supported by the observed metrics, not a causal proof. The
next experiment should calibrate prior mean and precision separately; it should
not add another unrestricted feature bank.

## Combined validation contract

The machine-readable design is
`research/experiments/combined_validation_interpretability_v1.yml`. One runner
produces three panels, but never averages them into one score.

| Panel | Exact evaluation unit | Aligned units | Folds | Meaning |
|---|---|---:|---:|---|
| Prior target fit | `(PLAYER_ID, Window_End)` | 5,911 per candidate | 6 seasons | Statistical-prior fit |
| Next-season prediction | `(rating_season, test_season, game_id)` | 6,141 per candidate | 5 seasons | Oracle-lineup future rating quality |
| Strict same-season reconstruction | `(season, fold, game_id)` | 466 per candidate | 5 folds | Retrospective engineering diagnostic |

Every panel rejects duplicate keys, intersects the candidate row universe
before scoring, verifies identical outcomes, and fails closed on Season 2027.
Next-season uncertainty resamples paired whole games within season and weights
seasons equally.

The strict same-season result is supportive but narrow: Box15 AIO has RMSE
`14.5157` versus `14.7605` for zero-prior RAPM, with paired MSE interval
`[-12.3020, -2.1471]`. The source filter retains only 466 of 1,079 games
(`43.2%`) and is not representative of the full season. It cannot be combined
numerically with the next-season result or used alone for production
promotion.

The next-season panel also is not a deployable pregame forecast. It uses
observed future lineups to isolate rating quality. Projected-minutes validation
remains separate.

## Exact player-level AIO accounting

The 2026 additive identity is now explicit. For player `i`, side `s`, and
feature group `g`, the standardized ridge contribution is

```text
c[i,s,g] = sum over j in g of coefficient[s,j] * standardized_feature[i,j].

centered_prior[i,s] = intercept[s] + centering_offset[s]
                      + sum_g c[i,s,g]

AIO[i,s] = centered_prior[i,s] + RAPM_update[i,s].
```

The saved centering offsets are `-0.310903` offense and `+0.071919` defense.
The generated ledger reconstructs offense, defense, and net AIO for all 582
active 2026 players with maximum absolute error `1.78e-15`.

The additive groups are:

| Component | Box15 fields | Additive to AIO? |
|---|---|---|
| Shooting/scoring | PTS, FTA, FTM, 2PA, 2PM, 3PA, 3PM per 100 | Yes |
| Creation | AST per 100 | Yes |
| Turnover | TOV per 100 | Yes |
| Rebounding | OREB and DREB per 100 | Yes |
| Disruption/fouls | STL, BLK, PF, PFD per 100 | Yes |
| RAPM update | Current-season lineup residual around the centered prior | Yes |

These numbers are exact model accounting, not causal effects. Correlated
features such as points, makes, and attempts mean that coefficient allocation
is conditional on the other included fields.

The global permutation audit answers a different question: how much the frozen
downstream prediction depends on each existing group. Mean game-MSE increases
were `7.063` for disruption/fouls, `6.076` for shooting/scoring, `3.059` for
creation/security, and `1.361` for rebounding. Permutation dependence is not a
causal importance estimate.

## True shooting, turnover, and rebounding skills

Box15 has no literal true-shooting feature, so an exact "TS points of AIO"
cannot be recovered honestly from its coefficients. The report instead keeps
two layers:

1. The exact additive AIO ledger above contains shooting/scoring, turnover, and
   rebounding contributions in points per 100 possessions.
2. A separate specialist layer reports true-shooting, turnover-avoidance, and
   opponent-offensive-rebound-prevention scores in each factor target's own
   units. These are explicitly marked `additive_to_aio = false`.

Across the five evaluated factor folds, the specialist models have mean R-squared
of `0.601` offense / `0.424` defense for true shooting, `0.645` / `0.664` for
turnovers, and `0.620` / `0.584` for rebounding. That confirms the specialist
scores are interpretable descriptions of the intended factors.

It does not confirm that adding them improves AIO. Within the matched factor
tournament, TS worsened mean RMSE by `0.0080` points/game; turnover improved it
by `0.0014`; rebounding improved it by `0.0136`; and all factors together
improved it by `0.0055`. Every result misses the frozen minimum improvement of
`0.05`, and no factor candidate passes the full promotion gate.

## Future correct AIO

The next AIO should be a dynamic current-strength model, not another static
five-year regression. A minimal bivariate state model is

```text
z[i,t] = F(delta_t, age[i,t]) z[i,t-1] + eta[i,t],
eta[i,t] ~ Normal(0, Q)

y[i,t,source] = H[source] z[i,t] + b[source] + epsilon[i,t,source],
epsilon[i,t,source] ~ Normal(0, R[i,t,source]),
```

where `z = [offense, defense]`. `Q` learns annual or weekly state volatility and
offense-defense covariance. `R` grows when possessions are low or a source is
missing. The filtered posterior supplies the historical rating; the transition
supplies the future rating and uncertainty interval.

The model should expose a point-valued bridge for the requested factors:

```text
predicted impact = TS impact + turnover impact + rebound impact
                   + other box/tracking impact + lineup residual.
```

Each factor first needs a basketball-value target in points per 100:

- TS impact: shotmaking points above an expected shot-quality baseline;
- turnover impact: expected possession value of turnovers avoided or created;
- rebound impact: expected possession value added by offensive rebounds and
  prevented by defensive rebounds.

The factor-to-points bridge must be trained only on earlier seasons and
cross-fitted before it enters the RAPM prior. The remaining RAPM signal becomes
an explicit residual component. Raw factor target units must never be summed.

Validation should follow the frozen current-strength contract:

1. Create Monday snapshots from November 1 through April 1 using only data
   available at the cutoff.
2. Predict the next 14 days with observed lineups for the oracle player-quality
   test.
3. Separately predict games with pre-tip roster status, injuries, and projected
   minutes for the operational test.
4. Score margin MSE/RMSE, win-probability log loss and Brier score, calibration,
   and exposure/source-era segments on identical games.
5. Freeze all choices before the single untouched Season 2027 confirmation.

The generated `player_aio_trajectories_2021_2026.parquet` already provides the
descriptive history needed for a trajectory view. It contains 2,235 positive-
exposure ratings for players active in 2026. Those rows are retrospective AIO,
not filtered historical current-strength estimates; the state model would
replace that ambiguity.

## Reproduction and outputs

Run from `New SPM/` with the repository environment:

```bash
PYTHONPATH=src .venv/bin/python research/run_combined_validation_interpretability.py
.venv/bin/python -m pytest tests/test_combined_validation_interpretability.py -q
```

The content-addressed artifact directory is
`artifacts/research/combined_validation_interpretability/combined_validation_interpretability_v1_0f84dac95b/`.
Its principal outputs are:

- `next_season_game_summary.parquet` and
  `next_season_paired_bootstrap.parquet`;
- `prior_target_fit_summary.parquet`;
- `strict_same_season_summary.parquet`;
- `player_aio_component_ledger_2026.parquet` and
  `player_aio_summary_2026.parquet`;
- `player_factor_skills_2026.parquet`, `factor_skill_quality.parquet`, and
  `factor_downstream_gate.parquet`;
- `player_aio_trajectories_2021_2026.parquet`;
- `row_set_audit.parquet` and `run.json`.

## Promotion status

This work completes the requested diagnosis, combined report-only validation,
and exact model accounting. It does not promote a production AIO. The richer
model did not clear the matched next-season gate, the strict same-season result
has selection-limited coverage, the operational pregame test is not yet built,
and Season 2027 remains untouched.
