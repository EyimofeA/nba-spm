# Precision-Aware Prior RAPM

Status: **revised contract; blocked from scored evaluation until historical
cross-fitted prior coverage is sufficient.**

## Question

Can a statistical plus-minus (SPM) center improve a normal RAPM fit without an
arbitrary global prior-amplitude search?

## Frozen candidate

For player `j` and side `s` (offense or defense), the candidate has an SPM
center `mu_js` and a side-specific prior error variance `tau_s^2`:

```text
beta_js ~ Normal(mu_js, tau_s^2)
```

The possession likelihood is the existing unweighted points-SSE normal RAPM
objective. Its training residual scale is `sigma^2`. The player-side MAP penalty
is therefore:

```text
lambda_s = sigma^2 / tau_s^2
```

The candidate replaces the 3000 player penalty with `lambda_s`; it preserves
the terminal-lineup assignment, zero-valued home center, and fixed home penalty
of 300. Offense and defense remain jointly fitted. Net is exactly offense plus
defense.

The SPM center is cross-fitted and converted to RAPM coefficient units. It is
recentered separately for each side with the current RAPM training exposures.
This removes unidentifiable side-level location from the player coefficients.

## Precision calibration

For earlier player-windows only, let `r_i` be cross-fitted RAPM minus
cross-fitted SPM, and let `v_i` be the game-cluster RAPM label-variance proxy.
Estimate each side using:

```text
r_i ~ Normal(mean, tau_s^2 + v_i)
```

with a heteroskedastic profile likelihood. This is better than subtracting one
pooled `v` from a pooled residual variance. It is still an empirical-Bayes
approximation: ridge bias, lineup connectivity, and imperfect sandwich variance
mean that `v_i` is a variance proxy, not a literal measurement-error truth.

A zero-boundary or failed `tau_s^2` invalidates the candidate. There is no
fallback amplitude, clipped large penalty, or scored-season tuning.

## Evaluation contract

Compare exactly four models on identical held-out games:

1. zero-prior normal RAPM;
2. statistical prior only;
3. fixed-center prior RAPM;
4. side-specific precision-aware prior RAPM.

Primary metric: equal-season game-margin RMSE. Report margin correlation, low-
and high-exposure diagnostics, and a paired whole-game interval. The candidate
must improve RMSE by at least 0.05 points per game, retain correlation within
0.01, and pass the frozen 2027 confirmation before promotion.

## Current block

The frozen three-season feature panel starts at window end 2016. Its purge rule
requires labels ending no later than `T - 3`, so the first valid cross-fitted
SPM prior is 2019. A scored RAPM season `Y` uses a prior ending `Y - 1`; three
strictly earlier calibration windows therefore first exist for `Y = 2023`.

This makes the planned 2018--21 selection schedule impossible under the frozen
three-season feature contract. It is a data-horizon fact, not an implementation
bug. The previous 2021--24 result is invalid for promotion and is not a basis
for retuning. Do not weaken the calibration rule just to score earlier seasons.
Either build a separately specified pre-2016 prior feature contract, or defer
this challenger until a new untouched confirmation schedule is available.
