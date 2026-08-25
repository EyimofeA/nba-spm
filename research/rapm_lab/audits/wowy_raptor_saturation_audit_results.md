# WOWY, RAPTOR, and RAPM saturation audit results

Date: 2026-08-25

Final local runs:

- `wowy_raptor_reproduction_v1_4983f2cd47`
- `raptor_onoff_proxy_v1_bb23b07cc8`

The source data came from the public [DARKO WOWY](https://www.darko.app/wowy)
player histories and FiveThirtyEight's official
[modern RAPTOR data](https://github.com/fivethirtyeight/data/tree/master/nba-raptor).

## Reproduction results

| Check | Scope | Pearson | Rank | Maximum error |
|---|---:|---:|---:|---:|
| DARKO season offense from game rows | 5,497 player-seasons | 1.0000 | 1.0000 | 7.55e-15 |
| DARKO season defense from game rows | 5,497 player-seasons | 1.0000 | 1.0000 | 6.66e-15 |
| DARKO season net from game rows | 5,497 player-seasons | 1.0000 | 1.0000 | 7.55e-15 |
| Local RAPTOR player CSV versus official CSV | 4,684 non-null rows | 1.0000 | 1.0000 | 0 |
| RAPTOR-style proxy versus held-out RS team target | 1,036 seasons at 1,000+ minutes | .9658 | .9575 | n/a |

DARKO's public season average is the unweighted mean of its public Final Cut
game rows. This reproduces the displayed average, not DARKO's private daily
model or smoothing.

The RAPTOR proxy trains one shared offense and defense coefficient vector on
2014-2018, then scores 2019-2022. It uses opposition-adjusted own on-court
rating, direct courtmates' ratings without the player, and second-order
courtmate context. It keeps traded-player context within team and excludes the
focal player from second-order context. The target is the published regular-
season team-stint file aggregated with possession weights. The separate player
CSV identity check combines regular season and playoffs.

The fitted proxy coefficients are `+.5919`, `-.5964`, and `+.2431`. Their signs
match the disclosed construction, but the inputs are correlated. Do not read
the coefficients as causal or individually identified effects. FiveThirtyEight
did not publish its coefficients, exact opponent adjustment, or complete
second-order weighting, so this remains a proxy.

## External agreement with CourtSignal RAPM

| Reference | Matched scope | Net Pearson | Net rank | Interpretation |
|---|---|---:|---:|---|
| Ryan Davis normal RAPM | 5,217 annual rows | .967 | .962 | Same-key normal RAPM check |
| Ryan Davis normal RAPM | 5,869 exact three-year rows | .980 | .970 | Same-window normal RAPM check |
| Ryan Davis normal RAPM | 5,513 exact five-year rows | .957 | .948 | Same-window normal RAPM check |
| xRAPM | 687 rows, 2024-2026 | .897 | .888 | Same scope, unequal season weights |
| RAPTOR on/off player file | 2,449 rows at 1,000+ minutes | .917 | .912 | Different estimand, RS plus playoffs |
| DARKO WOWY | 5,492 rows, 2017-2026 | .574 | .500 | Different estimand and game scope |

The same-key normal RAPM checks rule out gross sign, player join, and rolling-
window errors. Correlation alone does not validate rating magnitude. CourtSignal
has a 1.391 fitted scale slope against Ryan Davis annual RAPM, so out-of-sample
scale calibration remains worth one cheap diagnostic.

## Four-model review

The same frozen audit went to Cursor Grok 4.6 slow, OpenRouter ox-alpha,
Cursor Opus 5 slow, and Cursor Kimi K3, all at maximum reasoning.

- All four voted `FREEZE NORMAL RAPM`.
- Grok caught an ambiguous claim that treated the regular-season team target as
  the same table as the RS-plus-playoffs player CSV. The docs and run manifest
  now separate them.
- Opus caught cross-team pooling for traded players and focal-player leakage in
  second-order context. Both are fixed. Held-out net correlation moved from
  `.9626` to `.9658`.
- ox-alpha and Kimi found no blocking defect. They stressed that this is a
  construction check, not independent evidence that RAPTOR or CourtSignal is
  ground truth.

## Decision

Freeze the core normal RAPM estimator. Keep the current production candidate:
terminal-lineup, zero-prior, possession-level, rolling five-year RAPM with
`3000 / 3000 / 300` penalties.

RAPM-like research is not finished, but it belongs in a separate lane. The
highest-value order is:

1. run one out-of-sample scale-calibration diagnostic on saved predictions;
2. repair annual SPM defense and evaluate the AIO prior;
3. finish practical uncertainty for published ratings;
4. build dynamic current-strength projections;
5. return later to factor, WP-credit, pair, lineup, and teammate-event views.

Do not reopen broad lambda, rubber-band, age-control, or lineup-encoding
searches without a new failure or a predeclared test that can change a decision.
