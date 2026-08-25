# Points-channel RAPM

Updated 2026-08-25. This is a local research model. It does not replace public
RAPM.

## Why this comes before a multinomial model

Possession points are discrete, but a true multinomial model is not required to
learn an additive first decomposition. Let possession points be `y`. Define:

```text
y_one        = 1 * I(y = 1)
y_two        = 2 * I(y = 2)
y_three_plus = y * I(y >= 3)
```

For every possession:

```text
y = y_one + y_two + y_three_plus
```

Fit every target against the same offensive lineup, defensive lineup, and home
design, with the same ridge penalty. Ridge is linear in the target:

```text
beta_k = (X'X + P)^-1 X'(y_k - mean(y_k))
```

Therefore:

```text
beta_points = beta_one + beta_two + beta_three_plus
```

The same identity holds after exposure-weight centering because centering is a
linear transformation. One factorization of `X'X + P` solves all target columns.

## What the output means

Each channel reports offense, defense, and net points per 100 possessions.
Positive defense means preventing points in that channel.

The channel name describes the possession's final point total. It does not say
which player took the shot, drew the foul, passed the ball, or caused the miss.
For example, the three-plus channel includes made threes, three-shot fouls, and
rare possessions worth four or more points.

## Pilot result

Run `points_channel_rapm_v1_4507aab97c` uses the 2022--26 regular seasons:

| Check | Result |
| --- | ---: |
| Possessions | 1,229,744 |
| Games | 6,141 |
| Players | 1,029 |
| Target recomposition error | 0 |
| Rating recomposition error | 0 |
| Maximum channel offense + defense - net error | `8.88e-16` |
| Maximum difference from canonical RAPM | `1.32e-7` points per 100 |

Among players with at least 5,000 possessions on each side, Nikola Jokic leads
the total and two-point channels, Giannis Antetokounmpo leads the one-point
channel, and Sam Hauser leads the three-plus channel.

## Why this is not probability RAPM

A multinomial model estimates `P(y=k | X)`. Expected points are:

```text
E[y | X] = P(y=1 | X) + 2 P(y=2 | X) + sum(k P(y=k | X), k >= 3)
```

Adding `P(1) + P(2) + P(3+)` without point weights is not expected points.
Also, separate binary models do not automatically produce probabilities that
sum to one. A softmax model fixes that but costs more and its player
coefficients do not add back to linear RAPM.

The 2024 multinomial paper is useful as a model idea, but its published model
selection relies on All-NBA overlap, starter prevalence, minutes, and box-score
plausibility. Those are face-validity checks. CourtSignal would require
chronological probability calibration and held-out game-value tests before
promoting a multinomial model.

Paper: <https://arxiv.org/abs/2406.09895>

## Event-factor extension

Turnovers, shooting fouls, shot attempts, makes, misses, and offensive rebounds
are not mutually exclusive possession outcomes. Fitting their raw indicators
and adding coefficients will not recover points.

The next version needs a conserved value ledger. Each possession receives
component targets whose sum equals observed points or a predeclared expected
possession value. A useful first ledger is:

```text
turnover value
free-throw value
two-point shot value
three-point shot value
offensive-rebound continuation value
```

Miss and rebound values require state-transition estimates. Train those values
on earlier games, then freeze them before fitting player effects. Only after
this ledger recomposes exactly should we split rim, midrange, transition, shot
quality, and shotmaking.
