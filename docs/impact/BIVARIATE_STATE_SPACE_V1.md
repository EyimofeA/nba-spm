# Bivariate Annual State Space V1

## Result

The joint offense-defense process is a research challenger. The selected
process correlation is +0.90. It improves next-year annual normal-RAPM proxy
MSE against the independent filter in both reused evaluation blocks.

| Scope | Bivariate RMSE | Independent RMSE | Paired MSE difference | 95% interval |
|---|---:|---:|---:|---:|
| Selection, 2022--23 origins | 1.7022 | 1.7167 | -0.0497 | [-0.0972, -0.0026] |
| Diagnostic, 2024--25 origins | 1.7422 | 1.7616 | -0.0687 | [-0.1254, -0.0138] |

The artifact is `bivariate_annual_state_space_v1_c87f6bdbe8`. Season 2027 was
not loaded.

## Model

For player state

\[
x_t = \begin{bmatrix}O_t\\D_t\end{bmatrix},
\]

the transition is

\[
x_t = 0.9x_{t-1} + \eta_t,
\qquad
\eta_t \sim N\!\left(0,
0.25^2
\begin{bmatrix}
1 & \rho\\
\rho & 1
\end{bmatrix}
\right).
\]

The observation is annual 3000/3000/300 zero-prior RAPM. The observation
matrix uses analytic homoskedastic ridge covariance, including the same-player
offense-defense covariance. The process grid tested
`rho = -0.50, -0.25, 0, 0.25, 0.50, 0.75, 0.90`. The selection origins chose
`rho = 0.90`. One-year forecasts multiply the filtered state by 0.90.

## Interpretation

The analytic observation covariance has median offense-defense correlation
near zero and makes almost no difference by itself. Positive process covariance
creates the gain. Players whose latent offense improves also tend to have their
latent defense move in the same direction. This may represent health, role,
overall form, coaching context, or a shared measurement factor.

The selected correlation lies at the edge of the grid. Treat this as evidence
for a shared latent impact factor, not as a precise estimate that the true
correlation equals 0.90. The next model should parameterize common and
side-specific shocks directly.

## Boundary

The target is next-year annual RAPM, not game margin or true ability. The
observation covariance is not bias-aware. Selection and diagnostics reuse
historical seasons. The model needs Season 2027 confirmation before production.
