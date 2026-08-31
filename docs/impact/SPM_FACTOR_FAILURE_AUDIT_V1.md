# SPM factor failure audit

## Result

The 15-feature BoxPIPM-style bank estimates the 2026 factor RAPMs better than
the full 127-offense and 60-defense bank on five of six heads. The full bank
only improves turnover offense. That improvement is small and its paired
player-bootstrap interval crosses zero.

This result explains the larger model search. The extra features can fit more
of the five-year RAPM target, but they do not transfer reliably through the
single-season RAPM update. The full and Box15 factor residuals correlate from
`.93` to `.98`, so both models miss nearly the same players.

## Same-model factor comparison

Both candidates use ridge regression. Both train on 2024, select the penalty
on 2025, refit on 2024 and 2025, and diagnose on reused 2026. Both use the same
387 players with at least 1,000 offensive and defensive possessions. The audit
uses six targets:

- shooting true-shooting RAPM on offense and defense;
- turnover RAPM on offense and defense;
- offensive-rebound RAPM on offense and defense.

The table reports possession-exposure-weighted results. A negative MSE delta
favors the full bank.

| 2026 factor target | Box15 R² | Full R² | Full minus Box15 RMSE | Full minus Box15 MSE, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Shooting offense | .345 | .281 | +.023 | +.0235 `[+.0063, +.0411]` |
| Shooting defense | .129 | .102 | +.008 | +.0085 `[-.0036, +.0211]` |
| Turnover offense | .349 | .360 | -.003 | -.0017 `[-.0079, +.0043]` |
| Turnover defense | .379 | .371 | +.002 | +.0015 `[-.0037, +.0068]` |
| Offensive rebound offense | .380 | .351 | +.013 | +.0139 `[-.0171, +.0443]` |
| Offensive rebound defense | .120 | .075 | +.011 | +.0099 `[+.0014, +.0190]` |

Box15 wins shooting offense and offensive-rebound defense with intervals that
exclude zero. The remaining differences are unresolved. The full bank does
not win any factor head decisively.

## Where both models fail

Shooting defense and defensive effect on offensive rebounding remain the weak
heads. Box15 explains only 12.9% and 12.0% of their 2026 variation. The full
bank explains 10.2% and 7.5%. The missing 2026 defended-shot source limits the
shooting-defense result, but it does not explain the offensive-rebound failure.

Turnover impact is the most learnable two-way factor. Box15 explains 34.9% of
offensive turnover impact and 37.9% of defensive turnover impact. The full
bank reaches 36.0% and 37.1%.

The models also miss the same extreme cases. Their factor residual correlations
range from `.932` to `.982`. More columns change estimates at the margin. They
do not solve the omitted lineup, scheme, role, and assignment information that
drives the largest factor RAPM residuals.

## The selection-to-diagnostic reversal

On 2025, the full bank had lower RMSE for four factor heads. On 2026, it lost
five. The selected full models also use the maximum tested penalty of 10,000
for five of six heads. Those two facts point to weak transferable signal and
heavy shrinkage, not a need for a larger learner.

The normal-RAPM reconstruction tells the same story. Box15 factor predictions
reconstruct 2026 net RAPM at `1.751` RMSE and `.538` correlation. Full-bank
factor predictions reach `1.764` RMSE and `.530` correlation.

## What this does and does not establish

This audit isolates feature-bank capability because it holds the learner,
targets, rows, weights, and split fixed. It does not directly decompose errors
from the five-year SPM. The available factor targets are annual. A matched
five-year factor backfill would be required for that claim.

Season 2026 is reused diagnostic evidence. Season 2027 remains untouched. The
factor targets and normal RAPM share lineup data, so reconstruction is a
descriptive mechanism check rather than independent validation.

## Decision

Keep Box15 as the AIO research prior. Stop adding broad feature families. The
next defensible feature experiment must target one known failure with new
information. Defense shot assignment and box-out responsibility are the two
clear gaps. Do not build a generic residual model from the existing full bank.

Artifacts: `spm_factor_failure_audit_v1_23a5d60bca` under
`artifacts/research/spm_factor_failure_audit/`.
