# CourtSignal frontier research closeout

## Decision

Keep Box15 as the frozen research SPM prior. Keep terminal-lineup, zero-prior
`3000 / 3000 / 300` RAPM as the retrospective reference. Retain the bivariate
annual state-space model as a research challenger. Do not promote the other
challengers from this program. Season 2027 remains untouched.

## Results by planned stage

| Stage | Evidence | Decision |
| --- | --- | --- |
| Evidence harness | Corrected predictive-SPM audit; identical rows, chronological folds, hashes, calibration, spread, and paired game loss | Pass |
| Source lineage | Official defense-source transition changed observed rows but worsened mean held-out target RMSE by `0.000099` | Keep the existing frozen source |
| Target horizon | Five-year zero-prior RAPM labels scored `13.7681` downstream RMSE, ahead of one-, three-, six-year and centered variants | Use five-year labels for research SPM |
| Label precision | Inverse analytic-variance weighting worsened AIO RMSE from `14.3904` to `14.6485`; paired MSE interval `[6.06, 9.21]` | Keep square-root possession weights |
| Residual Box15 | Full residual worsened AIO. Downstream-tuned defense-only residual improved RMSE by `0.0147`, but paired MSE interval `[-1.01, 0.13]` crossed zero | Keep Box15; retain defense residual as a clue |
| Defense mechanisms | Repaired official defense rows and partial CARUSO mechanisms did not transfer decisively. Public NBA tables provide aggregate opponent shooting and closest-defender summaries, not a shot-level scorer-defender assignment or true player on/off rim deterrence | Block the missing mechanisms until a valid source exists |
| Feature controls | Registry covers 200 side-features. Gaussian noise worsened mean held-out target MSE. Target-fit gain and downstream game gain had Spearman correlation `0.071` | Pass falsification control; select downstream only |
| Shared-label control | Five different AIO candidates won five folds. Same-window label fit did not identify the downstream winner | Treat RAPM-label fit as diagnostic |
| RAPM publication | All active 2018–26 lineup graphs are connected. Ryan Davis net correlation is `0.967` annual and `0.957` on exact five-year rows. Scale slopes vary sharply by exposure | Publish points and exposure; do not globally rescale |
| Uncertainty | Two 1,000-draw whole-game pilots and nine analytic fixed-window panels exist. Analytic ridge intervals narrow mechanically at low exposure | Do not publish analytic intervals as true-impact or rank error bars |
| Dynamic model | Bivariate process correlation `+0.90` improved net MSE in selection and diagnostics; both paired intervals favored it | Research challenger; boundary estimate requires 2027 confirmation |
| Playoff portability | A heavily shrunk residual playoff deviation improved 2022 selection RMSE by `0.19`, then worsened 2023 diagnostic RMSE by `0.18`; paired MSE interval `[-27.88, 46.32]` | Null; use regular-season ratings as the playoff reference |
| Release bundle | Local API v2 bundle contains derived ratings, schemas, model cards, checksums, and a synthetic fixture. Row-set hash is `f552fc548105fe47a938ff1cd42ab748d705ed2691c1529a07d6ac8295c06ecd` | Valid local bundle; raw NBA rows excluded |

## Later expansion lanes

### Multi-target factor RAPM

The six-sided shooting, turnover, and rebounding surfaces and the conserved
one-, two-, and three-plus-point channels already exist. A learned mapper from
three factor ratings reconstructs direct RAPM closely on 2025 and 2026. This is
a mechanism diagnostic, not a better total-impact model.

With the same lineup matrix, penalty, and complete target rows, an unrestricted
residual covariance does not change the multivariate ridge point estimate. It
changes joint uncertainty. A point-estimate change requires a coupled
cross-target prior or different missingness. The current factor evidence does
not justify choosing that coupling. Do not add a nominal “shared covariance”
fit that reproduces the same coefficients under another name.

### Synergy and combinations

The additive-plus-bilinear embedding challenger lost to ridge. Exact residual
pair, trio, four-player, and lineup layers produced null results. Standalone
pair through lineup RAPM lost by `0.471`, `0.780`, `1.022`, and `1.226` RMSE on
the reused 2026 diagnostic. All paired intervals favored one-player RAPM.
Close this lane until new tracking or tactical context creates a testable
interaction mechanism.

### Roles

Roles remain descriptive usage summaries. Existing role-context models changed
next-season net RMSE by only `-0.017`; hard role experts worsened net RMSE by
`0.044`. One defense role lacked enough training support in every fold. Do not
publish role counterfactual value. Reopen the lane only with overlapping soft
roles, explicit support thresholds, and a causal intervention question.

### Event credit and win probability

Points-channel RAPM exactly recomposes direct points RAPM. The annual and
rolling WP-RAPM products conserve game outcome change to floating-point
precision. Their estimand is leverage-weighted retrospective lineup credit.
Annual WP-RAPM net stability is only `0.125`, so it is not portable player
strength. These products complete the additive credit prototypes. A named
Event Points product still needs a predeclared event-owner allocation rule and
earlier-game state values. Do not claim an ESPN Net Points reproduction.

### Injuries, contracts, draft, and roster projections

These projects need a current-strength model, future minutes, availability,
roster identity, and source-rights contracts. The measurement core has not
passed untouched confirmation. Keep these projects out of the impact-model
promotion path until Season 2027 confirms the selected current-strength model.

## External source boundary

The official NBA site exposes aggregate [opponent shooting](https://www.nba.com/stats/players/opponent-shooting)
and [closest-defender shot dashboards](https://www.nba.com/stats/players/shots-closest-defender).
Those tables can support player-season mechanism features. They do not expose
one shot row with the shooter, assigned defender, location, contest distance,
and lineup. Do not manufacture that join from game-level matchup totals.

## Reproducibility boundary

The current checkout lacks the 2024–26 silver `possessions.parquet` and
`possession_lineup_segments.parquet` files that produced several pinned
artifacts. Their source hashes and derived outputs remain recorded. The new
playoff pilot therefore uses complete cached 2019–23 possessions and says so in
its manifest. A 2024–26 refresh must restore the exact silver inputs or rebuild
them from pinned raw manifests before it claims reproduction.

## Frozen next confirmation

Do not inspect Season 2027 until the confirmation contract is locked. Score the
following once on identical games:

1. zero-prior normal RAPM;
2. Box15-centered AIO;
3. independent annual latent state;
4. bivariate annual latent state.

Use paired whole-game MSE as the decision statistic. Report margin RMSE,
correlation, calibration, exposure failures, and offense/defense subgroup
results. Do not redesign a losing model after the confirmation result.
