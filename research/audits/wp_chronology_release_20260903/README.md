# Corrected WP release

## Decision

Publish shared log-odds penalties **3000/3000** with home penalty 300 and fixed
2.5% clipping. The immutable run is `wp_chronology_release_v2_e3a3fbf4c2`.

The 2016–2021 development RMSE is 13.312710. Lambda 1000 is close at 13.324487,
but its simultaneous upper excess bound is 0.120078, above the frozen 0.05
tolerance. Lambda 3000 is the lowest setting that clears the declared rule.
This does not establish a uniquely optimal penalty.

On 6,141 matched 2022–2026 games, lambda 3000 has equal-season RMSE 14.703783.
Corrected lambda 150000 has 15.366597. The paired difference is −0.662814;
its one-sided 95% upper bound is −0.585270. Every later season improves;
the smallest improvement is 0.313396 RMSE. The publication gate passes.

| Model | Official final-margin RMSE, 2022–2026 |
| --- | ---: |
| PULSE | 14.441775 |
| Normal RAPM | 14.599081 |
| Log-odds WP, 3000/3000 | 14.703783 |
| Raw WP, one-season validation | 14.934896 |

WP remains behind PULSE and normal RAPM in this comparison. The change repairs
and publishes descriptive WP ratings; it does not replace PULSE.

## Method

The frozen contract is `research/experiments/wp_chronology_release_v2.json`.
The runner is `research/rapm_lab/run_wp_chronology_release.py`.

- Source audit: canonical terminal-lineup loaders, 1997–2026.
- WP surface: original six progress/score-proxy features, logistic C=1,
  stride 10, strictly earlier training seasons, official winner labels.
- WP targets: 2014–2026, chronological possession changes with game conservation.
- Log-odds clipping: fixed at 0.025 and 0.975.
- Shared player penalties: 100, 300, 1000, 3000, 10000, 30000, 150000.
- Home penalty: fixed at 300.
- Development outcomes: 2016–2021. Choose the lowest penalty within 0.05 RMSE
  of the best that also passes the simultaneous all-pair bootstrap bound.
- Later diagnostic outcomes: 2022–2026. The frozen choice must have an upper
  one-sided 95% RMSE difference no greater than 0.05 versus corrected 150000,
  and no single season may deteriorate by more than 0.20 RMSE.
- Bootstrap: 5,000 shared whole-game draws within each season, seed 20260903.
  RMSE is the square root of equally weighted season MSE.
- Every candidate uses the exact same next-season games and observed lineups.
  Candidate-specific affine margin calibration uses earlier outcomes only.
- Raw-WP public windows have five years; logit-WP public windows have one year.
  Both expose only endpoints 2024, 2025, and 2026.

## Source repairs

Legacy possession numbering restarted each quarter. Changes now follow the
game-global state index. Official scores also replace wrong cached winner labels.
Unplayed 0–0 schedule entries are excluded; every possession game must still
match a non-tied official outcome. One such entry exists in the source: 2013,
game `0021201214`.

The old WP-SPM, raw-WP comparison, and logit penalty results are superseded.
The old checkpoint and incomplete run outputs remain unchanged for audit.

## Interpretation

This is a descriptive research release, not a promoted production model.
Historical scoring states are cumulative offense-assigned possession-point
proxies, not exact scoreboard and clock measurements. Official winners do not
repair those intermediate features. No terminal-score correction is inserted
into earlier states.

Legacy 2014–2023 caches omit overtime possessions. Official final-result closure
therefore assigns missing overtime change to the last regulation lineup in those
games. The 2024–2026 source includes overtime. The frozen predictive comparison
does not establish complete historical possession attribution. This limitation
also affects the legacy seasons inside published five-year raw-WP windows.

The next-season outcome is official final margin, including technical free
throws. PULSE and normal RAPM predictions are frozen and rescored against that
same outcome. PULSE's existing validation table instead uses scoring margins
excluding technical free throws. These absolute RMSE values should not be mixed.

The historical outcomes have been inspected before. Bootstrap intervals are
conditional on fitted models and do not include refitting uncertainty. Observed
future lineups are available to the evaluation. It is not a roster-only forecast.
Affine calibration can absorb much of coefficient shrinkage, so equivalent
prediction error does not prove that larger player ratings are more accurate.
Season 2027 remains untouched.

## Verification companion

`verification.ipynb` reads the pinned output and reproduces the selection,
publication gate, equal-season RMSE, source hashes, and public rating windows.
It does not fit new models. Execute with the repository Python environment from
the repository root.

## Website verification and publication status

36 focused Python tests and 42 website tests pass. Website lint has no errors
(one existing image-element warning). The research-control validator and diff
whitespace checks pass. The verification notebook executed successfully.
Desktop and 390-pixel mobile checks cover the WP windows, search, stable
percentiles, readable units, and horizontally scrollable tables.

Cloudflare preview version `7e04be11-101a-40da-a169-96491eb7d34a` uploaded
successfully. Its automated public data verification was blocked by HTTP 403,
Cloudflare error 1010. Cloudflare documents 1010 as a client browser-signature
block; the specific triggering security rule has not been identified.

The user authorized publication without that preview check. At
2026-09-03 13:43:19 UTC, the same version was deployed to 100% of production
traffic. The deployment status confirmed the active version. A fresh browser
reload of `https://courtsignalnba.pages.dev` verified the corrected log-odds
board, the 3,000/3,000 note, latest three endpoints, raw-WP board, stable search
percentiles, and corrected Research benchmark. Live values matched the tested
release, including 2026 Shai log-odds net +0.973 and raw-WP net +0.164.
Security settings were not changed. The prior version
`4aa6ec99-2e57-40bf-b75c-a73be6d9c10e` remains available for rollback.
