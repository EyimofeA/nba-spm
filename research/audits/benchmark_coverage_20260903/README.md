# Benchmark coverage and temporal isolation

Audit date: 2026-09-03. Years below identify the season's ending year.
The executed companion is [verification.ipynb](verification.ipynb).
[source_coverage.json](source_coverage.json) records input files and SHA-256 hashes.

## Available external sources

| Model | Local rating years | Potential next-season overlap with PULSE validation |
| --- | --- | --- |
| EPM | 2002–2026 | 2015–2026 outcomes |
| LEBRON | 2010–2026 | 2015–2026 outcomes |
| DARKO DPM | 1997–2026, updated CSV | 2015–2026 outcomes |
| PIPM | 1974–2020, plus partial 2021 | 2015–2021 outcomes; 2022 only as a separately labeled partial-source test |
| RAPTOR, modern | 2014–2023 | 2015–2024 outcomes |
| MAMBA | 2015–2024 usable panels | 2016–2025 outcomes |
| BPM 2.0 | 2014–2026, newly collected | 2015–2026 outcomes |
| xRAPM | 1997–2026, newly collected | 2015–2026 outcomes |
| CourtSignal PIPM reconstruction | 1997–2026 | 2015–2026 outcomes |
| CourtSignal RAPTOR reconstruction | 2014–2026 | 2015–2026 outcomes, pooled historical fitting |

RAPTOR's historical file spans 1977–2022. Its pre-2014 reconstruction is a different
information regime from modern RAPTOR. The separate latest file supplies 2023.
These are source ratings, not CourtSignal reconstructions.

PIPM 2021 has 474 source rows and a maximum of 22 games played. It is not a
complete-season panel. MAMBA has one row per label from 2025 onward, including
impossible future labels; 2025 and 2026 fail same-season identity matching.
Do not repair those labels by guessing. The newly supplied **Full DPM History.csv**
contains 15,054 unique player-season rows for 1997–2026, with a per-player date.
It supersedes the older workbook that ended in 2024. The two separate current
DARKO leaderboard CSVs lack historical season fields and are not the source used here.

Availability does not certify an external model's historical training cutoff
or exact point-in-time snapshot. The requested main comparison uses CourtSignal
reconstructions instead of original PIPM/RAPTOR and excludes MAMBA's shorter panel.

### Updated requested comparison

The main panel uses PULSE, EPM, LEBRON, xRAPM, DARKO DPM, BPM 2.0,
CourtSignal PIPM reconstruction and CourtSignal RAPTOR reconstruction. Normal
RAPM remains an internal control. Keep MAMBA on a separately labeled shorter
panel: its usable data ends in **2023–24**, so its last next-season test is 2024–25.
The frozen main window is ratings 2015–2025, tested on 2016–2026 outcomes.
See [benchmark_plan.json](benchmark_plan.json) for the scoring contract.

EPM, LEBRON and the updated DARKO source all extend through 2026. After an earlier
fetch failure, ordinary direct requests to the user-supplied URLs succeeded.
All 43 annual pages are saved and hashed: 30 xRAPM pages and 13 BPM pages.
The xRAPM parser handles the new Team column and reverses its defensive sign.
It explicitly excludes the two ambiguous Marcus Williams rows in 2008.
The 2025 xRAPM page includes 771 rows; only 568 match same-season identities.
No unmatched historical names are guessed into the scored panel.

The existing RAPTOR reconstruction trains its box mapping on all 2014–2022
source years and its on/off map on 2014–2018. The user explicitly requested the
saved RAPTOR and PIPM reconstructions as-is. No fold-specific reconstruction was
fit. RAPTOR's early outcomes overlap its mapping training years, so the combined
table cannot claim clean out-of-time performance for every comparator.

PIPM uses fixed published coefficients and season-local normalization. Duplicate
selected source rows affected centering in 2014, 2015, 2018 and 2023. The input
repair removes identical selected rows before centering and rejects ambiguous
remaining player IDs. A deterministic rebuild with unchanged coefficients
created `pipm_reconstruction_v1_e0625de5fe`; source and builder hashes now enter
its identity. The old run remains intact. No model tuning or site export occurred.

## PULSE audit

Pinned source: `pulse_canonical_v1_cd3c14750a`.

| Check | Finding |
| --- | --- |
| Prior-label cutoff | All 12 folds have training end equal to rating year minus one. |
| Target window | Nine-year trailing RAPM labels end before the rating year. |
| Lineup likelihood | Only the rating year's stints enter its update. |
| Test games | Ratings 2014–2025 score outcomes 2015–2026. |
| Matched internal games | 14,439 identical games each for prior, PULSE, and RAPM. No duplicate keys or differing actual margins. |
| Descriptive display rows | Separate full-history mapping trained through 2026; prohibited as historical held-out predictions. |
| Calibration | Reported slope is a diagnostic, not a correction fitted on test outcomes. |
| Selection independence | Not established. All historical outcomes are development or selection exposed under `research/season_exposure.yml`. |
| Provenance | Config, features, targets and stint manifest match the recorded input hashes. The builder source hash differs from the recorded snapshot. |

The fixed PULSE prior penalties are 300 offense and 1000 defense. They are not
nested selections inside every validation fold. The referenced learner screen
supports the raw ridge family, not exact estimator equivalence. Horizon and
RAPM-penalty choices also used these historical outcomes. Coefficient chronology
therefore does not imply an untouched model-selection test.

## Why the removed table cannot simply be restored

The website catalog still contains old external run
`external_all_in_one_benchmark_v2_eac56750f7`; the fuller local report references
`external_all_in_one_benchmark_v2_6a898e99d9`. Both compare older Box15, rich-SPM
and defense-residual fits, not the current canonical PULSE release.

The external runner's strict intersection filters **prior rows**, then estimates
the internal RAPM update on the full player matrix. Internal AIO can consequently
score nonzero coefficients for players absent from the external metrics. Its
games are shared, but its final player support is not identical across arms.

Before restoring a current-PULSE table, rescore its fold-specific coefficients
and external ratings on identical games, with identical final player support and
an explicit missing-player rule. Keep the full-season common panel separate from
broader pairwise comparisons. Do not relabel old Box15+RAPM as current PULSE.

## Corrected common-support result

Run: `pulse_external_common_v1_b1faa64e9b`. Every model scores the same 13,209
official games from 2016–2026. The same finite source-year player IDs enter all
arms. Source-year side-possession weights center each component. Excluded
players receive zero after fitting, and all arms share the source-year RAPM
home coefficient and intercept. No test-season calibration is applied.

| Model | Equal-season RMSE |
| --- | ---: |
| xRAPM | 13.6791 |
| EPM | 13.6836 |
| DARKO DPM | 13.7056 |
| LEBRON | 13.7391 |
| PULSE | 13.7545 |
| BPM 2.0 | 13.8497 |
| RAPM | 13.8644 |
| CourtSignal RAPTOR reconstruction | 13.8695 |
| CourtSignal PIPM reconstruction | 14.0529 |

PULSE minus RAPM is -0.1098 RMSE, with paired 95% interval [-0.1347, -0.0862].
PULSE minus LEBRON is +0.0155, with interval [-0.0267, +0.0556]. Intervals use
5,000 shared whole-game draws within each fixed season, with equal season weights.
They do not account for historical model-selection exposure or future-era drift.

Matched player ratings cover 86.2–90.6% of target lineup exposure by season.
This is intentionally a common-support diagnostic, not the original full-support
PULSE score. The original internal table also uses a different point-margin target.
MAMBA's separate 2016–2025 comparison has 11,979 games. MAMBA scores 13.4697;
PULSE scores 13.5937 on that shorter common panel.

All 11 PULSE source-year fits reproduced saved predictions with maximum absolute
difference zero before masking and centering. Independent review verified the
coefficient signs, zeroed excluded players, source-season centering and exact
official-game key equality in every evaluated season. The public export verifier
checks hashes, shared games, masks, finite predictions and game-derived RMSE.

## Changes and verification

`load_pulse_validation` now guards the internal validation-release builder and
web snapshot exporter. It requires the explicit past-only contract, canonical
artifact names and candidate labels, complete declared folds, valid prior keys,
strict training cutoffs, next-season outcomes, finite predictions, identical
games, and game-derived agreement for MSE, correlation and calibration slope.
It rejects duplicate rows instead of silently discarding them. The web exporter
no longer falls back to a different model's summary.

The checked summary is numerically identical to the existing public PULSE table.
PIPM was recalculated with unchanged coefficients after the input repair before
the as-is request. PULSE's frozen lineup fits were replayed without retuning.
The Research page now reads a pinned summary-only external benchmark payload.
No ratings were replaced and no deployment occurred. These checks cannot undo
historical model-selection exposure or certify third-party training chronology.
