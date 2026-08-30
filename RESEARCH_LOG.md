# Research Log — NBA RAPM (New SPM project)

Append-only. Newest entries at the bottom.

## 2026-07-03 — Fixed the RAPM validation harness (it was wrong in 3 ways)

**Question:** Why did the ablation scorecard show ~0.0002 RMSE differences between specs — is the validation broken?

**What we did:**
- Audited `evaluate_rapm_models.py` + `standard_rapm.py`.
- Fix 1: garbage-time is now a per-row flag, all specs scored on the same non-garbage validation possessions (previously `no_gt` was graded on its own easier row set).
- Fix 2: lambda tuning now only sees the training side of each split (previously tuned on the full matrix including the held-out season — leakage).
- Fix 3: added game-margin metrics (RMSE + corr of predicted vs actual home-minus-away points per game); summary now ranks on margin RMSE.
- Added per-season parquet possession cache (`data/possession_cache/`): DB fetch of 2M rows went 50s → 4.7s.

**Result:** (2022–24 window, zero prior, default lambdas 3000/3000/300/100)
- Possession RMSE ~1.19 for every spec — confirmed useless as a discriminator [SOURCED: eval run final_v2].
- Game-margin metric separates specs by whole points: player+home wins (margin RMSE 13.9 CV / 14.6 chrono, corr ~0.63). Home effect worth ~0.05 pts over player-only.
- Rubberband term costs ~1.6–2.0 pts of margin RMSE and drops corr to ~0.38–0.44. Caveat: rubberband uses live score margin as a feature, so the margin metric is stacked against it — its real test is rating quality, not margin prediction.
- Season effects: no gain on a 3-year window.
- Training with garbage time included is slightly worse once scored fairly.

**Dead ends:** Possession-level RMSE as a model-selection metric — a 0/2/3-point outcome has so much irreducible noise that any sane spec ties. Do not use it to pick specs again.

**Lesson:** Validate at the level of the thing you ship. We ship player ratings and game-level impact, so the metric must aggregate to at least game margins (better: next-season retrodiction from frozen ratings). Also: any cross-model comparison is invalid unless the validation rows are identical.

**Promote?:** "Score all model variants on a common validation set" belongs in a general validation skill — second time leakage/comparability bugs have bitten this project (first: circular SPM prior).

## 2026-07-03 — Rating-level validation picks the spec; nonlinear probe clears ridge

**Question:** Which spec produces the best RATINGS (not possession fit)? Does a nonlinear model beat ridge? Is a model-based rubberband salvageable?

**What we did:**
- `validate_ratings.py`: split-half reliability (game-split refits, r across halves) + next-season retrodiction (freeze 2021–23 betas, predict all 1,201 games of 2024).
- `nonlinear_probe.py`: LightGBM on frozen lineup ratings ± live score margin, same retrodiction test.

**Result:** [SOURCED: rating_validation_ratings_v1, nonlinear_probe CSVs]
- player+home: split-half r=0.549, next-season margin corr=0.632, RMSE 14.49 (naive baseline 17.89). WINNER.
- +rubberband: split-half 0.565 but corr collapses to 0.419. Rejected.
- LightGBM with live margin: corr 0.424 — same collapse as linear rubberband. The live margin FEATURE is the problem (good teams lead more, so any model routes quality through it), not the linear functional form.
- LightGBM without margin: corr 0.613 — no nonlinear structure left over linear lineup ratings.
- Lambda sweep aborted early: λ∈[1000,8000] barely moves retrodiction. Fixed λ=3000.
- Caveat to check: probe reported def_sum feature importance = 0; verify feature construction.

**Dead ends:** (1) Rubberband/coasting adjustments via any margin-derived feature — endogeneity, not crudeness. A real fix needs an exogenous effort model, not the margin itself. (2) Tree models at the possession level on rating-derived features.

**Lesson:** When a control variable is caused by the thing you're measuring (leading ⟵ team quality), "controlling" for it removes signal, not bias. Endogeneity beats functional-form sophistication every time.

**Promote?:** Next-season retrodiction as the standard model gate for any player-value model (soccer RAPM too).

## 2026-07-03 — Overnight experiment queue: 21 variants, decisive results

**Question:** Which of the brainstormed RAPM knobs actually improve next-season retrodiction (train 2021-23, test all 1,201 games of 2024, baseline player+home ridge corr 0.6317 / RMSE 14.49)?

**What we did:** `experiments.py` — crash-safe sequential queue in tmux, one CSV row per experiment, Gobert/Jokic sign anchors, ESS logging, walk-forward priors only. `combos.py` follow-up merging winners.

**Result:** [SOURCED: outputs/diagnostics/experiments.csv]
- WON: recency decay hl=365d → corr 0.6535 (biggest single gain; ESS healthy at 493K/620K rows). hl=730 → 0.6447.
- WON: asymmetric lambdas, shrinking DEFENSE harder (off 2000 / def 4500 → 0.6347; off 1500 / def 6000 → 0.6353). The reverse direction loses (0.6216). Defense coefficients are noisier and want more shrinkage — now measured, not assumed.
- LOST: previous-window prior (2018-20 → 0.589–0.613, all below baseline). Old ratings are a *biased* shrinkage target — aging + role churn — not free information.
- LOST: infinite RAPM — walk-forward chain 2012→2015→2018→2021 scored 0.5933; degradation compounds with depth. (User's "infinite RAPM" question: answered, it degrades.)
- LOST: soft garbage-time weighting (0.6316 — dead even with the hard drop); replacement pooling at 250/500/1000 poss (~baseline); 5-year window (0.6095).
- REJECTED BY ANCHOR TEST: 1-year window (corr 0.647 but flipped a sign anchor and worst RMSE 14.64 — noisy ratings pointing the right way on average).
- INCONCLUSIVE: elastic net via SGD (0.16–0.32) — the optimizer underfit, not a verdict on sparsity.

**Dead ends:** Stale-prior lane (both variants). Do not revisit priors built from old windows; if priors return, they must be same-window information (i.e., an SPM) — upgraded that from "blocked" to "next-project candidate" in IDEAS.md.

**Lesson:** The two things that worked both amount to "trust recent, noisy-in-the-right-way data more, and shrink the noisiest block hardest." The things that failed all amount to "import information from a different distribution than the one you're predicting." Recency beats history; same-window beats stale.

**Promote?:** The tmux + crash-safe CSV + sentinel-wake pattern worked flawlessly for unattended overnight research; worth making a skill.

## 2026-07-03 — FINAL: production RAPM shipped, 26 windows, 1997–2024

**Question:** Ship the finalized regular-season RAPM with the night's winning config.

**What we did:** `final_windows.py` in tmux — 26 rolling 3-year windows (end seasons 1999–2024), config: player+home, garbage-time filtered, 365-day recency half-life, symmetric lambda=3000, zero prior, SEs/CIs + A/B/C confidence tiers. ~57 minutes total. Output: `rapm/outputs/rapm_results/final_20260703/` (per-window CSVs + `rapm_all_windows.csv`, 16,716 player-window rows + `run_meta.json`).

**Result:** Sanity checks pass everywhere [SOURCED: rapm_all_windows.csv]:
- 2022–24 top-15: Jokic clear #1 (+7.78), then George, D.White, Draymond, Embiid, Kawhi, Giannis, LeBron, Tatum — consistent with xRAPM-class lists. Low-minute Jonathan Isaac correctly flagged Tier B.
- Window ending 2010: LeBron #1 (+6.92), Wade, Dwight, Varejao (the classic RAPM darling), Nash — textbook.
- Window ending 2016: LeBron, Draymond, CP3, Curry, Kawhi.
- Combo round footnote: decay + asymmetric lambdas do NOT stack (0.6532 vs 0.6535 decay-alone); hl=250d raises corr (0.6596) but worsens RMSE — drifting toward the anchor-failing 1yr-window regime. hl=365 is the stable optimum.

**Dead ends:** none new; see previous entry.

**Lesson:** A finished model is a config + a gate + a ledger, not a bigger model. The whole night's improvement (corr 0.632 → 0.654) came from ONE knob (recency decay) surviving a fair gauntlet that killed eight fancier ideas.

**Promote?:** Next projects, in order of expected value: (1) clean same-window SPM prior (the one prior type the night did NOT kill), (2) win-probability RAPM target, (3) trio RAPM on a small window.

## 2026-07-03 — Decay function learned model-free; aging curve built; new champion hl=250

**Question:** What is the true decay shape (user predicted "between exponential and power law")? What does the aging curve look like from our own panel?

**What we did:**
- D0 `d0_bucket_decay.py`: 6 free bucket weights tuned on retrodiction (55 Nelder-Mead evals).
- D1+D2 `d2_decay_folds.py`: curve-fit families to the buckets, then confirmed all variants on TWO folds (2021-23→2024 AND 2020-22→2023).
- A1+A2 `aging_curve.py`: delta-method aging curve from the 1-yr RAPM panel (13,396 player-seasons, ages already joined), survivorship sensitivity, off/def split, level-dependence regression, year-ahead player gate.

**Result:** [SOURCED: decay_fold_confirmation.csv, aging_curve_delta.csv/png/meta]
- Learned buckets are NON-monotone [1.0, 1.18, 0.26, 0.07, 0.33, 0.49] — wiggle is single-fold overfit; on two folds the simple exponential beats them.
- Power-law fit DEGENERATED to the exponential limit (tau→huge with alpha/tau fixed): at 3-year horizons exp vs power law is empirically indistinguishable. User's fat-tail intuition untestable at this window length.
- NEW CHAMPION: exponential decay hl=250 days — wins BOTH folds (0.6596/0.5939, mean 0.627 vs hl365's 0.619, no-decay 0.592). Anchors pass (from overnight run).
- Aging curve: peak 25-27; ~-4.5 at 19; ~-4 by 36. AGING IS AN OFFENSE PHENOMENON — defense is nearly flat across career. Survivorship penalty steepens old-age decline (truth between curves). Level dependence b=-0.40 (higher-rated decline more; partly measurement-error mean reversion — needs SE-aware model to separate).
- Aging-translation gate: year-ahead player MAE improves 1.974→1.920; corr ticks down 0.522→0.511 (calibration gain, ordering wash).

**Dead ends:** Free-form bucket weights as a production choice (overfit single folds; use them only to license a parametric form). Power law at 3-yr windows (unidentifiable vs exponential).

**Lesson:** Fit free, read the shape, ship the simple form that survives multiple folds. And a non-monotone learned weight pattern is a leakage/overfit alarm, not a discovery.

**Promote?:** Two-fold confirmation should be the standard gate for any weighting/hyperparameter choice (single-fold wins are hypotheses, not results).

## 2026-07-03 — Aged prior: mechanism validated, strategy dominated

**Question:** Does translating the 2018-20 prior through the aging curve (user's idea) fix the stale-prior failure?

**What we did:** `s_aged_prior.py` — off/def coefficients of the 2018-20 fit translated by f_off/f_def over the 3-year gap (ages from the career panel), used as shrinkage target for 2021-23 with hl250 decay, strengths {1,2,4}, vs raw prior and vs zero-prior champion.

**Result:** [SOURCED: aged_prior_results.csv] Aged beats raw at every strength (0.6403 vs 0.6224 at best) — the staleness diagnosis and the aging cure are both CORRECT. But all prior variants remain below zero-prior + hl250 decay (0.6596). Recency weighting already extracts old information more cheaply than shrinking toward it.

**Dead ends:** Global-strength stale priors, even age-corrected, on top of a well-decayed likelihood.

**Lesson:** Two mechanisms can both be real and still not stack: decay and priors are substitute routes to the same old information, and the cheaper unbiased route (decay) wins. Test combinations, never assume additivity.

**Promote?:** Per-player adaptive prior strength (prior only where current data is thin) — the one untested corner of the prior lane, and the theoretically correct one (that's literally what a posterior does).

## 2026-07-03 — Production rerun with new champion config

final_v4 launched: 26 rolling 3-year windows, hl=250d decay (two-fold champion), otherwise identical to final_v3 (player+home, GT filter, lambda 3000, zero prior, SEs + tiers). Output: `outputs/rapm_results/final_20260703_hl250/`.

COMPLETED same day: all 26 windows green, 16,716 rows. Top-10 sanity passes (Jokic +6.97 #1; SEs slightly tighter than hl365 panel). README updated; hl250 panel is the production artifact.

## 2026-07-03 — Minutes-only prior: largest single win of the project; stale-prior verdict overturned

**Question:** Does ANY same-window prior help (machinery test with q=1 feature), under clean semantics?

**What we did:** `spm_minutes_prior.py` — pass-1 champion fit gives labels; OOF quadratic regression of player coefficients on log(possessions) gives prior center beta_0 and tau^2 (OOF residual variance); pass-2 refit with champion penalty UNCHANGED plus separate per-player pull c*sigma^2/tau^2 toward beta_0. Both folds. Stale 2018-20 prior re-baselined under same semantics as control.

**Result:** [SOURCED: experiments.csv mprior_* rows]
- Fold 2024: champion 0.6596 -> c=2 gives 0.7335 (+0.074). Fold 2023: champion 0.5939 -> c=2 gives 0.6953 (+0.101). Anchors pass everywhere. Largest improvement in project history (decay was +0.03).
- Prior quality is modest (OOF R^2 ~0.16, tau ~1.07/100) but applied where it matters: fringe players. Implied lam0 ~12,000 possessions of evidence.
- RMSE trade: corr rises monotonically in c but fold-2024 RMSE creeps up (14.55->14.86 at c=2); fold-2023 RMSE flat. Ordering improves, scale slightly inflates — calibration check pending. Extended grid c={4,8,16} running.
- CONTROL: stale 2018-20 prior under CLEAN semantics now BEATS champion (0.6644 vs 0.6596) — the overnight "stale priors lose" verdict was substantially a plumbing artifact (confounded lambda scaling + global strength). User's suspicion ("older experiments were done poorly") confirmed and consequential.

**Dead ends:** none new; overnight prior magnitudes formally retracted (directional aged>raw comparison stands).

**Lesson:** When an idea with sound theory loses, audit the plumbing before accepting the result. The difference between "prior loses by 0.05" and "prior wins by 0.07" was parameterization, not information.

**Promote?:** Clean prior semantics (separate per-player pull, tau^2-derived strength) is now THE pattern for all future prior work — soccer included.

## 2026-07-03 — SPM v1: a leakage class discovered; v1.1/v1.2 chase the minutes prior

**Question:** Does a 56-feature box+tracking SPM beat the minutes-only prior?

**What we did:** `build_spm_features.py` (audit: 100% coverage all 26 windows, possession-weighted; tracking tier valid from 2001, not 2014 as assumed). `spm_v1.py`: OOF weighted ridge per side → prior center + tau² → pass-2 refit, both folds.

**Result:** [SOURCED: spmv1/spmv11 logs, experiments.csv]
- v1 PARADOX: OOF R² 0.54/0.45 (3x the minutes prior) but gate WORSE than minutes (0.665/0.606 vs 0.734/0.695) and degrading in c.
- DIAGNOSIS: `OnOffRtg`/`OnDefRtg` are computed from the same possessions as the labels — the feature itself carries the label's noise. OOF-by-player cannot block this (leakage inside the feature, not the split). R² was inflated, tau² underestimated, prior over-trusted.
- v1.1 (features dropped): R² fell to 0.39 (confirming the diagnosis — the gap WAS shared noise) and the gate ROSE to 0.699/0.668. Still below minutes.
- v1.1 exposed: defensive SPM OOF R² ≈ 0.01-0.04 (box stats don't see defense) while sharing one tau² with offense — over-pulls defense; and features lacked the log-poss shape that made the minutes prior work. v1.2 (per-side tau², log-poss features) launched; verdict pending in `spmv12_run.log`.

**Dead ends:** Any feature computed from the same possessions as the label (on/off ratings, team ratings, plus-minus family). New banned class alongside rubberband.

**Lesson:** Leakage hides in features, not just in splits. "Better R², worse product" is the signature — R² measures fit to labels including their noise; the gate measures signal only. When a diagnostic improves while the product degrades, suspect contamination.

**Promote?:** Feature provenance check before any supervised fit: "could this feature share noise with the label?" — belongs in a validation skill.

## 2026-07-03 — Feature Foundry Phase 0 shipped; harness reproduces minutes; SPM v2 pooled still loses

**Question:** Can we build a Karpathy-style foundry with a fixed RAPM×SPM splice, team L2 report, and fair comparison vs the minutes prior?

**What we did:**
- `feature_eval.py`: two-pass splice (pass-1 hl250 → prior → pass-2), L1 game-margin gate, L2 team net/wins corr, folds f24/f23/vault.
- Baseline repro through new harness: minutes prior c=2 → **0.7346 / 0.6954** (matches prior runs 0.7335/0.6953).
- `spm_v2.py`: pooled ~13.7K player-windows (time leave-out), CV alpha (picked 1000), heteroskedastic τ²(log poss), residual SPM lane.
- Infra: `rapm/features/` (program.md, prepare.py, data_allowlist.yaml, results.tsv), `verify_keep.py`, `data_ingest.py`, `data_curator.py`, `feature_submit.py`, `feature_foundry.py`, `fig_foundry_progress.py`.
- Foundry gen 0 launched overnight (minutes baseline all folds).

**Result:** [SOURCED: feature_eval_baseline.log, spmv2_run.log, experiments.csv]
- Harness validated — do not trust foundry until this passed; it passed.
- SPM v2 pooled best: f24 **0.7304** (c=4), f23 **0.6983** (c=4) — still below minutes c=2 on both folds.
- Residual SPM: f24 0.7239, f23 0.6874 — no lift from box beyond minutes at prior granularity.
- Pooled OOF R² off/def: ~0.01/-0.09 (f24 pooled) — estimation fix did not unlock box signal on offense in this spec; defense still ~0.
- Team L2 (report-only): minutes c=2 f24 team_wins_corr **0.653**; pooled SPM c=2 f24 **0.652** (similar).

**Dead ends:** None new — confirms minutes prior remains interim product until foundry discovers something that clears f24+f23+vault with anchors.

## 2026-07-03 — Autoresearch loop: build.py candidates (not column subsets)

**Question:** Are we actually building new features for autoresearch, or re-filtering existing box parquet?

**What we did:**
- Honest audit: gen 2–5 and SPM v2 soup were **column ablations only** — not new research. Minutes re-runs wasted ~30+ min.
- Added `candidates/gen_NNN/build.py` pattern + `candidate_build.py` + `type: build` in foundry.
- Gen 006: 11 NEW derived features (ratios, z-scores). Gen 007: playtype PPP (sparse). Gen 008–009: tracking interactions + blend.
- `autoresearch_loop.py` + `autoresearch_proposer.py` + `scripts/run_autoresearch.sh` — pending queue → evaluate → propose → repeat.
- Stale lock cleanup in `run_lock.py` (dead pids were blocking all runs).

**Result:** Pipeline complete for v1 autoresearch. Overnight queue: gens 6–9 then proposer templates (residual SPM, etc.). Gate truth unchanged: beat frozen minutes 0.7335/0.6953 on f24+f23.

**Promote?:** Next increment is LLM proposer writing `build.py` — harness + loop are ready.

**Lesson:** The harness is the product; features are plugins. Log everything (`results.tsv` + `experiments.csv` + registry).

**Promote?:** `feature_eval.run_splice()` as the only gate entrypoint — ban ad-hoc prior scripts for comparisons.

## 2026-08-08 — Independent reset; Net Points and RL/Shapley scoped

**Question:** What should the NBA project do next after treating the existing RAPM,
SPM, validation conclusions, and frontend as stale or unverified? How should event
credit, playoffs, roles, cloud ingestion, and the 2026 Sloan RL/Shapley paper fit?

**What we did:** Read the principal's RAPM research note; independently audited the
available data and validation paths; reviewed ESPN's published Net Points explainer,
Dean Oliver's public credit-allocation discussion, the complete Sloan
DRL/temporal-difference/Shapley paper, and current Cloudflare storage/compute
pricing. Created `docs/planning/RESEARCH_BACKLOG.md` as the dependency-aware planning page.

**Result:**
- Net Points is a descriptive event-accounting product, distinct from RAPM. Our
  version must conserve team value and expose residual/unobserved credit.
- The Sloan paper is replication-worthy, not a benchmark to accept on authority.
  Its TD, replacement-coalition, neural-Shapley, synergy, and forecast claims need
  separate ablations and time-safe revalidation.
- Playoff work should first distinguish rotation tightening from per-possession
  translation and use cutoff-time projected lineups for deployable tests.
- Offense/defense components remain useful, but promotion is gated primarily on
  recombined out-of-time team/game prediction and component calibration.
- R2 is inexpensive object storage, not compute or a faster home upload. Cloud-side
  ingestion can bypass the slow residential connection; local overnight ingestion
  remains a fallback for access-sensitive sources.

**Dead ends / cautions:** Do not label passive value-function fitting as a solved
causal credit problem; Shapley allocates a model's prediction relative to a chosen
baseline. Do not make injuries, contracts, draft, or RL the current sprint before
the canonical event and identity layer exists.

**Promote?:** P0 in `docs/planning/RESEARCH_BACKLOG.md`: canonical data repair, independent simple
RAPM, new chronological folds, model registry, and resumable ingest.

## 2026-08-08 — P0 implementation: immutable ingest, quality gates, independent RAPM

**Question:** Can we begin with the data already on disk, patch the critical gaps,
and get an independent model/data spine running without inheriting the earlier
implementation's conclusions?

**What we did:** Built the clean `nba_impact` package, resumable manifest downloader,
SHA-256 sidecars, source-aware event and possession audits, whole-game structural
quarantine, DuckDB registry, independent zero-prior offense/defense RAPM, immutable
artifacts, and a one-fold regularization diagnostic. Downloaded an Apache-2.0
bootstrap of NBA Stats V3 events, PBPStats possessions, shot detail, and matchup
detail. Added tests and an operator guide in `docs/planning/NBA_IMPACT_BUILD.md`.

**Data result:**
- Current connection measured about 10.8 Mbps; the 12-file bootstrap completed
  locally with resume/checksum support.
- Complete NBA Stats V3 event partitions now exist for 2024 and 2025 regular
  seasons (1,230 games each) and playoffs (84 and 85 games). Shot and matchup
  partitions cover 2025 playoffs as well.
- Every downloaded file passes its source-level schema, identity-key, season, and
  season-type checks.
- Cross-source reconciliation finds one missing game in 2024 regular-season matchup
  detail: `0022400856`. The event and shot partitions contain all 1,230 games.
- Legacy RAPM cache quality: 2022 and 2023 pass; 2024 contains one structurally bad
  game (`0022300835`, duplicated home player ID) and is quarantined; 2025 is an
  empty cache and fails critically.

**Independent model result:** Fixed-lambda zero-prior RAPM trained on 2021–2023 and
lineup-conditioned on all valid 2024 regular-season games scored margin RMSE 14.268,
MAE 11.134, and correlation 0.409 across 1,229 games. Predicted margin SD was 6.30
versus 15.62 actual, with calibration slope 1.01. This is a reproducible baseline,
not a production rating. A six-candidate penalty diagnostic barely moved results;
best RMSE was 14.260 at offense/defense lambdas 2000/4500, while the highest
correlation was 0.412 at 1000/1000. No candidate is promoted from one fold.

**Important correction to prior evidence:** The legacy validation's reported
1,201-game 2024 sample was created by a date-based postseason cutoff that removed
late regular-season games. It also dropped estimated garbage-time possessions from
the holdout target before aggregating “game margin,” so that target was not the
actual final score margin. Its ~0.63 correlation is therefore not comparable to the
clean all-game 0.409 result and must not be used as a production gate.

**Engineering result:** Vectorizing lineup overlap checks and sparse design
construction reduced the four-season fit from roughly four minutes to about 53
seconds on this machine. Nine tests pass.

**Next:** Build the event-to-game/stint/possession silver layer with exact score and
minutes reconciliation. That bridge is required for current RAPM, honest
garbage-time experiments, win probability, Net Points, and WP-RAPM.

**Dead ends / cautions:** Player-name sanity checks are not validation. One
chronological season is a diagnostic, not hyperparameter selection. Do not mix the
source-native matchup-detail table with the legacy lineup possession table despite
their similar names.

## 2026-08-08 — Gabriel mirror refresh and four-fold RAPM evidence gate

**Question:** Can Gabriel's live data mirror patch recent aggregate coverage, and
how many model runs are enough to say a three-year RAPM idea is better?

**What we did:** Inspected `gabriel1200/site_Data` at pinned commit
`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd` (2026-07-05). Extended the safe
downloader to validate CSV schemas and widths. Downloaded 20 pinned assets with
atomic writes, hashes, row counts, source revision, and upstream-license status.
Implemented a multi-fold three-year-to-next-year RAPM evaluator with paired
within-season game bootstraps and explicit evidence labels.

**Data result:** The refresh adds complete-ID aggregate playtype (2014–2026),
tracking (2014–2026), player shooting (2014–2026), shot zones (2001–2026), hustle
(2018–2026), passing (2014–2026), and the LEBRON identity panel (2010–2026).
The DFG master remains 91% null on player ID and is not safe as an ID-keyed
canonical table. Contract assets include 389 salary rows, 389 option rows, 328 cap
holds, 11 dead-money rows, and cap tables; they are raw inputs without stable NBA
player IDs. The upstream repository declares no license, so the manifest records
`not-declared-by-upstream` rather than assuming reuse rights.

**Model result:** Four outer folds used 3-year training windows and test seasons
2021–2024. Baseline `off=3000/def=3000` mean margin RMSE was 13.893. Asymmetric
`off=2000/def=4500` reached 13.883: a 0.08% improvement, 2/4 fold wins, and 87.35%
paired-bootstrap probability of lower loss. This does not clear the research gate.
`1000/1000` was 1.12% worse on RMSE and won 0/4 folds.

**Answer on run count:** RAPM is deterministic, so repeated seeds on the same split
add no evidence. One baseline-vs-idea screen needs at least four chronological
outer folds (eight fits total). A publishable claim should use six to eight folds
plus paired uncertainty and one untouched confirmation season. Stochastic models
add at least three seeds per fold.

**Dead ends / cautions:** Do not run Gabriel's scraper scripts unchanged in
production: many are hard-coded to one year or season type, overwrite CSVs directly,
and lack timeouts, retries, atomic writes, and schema gates. Treat the repository as
a useful pinned mirror and endpoint notebook until each source is independently
wrapped.

## 2026-08-08 — Canonical game dimension exposes season-label mismatch

**Question:** Can the new event sources share one reliable game identity for win
probability, Net Points, current RAPM, and playoff analysis?

**What we did:** Built a silver `game_dim` from NBA Stats V3 events, shot detail,
PBPStats, and matchup detail. Added explicit source-season, start-year, end-year,
season-label, phase, date provenance, home/away IDs, final scores, overtime,
source-coverage flags, content-addressed lineage, and registry integration.

**Result:** 2,629 unique games across 2024–25 and 2025–26: 2,460 regular-season and
169 playoff games. All games have dates, final scores, and two consistent teams.
There are zero duplicate game IDs, duplicate V3 action IDs, source-date conflicts,
team-code conflicts, or matchup team-ID conflicts. Shot detail supplies 2,545 games,
PBPStats supplies 1,314, and matchup detail supplies 2,544; their complementary
coverage yields a complete game dimension. Play-in games are absent from the archive
and are stated as out of scope.

**Critical semantic finding:** `nba-data-archive/season=2024` means the season
starting in 2024 (2024–25), while legacy `matchups_2024.parquet` means the season
ending in 2024 (2023–24). Any join on the integer `season` alone is off by one year.
Canonical tables must use `season_start`, `season_end`, or `season_label` explicitly.

**Next:** Within-game score-state normalization, then lineup stints and possessions
with final-score and player-minute reconciliation.

## 2026-08-08 — Reconciled event states and chronological win probability v0

**Question:** Can the current event archive support a trustworthy score-state layer
and a first leakage-safe live win-probability baseline?

**What we did:** Normalized NBA Stats V3 actions using `actionId` feed order, parsed
clocks, created regulation/overtime time state, reconstructed pre/post scores,
joined canonical games, and added terminal reconciliation. Sampled one state per 30
seconds plus exact opening and terminal states. Fit a fixed logistic state model on
2024–25 and evaluated once on untouched 2025–26 data.

**Data result:** 1,313,486 actions across 2,629 games pass all critical gates: no
duplicate event IDs, missing games, invalid clocks, time-order reversals, terminal
score mismatches, or nonzero terminal clocks. Scores reconstructed from made field
goals and made free throws match all 2,629 final scores exactly. Sparse upstream
score snapshots disagree on 53 correction rows, and `actionNumber` backtracks on
11,980 rows; therefore `actionId` is chronology and `actionNumber` is metadata.

**Model result:** Test set contains 1,315 games and about 129K uniformly sampled
states. Overall Brier is 0.1622, log loss 0.4769, AUC 0.8367, calibration slope 1.04,
and Brier skill versus the constant training home-win rate is 34.4%. Checkpoint
Brier improves from 0.247 at game start to 0.169 at halftime, 0.124 at the start of
the fourth, 0.088 with six minutes left, 0.057 with two minutes left, and 0.045 with
one minute left. Game-start AUC is correctly 0.5 because v0 has no pregame-strength
inputs.

**Interpretation:** This is a calibrated state baseline, not a finished win model.
The clean next ablations are pregame team/player strength, possession indicator,
timeouts/bonus, and availability—each added one at a time against the same test
contract. Final states are set deterministically from the completed score.

**Dead ends / cautions:** Sorting by `actionNumber` creates thousands of clock
reversals because the feed retroactively renumbers corrected actions. Forward-filling
the source score snapshots also creates false score decreases. Both approaches are
rejected.

## 2026-08-08 — Current starter seeds and five-player lineup stints

**Question:** Can 2024–25 and 2025–26 lineups be reconstructed accurately enough
to unblock current RAPM without relying on the stale legacy possession cache?

**What we did:** Added a pinned, resumable ingest for CDN NBA event partitions and
the `llimllib/nba_data` NBA/ESPN mirrors. NBA player-game positions supply starter
seeds; ESPN player boxes add independent starter checks plus Net Points and WPA
benchmarks. Reconstructed substitution stints, inferred omitted between-period
lineups in the late-playoff V3 fallback, and reconciled every player's stint seconds
against official box minutes.

**Result:** The canonical player-game table contains 69,517 rows and all 2,629
games. Every team-game has exactly five starters, team minutes reconcile, and the
2,627 ESPN-overlap games have zero starter disagreements. The lineup table contains
81,893 validated stints covering 2,622 games (99.73%). Seven games (0.27%) are
quarantined for feed/substitution or minute-reconciliation anomalies. Emitted rows
have zero duplicate stint IDs, nonpositive durations, invalid five-player sides, or
home/away player overlap. The five-second player-minute tolerance and 0.5% maximum
quarantine budget are explicit pipeline parameters.

**Source/legal note:** `cdechoch/nba-data-archive` declares Apache-2.0. The
`llimllib/nba_data` repository declares no license, so its NBA/ESPN-derived fields
are research-only and must not be redistributed until rights are clarified. ESPN
Net Points/WPA are benchmarks, not labels to copy into our own metric.

**Next:** Convert validated stints plus event states into possession outcomes, then
fit a current zero-prior RAPM baseline. Keep the seven quarantined games excluded
until a second source or a provably minute-consistent repair resolves them.

## 2026-08-08 — Ordinal possessions and current RAPM sensitivity

**Question:** Can the current two-season event archive replace the stale legacy
RAPM cache without silently misordering events or assigning same-clock actions to
the wrong lineup?

**What we did:** Built CDN possessions in physical `orderNumber` order and retained
every within-possession lineup change in a separate ordinal segment table. Used
`actionNumber` only as a guarded V3 score join when game, period, and clock agree.
Replayed substitutions in source order, keeping the prior lineup active until a
complete out/in transaction exists. Fit identical zero-prior RAPM models using the
lineup at possession start and at possession end as explicit sensitivity variants.

**Data result:** 525,279 possessions and 633,568 lineup segments cover 2,598 of
2,629 games (98.82%). The 31 excluded games are 25 CDN games after the May 9, 2026
source cutoff plus lineup-quality quarantines, with one overlap. All 2,598 included
games reconcile exactly to final scores. There are zero duplicate IDs, invalid
point outcomes, implausible game possession counts, or possession/segment point
mismatches. 76,354 possessions cross at least one lineup change, which confirms
that a single implicit clock-based lineup would be a lossy contract.

**Model result:** On 2024–25 regular season training and 2025–26 regular season
lineup-conditioned retrodiction, the start-lineup model has margin RMSE 15.798 and
correlation 0.285; terminal-lineup has RMSE 15.733 and correlation 0.294. A paired
10,000-draw game bootstrap gives a 99.73% probability that terminal-lineup squared
error is lower, with mean MSE delta -2.05 and 95% interval [-3.50, -0.58]. The
effect is real but small: about 0.4% relative RMSE, not a production verdict.

**Cautions:** This is one outer season, uses observed future lineups, and knows only
82.8% of test-season players from the training season. It is retrodiction, not a
forecast. CDN coverage stops during the 2026 playoffs, and the existing V3 fallback
must be validated prospectively before those 25 games enter possession RAPM.

**Next:** Add time-safe pregame team strength to win probability, then compare
possession-start state models. For RAPM, add a formal start-vs-terminal-vs-segment
comparison harness and more seasons before selecting a production lineup policy.

## 2026-08-08 — Time-safe pregame Elo materially improves win probability

**Question:** Does a simple pregame strength signal improve the state-only win
model without using current-game or future information?

**What we did:** Added a fixed Elo ablation (1500 initial rating, K=20, 60-point
home advantage, 25% offseason regression). Every game on the same date receives
ratings from the start of that date; results update ratings only after the entire
date is scored. Compared state-only and state-plus-Elo logistic models on identical
30-second states, training on 2024–25 and holding out all 2025–26 games. Terminal
states were excluded from the comparison, and uncertainty resampled whole games.

**Result:** Across 128,232 nonterminal test states and 1,315 games, Elo reduces
Brier from 0.16385 to 0.14987 (8.54%), log loss from 0.48175 to 0.44855, and raises
AUC from 0.83339 to 0.86294. The 5,000-draw paired game bootstrap favors Elo in
100% of draws; mean game-level Brier delta is -0.01404 with 95% interval
[-0.01949, -0.00875]. At game start, Brier improves 0.24708 to 0.21175 and AUC
improves from the expected 0.500 to 0.720. Elo's test calibration slope is 0.971.

**Interpretation:** Pregame strength is a large missing input, not a marginal
tweak. This is still one outer season and basic team Elo will lag trades, injuries,
rookies, and lineup changes. Keep it as the first promoted research challenger,
not a final production model. Older WP artifacts without a source-code hash are
superseded for provenance; new runs fingerprint both data and implementation.

**Next:** Build a possession-start WP table and add current possession as a
same-state ablation. Then replace team-only strength with frozen pregame player and
projected-lineup strength once availability/projection data exists.

## 2026-08-08 — ESPN benchmark and prior-season starter RAPM ablation

**Question:** Where does the local WP model lag ESPN, and does frozen prior-season
starter RAPM close the pregame gap without using current-game outcomes or minutes?

**What we did:** Cached ESPN play-level WP for 1,314 of 1,315 2025–26 games and
matched ESPN and local post-action states on game, period, score, and clock. The
single unavailable game has ESPN play-by-play but no WP records. For the lineup
ablation, 2024–25 games use RAPM fit only through 2023–24; 2025–26 games use RAPM
fit only through 2024–25. Official starters are treated as tipoff-time information.
Missing prior ratings are centered at zero; actual minutes and current-game results
are excluded. All comparisons resample whole games.

**External benchmark:** The validated run `wp_espn_benchmark_v1_ca79cde82d`
scores 631,380 matched nonterminal plays across 1,313 games with 99.26% raw
play-match coverage. ESPN Brier is 0.14759 versus 0.14883 for local state plus Elo.
The equal-game Brier-difference interval crosses zero. ESPN's clear advantage is at
tipoff: 0.20210 versus 0.21181, with local-minus-ESPN 95% interval
[0.00451, 0.01495]. From halftime onward the models are statistically tied at the
predeclared checkpoints.

**Starter-RAPM result:** Run `wp_lineup_ablation_v1_7570ad01c9` reduces overall
Brier from 0.14987 to 0.14922 and tipoff Brier from 0.21181 to 0.21057; tipoff AUC
rises from 0.7195 to 0.7226. Evidence is inconclusive: the tipoff paired bootstrap
favors the starter model in 70.24% of draws and its 95% delta interval
[-0.00562, 0.00319] crosses zero. It remains clearly worse than ESPN at tipoff:
local-minus-ESPN Brier delta 0.00847, 95% interval [0.00322, 0.01393]. Prior ratings
cover 91.96% of 2024–25 starters and 92.34% of 2025–26 starters on average; some
games cover only four of ten starters.

**Verdict:** Do not promote the starter-RAPM feature. The direction is plausible,
but one outer season and incomplete rookie/new-player coverage are insufficient.
The next pregame challenger should use expected rotation minutes, rookie/translation
priors, injuries/availability, and rest, while keeping ESPN as an external benchmark.

## 2026-08-08 — Rolling team context closes most of the ESPN tipoff gap

**Question:** Does a strictly time-safe rolling margin and rest signal add value
beyond Elo and frozen prior-season starter RAPM?

**What we did:** Added exponentially updated team point differential (10% update
weight, 75% offseason retention) plus rest advantage capped at seven days. Features
for every game on a date are computed before any result from that date updates the
ratings. The candidate retains Elo and prior-season starter RAPM and uses the same
2024–25 train / 2025–26 outer test split and whole-game bootstrap.

**Result:** Run `wp_pregame_ablation_v2_522e1a36f2` lowers overall Brier from
0.14922 for Elo plus starters to 0.14731. The paired mean game-level delta is
-0.00195, 98.56% of draws favor the context model, and the 95% interval
[-0.00373, -0.00020] excludes zero. At tipoff it improves Brier from 0.21057 to
0.20592 and AUC from 0.7226 to 0.7348; all 5,000 paired draws favor context over
starter RAPM. ESPN remains directionally better at 0.20210, but the paired
local-minus-ESPN interval [-0.00063, 0.00839] crosses zero.

**Verdict:** Promote rolling team context as the next research challenger, not as
production. It closes about 61% of the original Elo-to-ESPN tipoff Brier gap, but
only one outer season exists and the fixed smoothing choices have not been tested
across folds. Next acquire an earlier event season, freeze a small smoothing grid,
and require multi-season improvement before promotion.

## 2026-08-08 — Inpredictable neutral-state surface audit

**Question:** Does the local score/time probability shape agree with an independent
published calculator, and what missing state matters most?

**What we did:** Queried Inpredictable's public zero-spread calculator at eight
checkpoints, eleven margins from -15 to +15, and possession/no-possession. Removed
the local model's home intercept by averaging mirrored team orientations, then
compared it with the midpoint of Inpredictable's two possession states. This is a
reference-surface comparison, not outcome validation.

**Result:** Run `wp_inpredictable_surface_v1_56696b0386` covers 88 neutral states.
Surface correlation is 0.9983, mean absolute difference 1.64 percentage points,
RMSE difference 2.32 points, and maximum difference 5.97 points. Inpredictable's
possession input moves probability by 2.87 points on average; in a tie with ten
seconds left it moves probability by 23.4 points.

**Verdict:** The local score/time functional shape is independently plausible.
Possession/control is the highest-value missing in-game state. Build a causal
possession-start table before adding it; do not join the raw action possession tag
naïvely or interpret this surface agreement as predictive accuracy.

## 2026-08-08 — Causal possession-start WP improves late-game accuracy

**Question:** Does knowing which team controls the ball improve WP on held-out
games after score, clock, Elo, prior starter RAPM, rolling margin, and rest?

**What we did:** Rebuilt the canonical possession table with separate home and
away point deltas. A possession-start score is the cumulative sum of completed
prior possessions only; the current possession's points and every future result
are excluded. The first attempted shortcut—assigning all possession points to the
tagged offense—failed score conservation in 417 games because technical/free-throw
sequences can score for the other side. That representation was rejected. Three
logistic models were fit on identical 2024–25 starts and tested on 2025–26:
pregame context, context plus a possession sign, and possession with time-pressure
interactions. Uncertainty resampled whole games 5,000 times.

**Result:** Run `wp_possession_start_v1_9af34729ef` covers 261,222 test starts in
1,288 games. The time-interacted possession model lowers Brier from 0.146513 to
0.146325; mean game-level delta is -0.000189 with 95% interval
[-0.000214, -0.000163], and all draws favor the candidate. In 2,497 regulation
states inside two minutes with margin at most three, Brier improves from 0.17122
to 0.16719 and AUC from 0.82594 to 0.83474. Its fitted home-possession swing is
2.04 percentage points overall, 11.57 points in close last-two-minute states, and
19.67 points when tied inside ten seconds; Inpredictable's corresponding public
tied-ten-second swing is 23.4 points.

**Verdict:** Possession/control is retained as a research feature and the
time-interacted form beats a constant possession effect. It is not production yet:
this is one chronological outer fold, only CDN-covered lineup-reconciled games are
eligible, and the 2023–24 confirmation fold is not downloaded. The causal state
contract and failed score-attribution shortcut are now regression-tested.

## 2026-08-08 — Fractional and terminal lineup RAPM beat start-lineup attribution

**Provenance:** Fractional exposure was designed and implemented inside this
repository in commit `db4cb02`. It was motivated by the canonical segment table,
which preserves substitutions inside a possession. It is not attributed to a
paper or claimed as an established RAPM method. Treat it as a project-specific
sensitivity analysis unless later literature review finds a direct precedent.

**Question:** When substitutions occur inside a possession, should RAPM credit the
lineup at the start, the lineup at the terminal event, or every lineup fractionally?

**What we did:** Compared fixed zero-prior ridge RAPM models on the exact same
497,177 regular-season possessions. All use 2024–25 for training and retrodict game
margins in 2025–26 with observed lineups. Start and terminal select one lineup.
Fractional assignment keeps one outcome row per possession and gives each player
their elapsed-time share across ordinal lineup segments; possessions with zero
clock span fall back to action-count shares. Every fractional row sums to five
offensive and five defensive player exposures. Uncertainty resamples 1,228 whole
held-out games 5,000 times.

**Result:** Run `rapm_lineup_policy_v1_23149bbb29` gives held-out game-margin RMSE
15.798 for start, 15.733 for terminal, and 15.723 for fractional. Terminal minus
start squared-error delta is -2.048 with 95% interval [-3.463, -0.665]; fractional
minus start is -2.342 with interval [-3.303, -1.420]. Fractional is only
directionally better than terminal: delta -0.294, 74.5% probability better, 95%
interval [-1.082, 0.510]. Net-rating correlations with start remain 0.970 for
terminal and 0.984 for fractional, so the policy changes some rankings without
creating an unrelated metric.

**Verdict:** Reject start-lineup attribution for current RAPM. Use terminal lineup
as the simplest provisional production policy because fractional exposure has not
shown a reliable advantage over it. Keep fractional exposure as the research
challenger: it is conceptually fairer for possessions spanning substitutions, but
its sub-second timing is approximated and only one outer season exists. Neither
result is a true forecast; 82.8% player carryover and observed test lineups make it
lineup-conditioned retrodiction.

## 2026-08-08 — Nonlinear WP architecture ladder frozen before testing

**Question:** Should the next WP model be a CNN, recurrent network, transformer,
or a stronger tabular model?

**Evidence:** Published NBA work supports dynamic Bayesian smoothing and reports
small neural/recurrent mixture-density gains over logistic and random-forest
baselines, but the latter study's preprocessing data are proprietary. Generic TCN
research makes causal dilated convolutions a credible efficient sequence baseline.
TFT, NBA2Vec, and Set Transformer motivate heterogeneous temporal inputs and
permutation-invariant lineup embeddings; the Sloan distributional TD/Shapley work
is more directly relevant to the later player-credit lane than to the first WP
replacement. These sources do not establish that a transformer beats simpler NBA
WP models on our states or seasons.

**Decision:** Freeze the comparison order before the third rich season arrives:
dynamic Bayesian/GAM, tree boosting, small MLP, causal TCN, parameter-matched GRU,
then a two-layer causal transformer. A player-set encoder and recurrent mixture
density/value model follow only after the core comparison. Candidates receive
identical states, two chronological outer folds, whole-game bootstrap, calibration
gates, and five fixed seeds for optimizer noise. A seed is not an independent run.
Production remains the smallest model that clears both folds.

## 2026-08-08 — 2023–24 rich-event backfill clears the canonical gates

**Question:** Can the earlier season support an independent WP fold without
weakening lineup or score-conservation validation?

**What we did:** Downloaded the pinned ten-file 2023–24 event batch (86.96 MB)
with checksums and resumability. Added the missing CDN schema contract, supported
the older V3 schema without `shotValue`, and derived action scores from sparse
scoreboard snapshots. The historical NBA box cache contained 15,261 exact
duplicate rows and almost no 2023–24 coverage, so exact duplicates were removed
and ESPN starters/minutes were used as explicitly marked research fallbacks.
The first lineup rebuild quarantined 63 games. We downloaded official
BoxScoreTraditionalV3 JSON for the 56 affected 2023–24 games, then rebuilt every
silver table. Possession scoring was changed from broad V3 action-number
replacement to CDN `orderNumber` score deltas with V3 repair only when an
official terminal-score gap proved the CDN row wrong.

**Result:** Game dimension and event states pass for 3,941 games and 1,950,498
actions. Player-games cover all 3,941 games with no duplicate keys or missing
boxes. Lineups pass 3,931 games; ten remain transparently quarantined (0.254%,
below the 0.5% budget). Possessions pass for 3,907 games: 787,579 possessions and
946,768 lineup segments, with zero score mismatches, negative/implausible point
rows, duplicate IDs, or segment-conservation failures. Only two one-point CDN
score gaps were repaired from clock-and-period-aligned V3 free throws.

**Dead ends:** Treating `actionNumber` as a trustworthy scoring replacement.
Although 83.1% of CDN rows aligned with V3 on action number, clock, and period,
broad replacement created 37 final-score failures. `orderNumber` plus the CDN
scoreboard is the canonical path; V3 is a narrow validator/repair source.

**Verdict:** The data constraint for the second WP fold is removed. Run the
frozen 2023–24 → 2024–25 comparison next; do not tune architectures against the
existing 2025–26 test season.

## 2026-08-08 — Frozen second WP fold confirms context and possession

**Question:** Do rolling team context and causal possession control repeat on an
independent season, or were their 2025–26 gains one-fold noise?

**What we did:** Parameterized the existing frozen comparisons by explicit train
and test season without changing features, regularization, state sampling, or the
5,000-draw whole-game bootstrap. Trained on 2023–24 and tested on 2024–25, then
compared the result with the already-frozen 2024–25 → 2025–26 fold. ESPN was
skipped on the earlier fold because it is an external later-season benchmark.

**Result:** Run `wp_pregame_ablation_v2_3026c4a4b9` lowers Brier from 0.15496
with Elo plus starters to 0.15378 with rolling margin and rest. Its paired delta
is -0.00118, 95% interval [-0.00198, -0.00040]. The later fold lowers Brier from
0.14922 to 0.14731, delta -0.00195, [-0.00373, -0.00020]. Starter RAPM versus
Elo crosses zero in both folds. Run `wp_possession_start_v1_f4a1c8a2d2` lowers
2024–25 Brier from 0.15333 to 0.15314 with time-interacted control, delta
-0.000196, [-0.000230, -0.000164]. The later possession fold repeats the gain:
0.14651 to 0.14632, delta -0.000189, [-0.000214, -0.000163]. Close-last-two-minute
Brier improves in both seasons, with fitted possession swings rising above 11
percentage points.

**Null result:** Prior-season starter RAPM does not independently clear either
fold. It is still included inside the current rolling-context candidate, so that
exact combined implementation is larger than the evidence justifies.

**Verdict:** Promote rolling margin plus rest and causal possession-start control
as confirmed feature blocks. Do not freeze the exact production model until a
starter-free rolling-context ablation passes both folds. Then begin the fixed
GAM/GBM architecture ladder on identical states.

## 2026-08-08 — Starter-free WP becomes the frozen Stage 0 baseline

**Question:** Does prior-season starter RAPM add enough value to justify its
coverage gaps and dependencies once rolling team margin and rest are present?

**What we did:** Added an explicit starter-free model with the same state, Elo,
rolling margin, rest, interactions, regularization, rows, and whole-game
bootstrap as the combined model. Scored both frozen folds. Then removed player
games, lineup segments, legacy possessions, and starter ratings from the
possession-start model and repeated both possession folds.

**Result:** Pregame runs `wp_pregame_ablation_v3_30ab68d381` and
`wp_pregame_ablation_v3_cdbcea84ee` show starter-free context beating Elo on
2024–25 (0.15502 → 0.15302; delta -0.00201, 95% interval
[-0.00365, -0.00042]) and 2025–26 (0.14961 → 0.14777; delta -0.00189,
[-0.00377, -0.00008]). Against context plus starters, the smaller model wins the
first fold by 0.00077 and loses the second by 0.00053; both intervals cross zero.
Possession runs `wp_possession_start_v2_1db472e450` and
`wp_possession_start_v2_0a5d626234` retain the time-interacted control gain on
both folds, with Brier deltas -0.000177 and -0.000175 and both intervals excluding
zero. Late close-game gains remain materially larger.

**Verdict:** Freeze starter-free Elo plus rolling margin and rest as Stage 0.
Starter RAPM remains a research feature, not a production dependency. The
possession-start extension is separately validated and now uses the same smaller
context. Begin the fixed GAM/GBM comparison on identical features.

## 2026-08-08 — Spline GAM and histogram GBM fail state parity

**Question:** Can generic nonlinear shape or interactions beat the frozen
starter-free logistic WP model without adding information?

**What we did:** Run `wp_stage1_v1_7e6c77d51a` fits three models on identical 12
features, 30-second states, labels, and chronological folds: the frozen scaled
logistic control; an additive logistic model with five-knot cubic splines; and a
fixed histogram GBM with 200 trees, learning rate 0.05, at most 15 leaves and
depth six. No hyperparameter was selected on an outer test season. Uncertainty is
the same 5,000-draw whole-game bootstrap.

**Result:** Logistic Brier is 0.15302 on 2024–25 and 0.14777 on 2025–26. The
spline model is worse at 0.15560 and 0.14943; its pooled candidate-minus-logistic
delta is +0.00214, 95% interval [0.00088, 0.00340]. HistGBM is substantially
worse at 0.17030 and 0.15661; pooled delta +0.01303 [0.00970, 0.01639]. Both lose
AUC in both folds. The spline calibration slope is 0.876 on the first fold;
HistGBM slopes are 0.717 and 0.849.

**Verdict:** Reject both candidates. Their failures are not merely calibration:
ranking quality also declines. Do not tune around the result using 2024–25 or
2025–26. Run the preregistered residual MLP next; treat TCN/GRU/transformer models
as tests of causal sequence history, not automatic upgrades over tabular logistic.

## 2026-08-08 — Fixed feed-forward MLP fails across five seeds

**Question:** Does a small neural network improve the same frozen tabular WP
states where splines and trees failed?

**What we did:** PyTorch is unavailable locally, so the residual architecture was
not silently approximated. Run `wp_mlp_v1_7a7825bf09` explicitly tests a scaled
scikit-learn feed-forward MLP with two 64-unit ReLU layers, Adam, batch size 1024,
early stopping, a frozen 100-epoch cap, and seeds 7/17/29/43/71. Both outer folds
use identical rows and the five-seed probability ensemble; seeds measure optimizer
variance only.

**Result:** On 2024–25 the ensemble Brier is 0.18965 versus logistic 0.15302; on
2025–26 it is 0.17448 versus 0.14777. The pooled candidate-minus-logistic delta is
+0.03171, 95% interval [0.02619, 0.03745]. AUC falls from 0.8582 to 0.8092 and
from 0.8666 to 0.8255. Calibration slopes are 0.411 and 0.485. Nine of ten seed
fits hit 100 epochs, but every seed is materially worse.

**Verdict:** Reject this feed-forward implementation and do not tune it against
the outer seasons. This is not evidence against residual networks or sequence
history. The next meaningful experiment is a prefix-invariant completed-
possession token table, followed by parameter-matched TCN/GRU/transformer tests.

## 2026-08-08 — WP frozen after playoff and temporal audit

**Question:** Is the current model good enough to close the WP lane, including
playoffs and older-season evidence?

**What we did:** Segmented both frozen action-state and possession-start artifacts
by regular season versus playoffs and recomputed Brier, AUC, calibration slope,
and whole-game paired intervals. Also segmented the 2025–26 ESPN matched-play
benchmark and audited whether the legacy cache can support an older WP backtest.

**Result:** Regular-season action-state performance is stable: 2024–25 Brier
0.15096/AUC 0.862/slope 0.954 and 2025–26 Brier 0.14696/AUC 0.868/slope 1.047.
The earlier context-versus-Elo delta is -0.00250 with interval
[-0.00417, -0.00085]; the later interval narrowly crosses zero. Playoff samples
are only 84 and 85 games. Their Brier/slope pairs are 0.18317/0.648 and
0.15945/0.937; context-versus-Elo intervals cross zero. ESPN playoff Brier is
0.15696 versus 0.16182 for the older local matched-play benchmark, also unresolved
by whole-game bootstrap. Possession improves the 60-game later CDN playoff slice,
but the source omits the final 25 playoff games. Legacy pre-2023 possessions lack
clock/state fields and cannot evaluate this estimand.

**Verdict:** Freeze WP as good enough for regular-season platform use with an
explicit playoff calibration caveat. Do not fit a playoff-only model from this
sample, continue local neural training, or claim a pre-2023 backtest. Move active
work to RAPM and the all-in-one impact model.

## 2026-08-08 — Fractional lineup attribution wins the two-fold RAPM check

**Question:** Which lineup should receive credit when a substitution occurs
inside a possession: the start lineup, the terminal lineup, or fractional
exposure across every lineup segment?

**What we did:** Run `rapm_lineup_policy_v2_911d8bfce1` fits the same zero-prior
ridge specification for all three policies. It uses 743,946 identical regular-
season possessions and two one-season-train chronological folds: 2023–24 to
2024–25 and 2024–25 to 2025–26. It resamples whole games 5,000 times. Fractional
weights use elapsed-time share, with action-count share only when the clock does
not move. The policy does not use the possession outcome.

**Result:** On the first fold, RMSE is 15.0107 for start, 15.0541 for terminal,
and 15.0101 for fractional. Fractional versus start is a tie: squared-error
delta -0.016, 95% interval [-1.028, 1.021]. Fractional beats terminal by -1.321,
[-2.196, -0.443]. On the second fold, RMSE is 15.7975, 15.7326, and 15.7232.
Fractional beats start by -2.342, [-3.306, -1.393], and is unresolved versus
terminal at -0.294, [-1.107, 0.519]. Across 2,454 disjoint test games,
fractional beats start by -1.180, [-1.876, -0.487], and terminal by -0.807,
[-1.426, -0.208].

**Verdict:** Freeze fractional exposure as the working current-RAPM attribution
policy. Keep start and terminal as mandatory sensitivity outputs. This is a
small, project-specific engineering decision. It is not a published standard.
The current evidence has only two outer folds, and one fractional-versus-start
fold is effectively tied. Select ridge penalties next without using 2025–26 for
model selection.

## 2026-08-08 — Normal RAPM keeps the original ridge penalties

**Question:** Do new offensive, defensive, and home penalties improve normal
RAPM outside the season used to select them?

**What we did:** Run `normal_rapm_v1_85e0cc8e27` compares 20 predeclared penalty
triples on 2023–24 to 2024–25 game-margin retrodiction. It then evaluates the
selection winner once on 2025–26 after fitting 2023–24 and 2024–25. The run uses
terminal lineup assignment, 743,946 possessions, 3,681 games, and no uncertainty
estimation.

**Result:** Selection chose 4500/4500/1000 with RMSE 15.05339, only 0.00067
better than 3000/3000/300. On untouched confirmation, the original penalties win:
RMSE 15.47320 versus 15.50980, correlation 0.3344 versus 0.3230, and MAE 12.1970
versus 12.2262.

**Verdict:** Reject the selected challenger. Normal RAPM remains zero-prior ridge
with offensive/defensive/home penalties 3000/3000/300. Park fractional lineup
attribution as research-only by user direction.

## 2026-08-08 — First three-year statistical impact baseline

**Question:** Do the available advanced features and on/off add predictive signal
beyond box rates for three-year normal RAPM?

**What we did:** Run `statistical_impact_v1_6dff345dc2` joins 6,513 player-window
rows from 2016–2024 to separate offensive and positive-is-good defensive RAPM
targets. It compares ridge models using box rates, 49 allowed advanced features,
and the same advanced panel plus OnOffRtg/OnDefRtg. Age, experience, height,
position, minutes, games, and possession counts are not predictive features.
Square-root possession counts weight noisy labels. Test windows end in 2022,
2023, and 2024; a two-window purge prevents three-year target overlap.

**Result:** Advanced features improve net weighted RMSE over box rates in all
three folds by 0.0342, 0.0466, and 0.0410. Mean net RMSE is 1.3594 versus 1.4000.
Adding on/off improves every fold again by 0.2815, 0.2276, and 0.2206; mean net
RMSE is 1.1162 and correlation is 0.5629. The independent advanced model has
mean offensive/defensive correlations 0.4341/0.3092. The on/off-assisted model
has 0.5242/0.4800.

**Verdict:** Keep the advanced ridge model as the independent baseline. Keep the
on/off-assisted model as a distinct, stronger impact-assisted challenger; do not
mislabel it as an independent statistical prior. Before further model comparison,
rebuild percentages and average tracking fields with natural attempt or touch
denominators because the inherited panel minute-weights those fields.

## 2026-08-08 — Direct net target adds no value to the ridge baseline

**Question:** Should the statistical model predict net RAPM directly, or predict
offensive and defensive RAPM separately and add them?

**What we did:** Run `statistical_impact_v2_48f6ad776f` adds a direct three-year
net RAPM target to the same purged chronological folds, feature sets, reliability
weights, preprocessing, and ridge alpha search used by the component models.

**Result:** On the advanced features, direct net and summed components have the
same mean weighted RMSE to floating-point precision: 1.3593836544. Their mean
correlations are also the same: 0.4159608624. The on/off-assisted variants are
also identical: RMSE 1.1161642634 and correlation 0.5629409007. This is expected
because a ridge solution is linear in its target when the design, weights,
preprocessing, and penalty are the same. The box-only variants differ by 0.00085
RMSE because their independently selected penalties are not identical in every
fold.

**Verdict:** Use separate offensive and defensive ridge models, then add their
predictions for net. The direct target gives no practical gain and cannot explain
the offense/defense split. Reconsider a direct net target only for a nonlinear
model, a different loss, or a different feature set.

## 2026-08-08 — Natural-denominator statistical feature rebuild

**Question:** Does replacing the legacy minute-weighted feature panel with
pooled counts and natural attempt/touch denominators change the ridge baseline?

**What we did:** Run `statistical_features_v1_940f99ed54` builds 97 box,
tracking, shot-zone, creation, turnover, and rebound features for 6,689
player-windows ending 2016–2024. It excludes age, minutes, games, position, and
experience. Possessions remain reliability weights only. The builder collapses
four duplicate source rows that agree on the declared feature contract. It
rejects conflicting feature rows, duplicate output keys, and invalid bounded
ratios. Rebound-chance conversion was removed because total rebounds and
tracking chances do not have consistent coverage.

Run `statistical_impact_v2_5224a3b4a6` evaluates the rebuilt table on the same
three purged outer folds as the legacy baseline.

**Result:** Independent advanced net RMSE is 1.3578 with correlation 0.4212,
versus 1.3594 and 0.4160 on the legacy panel. It beats box-only RMSE in all three
folds by 0.0202, 0.0517, and 0.0518. The on/off-assisted model remains much
stronger at RMSE 1.1157 and correlation 0.5583, but it is not an independent
statistical prior. The clean rebuild is a correctness improvement; its aggregate
metric lift is small.

**Verdict:** Promote the rebuilt table as the feature baseline. Do not claim a
large accuracy win. Use it for the fixed ridge/elastic-net/tree comparison and
for the user's future feature challenge.

## 2026-08-09 — First statistical model-family comparison

**Question:** Do elastic net or a small nonlinear tree model beat ridge on the
validated 97-feature table?

**What we did:** Run `statistical_model_comparison_v1_dd31e7957d` compares
ridge, elastic net, and histogram gradient boosting on identical player-windows,
three-year offensive and defensive normal RAPM targets, possession reliability
weights, and 2022–2024 purged outer folds. Hyperparameters are selected only in
each fold's inner chronological validation window. The histogram models use at
most 15 leaves; this is not a neural model.

**Result:** Histogram GBM improves component-summed net RMSE in all three folds
by 0.0412, 0.0327, and 0.0263. Mean net RMSE is 1.3259 versus 1.3593 for ridge;
correlation is 0.5169 versus 0.4265. The gain is offensive: histogram offense
beats ridge in all three folds, with mean RMSE 0.9349 versus 0.9725 and
correlation 0.5448 versus 0.4699. Histogram defense loses RMSE in all three folds
(0.9083 versus 0.9028) despite higher correlation. Elastic net has mean net RMSE
1.3804 and beats ridge in only one fold.

**Verdict:** Promote histogram GBM as the offensive challenger. Keep ridge as
the defensive baseline. Reject elastic net. The mixed histogram-offense plus
ridge-defense model is the next component candidate; compare it with direct
nonlinear net prediction before any production choice. Three chronological fold
wins are initial evidence, not proof across every NBA era.

## 2026-08-09 — Direct nonlinear net loses to decomposed AIO

**Question:** Does a histogram GBM trained directly on net RAPM beat histogram
offense plus ridge defense?

**What we did:** Run `statistical_direct_net_v1_286a104216` tunes the direct
histogram model only inside each chronological training fold. It compares direct
net with saved component predictions from
`statistical_model_comparison_v1_dd31e7957d` on identical test players.

**Result:** Direct net loses weighted RMSE in all three folds. Mean RMSE is
1.3413, versus 1.3257 for histogram offense plus ridge defense and 1.3593 for
ridge components. Direct net has higher correlation than the hybrid, 0.5093
versus 0.4825, but worse magnitude calibration. The hybrid is better on the
primary loss in 2022, 2023, and 2024.

**Verdict:** Reject direct nonlinear net for this feature set. Keep the AIO
decomposed as histogram GBM offense plus ridge defense. Next, ablate feature
families before adding complexity.

## 2026-08-09 — Frozen-model feature-family optimization

**Question:** With histogram GBM offense and ridge defense frozen, which broad
feature families add chronological signal?

**What we did:** Run `statistical_feature_ablation_v1_918be14a38` removes core
box, shot profile, creation/role, turnover detail, and tracking rebound/defense
families one at a time. Model families and hyperparameters never change. The
2022–23 folds select combined removals; 2024 is scored once as the feature-level
confirmation fold.

**Result:** Offense removes the 25-feature creation/role block. Defense removes
35 shot-profile and 10 turnover-detail features. Core box and tracking
rebound/defense survive. On 2024, offensive RMSE improves 0.8846→0.8822,
defensive RMSE improves 0.9055→0.8998, and combined net RMSE improves
1.3096→1.3020. Net correlation improves 0.5155→0.5410. The resulting models
use 70 offensive and 50 defensive features and are saved in
`statistical_aio_v1_b0295558c6`.

**Verdict:** Freeze this statistical feature contract as a research challenger.
Do not continue subset search on the same three outer folds. Another search would
reuse the only tracking-era evidence and overfit research choices. The next AIO
step is cross-fitted prior generation and prior-informed RAPM, not another model
or feature sweep.

## 2026-08-09 — Basketball-domain feature engineering improves offense only

**Question:** Can stabilized rates, era context, scoring topology, creation
quality, behavioral role, and within-window temporal shape improve the frozen
statistical AIO without age, experience, height, position, minutes, games, on/off,
or plus-minus inputs?

**What we did:** Run `statistical_features_v2_6f7b3c5c57` adds 131 engineered
features to 6,689 validated player-windows. It recalculates empirical-Bayes
percentages from pooled natural denominators; adds possession-weighted
era-relative rates; separates latest-season level, linear trend, and volatility;
and adds scoring-topology, creation, role, rebound, and defensive interactions.
Missing engineered values receive a neutral within-window fill. The table has
zero duplicate keys, infinities, bounded-feature failures, or remaining missing
engineered values.

Run `statistical_feature_v2_comparison_e1cba6dd1d` holds the learners fixed:
Histogram GBM for offense (`learning_rate=0.03`, seven leaves, L2=1) and ridge
for defense (`alpha=3000`). It tests predeclared blocks on 2022 and 2023. A block
must reduce component RMSE in both folds. The selected combination is scored once
on 2024.

**Result:** Offense selects era-relative rates, latest-season levels, and
trend/volatility. Discovery offense RMSE improves 0.97270 to 0.90945 in 2022 and
0.92624 to 0.88170 in 2023. No defensive block passes both folds. On 2024,
offense RMSE improves 0.88224 to 0.84528 and correlation improves 0.56423 to
0.58908. Recombined net RMSE improves 1.30196 to 1.27987 and correlation improves
0.54104 to 0.55221. The saved challenger uses 117 offensive and 50 defensive
features.

**Null results:** Stabilized percentages, scoring topology, and creation-quality
blocks do not improve offense in both discovery folds. Era context, recent level,
temporal dynamics, direct defensive interactions, and offensive role context do
not improve defense in both folds. The available public panel still contains
little direct defensive matchup information.

**Verdict:** Keep v2 as the current research challenger. Do not call it final or
optimal: 2024 was held out from this feature selection but was inspected during
earlier model-family work, and the target RAPM run ends in 2024. Stop subset
search on 2022–24. Next, create cross-fitted priors and test whether this feature
gain improves the downstream possession model.

## 2026-08-10 — Public all-in-one methods support a factor challenger

**Question:** Should the statistical AIO estimate direct offensive and defensive
RAPM only, or also estimate interpretable team-factor effects?

**What we reviewed:** Public methodology for MAMBA and its six-factor RAPM,
Thinking Basketball's Box Creation, Offensive Load, Passer Rating, ScoreVal, and
AuPM, BPM 2.0, RAPTOR, LEBRON, EPM, DARKO, PIPM, the Dean Oliver Four Factors,
and the linked APBR RAPM implementation discussion. Exact source links and the
resulting model contract are in `docs/impact/FACTOR_DECOMPOSITION.md`.

**Verified findings:** Teemo's factor RAPM uses six targets: TS, turnovers, and
rebounds on offense and defense, then learns a linear scale back to offensive and
defensive RAPM. EPM and DARKO both support stat-specific stabilization and decay
instead of one shared rolling average. RAPTOR supports expected shot value,
assisted-shot, contested-rebound, and spacing features. BPM and LEBRON support
behavioral role interactions and role-based stabilization. Ben Taylor's work
supports creation and turnover features that distinguish responsibility from
efficiency. It does not provide a reproducible current all-in-one formula.

**Decision:** Make an eight-head eFG, turnover, rebound, and free-throw model the
primary factor challenger. Make the six-head TS version an ablation. Do not use
TS with a separate free-throw head. Keep the direct offense and defense model as
the production baseline and allow a cross-fitted residual for effects the four
factors do not assign. Current rich events are sufficient to validate the target
builder for 2023–24 through 2025–26, but historical event ingestion is required
before production promotion.

## 2026-08-10 — Correction: factors structure features, not required targets

**User clarification:** The shooting, turnover, rebound, and free-throw factors
are the user's mental decomposition of basketball value. The first AIO does not
need a separate supervised RAPM target for every factor.

**Additional source review:** The CraftedNBA glossary publishes Box Creation,
Offensive Load, Shooting Proficiency, Spacing, and an explicit Passer Rating
formula. CraftedNBA labels its Passer Rating as inspired by Ben Taylor, not as
Taylor's exact formula. It uses standardized Load, assist-to-Load,
position-standardized assist-to-Load, turnover-to-Load, creation-to-Load, and
height. The glossary does not specify the standardization population, padding,
position assignment, or final 1–10 transform. CraftedOPM, CraftedDPM, and
Portability include external impact metrics and are not valid independent prior
inputs.

**Corrected decision:** Keep direct three-year offensive and defensive RAPM as
the primary targets. Use the factors as feature families, diagnostics, and the
published explanation layer. Add versioned public formulas as candidate inputs.
Test the exact and behavioral passer ratings as separate challengers. Raw height
and listed position remain excluded as general columns; the exact composite is
blocked until canonical metadata and its standardization contract exist. Factor
RAPM remains optional research.

## 2026-08-10 — Public basketball features pass an exploratory offense test

**Question:** Do versioned Box Creation, Offensive Load, passing-ratio,
Shooting Proficiency, and Spacing features add signal to the frozen direct
offensive RAPM model?

**What we did:** Feature run `statistical_features_v2_8b2566243f` adds exact
public-formula benchmarks plus possession-shrunk model variants. The model-safe
block has eight features: Shooting Proficiency, Box Creation, Offensive Load,
assist-to-Load, turnover-to-Load, creation-to-Load, a behavioral passer score,
and a stable spacing proxy. The feature table has 6,689 player-windows, zero
duplicate keys, zero infinities, and zero missing engineered values after the
documented neutral fill. The exact Crafted passer variant is not present because
the current canonical table lacks height and position metadata.

Run `statistical_feature_v2_comparison_9b8d0555e0` holds the learner and target
fixed. It evaluates the public block separately from generic empirical-Bayes
ratios. The public block reduces offense RMSE from the 0.94912 discovery mean to
0.92292 and improves both 2022 and 2023. The final selected offense combination
also includes stabilized ratios, era-relative rates, recent level, and temporal
dynamics. No defensive block passes.

**Result:** On the reused 2024 check, offense RMSE improves from 0.87881 to
0.82701 and correlation improves from 0.56890 to 0.62133. Recombined net RMSE
improves from 1.29843 to 1.26244 and correlation improves from 0.54462 to
0.57485. These are exploratory estimates because 2024 informed earlier model
choices and the historical RAPM labels may not be the final target run.

**Verdict:** Keep the public block in the research challenger. Do not promote it
from these reused seasons or search individual public features on the same
folds. The next valid test is whether cross-fitted statistical priors improve a
downstream prior-informed RAPM on chronological future games.

## 2026-08-10 — Cross-fitted statistical priors are ready for RAPM

**Question:** Can the selected statistical AIO generate a prior for every
historical player-window without allowing a target window to train itself?

**Method:** Run `statistical_priors_v1_2c81b23662` freezes the selected 162-feature
offense histogram GBM and 50-feature defense ridge. For prediction window `T`,
training includes only RAPM target windows ending by `T-3`. The three-year train
and prediction targets therefore share no seasons. Same-window box and tracking
features are observed before prediction, so this is an end-of-window
retrodiction rather than a forecast. Models predict all feature rows; labels are
joined only afterward for evaluation.

**Quality:** The artifact contains 4,656 unique player-windows across 2019–24
and 1,270 players. It scores every eligible feature row. Minimum label coverage
is 96.21%; the remaining feature-covered players still receive priors. There are
zero duplicate keys, missing predictions, non-finite predictions, purge
violations, or offense-plus-defense identity errors. A test confirms that
changing a window's target cannot change that window's prior.

**Result:** Across six chronological folds, offense RMSE is 0.85494 with 0.60664
correlation, defense RMSE is 0.87365 with 0.29882 correlation, and net RMSE is
1.25131 with 0.51980 correlation. Defense remains the weak component. The 2019
fold has only one older target window, and the 2022–24 metrics are reused
research evidence.

**Verdict:** The priors are valid inputs for a prior-informed RAPM experiment.
They are not yet a final all-in-one. Next compare zero-prior RAPM, prior-only
ratings, and prior-informed RAPM on identical chronological possession windows.

## 2026-08-10 — Prior-informed RAPM does not clear confirmation

**Question:** Does the purged statistical AIO improve three-season normal RAPM
when used as the ridge center rather than as a standalone rating?

**Method:** Run `prior_informed_rapm_v1_122ef63045` uses matched regular-season
games and three-season training windows. It maps positive-good offense to the
offensive coefficient and positive-good defense to the negative points-allowed
coefficient, centers both blocks by training possession exposure, and compares
zero-prior, prior-only, and center scales 0.25/0.50/0.75/1.00. Scale selection
uses test seasons 2020–22. Seasons 2023–24 are a later research check. A
2,000-draw whole-game bootstrap resamples within season and weights seasons
equally.

**Result:** Scale 1.0 wins selection (13.8002 RMSE versus 13.8800 zero-prior).
On 2023–24 it records 13.5263 RMSE versus 13.5296, a 0.00327-point difference,
and wins one of two folds. The paired MSE delta is -0.195 with 95% interval
[-1.119, +0.729] and 0.6425 bootstrap probability of improvement. Prior-only is
worse at 13.7952 RMSE. The comparison covers 2,459 matched confirmation games;
minimum test lineup-slot prior coverage is 89.51%.

**Verdict:** Improvement is not demonstrated. Keep terminal-lineup zero-prior
normal RAPM with penalties 3000/3000/300 as production. Keep the statistical
model separate and labeled as research. Do not tune more scales on these reused
seasons. Revisit integration only with new seasons or one predeclared
sample-size-adaptive rule.

## 2026-08-10 — SPM benchmarked against BPM and xRAPM

**Question:** Does the current three-season SPM resemble established box and
prior-informed adjusted-impact metrics on matched historical windows?

**Method:** Run `external_impact_benchmark_v1_bab43a4087` downloads and hashes
annual 2017–24 Basketball Reference advanced tables and xrapm.com historical
tables. It resolves traded-player BPM rows to the multi-team total, converts
xRAPM defense from lower-is-better to positive-good, normalizes accents and
suffixes, and minutes-weights annual values over each `T-2:T` SPM window. The
comparison covers all 4,656 SPM rows and separately reports rows with at least
3,000 offensive and defensive possessions.

**Quality:** External source-to-source name matching is at least 99.43% in every
season. SPM-to-BPM and SPM-to-xRAPM coverage is at least 98.52% and 98.47% per
window. There are zero duplicate player-window keys. Seventeen normalized names
are ambiguous in the historical NBA name dimension and are quarantined rather
than guessed.

**Result:** Across 2,295 high-exposure player-windows, SPM net correlates 0.876
Pearson / 0.841 Spearman with BPM and 0.756 / 0.692 with xRAPM. Offensive SPM
correlates 0.831 with xRAPM offense. Defensive SPM correlates only 0.630 with
xRAPM defense, the clearest component gap. Including all tiny samples lowers net
Pearson correlation to 0.541 versus BPM and 0.610 versus xRAPM, showing that
sample/reliability handling materially changes any leaderboard comparison.

**Verdict:** The SPM has strong external face validity for established players
but remains closer to BPM than to the possession-informed xRAPM, especially on
defense. Use player-level defensive disagreements as the next feature diagnostic.
Do not train on BPM or xRAPM and do not treat either as ground truth.

## 2026-08-10 — Single-season SPM baseline and defensive disagreement report

**Question:** Can one global statistical model turn each season's own box and
tracking evidence into an annual SPM, while keeping annual xRAPM only as a
multi-window comparator?

**Method:** Target run `single_season_rapm_targets_v1_fd876680da` fits separate
zero-prior normal RAPM models for 2014–24 with penalties 3000/3000/300. Feature
run `statistical_features_v2_0e1350d95a` uses one-season pooling. It excludes
age, experience, height, position, minutes, games, on/off, plus-minus, and all
three-season temporal features. Run `single_season_spm_v1_51adc53061` keeps the
frozen offense histogram GBM and defense ridge. For each reported 2017–24
season, it trains on every other 2014–24 season. The final leaderboard refits on
all 2014–24 labels. This is descriptive leave-one-season-out evaluation, not a
forecast.

**Quality:** The panel has 5,791 unique player-seasons and the reported table has
4,341 rows. It has zero duplicate keys, missing names, non-finite predictions,
or offense-plus-defense identity errors. BPM matches 98.89% and xRAPM matches
98.99% of reported rows. The high-exposure comparison has 2,860 rows matched to
both sources. The 2024 target cache has 1,229 regular-season games, one fewer
than expected, and all legacy targets stop after 2024.

**Result:** Mean held-out weighted RMSE/correlation is 1.0060/0.6178 on offense,
1.0578/0.3091 on defense, and 1.4611/0.5314 on net. For high-exposure rows, net
Pearson correlation is 0.897 with BPM and 0.762 with xRAPM. Defensive correlation
with xRAPM is 0.590. The annual net xRAPM correlation remains between 0.725 and
0.795 in every season. Stable disagreement examples include Dillon Brooks and
Robin Lopez ranked higher by xRAPM defense, and Damian Jones and Mo Bamba ranked
higher by SPM defense. These are audit leads, not evidence that one metric is
correct.

**Verdict:** Share with caveats as the first annual SPM baseline. It is not the
final all-in-one because the target is noisy and the evaluation uses later as
well as earlier seasons. Next use annual out-of-fold SPM as the prior mean in a
one-season RAPM comparison. Estimate or tune prior precision inside training
seasons; do not restore the earlier arbitrary amplitude scale.

## 2026-08-10 — Annual zTS and playtype challenger

**Method:** `playtype_features_v1_db63ed1132` rebuilds 2014–24 zTS from the
newest local Gabriel O'Connell playtype snapshot and annual box totals. zTS is
player TS% minus expected TS% from the player's playtype mix. It also calculates
overall playtype POE, transition share, and transition POE contribution per 75
total Synergy possessions. Existing thresholds remain 250 minutes, 50 player
Synergy possessions, and 20 possessions per player-playtype row for the league
playtype TS calculation. Models and eight leave-one-season-out folds are frozen.

**Quality:** The table has 4,299 unique player-seasons. Every key matches the old
rounded `zts_results.csv`; zTS correlation exceeds 0.9999998 and maximum absolute
difference is 0.005 percentage points. Feature run
`statistical_features_v2_ab9646062b` has no duplicate keys or non-finite values.

**Result:** Baseline offense RMSE/correlation is 1.0060/0.6178. Adding zTS alone
(`single_season_spm_v1_fcdb9559f6`) improves it to 0.9972/0.6302; net improves
from 1.4611/0.5314 to 1.4553/0.5383. Adding overall POE, transition share, and
transition POE (`single_season_spm_v1_1f2125a38b`) records offense
0.9967/0.6301 and net 1.4558/0.5376. Defense is unchanged.

**Verdict:** Keep zTS in the annual offensive research model. The larger block's
tiny RMSE gain and slightly worse correlation do not justify promotion. These
seasons are now inspected, so require new data or a predeclared nested rule for
promotion. Next canonicalize DFG, rim-defense, and hustle features.

## 2026-08-10 — Canonical defensive tracking block

**Question:** Do direct shot-defense and hustle measurements repair the weak
annual defensive SPM?

**Source and quality:** The existing manifest already pinned and downloaded
Gabriel O'Connell's current `site_Data` commit
`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`. The aggregate rim-defense files
were missing from the manifest, so they were added and downloaded through the
resumable ingest path. Upstream declares no license; use these inputs for local
research only. Run `defensive_tracking_features_v1_9f66c664eb` contains 5,791
unique player-seasons with no non-finite values. DFG and rim source-name joins
cover 99.60% and 99.61% of rows; hustle joins cover 100%. Hustle begins in 2018,
so earlier seasons receive the global neutral median rather than a zero-valued
era marker.

**Features:** The predeclared block contains defended attempts per 100,
empirical-Bayes overall DFG differential, rim DFGA per 100, empirical-Bayes rim
DFG differential, rim points saved per 100, deflections, charges, contested
2-point and 3-point shots, and defensive loose balls recovered. Hustle counts
use defensive possessions. DFG differentials shrink toward zero with 200 total
or 100 rim attempts. Positive rim points saved means better defense.

**Result:** `single_season_spm_v1_bff6060df6` keeps the zTS offense challenger
and adds all ten fields to the frozen defensive ridge. Defense held-out RMSE
improves from 1.0578 to 0.9595 and correlation from 0.3091 to 0.4964. Net RMSE
improves from 1.4553 to 1.3859 and correlation from 0.5383 to 0.5991. Defense
and net improve on both metrics in all eight 2017–24 season folds. High-exposure
defense correlation with xRAPM rises from 0.590 to 0.701, while correlation with
BPM defense falls from 0.803 to 0.690. This is consistent with adding information
that is less box-score-like; neither comparator is ground truth.

**Verdict:** Retain the full defensive tracking block in the annual research
model. Do not search its subsets on these folds. Confirm on new seasons or use a
predeclared nested historical selection design before production promotion.

## 2026-08-10 — Gabriel assist audit and locked annual AIO integration

**Assist audit:** Gabriel O'Connell's pinned `assist_data` table defines adjusted
potential assists as potential assists plus 0.88 times free-throw assists. Its
assist efficiency is adjusted assist points divided by adjusted potential
assists. It is points per adjusted potential assist, not true shooting. The
clean builder removes four identical duplicate player-season rows, repairs three
infinite ratios, and emits 5,791 unique finite player-seasons. The two
nonduplicate candidates are free-throw assists per 100 and empirical-Bayes assist
points per adjusted potential assist.

**Assist result:** `single_season_spm_v1_d6de68348c` slightly improves the
eight-fold average, but offense and net lose both RMSE and correlation in 2023
and 2024. Do not promote the assist candidates. Contract
`configs/models/annual_spm_v1.json` freezes `single_season_spm_v1_bff6060df6`:
offense histogram GBM plus zTS, defense ridge plus all ten canonical defensive
tracking fields, and net equal to offense plus defense.

**Time-safe method:** Leave-one-season-out SPM is valid for descriptive mapping
tests but leaks later labels into a next-season experiment. Run
`annual_spm_priors_v1_1107680642` instead trains each SPM(T) mapping only on
seasons before T. It creates 3,769 priors for 2017–23 with no duplicate keys or
nonfinite values. Run `prior_informed_rapm_v1_e1239679c1` fits one-season normal
RAPM(T) with the fixed 3000/3000/300 penalties. It compares zero center, the
exact full SPM(T) center, and SPM alone on T+1 regular-season game margins. No
prior-amplitude grid is used.

**Result:** Full SPM centering wins RMSE in all four 2018–21 selection folds and
all three 2022–24 later folds. On 2022–24, mean RMSE is 13.8118 for the combined
model, 13.8892 for normal RAPM, and 14.0128 for SPM alone. Correlation is 0.3656,
0.3486, and 0.3412 respectively. Across 3,689 matched games, the equal-season
MSE delta for combined minus normal RAPM is -2.143. The paired game bootstrap
95% interval is [-3.270, -0.965].

**Verdict:** The annual SPM-centered normal RAPM is the leading research AIO.
This is not a production promotion because the later seasons already influenced
feature design. Freeze the specification. The next deliverable is its annual
offense, defense, net, SPM-center, RAPM-update, and possession-exposure table.

## 2026-08-10 — Decomposed annual AIO ratings

**Method:** Run `annual_aio_ratings_v1_23c4895f8f` fits the frozen full-SPM-center
normal RAPM separately for each 2017–24 regular season. Each row records the raw
SPM value, the possession-weighted centered SPM coefficient, zero-prior normal
RAPM, final AIO, and the RAPM update equal to final AIO minus centered SPM. It
also records offensive and defensive possession exposure and within-season net
rank.

**Quality:** The panel has 4,341 unique player-seasons and 1,270 players. All
player names and SPM priors match. The minimum lineup-slot prior coverage is
100%. Offense plus defense equals net, and SPM center plus RAPM update equals
final AIO, with maximum error below 9e-16. The 2024 input has 1,229 regular-season
games and remains one game short.

**Distribution check:** The 2024 AIO net distribution spans -6.07 to +6.74 and
has standard deviation 2.06. It is mildly right-skewed (0.59), not exactly
normal. Its unweighted player mean is -0.63 because many low-exposure players
are below average; its possession-weighted mean is effectively zero (-0.002).
The AIO is wider than zero-prior normal RAPM (standard deviation 1.66) because
the statistical center restores signal that zero-centered ridge suppresses.

**2024 high-exposure leaders:** Nikola Jokic (+6.74), Shai
Gilgeous-Alexander (+6.39), Paul George (+6.16), Joel Embiid (+5.49), OG Anunoby
(+5.16), and Jalen Brunson (+5.15), using at least 1,000 possessions on both
sides. Treat the precise order as exploratory, not ground truth.

**Verdict:** Use this versioned panel for annual leaderboards and decomposition.
Do not add uncertainty labels until they are estimated. Build three-year and
five-year peak tables next.

## 2026-08-10 — Independent three-year and five-year normal-RAPM peaks

**Contract:** `configs/models/rolling_normal_rapm_peaks_v1.json` freezes 1997–2024
regular-season windows, zero-prior ridge penalties 3000/3000/300, terminal
lineups, and a minimum 1,000 offensive and defensive possessions per included
window season. Peak selection is descriptive and winner's-curse biased. The
model removes each season's mean points per possession before fitting player
effects so changing league scoring environments do not become player credit.

**Implementation and quality:** Run `rolling_rapm_peaks_v1_584adf4f3d` fits 26
three-year and 24 five-year windows. It saves 36,530 rolling ratings and 7,866
offense/defense/net player peaks. Rating and peak keys are unique; component
identity error is below 9e-16. Annual player sheets fill historical crosswalk
gaps, including Sasha Danilović (ID 390), so every eligible peak has a name.
Legacy IDs 471 and 775 remain unresolved in four non-eligible rating rows. ID
775 appears across unrelated teams in 1997 and is likely a parser placeholder;
do not assign it a player name. Window fits checkpoint atomically after the
full archive audit.

**Leaders:** Three-year net peaks start with LeBron James 2009–11 (+9.19), Kevin
Garnett 2007–09 (+8.96), Nikola Jokić 2022–24 (+8.82), Stephen Curry 2016–18
(+8.77), and Kawhi Leonard 2020–22 (+8.32). Five-year net peaks start with Curry
2015–19 (+9.58), LeBron 2007–11 (+9.34), Garnett 2003–07 (+9.23), Chris Paul
2014–18 (+8.67), and Steve Nash 2007–11 (+8.16). Five-year offense is led by
Nash 2006–10 (+8.93); five-year defense is led by Garnett 2007–11 (+6.37).

**Validation:** For 1,223 players eligible in both window lengths, three-year
and five-year peaks correlate 0.963 for net, 0.964 for offense, and 0.950 for
defense. The independent three-year ratings correlate 0.842 with the legacy
three-year net panel across 16,716 matched rows. The difference is expected:
the legacy pipeline removes garbage time and adds score-margin and penalized
season terms, while this model uses all regular-season possessions and a direct
season scoring-level adjustment.

**Verdict:** Use this artifact for the first all-time peak product. Keep it
labeled normal RAPM, not AIO, because it has no statistical prior. Next build
the read-only query layer and player trajectory/decomposition schemas.

## 2026-08-10 — Read-only ratings API v1

- Added a pinned read-only query contract over the annual AIO and rolling normal
  RAPM peak artifacts. Artifact selection is explicit by run ID, never filesystem
  recency.
- Routes expose metadata/caveats, annual leaderboards, three- and five-year peak
  leaderboards, player search, and a player payload containing annual SPM center,
  RAPM update, AIO decomposition, rolling history, and component peaks.
- Added schema/key validation, deterministic ordering, pagination limits,
  possession filters, JSON null handling, HTTP 400/404 behavior, and permissive
  local CORS.
- Observable validation used the real pinned artifacts: the 2024 AIO net endpoint
  returned Nikola Jokic, Shai Gilgeous-Alexander, and Paul George at the top with
  a 1,000-possession-side filter; the five-year peak endpoint returned Stephen
  Curry, LeBron James, and Kevin Garnett. LeBron search and the full player payload
  also passed.
- This is a local-development server, not an internet-facing production server.
  Public hosting still needs a managed runtime, caching/compression, and restricted
  CORS.

## 2026-08-10 — First player trajectory product slice

- Replaced the discarded dashboard direction with one focused player page in
  `web/`. It consumes the pinned ratings API rather than copying model values
  into frontend code.
- The page shows annual AIO offense/defense/net, SPM-center plus RAPM-update
  decomposition, 3Y and 5Y normal-RAPM trajectories, component peak ranks,
  possession exposure, and the known research caveats.
- Search and quick-player controls use canonical NBA IDs. LeBron James is the
  default inspectable case.
- The Cloudflare-compatible production build and rendered product-shell test
  pass. This slice remains local: deploying only the page would be broken until
  the ratings API also has a managed public runtime.

## 2026-08-10 — Current possessions and frozen 2024–26 normal RAPM

- Re-audited the canonical current possession and ordinal lineup tables for the
  regular-season RAPM estimand. They contain 743,946 possessions across 3,681 of
  3,690 source games (99.76%). Nine games fail minute/substitution reconciliation
  and remain quarantined.
- Possession and segment keys are unique, player lineup slots have zero nulls,
  offense/defense team domains are valid, game scores and child-segment points
  conserve exactly, and there are no negative or greater-than-seven-point rows.
- Added current-box fallback names to the RAPM runner. This replaces stale legacy
  names when available and yields zero missing names across 802 fitted players.
- Frozen run `rapm_v0_01b5084f0a` uses terminal-lineup zero-prior ridge at
  3000/3000/300 over the 2023–24 through 2025–26 regular seasons. Its ratings are
  numerically identical to the earlier validated frozen run. The new run adds
  explicit input hashes, corrected research status, current names, and caveats.
- On 2025–26 observed-lineup retrodiction, margin RMSE is 15.473 and correlation
  is 0.334. Earlier-season player coverage is only 86.5%, so this remains a
  descriptive baseline, not a forecast.
- Start-versus-terminal net ratings correlate 0.971 with mean absolute difference
  0.358. This confirms that lineup policy matters; terminal remains the active
  simple contract by prior validation and user direction.
- The pinned API now exposes `/v1/leaderboards/current` and embeds current normal
  RAPM in each player payload. With at least 3,000 possessions per side, the top
  five are Shai Gilgeous-Alexander (+8.78), Nikola Jokić (+8.69), Kawhi Leonard
  (+6.79), Giannis Antetokounmpo (+6.70), and Victor Wembanyama (+5.90). Treat the
  order as a research result.

## 2026-08-10 — Current feature audit and untouched 2025 SPM confirmation

**Question:** Does the frozen 2014–24 annual SPM mapping retain its held-out
performance on the first new complete season?

**Data:** Rebuilt one-season features through 2025. Run
`statistical_features_v2_9e7c27e281` has 265 features, 6,360 player-seasons,
unique keys, and no non-finite or bounded-ratio failures. Playtype and defensive
tracking sources extend through 2025. The 2026 base sheet has only 81.8% of the
prior two-season median possession exposure and is marked partial.

**Method:** Applied saved offense and defense models from
`single_season_spm_v1_bff6060df6` to 2025 without refitting. Built the 2025
target from 1,226 regular-season games and 247,630 canonical possessions using
terminal-lineup normal RAPM at 3000/3000/300. All 569 target players match.

**Result:** Run `current_spm_confirmation_v1_9b4cca0b12` produces offense RMSE
1.102/correlation 0.619, defense 1.154/0.331, and net 1.610/0.500. Every RMSE is
worse than the worst 2017–24 held-out fold. Defense and net correlation are also
below the prior held-out range.

**Pipeline checks:** The rebuilt and frozen 2024 selected features and saved
predictions are numerically identical. Legacy and canonical 2024 annual RAPM
targets correlate 0.964–0.975. The target change adds about 0.03 RMSE on the same
2024 out-of-fold predictions, which does not explain the larger 2025 loss.

**Verdict:** Do not promote the frozen annual SPM or publish a 2025 AIO from it.
Do not tune it on 2025. Use 2025 for failure diagnosis, develop the next defense
model inside nested or forward-only 2014–24 validation, and reserve a new season
for confirmation. The historical-range rule was recorded after observing 2025;
it is a diagnostic comparison, not a predeclared gate.

## 2026-08-10 — No-tuning diagnosis of the 2025 defensive SPM miss

**Question:** Is the 2025 defensive regression caused by low exposure, unstable
RAPM targets, schema drift, or the new defensive tracking block?

**Method:** Run `current_spm_diagnostics_v1_59632783de` keeps the frozen model
unchanged. It stratifies errors by possession exposure, compares selected-feature
distributions with 2014–24, fits 2024 and 2025 first-half/second-half normal RAPM
diagnostics, and neutralizes saved defensive feature blocks at historical
medians. Neutralization is interpretation, not a retrained challenger.

**Result:** The 276 players above 2,000 possessions still have defense RMSE 1.232
and correlation 0.433. Their target standard deviation is 1.346 versus 0.649 for
predictions. At a 1,000-possession floor in each half, split-half defensive RAPM
correlation is nearly unchanged: 0.329 in 2024 and 0.331 in 2025. Neutralizing
all ten DFG/rim/hustle features worsens RMSE from 1.154 to 1.182 and correlation
from 0.331 to 0.230. Neutralizing DFG/rim alone is worse than neutralizing hustle
alone. The largest median shift is contested-threes per 100 at 1.32 historical
IQR, but its 2025 values remain inside the historical player range.

**Verdict:** The current evidence rejects two simple explanations: low-minute
noise alone and a harmful defensive-tracking block. The frozen model lacks enough
stable, calibrated defensive separation in 2025. Keep tracking. Design the next
defensive challenger under nested or forward-only validation and reserve a new
season for confirmation.

## 2026-08-10 — Nested forward defensive ridge selection

**Question:** Is the frozen defense model under-dispersed because its ridge
penalty is too strong?

**Contract:** `configs/models/annual_defense_ridge_nested_v1.json` freezes ridge
alphas 300, 1000, 3000, and 10000 before the run. For outer test seasons 2020–24,
each alpha is scored on the two immediately prior validation seasons, with model
training restricted to seasons before each validation year. The selected alpha
is then refit on all seasons before the outer test. Fixed alpha 3000 is the
baseline. The gate requires at least four of five RMSE wins, lower mean RMSE,
and noninferior mean correlation. No 2025 evidence enters selection.

**Result:** Run `annual_defense_ridge_nested_v1_5b06407982` selects alpha 300
once, 3000 once, and 10000 three times. It beats fixed 3000 on only one of five
outer folds. Selected-minus-fixed mean RMSE is +0.0015 and mean correlation is
-0.0004. The predeclared promotion gate fails.

**Verdict:** Keep fixed alpha 3000. Lower regularization does not fix the 2025
under-dispersion and is not supported by older forward folds. The next defensive
experiment must add stable signal rather than retune ridge strength.

## 2026-08-10 — Nested annual defense feature-block experiment

**Question:** Do interpretable defensive interactions improve the frozen annual
defense model under forward-only nested selection?

**Features:** Added overall DFG two-point-equivalent points saved per 100, rim
matchup-attempt share, and contested-three share to the defensive tracking
builder. Combined these with pre-existing stocks/foul, rebound-contest,
block-recovery, interior-role, and era-relative defensive features. On/off,
plus-minus, team ratings, games, minutes, age, experience, height, and position
remain forbidden.

**Contract:** `configs/models/annual_defense_features_nested_v1.json` freezes
five variants before the valid run: baseline, seven defensive interactions,
three matchup interactions, four era-relative rates, and all 14 additions. Ridge
alpha remains 3000. Each 2020–24 outer fold selects on the two prior validation
seasons. The promotion gate requires four of five RMSE wins, lower mean RMSE,
and noninferior mean correlation.

**Leakage correction:** Initial run `annual_defense_features_nested_v1_d913f807c5`
used a feature artifact built through 2025. Although outer rows ended in 2024,
global defensive-source fallback medians could include 2025. That run is invalid
for the stated contract. Rebuilt defensive and combined feature panels strictly
through 2024 before the final run.

**Result:** Valid run `annual_defense_features_nested_v1_22b677e1ef` selects the
combined block three times, defensive interactions once, and matchup interactions
once. It beats baseline in two of five outer folds. Selected-minus-baseline mean
RMSE is +0.00024 and mean correlation is -0.00274. The gate fails.

**Verdict:** Keep the frozen 60-feature defense model and the new derived fields
as research features only. Existing annual aggregates appear near their current
signal limit. The next defense experiment should start with new matchup-level or
spatial assignment data, not another subset search on 2014–24.

## 2026-08-10 — Licensed player-matchup archive ingest

**Question:** Is a new matchup-level defensive source available without a slow
NBA endpoint crawl?

**Audit:** The local Gabriel mirror contains 121 regular-season defensive
playtype files (`*d.csv`), and all 121 are header-only. Its consolidated tables
are player-season aggregates, not player-versus-defender matchups.

**Source:** Pinned revision `e829d467...` of
https://github.com/shufinskiy/nba_data. The repository declares Apache-2.0 and
publishes NBA Stats matchup archives from 2017–18 through 2024–25. Added manifest
`configs/ingest/shufinskiy_matchups_2017_2024.json` and `.tar.xz` member
validation to the resumable downloader.

**Result:** Downloaded and validated all eight regular-season archives: 30.38 MB
compressed and 1,769,658 game/offensive-player/defender rows. Per-season rows
range from 181,840 to 232,985. Every archive passes byte size, member name,
minimum row count, header width, and required field checks. Sidecars record the
actual SHA-256 values.

**Verdict:** This is the next defensive feature source. Build opponent-quality-
adjusted, sample-shrunk defender features before any new SPM comparison. Matchup
assignment is not optical tracking and must not be interpreted as sole causal
credit.

## 2026-08-10 — Opponent-adjusted matchup-defense feature test

**Question:** Does primary-defender assignment data add stable annual defensive
SPM signal beyond the frozen box, DFG, rim, and hustle model?

**Feature layer:** Run `matchup_defense_features_v1_86d13d7357` aggregates
1,769,658 source rows into 4,409 defender-seasons for 2018–25. The main feature
compares each scorer-defender pair with the scorer's leave-one-defender-out
points rate, centers the result within season, and shrinks it with 500 assigned
matchup possessions. Point reconstruction is exact. Exposure-weighted ID match
is at least 99.46%. The output has no duplicate or non-finite feature rows.

**Contract:** `configs/models/annual_defense_matchup_nested_v1.json` freezes the
baseline and four matchup blocks. It keeps ridge alpha 3000 and excludes raw
exposure. For outer seasons 2022–24, it selects a block using the two prior
seasons. The gate requires at least two RMSE wins, lower mean RMSE, and
noninferior mean correlation. Season 2025 is excluded.

**Result:** Run `annual_defense_features_nested_v1_eaeca704eb` selects the
event-context block in all three folds. It improves weighted RMSE in all three:
0.966 to 0.944 in 2022, 0.940 to 0.932 in 2023, and 0.968 to 0.959 in 2024.
Mean RMSE improves by 0.0131. Mean correlation falls by 0.0202. Prediction spread
moves closer to target spread, but player ordering becomes worse.

**Verdict:** The gate fails. Keep the canonical matchup data and feature panel,
but do not promote this block to the frozen SPM. The result is not evidence that
matchup assignments are useless. It shows that scorer-quality adjustment alone
does not remove shot-role, scheme, help, and lineup context. Add those controls
before one new predeclared test. Do not search more subsets on 2022–24.

## 2026-08-10 — Matchup-defense factor residuals

**Question:** Can the matchup panel express defensive mechanisms separately
instead of using one total points-allowed feature?

**Method:** Run `matchup_defense_features_v1_09829b48c8` adds six scorer-adjusted
factor residuals. For each scorer-defender pair, the expected event rate uses the
scorer's results against every other defender. Positive values mean the defender
suppressed attempts, saved shot-making points, suppressed threes, forced extra
turnovers, suppressed assists, or prevented shooting fouls. Each factor is
season-centered and empirical-Bayes shrunk.

**Result:** All six residuals conserve their season centers to numerical
precision. On same-season 2018–24 defensive RAPM, shot-making points saved has
correlation 0.407–0.457 in every season. Turnovers forced has 0.198–0.234.
Attempt and three-point-attempt suppression have small negative correlations.

**Verdict:** The factor layer is usable for explanations and future research.
The correlations are descriptive and the target seasons are already inspected.
Do not run a new subset promotion search on them. The negative volume result is
a warning that assignment, scheme, and shot-location context remain important.

## 2026-08-12 — Research control plane and rolling-peak eligibility repair

**Question:** Which recommendations from the supplied GPT Pro static diagnosis
survive contact with the live checkout, and is its peak-eligibility concern real?

**Evidence boundary:** The source diagnosis did not mount local artifacts or run
tests. Its numerical claims remain documented claims. The live checkout started
at commit `6c00463e`; its baseline suite passed 98 tests. The accepted governance
decisions are recorded in `research/estimands.yml`,
`research/season_exposure.yml`, and `research/pinned_artifact_audit.csv`.

**Verified bug:** The rolling-peak contract required at least 1,000 offensive and
defensive possessions in every constituent season. The implementation instead
required only 3,000 or 5,000 total-window possessions per side. This allowed a
player to miss or fall below the threshold in a constituent season.

**Repair and result:** The builder now counts offense and defense by player and
season and requires both per-season minima to meet the threshold. Rebuilt run
`rolling_rapm_peaks_v1_a8a612143c` fits the same 50 windows and 36,530 rating
rows. It removes 8,715 falsely eligible player-windows and produces 5,505
player/component peak rows. There are 1,088 eligible three-year players and 747
eligible five-year players. Net peaks correlate 0.956 among the 747 eligible in
both. Component identity error remains below `9e-16`; peak keys and names pass.

**Verdict:** Retire `rolling_rapm_peaks_v1_584adf4f3d` and pin the Ratings API to
the corrected run. Keep peaks research-only because selection uncertainty and
winner's-curse correction remain absent. Keep terminal-lineup, zero-prior normal
RAPM as the production reference. Preregister precision-aware prior work, but do
not tune it on the inspected 2025 failure or partial 2026 season. Reserve Season
2027 for one untouched annual confirmation.

## 2026-08-12 — Research control plane strengthened

**Question:** Can the pinned rating API expose an artifact without complete,
portable lineage?

**What we did:** Added a machine-readable pinned-artifact contract and a
`validate-research-control` command. The gate requires an estimand, evidence and
uncertainty status, season scope/completeness, relative artifact location, code
and configuration hashes, data-hash status, and a forbidden interpretation for
every API pin. It rejects incomplete hashes, machine-specific release paths,
research-only artifacts marked as production, and explicit Season 2027 use.

**Result:** The current four API pins pass the gate. This validates their
lineage records. It does not add uncertainty or upgrade their scientific status.

**Next:** Freeze and implement game-cluster uncertainty for normal RAPM before
publishing any rank precision.

## 2026-08-12 — Normal RAPM uncertainty: 2025 pilot complete

**Question:** Does the frozen terminal-lineup zero-prior normal RAPM have
reproducible game-level uncertainty on a complete annual slice?

**What we did:** Fit 2025 regular-season terminal-lineup normal RAPM with fixed
3000/3000/300 penalties. Resampled complete games with replacement within the
season for 1,000 deterministic, atomically checkpointed draws. Estimated a CR0
game-cluster ridge sandwich only as a diagnostic.

**Result:** The run covers 1,226 lineup-quality-passing games and 569 players.
It completed all 1,000 draws. Component identity holds to `4.4e-16`. For players
with at least 2,000 possessions on each side, median analytic/bootstrap 95%
interval-width ratios are 1.011 (offense), 1.011 (defense), and 1.009 (net).
The bootstrap remains the publication method. Only 521 players have complete
offense/defense/net coverage in every draw; players absent from a draw remain
missing rather than being treated as zero evidence.

**Next:** Let the 2022-24 three-year historical pilot finish. Do not publish
rank precision until synthetic coverage and the historical run also pass.

## 2026-08-12 — Normal RAPM uncertainty: historical 2022-24 pilot complete

**Result:** The three-year terminal-lineup normal RAPM run completed 1,000
season-stratified whole-game draws across 3,689 games, 728,269 possessions, and
818 players. Component identity holds to `4.4e-16`. For the 451 players with at
least 2,000 possessions on each side, median analytic/bootstrap 95% width ratios
are 1.004 (offense), 1.004 (defense), and 1.006 (net). There are 704 players with
complete joint draw coverage.

**Interpretation:** This passes the stated analytic/bootstrap-width diagnostic.
It does not establish empirical coverage or justify rank precision by itself.
Bootstrap intervals remain the publication method. The selection-aware rolling
peak run has started separately and remains research-only.

## 2026-08-12 — Error quantification audit for normal RAPM

**Question:** Does the uncertainty contract retain the RAPM covariance that is
needed for net ratings and rank claims?

**External check:** Squared Statistics' RAPM error analysis derives a ridge
posterior covariance, shows that offensive-defensive covariance is required for
a net interval, and demonstrates that a marginal player leader can be
statistically indistinguishable from many other players. It also warns against
extrapolating to unobserved lineups. Source: "Exercising Error: Quantifying
Statistical Tests Under RAPM (Part IV)," 2019-10-03,
https://squared2020.com/2019/10/03/exercising-error-quantifying-statistical-tests-under-rapm-part-iv/

**Decision:** Keep the analytic game-cluster ridge sandwich as a diagnostic,
not as the public method. Continue to use whole-game bootstrap draws as the
publication method, retain joint offense/defense draws before forming net and
ranks, and do not publish exact rank precision from marginal intervals.

**New automated check:** The uncertainty test suite now simulates correlated
game-level shocks around a known regularized, recentered RAPM estimand. It
checks directional 80% and 95% coverage for offense and joint net intervals.
This is a regression test of the resampling implementation, not a claim that
the simulator is an NBA data-generating process or a full calibration study.

## 2026-08-12 — Precision-aware prior promotion review

**Result:** `precision_aware_prior_rapm_v1_c4b0878571` is invalid for the
preregistered promotion gate. The run scored 2021-24, not the frozen
2018-21 selection and 2022-24 diagnostic schedule. The available three-year
cross-fitted prior/calibration history cannot provide the fourth model in the
omitted earlier seasons.

**Descriptive result:** It also loses on its non-preregistered 2023-24 slice:
mean RMSE improvement is -0.1425 points per game, correlation changes by
-0.0210, and candidate-minus-zero MSE has a paired 95% interval of
[+0.675, +7.502]. Zero-prior normal RAPM remains the reference. Do not retune
this candidate on these outcomes.

## 2026-08-13 — Stop all-time peak bootstrap; retain analytic RAPM covariance

**Decision:** Stop `rolling_peak_uncertainty_v1_af2fd07f284e` after 65 of the
planned 1,000 draws. Each draw refits all 50 historical three-year and five-year
windows. The local runtime cost is disproportionate to the current product
value. Completed checkpoints remain local and resumable, but are incomplete and
must not enter an artifact, API response, or research result.

**What remains useful:** For one fixed normal-RAPM fit, the analytic
game-cluster ridge sandwich supplies offense, defense, and joint-net standard
errors quickly. It already agrees closely with our whole-game bootstrap pilots
for high-exposure players: median 95% interval-width ratios are 1.009--1.011 in
2025 and 1.004--1.006 in 2022--24. This supports using analytic covariance as a
fast fixed-window diagnostic.

**Boundary:** Analytic covariance does not include the extra uncertainty from
choosing a player's maximum among many correlated windows, reapplying
eligibility, or interpreting ranks. Therefore rolling peak tables remain
descriptive and have no rank intervals. Justin Jacobs-style public peak tables
are suitable external descriptive comparators, not a replacement for missing
selection-aware uncertainty.

## 2026-08-13 — Pin role-context bronze inputs; do not promote them yet

**Question:** What small, current data addition can support future continuous
role research without reopening the frozen all-in-one feature search?

**What we did:** Audited the latest pinned `gabriel1200/site_Data` revision
(`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`). The existing core archive and
the repository's current aggregate-source manifest already verify, so no broad
redownload was warranted. Added a separate resumable manifest for two additive
player-season-dribble-bucket inputs: shooting by dribble count (31,376 rows,
1,692 IDs) and jump-shot dribble context (28,435 rows, 1,656 IDs). Both cover
2014–26 and have unique `(PLAYER_ID, year, dribbles)` keys.

**Data-quality finding:** the jump-shot source provides populated makes and
attempts for every year but leaves supplied percentage and team display fields
blank in 2014–24. A future silver transform must derive shares from counts and
carry explicit missingness; it must not fill those fields with zero.

**Decision:** Keep both assets bronze and research-only. Do not add them to the
frozen AIO or interpret them as a rating. The next valid use is a time-safe
role table followed by a preregistered chronological role-block experiment
against the frozen baseline.

**Follow-through:** `build-role-context-features` now produces that
research-only player-season table from counts: 6,326 rows covering 1,590 IDs
in 2014–25, with no duplicate keys or infinite values. It excludes incomplete
2026 and deliberately drops source age, games, minutes, teams, and upstream
percentage display fields.

## 2026-08-13 — Current event-source coverage contract

**Finding:** NBA Stats V3 is locally complete for 2023–25: each regular season
has 1,230 games and the playoff slices have 82, 84, and 85 games respectively.
CDN NBA is complete through 2024 but has only 60 of 85 2025 playoff games;
PBPStats ends after 2024; 2024 playoff shot-detail and matchup slices are
absent.

**Decision:** V3 is the primary raw event source for 2023–25. CDN remains a
possession-tag validation source and must use `orderNumber`, not
`actionNumber`. No model may substitute any absent source slice with zero rows
or silently join a regular-only table as playoff-complete. The documented
coverage is a data-availability claim, not a licence grant.

## 2026-08-13 — Two-part shot-defense design revised; team pilot is null

**Sol checkpoint:** The proposed primary-defender, exact-zone, exact-lineup
model is not identified by the available licensed matchup aggregates. Only
about 36--37% of matched scorer-games exactly reconcile official makes and
attempts, allocated matchup attempts exceed official shots by about 9--11%, and
87% of shooter-games allocate attempts to multiple defenders. The revised
estimand is observed defensive-unit association with shot-zone mix and make
probability. Primary-defender and causal player-defense claims are forbidden.

**Data result:** `build-shot-defense-events` produced 654,376 regular-season
shots across 3,681 games in 2023--25. It aligns official shots to V3, CDN
`orderNumber`, stable event `actionId`, and exact ordinal five-player lineups.
All output gates passed: unique shot IDs, valid identities, known zones, finite
model fields, and 99.75% exact segment coverage.

**Pilot result:** `shot_defense_team_pilot_v1_a1d8880794` compared sparse L2
logistic zone and make models with and without defense-team context on a strict
chronological 2024-season split. Combined held-out log loss improved by 0.0893%
(2.04882 to 2.04699), below the frozen 0.5% practical gate. Make Brier improved
slightly. Only 41.5% of test shots used a lineup observed in training.

**Decision:** Classify the model `research_null`. Keep the validated event panel
for future exact-guarding data, but do not bootstrap this pilot, fit individual
defender rankings, merge it into the AIO, or publish a defense leaderboard. Move
the active model task to dynamic time-decayed impact trajectories.

## 2026-08-14 — Time-decay baseline passes forward proxy checks

**Question:** Can a small, no-future-leakage smoothing rule predict the next
annual normal-RAPM observation better than the latest annual value alone?

**Method:** Built `time_decayed_trajectory_v1_4706719bfb` from 2014--24 annual
zero-prior normal-RAPM targets. For each observed player-season it applies an
exponentially decayed filter using only that season and earlier observations,
separately on offense and defense; net is their exact sum. A frozen twelve-cell
grid varied annual decay (0.50, 0.65, 0.80, 0.90) and the possession exponent
(0.0, 0.5, 1.0). Selection used 2018--21 origins to predict the next annual
target; 2022--23 origins are later diagnostics. Players need 1,000 possessions
per side in the target season. One source row with zero defensive possessions is
explicitly excluded.

**Result:** Decay 0.80 and exponent 0.0 won selection. Equal-season next-year
net RMSE improves from 1.9481 to 1.7166 in selection and from 2.0285 to 1.7758
in later diagnostics. Net correlation is 0.419 and 0.435, respectively.

**Decision:** Retain this as a research trajectory baseline. It is not a full
latent-state model, has no trajectory intervals, and uses a legacy target
archive ending in 2024. Do not interpolate missing seasons, expose it in the
API, or call it current NBA strength. A later state-space model must beat this
baseline on a frozen forward test.

## 2026-08-14 — Expected-possession residual RAPM: causal state is ready

**Question:** Can normal RAPM be challenged with actual-minus-expected possession
points, without leaking the current possession or fitting away player effects?

**Data result:** `build-possession-start-context` created
`possession_start_context_ebcae214e662d404`: 787,579 canonical possessions
across 3,907 games. Start scores are reconstructed from completed prior canonical
possessions, rather than an incomplete cross-source action-number join. The
contract provides only period/time, score differential, home/away side, and
prior-possession context; it forbids player/team/lineup identity and all current
possession actions or outcomes as expected-points inputs.

**Decision:** The data contract is ready. Do not fit residual RAPM until expected
points are cross-fitted chronologically and player-neutral. The first model is a
simple Poisson or multinomial baseline; compare residual RAPM with normal RAPM
on identical games using equal-season margin RMSE and paired whole-game error.

## 2026-08-14 — Possession-start expected points is a useful null

**Method:** `expected_possession_points_v1_c9581a23b1` fits a player-neutral
Poisson expected-points model using only the frozen possession-start fields. It
uses whole-season chronological cross-fitting: train 2023 and test 2024, then
train 2023--24 and test 2025. No player, team, lineup, current-possession event,
or current-possession outcome is an input. The run yields 497,177 out-of-fold
regular-season predictions over 2,454 games.

**Result:** Context beats a training-mean constant in both folds, but only
slightly. 2024 RMSE is 1.19556 versus 1.19587 and Poisson deviance is 1.63425
versus 1.63493; 2025 RMSE is 1.19258 versus 1.19306 and deviance is 1.61978
versus 1.62078. Mean context improvement is 0.00039 RMSE and 0.00084 deviance.
Mean bias remains under 0.01 points per possession and both fits converge.

**Decision:** Do not fit residual RAPM. The contextual change is too close to a
constant subtraction to justify another RAPM evaluation. A prospective reopening
gate now requires at least 0.25% mean Poisson-deviance improvement in both
chronological folds, small bias, and player-neutral out-of-fold predictions. New
causal context—not extra model complexity—is needed before retrying.

## 2026-08-14 — Canonical annual normal RAPM extends the trajectory safely

**Question:** Can the current canonical possession/ordinal-lineup source extend
the annual normal-RAPM trajectory without silently mixing incompatible ratings?

**Method:** Built `current_single_season_rapm_targets_v1_9c0cdda919` for the
2023--24, 2024--25, and 2025--26 regular seasons (ending labels 2024--26). Each
season uses terminal lineups, a zero prior, and frozen 3000/3000/300 penalties.
The fits cover 1,227, 1,226, and 1,228 lineup-quality-passing games. Before the
join, `canonical_annual_target_panel_v1_2d9ff74ca3` checked the shared 2024
season against the legacy target. It explicitly excluded one legacy zero-
defensive-possession row, then applied an explicit 0.95 component correlation and
coverage, a 0.80--1.25 scale ratio, and a mean difference no larger than 0.10
points per 100.

**Result:** The 571 matched 2024 players pass all gates. Canonical-versus-legacy
Pearson correlation is 0.974 offense, 0.964 defense, and 0.975 net. Net RMSE is
0.377 points per 100, net mean difference is 0.010, and the net scale ratio is
1.019. The combined panel has 6,942 player-seasons from 2014--26. Applying the
already-selected 0.80 time-decay filter in `time_decayed_trajectory_v1_8ed684a8aa`
does not retune it. The historical selection remains 1.9481 to 1.7166 net RMSE;
the later diagnostic remains favorable at 2.0549 to 1.8086.

**Decision:** Retain the updated trajectory as a research-only dynamic baseline.
The 2025--26 endpoint is descriptive. It does not use Season 2027 or create new
confirmation evidence. The next model task is a separately reviewed,
side-specific precision-aware SPM-prior challenger, not more trajectory tuning.

## 2026-08-14 — Sol review revises the precision-aware AIO contract

**Question:** Is the proposed side-specific empirical-Bayes SPM prior coherent
enough to run as the next normal-RAPM challenger?

**Sol review:** `lambda = sigma_squared / tau_squared` is coherent only when
the RAPM likelihood, SPM center, residual scale, and prior variance all use the
same coefficient units and unweighted possession-SSE objective. The original
pooled calculation of `tau_squared` was not sufficient: game-cluster RAPM label
variance differs by player-window and is not exact classical measurement error.

**Revision:** `PRECISION_AWARE_PRIOR.md` and the experiment contract now require
a heteroskedastic profile likelihood for earlier cross-fitted residuals,
`RAPM-minus-SPM ~ Normal(mean, tau_squared + label_variance_proxy)`. The
candidate replaces the player-side 3000 penalty with the resulting MAP penalty;
the home penalty stays 300. A zero-boundary or failed precision estimate
invalidates the candidate. The training zero-prior residual MSE supplies
`sigma_squared`; there is no per-100 conversion or amplitude grid.

**Feasibility audit:** The frozen three-season statistical feature panel begins
at window end 2016. Its non-overlap purge makes 2019 the first cross-fitted SPM
prior. Since scored season `Y` uses the prior ending `Y - 1`, three strictly
earlier calibration windows first exist only for `Y = 2023`. The frozen
2018--21 selection schedule is therefore impossible under this feature
contract. The prior 2021--24 run is invalid for promotion and is not rerun.

**Decision:** Do not inspect or tune another AIO result. Do not weaken the
calibration horizon to force an earlier score. A new pre-2016 prior contract is
a separate research design; otherwise retain the schedule block and move to the
next controlled research lane.

## 2026-08-14 — Annual state-space trajectory beats frozen time decay

**Question:** Does a small causal state-space filter improve next-year annual
normal-RAPM proxy prediction over the frozen 0.80 time-decay baseline?

**Method:** Built `annual_rapm_observation_variance_v1_03f6a17336` by deriving
side-specific CR0 game-cluster ridge covariance diagnostics for every annual
normal-RAPM target from 2014--26. The result has 6,942 player-seasons, no
duplicate keys, and each newly computed annual fit reproduces its frozen
offense/defense point estimate within 9e-16. `annual_state_space_trajectory_v1_f150bcde08`
then fit causal side-specific AR(1) filters. The 16-cell `phi`/process-SD grid
selected `phi=0.90`, process SD 0.25 using only 2018--21 origins. Offense plus
defense equals net in every row. The challenger and frozen 0.80 time decay are
scored on identical, at-least-1,000-per-side-possession player-season rows.

**Result:** State space reduces mean next-year net RMSE from 1.7166 to 1.5917
in selection (-0.1249) and from 1.8086 to 1.7031 in the later 2022--23
diagnostic (-0.1055). It wins every individual origin. Mean net correlation is
0.427 in selection and 0.440 in diagnostics. The annual-latest baseline is
weaker still (1.9481 selection and 2.0549 diagnostic RMSE).

**Decision:** Keep state space as the leading dynamic-impact research
challenger and time decay as the frozen baseline. Do not tune after this result,
publish latent-strength claims, expose it in the API, or use Season 2027. The
observation covariance is a fast ridge diagnostic, not public rating
uncertainty; confirmation needs a separately approved untouched annual season.

## 2026-08-17 — AIO diagnosis identifies defense and skill measurement as the next bottlenecks

**Question:** What does the current all-in-one measure, which public-model ideas
are missing, and what feature work is justified before another model search?

**Audit:** The frozen three-season AIO uses 162 offense features in a bounded
histogram GBM and 50 defense features in ridge. Targets are purged three-season
normal RAPM offense and defense. Run
`statistical_interpretability_v1_94d3f2c24b` refit the frozen specification on
windows ending by 2021 and used grouped permutation on the reused 2024 fold.
Baseline offense RMSE/correlation is 0.82701/0.62133; defense is
0.89982/0.36705. Permuting shooting/scoring raises offense RMSE by 0.27185 and
permuting public composites raises it by 0.10815. On defense, disruption,
creation/role, and rebounding groups raise RMSE by 0.11673, 0.08159, and
0.07927. The role result is a warning that the rolling defense model still uses
indirect proxies. Permutation is model reliance, not causal player credit.

**Feature build:** `player_skill_features_v1_cf800d4e7e` creates 12 annual
opportunity-adjusted features from ID-based defender-distance shooting,
passing, hustle, and shot-zone tables. It separates shot difficulty from
shot-making above expectation and adds pass value, high-value assist share,
bad-pass cost, screen value, deflections, charges, defensive boxouts, and loose
balls. The artifact has 5,791 unique player-seasons from 2014--24, 1,499
players, and no infinite values. Eleven fields are model candidates. Absolute
shot difficulty is audit-only because the season profile found a 1.60-IQR
median shift from 2018 to 2019; the era-relative version is the candidate.
The partial 2025 snapshot is excluded. Hustle coverage begins in 2018 and
missing history remains missing. Integrations `statistical_features_v2_d67bb64ac7`
and `statistical_features_v2_2515b57958` add the 11 candidate fields to rolling and annual
tables with unique keys, no infinite values, and no missing values after
season-neutral imputation. No AIO model was fit or selected.

**Public research:** LEBRON/PIPM, RAPTOR, current EPM, xRAPM, old ESPN RPM, BPM
2.0, MAMBA, DARKO, BBall Index skill models, ESPN Net Points, and Six-Factor
RAPM were compared in
`docs/impact/AIO_DIAGNOSIS_AND_FEATURE_BLUEPRINT.md`. Public disclosures support
the same direction: stat-specific stabilization, expected-versus-actual skill,
role-relative evaluation, and separate retrospective versus predictive
contracts. They do not disclose enough detail for an exact clone.

**Decision:** Do not change normal RAPM. Do not search more subsets on 2022--25.
Integrate the validated skill layer, then build aging-residualized forward and
reverse diagnostics. Direct offense and defense RAPM remain the supervised
targets. Factor RAPM and role-fit counterfactuals remain separate research
branches. Season 2027 stays untouched.

## 2026-08-17 — Forward/reverse testing does not show a simple youth-proxy win

**Question:** Does next-year validation make the frozen annual SPM look better
mainly because its inputs proxy the direction of aging?

**Method:** `aging_balanced_validation_v1_ec5122d5a3` joins the frozen annual
OOF SPM at season `T` to annual normal RAPM at `T+1` and `T-1`. For each scored
origin, a fixed-knot ridge age curve is fit only on earlier origin seasons. The
curve predicts adjacent-minus-current RAPM and is removed from the adjacent
target. Age never enters the SPM. Four scored origins contain 1,768 matched
transitions in each direction.

**Result:** Raw forward net RMSE/correlation is 1.6277/0.4095. Raw reverse is
1.5860/0.4046. Forward age adjustment changes this to 1.6380/0.4399; reverse
changes to 1.5660/0.3837. Offense and defense show the same mixed pattern: the
adjustment can improve correlation while worsening scale error, or the reverse.

**Decision:** The similar raw forward and reverse correlations do not support a
simple claim that the current SPM wins only by encoding youth. Keep both
directions and the earlier-only adjustment as diagnostics. Do not replace the
target, add age to retrospective SPM, or tune the spline on inspected seasons.

## 2026-08-17 — Behavior-only roles pass frozen stability gates

**Question:** Can the AIO represent player role without age, size, listed
position, opportunity totals, efficiency, impact, or team-outcome inputs?

**Method:** `behavior_roles_v1_e0fb51c026` uses 32 season-relative behavior
descriptors covering shot allocation, creation, passing, interior activity,
rebounding activity, and dribble context. It fits six deterministic PCA axes
and eight K-means clusters on 2014--18 only, then applies the map unchanged
through 2024. A player-season needs at least 80% observed descriptors. Frozen
gates require at least 90% row coverage, 0.90 seed adjusted Rand, 50% adjacent-
season exact-role persistence, 0.75 adjacent-axis cosine, and later-period
cluster shares between 2% and 30%.

**Result:** All gates pass. Coverage is 93.71% over 5,427 eligible rows. The six
axes explain 83.18% of development variance. Median seed adjusted Rand is
0.9845. Across 2,397 out-of-sample adjacent player-season pairs, exact role persists 61.66%
and median axis cosine is 0.9149. Later-period cluster shares range from 6.27%
to 18.93%.

**Integration:** Continuous axes and seven drop-one soft affinities enter
rolling table `statistical_features_v2_2bb78bc737` and annual table
`statistical_features_v2_d8dd1d8dc2`. The tables have 6,689 player-windows and
5,791 player-seasons, 270 feature columns, unique keys, and no infinite or
bounded-value failures. No impact model was fit or selected.

**Decision:** Use axes and affinities as research candidates. Keep the hard
cluster descriptive. Do not call clusters positions, value, talent, or causal
role fit. The next controlled task is a frozen factor-group and role-interaction
AIO contract, not a role-counterfactual leaderboard.

## 2026-08-17 — Split roles pass stability, but defense improves without them

**Question:** Do separate offense and defense deployment roles improve the
annual defense SPM, and can scorer-adjusted matchup factors improve the weak
defense target?

**Method:** `side_roles_v1_2c228f4b9e` fits offense behavior on 2014--18 and
defense deployment on 2018--21. Cluster count is selected without RAPM from
silhouette, seed stability, and later cluster shares. The defense challenger
uses fixed ridge alpha 3000 and selects a predeclared block only on 2020 and
2021. It reports 2022--24 only after selection. Five newly exposed matchup
activity fields are constant zero in all source archives and are excluded.

**Result:** The role maps select six offense and five defense clusters. Seed
ARI is 0.9809 and 0.9910. Adjacent exact-role persistence is 71.30% and 70.44%.
The eight-feature scorer-adjusted matchup block wins both selection seasons.
It also wins all three diagnostic seasons. Mean diagnostic RMSE changes by
-0.0392 and correlation by +0.0601. Defense roles alone, roles plus matchup,
and role-conditioned interactions all lose to the matchup-only block on the
fixed selection seasons.

**Decision:** Keep separate roles as descriptive research outputs. Do not add
them to the first defense AIO challenger. Carry the eight matchup features into
the rolling three-season factor contract after correct count-level pooling.
The result is research-only because the feature family and 2022--24 are already
inspected. Keep Season 2027 untouched.

## 2026-08-17 — Full annual SPM and decomposed AIO are complete

**Question:** Does the selected scorer-adjusted matchup block improve the full
annual defense SPM, and can its held-out predictions safely center historical
one-season RAPM?

**Method:** Run `single_season_spm_v1_18496a1348` keeps the frozen offense
histogram GBM and defense ridge. It trains on annual player rows from 2014--24,
holds out each scored 2017--24 season in turn, and refits the final leaderboard
on all labels. Run `annual_spm_oof_priors_v1_7810d88ec3` converts only held-out
predictions into historical SPM centers. Run `annual_aio_ratings_v1_b52b5aecd9`
fits fixed 3000/3000/300 one-season RAPM around those centers.

**Result:** Defense weighted RMSE/correlation is 0.9210/0.5526 versus the prior
0.9595/0.4964. Defense RMSE improves in all eight folds. Net is 1.3556/0.6219
versus 1.3859/0.5991 and wins RMSE in seven of eight folds. The AIO contains
4,341 player-seasons, 100% prior coverage, no duplicate keys, and component
identity error below `9e-16`. The 2024 cache remains one regular-season game
short. High-exposure defense correlation improves against xRAPM but declines
against BPM.

**Decision:** Publish the result as a retrospective research leaderboard. Keep
zero-prior normal RAPM as the reference. Do not claim untouched promotion.
Do not attach SPM or AIO intervals until they are calibrated. The UI may show
existing RAPM intervals only at their exact 2022--24 and 2025 scopes.

## 2026-08-17 — Role stabilization and lazy ratings product

**Question:** Can role labels become less noisy without using impact outcomes,
and can the public client expose ratings and roles without a heavy first load?

**Method:** Run `role_stabilization_v1_f5b426dd5d` selects the current-season
weight that best predicts next-season raw role affinities on the original role
development seasons. It applies the frozen forward filter to later seasons and
resets after gaps. The web snapshot now splits annual leaderboards and role maps
by season. Player detail remains sharded.

**Result:** Both sides select a 0.70 current-season weight. Later exact-role
persistence improves from 71.30% to 79.89% on offense and from 70.44% to 76.97%
on defense. Stable/raw disagreement is 7.55% and 6.53%. The first ratings view
loads only the player index and 12 KB catalog. Season files are 0.33--0.43 MB.
Every table, role map, and pruned player-detail shard loads only when requested.

**Decision:** Default the product to stable roles, with a raw toggle. Keep role
stabilization out of SPM. Put the RAPM level curve and RAPM/AIO year-over-year
change under Research. Treat both aging summaries as descriptive.

## 2026-08-18 — Historical ESPN player-game subset is valid but incomplete

**Question:** Can local sources produce a strict 2017--23 player-game table
with team identities, starters, and minutes without a new download?

**Method:** Reconcile the pinned ESPN player-box mirror to official game IDs and
scores, then use pinned V3 play-by-play only to verify each team ID and expected
team minutes from the actual number of periods. Require one player-game row,
five starters, two home/away teams, valid minutes, and no more than five seconds
of team-minute error. Preserve every rejected game in a separate quality ledger.

**Result:** The strict subset accepts 5,611 games and 118,953 player-game rows:
2019 1,230; 2020 906; 2021 1,093; 2022 1,193; and 2023 1,189. The mirror has no
2017 or 2018 rows and lacks 171 official 2020 games. It does not support a
complete 2017--23 table. The output is separate from canonical `player_games`.

**Decision:** Use the new historical ESPN builder only for research inputs that
can tolerate explicit coverage gaps. Do not use its rows to infer historical
lineups or to replace the canonical player-game table. When a locally cached
official BoxScoreTraditionalV3 JSON is available, it takes priority over ESPN;
an invalid official box rejects the game rather than falling back silently.

## 2026-08-17 — Canonical annual RAPM bridge passes; 2025 SPM still fails

**Question:** Can the current event and lineup pipeline replace the legacy
annual RAPM labels without changing the estimand, and does that repair the weak
2025 SPM result?

**Method:** Run `canonical_annual_target_panel_v1_4586bd2f72` joins the legacy
2014--24 labels to canonical terminal-lineup, zero-prior 3000/3000/300 labels
for 2024--26. The transition gate checks 2024 player coverage, correlation,
standard-deviation ratio, and mean shift separately for offense, defense, and
net. Run `single_season_spm_v1_c4be58c72e` then repeats the fixed annual SPM
with 2014--25 labels, the same 127 offense and 60 defense features, and
leave-one-season-out scoring for 2017--25. The 2025 BPM and xRAPM pages were
downloaded and hashed before the run.

**Result:** The 2024 bridge passes every gate. Legacy/canonical correlations are
0.9740 offense, 0.9639 defense, and 0.9748 net; coverage is at least 99.8% and
scale ratios are 1.019--1.029. The refreshed 2017--24 mean RMSE changes from
0.9972/0.9595/1.3859 to 0.9993/0.9643/1.3924 for offense/defense/net. The 2025
fold remains weak at correlation 0.6140/0.3336/0.5030 and RMSE
1.1049/1.1527/1.6039.

**Decision:** Accept the canonical target panel as the research transition
path. Do not replace the public 2017--24 leaderboard. The label migration does
not explain the 2025 failure, so the next model work must diagnose feature and
defensive-target drift instead of refitting the same specification again.

## 2026-08-18 — Full 2026 data repairs exposure, not defensive SPM

**Question:** Was the weak current SPM result caused by the partial 2026 player
sheet, and does a full 2014--26 refit improve the frozen public model?

**Method:** Pin 2025/2026 player sheets at Gabriel revision `a86cbe4`. Rebuild
one-season base, playtype, defensive tracking, player-skill, and expanded
features. Run `single_season_spm_v1_47b3bd9b17` with the same histogram-GBM
offense model, ridge defense model, fixed 127/68 inputs, and leave-one-season-
out evaluation. Compare only the common 2017--24 folds to the public annual
SPM. Also audit clean possession-lineup coverage and download only the 10
Gabriel team files needed for quarantined-game repair.

**Result:** The 2026 player sheet has 582 unique players and an exposure ratio
of 1.005 versus the 2024--25 median. The 6,942-row base and expanded panels
have no duplicate keys or invalid bounded values. On common 2017--24 folds,
net RMSE/correlation changes from 1.3556/0.6219 to 1.3591/0.6206. Defense
changes from 0.9210/0.5526 to 0.9267/0.5475. The 2025 and 2026 defense
correlations are 0.3322 and 0.3782. The clean RAPM input covers 1,227/1,230,
1,226/1,230, and 1,228/1,230 regular games in 2024--26. The 2026 playoff CDN
source still stops at 60/85 games.

**Decision:** The partial 2026 sheet was not the main problem. Keep the new SPM
as a null result and keep public SPM/AIO at 2017--24. Publish Normal RAPM through
2026. Repair the 10 targeted lineup failures without weakening QA. Current
defense needs new information, especially 2026 DFG and scorer-matchup coverage,
not another identical refit.

## 2026-08-18 — Strict Gabriel repair closes one game

**Method:** Use canonical CDN possession ownership and event order. Join the
Gabriel on-court states through canonical player-game teams. Require final-score
conservation, ten unique players per segment, unique keys, and player-minute
errors no larger than five seconds.

**Result:** Game `0022300535` passes with 185 possessions and 219 lineup
segments. Nine targets fail without guessed lineups. The repaired 2024 fit has
1,228 regular-season games. On 572 common players, net correlation versus the
prior fit is 0.99975, mean absolute change is 0.0079 points per 100, and maximum
absolute change is 0.4982.

**Decision:** Keep the one passing repair. Keep all nine failures quarantined.
Use `current_single_season_rapm_targets_v1_b4cdb51de8` for the current Normal
RAPM snapshot. Better source data, not looser QA, is the next repair path.

## 2026-08-18 — Official Live source closes the 2026 playoff tail

**Question:** Can the 25-game 2026 playoff cutoff be repaired without inferring
possession ownership or weakening lineup-minute QA?

**Method:** Download official NBA Live JSON for all 85 playoff games. Compare
the 60-game overlap event by event with the pinned CDN archive. Build the same
CDN schema, rerun the normal lineup and possession builders, then add only
strictly passing regular-season repairs to a separate integrated table.

**Result:** All 34,579 overlapping actions match on event key, action number,
period, clock, possession owner, score, and action type. The 85-game output has
49,727 actions, 16,648 possessions, and 20,639 lineup segments. All 85 games
pass official-minute, score, player-count, and point-conservation gates. The
integrated regular-season coverage is 1,229 / 1,230 in 2024, 1,227 / 1,230 in
2025, and 1,228 / 1,230 in 2026.

**Decision:** Accept the official Live completion as the current 2026 playoff
source. Use Normal RAPM run `current_single_season_rapm_targets_v1_8f2a6f2e0a`.
Do not rerun SPM/AIO selection: the added regular games move net ratings by only
0.0057 and 0.0071 mean absolute points per 100 in 2024 and 2025.

## 2026-08-18 — Official Live does not close the six regular quarantines

**Question:** Can a fresh copy of the official NBA Live actions repair the six
remaining 2024--26 regular-season lineup failures?

**Method:** Download and validate the current official Live JSON for the exact
six games. Rebuild scoped CDN-schema partitions, then rerun the unchanged
starter, substitution, five-player, official-minute, and five-second gates.
Probe one midpoint regular game per project season from 2017 through 2023 to
bound whether the same source can support a historical backfill.

**Result:** The six files contain 3,477 valid ordered actions, but zero games
pass. Maximum official-minute errors are 156.1, 115.0, 611.2, 25.0, 93.0, and
6.9 seconds; three games also retain substitution-transition errors. The
sampled 2020--23 games return nonempty Live actions, while sampled 2017--19
games return HTTP 403.

**Decision:** Keep all six games quarantined and keep the published current
Normal RAPM unchanged. Do not spend bandwidth on a historical Live sweep until
historical starter and official player-minute inputs are pinned. The next
repair must improve lineup evidence or validate a new ordinal reconstruction,
not repeat the same clock-based replay.

## 2026-08-18 — V3 possession inference clears its frozen validation gate

**Question:** Can the complete local V3 action archive supply defensible
historical possession owners when the retired PlayByPlayV2 endpoint returns no
data and pre-2024 CDN possession tags are unavailable?

**Method:** Order primary V3 actions by `actionId`. Infer owner changes from
shots, free throws, turnovers, rebounds, fouls, and jump balls. Fix the rules on
project season 2024, then apply them unchanged to project season 2025. Compare
to CDN possession owners by the guarded action-number key. Build action points
from made-shot and made-free-throw descriptions and require exact official
team-score conservation.

**Result:** Core action-owner agreement is 99.934% in 2024 and 99.932% in the
untouched 2025 validation. Exact full-game owner sequences are 93.577% and
91.870%; possession counts are within two for 99.756% and 99.106% of games.
Mean count bias is -0.056 and -0.111 possessions per game. All frozen gates
pass. The complete 2017--2023 candidate accepts 8,863 games and 1,768,472
possessions. Eight 2017 regular-season games fail exact team-score conservation
and remain rejected; every other regular-season and playoff partition passes.

**Decision:** Build a separate historical V3 possession candidate. Do not call
it exact ground truth or RAPM-ready. Promotion requires ordinal ten-player
lineups, official-minute reconciliation, and a matched comparison with the
independent legacy terminal-lineup migration.

**Playoff check:** The unchanged rules pass the 2025 playoff gate. The 2024
check has 99.98% core action-owner agreement and 92.7% exact full-game
sequences, but only 80 of 82 games are within two possessions. Games
`0042300134` and `0042300163` fail that gate. Keep historical playoffs
research-only and exclude them from the first Normal RAPM fit.

## 2026-08-18 — Matched historical V3 versus legacy RAPM (superseded)

**Question:** Do the historical V3 possession and terminal-lineup candidate and
the legacy terminal-lineup cache produce materially different RAPM ratings on
the same regular-season games?

**Method:** Restrict both sources to the exact intersection of `002` regular-
season game IDs in 2018–23. Use the V3 terminal ordinal segment for each
possession. Fit each source separately with the frozen zero-prior ridge
penalties 3000/3000/300. Compare matched-player offense, defense, and net
correlations, RMSE, and Spearman rank correlations. Also report a chronological
20% within-season held-out game-margin retrodiction.

**Result:** Every V3 game was present in the legacy cache. Net-rating Pearson
correlations were 0.971–0.982 and net-rating RMSE was 0.233–0.348 points per
100. Candidate possession counts and point totals were consistently higher on
the matched games. Held-out metrics are source-specific retrodictions using
observed test lineups, not forecasts.

**Decision:** Superseded by the corrected 2017--23 run below. Keep this as a
provenance record only. The sources have similar player ordering but are not
interchangeable at the possession level. V3 must
still pass historical ordinal-lineup and official-minute reconciliation before
it can replace the legacy research cache. Artifact:
`artifacts/research/historical_matched_rapm/historical_matched_rapm_v1_f49c0fdc102e`.

## 2026-08-18 — Corrected 2017--23 matched historical RAPM

**Question:** Does the source-compatibility result survive the Unicode name fix,
the validated 2017 starter rule, and the final seven-season strict lineup build?

**Method:** Add 2017 to the frozen comparison. Restrict both sources to identical
regular-season game IDs. Fit the same terminal-lineup, zero-prior
`3000/3000/300` ridge separately to each source. Persist one chronological 20%
holdout split and score both sources against the same official final margin.
Hash the comparison code, RAPM code, official scores, source tables, and QA
ledgers. Repeat rating agreement at 500, 1,000, and 2,000 possessions per side.

**Result:** The V3 candidate accepts 7,136 matched games and 1,425,380
possessions. Net-rating Pearson correlation versus legacy is 0.971--0.981;
net-rating RMSE is 0.298--0.350 points per 100. At 2,000 possessions on each
side in both sources, net Pearson remains 0.970--0.982. V3 reconstructs official
held-out margins more accurately, with 0.60--0.97 points RMSE versus
1.32--1.97 for legacy. V3 has lower held-out prediction RMSE in two of seven
seasons. Its mean prediction RMSE is 0.085 points per game higher, mean MAE is
0.059 higher, and mean margin correlation is 0.013 lower.

**Decision:** The corrected run passes a narrow source-compatibility check but
does not support a predictive promotion. Keep V3 research-only until the full
official player-game cache is complete and the official-preferred rebuild
reproduces the accepted-game and comparison gates. Artifact:
`artifacts/research/historical_matched_rapm/historical_matched_rapm_v1_9fb68e0fd785`.

## 2026-08-18 — Official-box reproducibility finds one exact identity gap

**Question:** Does replacing the ESPN fallback with official player boxes keep
the strict historical lineup accepted-game set stable?

**Method:** Rebuild official-preferred player games for the complete 2017--20
cache. Re-run the frozen 2019 and 2020 regular-season lineup contracts. Compare
accepted game IDs and inspect every newly unresolved substitution.

**Result:** The player-game builder accepted 5,075/5,075 games. Season 2020
reproduced 949 lineup passes. Season 2019 initially fell from 1,139 to 1,102.
All 103 new substitution parse failures across 39 lost games were player ID
2403: official boxes use `Nene Hilario`, structured V3 names use `Hilario`, and
substitution text uses `Nene`. Adding exact same-game/team event aliases and the
versioned identity alias `2403 -> nene` raised 2019 to 1,141 passes. The
possession-lineup adapter emitted all 1,141 games, 230,905 possessions, and
279,555 segments with zero attachment rejects.

**Decision:** Keep the exact alias fix. Do not add fuzzy or general first-name
matching. The full 2017--23 official-preferred rebuild remains the final
reproducibility gate.

## 2026-08-18 — Official-box DNP rows cannot be substitution aliases

**Question:** Why did the official-only 2021 lineup rebuild lose five games
that passed with the earlier mixed player-game source?

**Method:** Compare the accepted game IDs, inspect every new substitution parse
failure, and reconcile the ambiguous names to official minutes and structured
V3 identities.

**Result:** All eight new failures were surname collisions caused by zero-minute
DNP rows retained in official boxes: Grant/Robert Williams or Moses/Charlie
Brown. Only the positive-minute player could have entered the observed game.
Excluding zero-minute rows from the substitution alias map restored 892/1,080
regular-season passes, exactly matching the prior accepted set. The attachment
stage emitted all 892 games, 179,181 possessions, and 216,916 segments with
zero rejects, duplicate keys, or invalid ten-player segments.

**Decision:** Keep positive official minutes as an exact substitution-alias
eligibility rule. This is not fuzzy matching and does not relax the five-second
minute reconciliation gate. Retain zero-minute players in the underlying
official player-game ledger for provenance.
# 2026-08-18 — Complete official-box historical rebuild and matched source audit

**Question.** Does a complete official 2017–23 starter/minute cache improve
historical V3 RAPM readiness?

**Method.** Downloaded 8,871 `BoxScoreTraditionalV3` JSON responses with
resumable retries, atomic writes, and hash validation; rebuilt official-only
player-games, strict V3 `actionId` lineups, and possession attachments; fit
seven frozen terminal-lineup, zero-prior RAPM models; then compared each to the
legacy source on the identical official-margin holdout games.

**Result.** All 8,871 official boxes passed. The rebuild emitted 7,250 regular
season games, 1,448,146 possessions, and 1,756,230 lineup segments, with zero
attachment rejects. Net-rating Pearson agreement with legacy was .971–.981 and
RMSE .299–.355 points per 100. The V3 parser reconstructed official final
margins better in all seven seasons (mean 0.816 lower RMSE), but V3 RAPM won
held-out prediction RMSE in only two seasons: mean delta was +.080 RMSE,
.050 MAE, and −.012 correlation versus legacy.

**Decision.** The official source spine is complete and is a compatibility
proof, not a public-source replacement. Keep V3 historical RAPM research-only;
legacy remains the public historical reference because the frozen prediction
gate did not pass.

## 2026-08-18 — Unified 2014--26 annual timeline and chronological SPM windows

**Question.** Can one audited terminal-lineup interface produce annual RAPM,
SPM, and AIO for every 2014--26 season, and how much history should SPM use?

**Method.** Used the verified annual source transition: legacy score-conserved
terminal possessions for 2014--23 and canonical event terminal lineups for
2024--26. Refit the zero-prior RAPM side of the AIO for each season, trained
leave-one-season-out retrospective SPM across all 13 seasons, and fit the
centered AIO with those OOF priors. Then compared strictly earlier expanding,
one-year, three-year, and five-year SPM training histories on identical
2017--26 player-season targets.

**Result.** The unified artifact has 6,942 player-seasons, 1,706 players, full
prior coverage, no duplicate keys, no missing names, and exact component
identities. Its zero-prior components reproduce the audited annual target panel
exactly. Expanding history has the best mean chronological RMSE: offense
1.0253, defense .9908, net 1.4307. Five-year is close but has worse RMSE on all
three components; one-year is worst.

**Decision.** Use expanding-history SPM for this research timeline and retain
five-year as a sensitivity. Keep 2025--26 SPM/AIO research-only because the
current defensive feature families remain incomplete and the pre-existing
current-season validation gate failed. Full detail:
`docs/impact/UNIFIED_TIMELINE_2014_2026.md`.

## 2026-08-19 — Matchup Elo-scale v1

**Question.** Can the existing scorer-versus-listed-defender matchup rows give
one simple offensive score and one simple defensive score before a more complex
matchup model is considered?

**Method.** Fit a separate regularized two-way log-rate model for each regular
season from 2018--26. The only inputs are scorer ID, listed defender ID,
assigned partial possessions, and player points. The fitted equation is
`log(PPP scorer,defender / league PPP) = offense scorer - defense defender`.
Scores use an Elo display scale centered at 1500 on each side. This is a static
Elo-scale transformation, not sequential Elo.

**Result.** The run emitted 4,991 player-seasons for 1,394 players from nine
source seasons and passed row identity, non-negative exposure, source-order,
and synthetic recovery tests. The 2026 offensive top five were Giannis
Antetokounmpo, Shai Gilgeous-Alexander, Luka Doncic, Kawhi Leonard, and Joel
Embiid. The defensive ordering contains clear matchup-assignment and context
signals, including implausible-looking high ranks for some players.

**Decision.** Retain `matchup_elo_v1_09b1ed8860` as a descriptive research
artifact only. Do not add it to RAPM, SPM, AIO, or the public site. Before any
predictive test, define a chronological holdout and compare it with the
existing scorer-adjusted matchup-feature baseline on identical rows. Details:
`docs/impact/MATCHUP_ELO_V1.md`.

## 2026-08-19 — Annual SPM versus BPM, xRAPM, and BoxPIPM-style baseline

**Question.** On identical player-season rows, do the current annual SPM, BPM,
xRAPM, and a transparent BoxPIPM-style baseline predict annual zero-prior RAPM
best?

**Method.** Used the pinned 2017--24 SPM OOF table. Restricted every comparison
to the same 2,860 player-seasons with at least 1,000 offensive and defensive
RAPM possessions and non-missing SPM, BPM, and xRAPM. Scored SPM and the
BoxPIPM-style baseline natively. For BPM and xRAPM, fit a component-specific
affine scale only on the other seven seasons before scoring the held-out season.
The BoxPIPM-style baseline is a LOSO ridge using only 15 traditional per-100 box
rates; it is not full PIPM.

**Result.** Mean held-out net RAPM scores were: BoxPIPM-style RMSE 1.5585,
correlation .5453; BPM 1.4944, .5792; SPM 1.4003, .6483; xRAPM 1.1080, .7936.
SPM also beat BPM and BoxPIPM-style for offense and defense. xRAPM was strongest
for all three components. It is not a clean box-only winner because it contains
an adjusted-plus-minus prior and multiple information windows.

**Decision.** BPM is the primary external box-model comparator. SPM has a real
matched-row advantage over it. Keep xRAPM as a stronger but non-independent
impact comparator. Keep BoxPIPM-style as a documented baseline and do not call
it PIPM. Details: `docs/impact/BOX_PIPM_STYLE_V1.md`.

## 2026-08-19 — Time-decayed matchup Elo challenger

**Question.** Does a fixed three-season, time-decayed scorer/listed-defender
rate model provide a stable alternative to the annual matchup Elo display?

**Method.** Fitted trailing three-season windows ending 2020--26. Each row was
weighted by assigned partial possessions times `0.70^(rating season - source
season)`. Each source season was normalized by its own league matchup scoring
rate before fitting the shared two-way ridge model.

**Result.** The run emitted 5,472 player-seasons across seven complete windows,
with 2.37--2.65 million effective assigned possessions per window. Unit tests
verified row-order invariance, synthetic scorer/defender ordering, complete
window requirements, and component centering.

**Decision.** Stability is not validation. Retain
`matchup_elo_time_decay_v1_f71da3382c` as research descriptive only. It does
not fix defensive assignment, help, scheme, or context confounding. Do not add
it to RAPM, SPM, AIO, or the public site before a predeclared chronological
matchup-outcome test. Details: `docs/impact/MATCHUP_ELO_V1.md`.
# 2026-08-19 — Shot-quality matchup and combination RAPM projects registered

- The current event archive has coordinates, shot distance, action type, clock,
  shooter, and result, so it can support a basic pre-shot expected field-goal
  model. It does not have a defender credited to every shot.
- The current matchup endpoint is player–defender aggregate data, not a
  shot-level join. Assigning a shooter’s season shot profile to individual
  defenders by overlap would be circular, so defender-specific shot-quality
  Elo is blocked pending a permitted shot-level defender source.
- Registered two projects in
  `docs/impact/SHOT_QUALITY_MATCHUPS_AND_COMBINATION_RAPM.md`: rim/non-rim
  expected-shot matchup research and staged 2–5-player interaction RAPM.
  Neither is a production rating, SPM input, or AIO input.

# 2026-08-19 — Player-neutral expected-shot v0 and source audit

- Built `expected_shot_quality_v1_f5d343a852`: a logistic location/context
  model with no shooter, defender, team, lineup, or outcome feature. It trained
  on 2024, was isotonic-calibrated on 2025, and was evaluated once on untouched
  2026 (season-end labels): 218,722 shots, Brier .23075, log loss .65413.
  Rim and non-rim outputs remain separate.
- Audited the public matchup endpoint and confirmed it is player-pair aggregate
  data, not an event-level defender assignment. The permissible public route
  supports shooter shot quality only. Defender-specific expected-shot matchups
  remain blocked pending an explicitly licensed/rights-reviewed tracking source.

# 2026-08-19 — Lineup-adjusted expected-shot residual fallback: research null

- Built `lineup_shot_residual_v1_aeb57da06b` on the 2026 shot panel. The
  player-neutral expected-shot baseline used only 2024 training and 2025
  calibration data. The residual model used the ten observed players on court,
  split all shots, rim, and non-rim, and held out complete 2026 games.
- On 47,367 held-out shots, all-shot RMSE improved from 1.18082 to 1.18027
  (0.05%). Rim improved from .92457 to .92104 (0.38%); non-rim improved from
  1.26713 to 1.26691 (0.02%).
- Decision: do not publish its player rows or treat its defense side as
  individual shot contest. The current data identifies a five-on-five lineup,
  not a defender at the shot. Retain this as a documented null and require a
  permitted shot-level defender assignment before a defender-specific model.

# 2026-08-19 — Predictive SPM v1: RMSE gates pass, calibration gate fails

- Branch `codex/glm-predictive-spm`. Predeclared spec:
  `research/experiments/predictive_spm_v1.yml`. Run:
  `predictive_spm_v1_cb5666f6db`. New CLI command `build-predictive-spm`
  (`src/nba_impact/models/predictive_spm.py`).
- Design: consecutive-season pair panel (features at season s, canonical
  one-season normal RAPM targets at s+1, 5,060 pairs / 978 players).
  Expanding-window training; frozen annual_spm_v1 learners and features
  (reference run `single_season_spm_v1_bff6060df6`, unified feature panel
  `statistical_features_v2_b808fc1bf1`). Arms: persistence (last-season RAPM),
  raw, and OOF-affine-calibrated. Rookies are out of scope by design.
- Gate 1 (diagnostics 2019–24): PASS. Mean net weighted RMSE 1.6453 vs
  persistence 1.9945; per-fold deltas −0.24 to −0.42, far beyond the +/−0.03
  predeclared margin. Every fold favors the predictive SPM.
- Gate 2 (confirmation 2025–26): PASS. Raw-vs-persistence net RMSE delta
  −0.3853 (2025) and −0.2709 (2026). Scored once, no retuning.
- Gate 3 (defense dispersion in [0.85, 1.15] on confirmation folds): FAIL.
  Dispersion ratios: offense 0.452–0.462, defense 0.349–0.356. The OOF affine
  calibration barely moves slopes (≈0.91–0.99) because within-training OOF
  residuals carry same-season-style signal that true next-season forecasts
  lack. Predictions are strongly under-dispersed against noisy next-season
  labels.
- Defense remains the weak lane: next-season correlation 0.18–0.35 versus
  0.38–0.50 for offense.
- Decision: no promotion. Gates 1–2 establish real next-season signal over
  persistence; gate 3 requires a new predeclared uncertainty-aware calibration
  experiment (new id), which may also unblock the precision-aware prior
  contract by extending forward-only prior history below 2019.
- Dead end recorded: same-run identity initially omitted the source hash and
  the first persistence join keyed same-season targets (RMSE 0.0 artifact,
  discarded before evaluation). Both fixed in this commit.

# 2026-08-19 — Predictive backbone race: stat model and state space tie

- Predeclared spec `research/experiments/predictive_backbone_race_v1.yml`.
  Frozen comparison run `predictive_backbone_race_v1_43db6c6446`; no refits,
  no tuning. Identical-row population with both-side possessions >= 1,000.
- Mean weighted net RMSE, forecast seasons 2019-2024:
  state_space_filtered 1.6827 | predictive_spm_raw 1.6832 |
  calibrated 1.6843 | time_decay 0.80 filter 1.7819 | persistence 2.0276.
- Top two are 0.0005 apart: a tie under the predeclared 0.05 rule. They trade
  fold wins (state space takes 2019, 2020, 2023, 2026; predictive SPM takes
  2021, 2022, 2024, 2025), which suggests complementary information.
- Confirmation folds scored once: 2025 predictive SPM by 0.0016; 2026 state
  space by 0.0284. No overall winner.
- Simple 0.80 time-decay loses to both; persistence loses badly to everything.
- Decision per the predeclared rule: tie -> the next predeclared experiment is
  the combination (history backbone plus current-season stat features), not
  another single-model comparison.

# 2026-08-19 — Predictive-history combo v1: equal blend beats both parents

- Predeclared spec `research/experiments/predictive_history_combo_v1.yml`.
  Run `predictive_history_combo_v1_0a9c938a1e` on the race's identical rows.
- Fixed-weight blends of predictive SPM raw and state-space filtered net;
  weights fixed in the spec, estimated nowhere. Equal blend selected on
  2019-24 (mean weighted net RMSE 1.6076 vs parents 1.6438 / 1.6465).
- Gate 1 PASS: combo beats both parents across selection folds and wins every
  individual selection fold.
- Gate 2 PASS: confirmation scored once. 2025 combo 1.7206 vs parents 1.7514 /
  1.7530; 2026 combo 1.8081 vs 1.8462 / 1.8246. Combo wins both.
- Gate 3: blend moves defense dispersion further below 1.0 (0.35 -> 0.31),
  i.e. it does not improve calibration; recorded as failing-if-read-literally.
  Blending was never expected to fix dispersion; the uncertainty-aware
  calibration experiment remains open.
- Decision: equal blend is the research champion for next-season net impact,
  pending an untouched Season 2027 check. Still research-only: no API or site
  exposure. Next predeclared step when resumed: uncertainty-aware calibration
  of the combo, then the 2027 confirmation contract.

- User directive 2026-08-19: predictive-SPM/combo outputs stay local-only. If
  any UI work happens, it is the local `web/` view against pinned runs. No
  Worker origin, no public site, no API route until the 2027 confirmation
  contract passes.

# 2026-08-19 — Combo production-readiness push: side null, intervals pass

- Side-specific blend spec `predictive_combo_sides_v1.yml`, run
  `predictive_combo_sides_v1_2928ef3fb1`: NULL. Offense and defense each
  independently selected the 0.50 stat weight from five predeclared values,
  making the candidate identical to the incumbent (max diff 4.4e-16).
  Hypothesis that defense leans more on rating history is rejected at
  fold-mean level. Incumbent equal blend stands as champion.
- Interval calibration spec `predictive_combo_intervals_v1.yml`, run
  `predictive_combo_intervals_v1_efb18006be`: PASS on all three gates.
  Empirical residual percentiles by exposure bucket, estimated on 2019-24
  only. Pooled 2025-26 coverage: 80% interval 0.777 (gate [0.72, 0.88]),
  50% interval 0.467 (gate [0.42, 0.58]); all bucket coverages inside their
  bands. Point forecasts unchanged.
- Frozen `predictive_combo_2027_confirmation_v1.yml` contract: when Season
  2027 targets exist, the champion and interval table are scored exactly once
  against both parents plus coverage bands; pass opens promotion review.
- Exposure stays local_ui_only per user directive. Two dead scripts in /tmp;
  one pandas ordering bug (selection slice copied before derived columns) and
  one leftover dead line caught before any result was read.

# 2026-08-22 — Independent backbone-combo replication and parametric intervals

- Written on `codex/predictive-spm` in parallel with the
  `predictive_history_combo_v1` line. Runs below are independent replications
  with different frozen conventions; they corroborate, not replace, the
  incumbent champion. The local-ui-only exposure boundary is respected.
- Spec `research/experiments/predictive_backbone_combo_v1.yml`, run
  `predictive_backbone_combo_v1_38c40ac73e`: frozen 50/50 mean of
  predictive_spm_raw and state_space_filtered on identical primary rows
  (2,619). Convention note: this experiment uses equal-fold-mean weighted net
  RMSE; the earlier race pooled folds. Ranking is identical under both.
- Selection 2019-24: combo 1.6076 vs spm_raw 1.6438, state_space 1.6465,
  time_decay 1.7613, persistence 1.9958. Confirmation 2025-26 scored once:
  combo 1.7644 vs 1.7888 and 1.7988. Predeclared decision: combo promoted.
  Sensitivity weights 0.25/0.75 lose to 0.50 on selection and confirmation.
- Dead end caught before acceptance: the first join read both filters at
  `Window_End`, which equals the forecast season itself; parent RMSEs came
  out ~0.3 better than the frozen race on identical rows. Cross-checking
  against race `scored_rows` matched exactly only at
  `Season == Target_Season - 1`. Join fixed; parents now reproduce the race's
  pooled numbers up to the declared fold-mean convention.
- Spec `research/experiments/forecast_dispersion_calibration_v1.yml`, run
  `forecast_dispersion_calibration_v1_c825a1ef56`: per-side
  `sd^2 = max(a + b/n, 1e-6)`, WLS fit on 2019-24 squared residuals only.
  The exposure slope clips to zero on both sides: next-season error is
  model-dominated, not label-noise-dominated. Confirmation 2025-26 gates all
  pass once: dispersion 1.0035 / 1.0351 / 1.0251 (off/def/net, gate
  [0.85, 1.15]); coverage68 0.6765 / 0.6913 / 0.6854 (gate +/-0.05). This is
  a second, parametric interval calibration agreeing with the empirical
  bucket table from `predictive_combo_intervals_v1`.
- DARKO-style figures rendered by `build-projection-figures` into the
  calibration run's `figures/`: top-6 fan charts with history line, forecast
  dots, 68%/95% bands, actuals; plus coverage-by-season. matplotlib remains
  outside the locked runtime; render with `uv run --with matplotlib`.
  CLI import no longer requires it (lazy import in projection_figures).
- Tests: `tests/test_predictive_backbone_combo.py` covers weighted moments,
  predeclared decision branches, slope clipping, and gate summaries.
- Next: hold. Season 2027 confirmation contract governs; no further tuning
  on 2019-26.

# 2026-08-22 - RAPM lab kickoff: charter, data audit, full-era simple APM

- Chartered `rapm-lab/` in AGENTS.md as the from-scratch RAPM optimization
  program (principal decision). Scripts: scrape_pbp.py (parked, see below),
  apm.py. Writes only inside rapm-lab/ and outputs/.
- Data audit result: the pre-2014 RAPM gap was already closed on disk.
  rapm/data/possession_cache/matchups_{1997..2024}.parquet hold possession-
  level rows with full 10-man lineups, points, and home flag (29 files;
  2025 empty). Lockout seasons show the expected row dips (1999: 142k,
  2012: 197k) - internal consistency check passed.
- Direct NBA endpoint scraping is parked: stats.nba.com hangs from this
  network (curl 0 bytes/40s) and cdn.nba.com liveData 403s old games. The
  scraper keeps probe/dry-run rails (--probe-games, --max-games, manifest,
  shape validation) for when a route exists. Event-grade 1997-2013 has a
  proven public source: gabriel1200/merged_playbyplay old_data.
- Simple APM (OLS, damp=1e-9) fit on 1997-2025: 6,644,989 possessions x
  5,309 players, LSQR converged (414 iterations). Outputs/apm_1997_2025.csv.
  High-exposure leaderboard passes anchors: Jokic +14.67, Embiid +13.67,
  LeBron +13.14 (132k poss), CP3/KG/Draymond/Stockton next tier; bottom =
  fringe bigs. This is the unregularized baseline for the whole program.
- Loader-based OLS on 2014-26 (apm_2014_2026_loaderbase.csv) kept as a
  teaching artifact: with ~3k-possession players it produces +111 NET
  outliers - the visual argument for ridge.
- Ledger: added shot-profile RAPM family, JE 6/8-factor, RAPTOR On-Off,
  SPM-vs-BPM head-to-head, GPM/WOWYR blocked row, DRIP verification row
  (expansion unverified), playtype backburner status, external sources and
  verification-target section.

- Advisory adopted (2026-08-22): future event-grade 1997-2013 ingest must pull official minutes/team totals alongside PBP on day one; lineup QA and 6/8-factor factor targets depend on them. Logged in IDEAS.md.

- Terminology guard (2026-08-22): 6/8-factor work is a JE-style faithful
  reconstruction unless his exact spec (targets, filters, lambda, luck
  adjustment) is replicated and verified against his tables. Logged.

- Provenance scan (2026-08-22): three RAPM sources stitched. (1) foundry
  matchups caches 1997-2024 (complete 10-man lineups, 100% of rows); (2)
  legacy terminal-lineup cache also covering 2014-23; (3) canonical CDN
  possessions 2024-26. Measured seams: legacy vs cache on the SAME seasons
  differ 195.0 vs 190.4 poss/game (~2.4% definition drift), pts/poss 1.090
  vs ~1.075. 2023-24 seam: cache 245,167 rows / 1,243 games / 1.146 ppp vs
  canonical 264,057 / 1,310 / 1.126. Rule: never mix sources inside one fit
  window; single-source-per-season enforced in apm.py. Lockout/COVID dips
  are real-world. League scoring drifted 1.05 -> 1.15 pts/poss over the era.
- Identifiability note: pure OLS APM leaves one free direction (add c to all
  offense coefs, subtract from all defense); LSQR settles it arbitrarily ->
  uniformly negative DEF column. NET is identified; OFF/DEF split needs
  ridge (unique min-norm solution) or an explicit normalization. Matches the
  comment already in nba_impact/models/rapm.py.
- Exact DREB/OREB RAPM confirmed feasible: rebounds are attributed events
  (Gabriel PBP actionType=rebound with person_id, players_on, defender_id;
  event_states carries actionType/personId). Target = 1 if defense secures
  the miss; same lineup design. Historical path: REB factors 1997-2013 in
  Gabriel old_data; 2014-16 is the thin era (terminal-lineup cache lacks
  event attribution).

- Gabes merged_playbyplay audit (2026-08-22): upstream = per-GAME csvs,
  {season_end_year}_{game_id}.csv, regular (2...) + playoffs (4...).
  Schema supersedes our local bronze copy: adds off_players_on /
  def_players_on (5v5 split at every row), xLegacy/yLegacy shot coordinates,
  poc_ok exactly-10-tracked QA flag, qualifier tags (2ndchance,
  pointsinthepaint), full attribution chain (assister, steal, block,
  foulDrawn person ids). Confirmed coverage: old_data archive 1997-2013
  (RussDT ingest spec) + data/ 2021-2026 observed across all teams; exact
  pre-2021 floor in data/ unverified pending clone/--list-years.
  Candidate single backbone for ALL RAPM variants 1997-2026; canonical CDN
  stays as independent cross-check for overlap seasons. Local bronze holds
  only project_season=2024/2026 team-file aggregates.

- Correction (2026-08-22): Gabriel upstream pbp_data holds per-game csvs from
  season-end 2014 onward (e.g. 2014_21300001.csv), so 2013-14 through
  2015-16 are NOT thin at the source - only not yet downloaded locally.
  Single-source span confirmed: old_data (1997-2013) + pbp_data (2014-2026)
  = full 1997-2026 in one schema. Gabriel is the CANDIDATE backbone, gated
  on an audit vs official schedule/box scores (per-season game counts,
  poc_ok rate, final-score agreement) before any fit uses it as sole input.

- Gabes old_data schema audit (2026-08-22): NOT a reduced format. Team-file
  parquets (e.g. ATL_1997_rs) carry the FULL modern schema - players_on,
  off_players_on/def_players_on, xLegacy/yLegacy shot coordinates (verified
  on a 1997-11-01 miss), poc_ok, qualifier tags, complete attribution chain.
  Single-source span upgraded: shot-quality/luck-adjusted/playtype-adjacent
  RAPMs are buildable 1997-2026 from Gabes alone.
- BBRef crawl anomaly (2026-08-22): stage-5 box-link harvest returns low
  counts for pre-1978 seasons (1974: 162, 1976: 62) vs expected ~600+ games.
  Suspected monthly-page split or link-pattern drift on old _games pages.
  Also: transient 502s are skip-not-retry in the current crawler. Follow-up:
  per-season link-count audit vs official_game_scores after queue completes,
  then a targeted top-up pass with monthly-page handling.

- Lake-build requirement (2026-08-22): old_data is TEAM-FILE shaped - every
  game appears from both teams' perspectives (ATL_1997_rs and MIA_1997_rs
  each carry the same game). Any season-level lake build must de-duplicate
  on (game_id, actionNumber) keeping a single perspective before row counts,
  game counts, or fits. Verified: ATL_1997_rs alone = 39,525 rows (~482
  actions/game), consistent with single-perspective per file, so dedupe key
  above suffices.
- Queue health check: overnight-pull confirmed ALIVE and progressing
  (bbref_games reached 1989; cursor advancing) - the xrapm-timeout-exit
  report was stale/misinformed; no restart needed. fetch() already logs and
  skips individual failures by design.

# 2026-08-22 - Normal RAPM v0 (lab): ridge, two windows, forward folds only

- rapm-lab/rapm_ridge.py: lambdas 3000/3000/300, augmented LSQR, shared
  train/test player universe. Windows use END-YEAR labels (canonical loader
  convention) - LAST3 = canonical end-years 2024-2026 (743,946 possessions,
  all event-grade); CAREER full fit = cache labels 1997-2023 +
  canonical 2024-2026 (7,143,768 possessions). Single source per season.
- Predictiveness is FORWARD-only after a leakage audit voided the earlier
  leave-one-season-out variant (it trained on future seasons). Forward
  folds train on strictly prior seasons plus the pre-window cache.
- Results (game-margin corr | MAE, cold):
  forward -> 2024: 0.382 | 5.74   forward -> 2025: 0.371 | 5.77
  forward -> 2026: 0.362 | 5.95
  Identical across windows because forward training sets converge.
- Leaderboards. LAST3 top NET (>=6k poss): SGA +9.59, Jokic +9.48,
  Kawhi +7.56, Giannis +7.49, Wembanyama +6.69, Derrick White +6.36,
  Hartenstein, Diabate - defensive specialists rank correctly and the DEF
  column carries real signal under ridge. CAREER top NET (>=20k poss):
  Jokic +10.61, Garnett +9.35, LeBron +8.91 (141k poss), Paul +8.18,
  Duncan +8.19, Stockton +7.75 - era-straddling anchors pass.
- Known conventions/caveats: symmetric lambdas pin the OFF/DEF level split
  by min-norm (NET identified; asymmetric lambda row already queued);
  possession-level R-squared of lineup-only models is inherently small -
  judge via aggregated game margins, not possession residuals.
- Artifacts: outputs/rapm_last3.csv | outputs/rapm_career.csv |
  rapm-lab/outputs/rapm_top15_last3.png | rapm_top15_career.png.

# 2026-08-22 - Normal RAPM v0.1: intercept fix (supersedes v0 leaderboard)

- Bug (principal caught it): no intercept column in the design. The league
  scoring constant (~1.12 pts/poss) leaked into both player blocks via ridge
  min-norm centering, pinning OFF at ~+11 and DEF at ~-11 for an AVERAGE
  player - every defender looked elite, every scorer looked like a 20-pt
  offensive engine. NET was unaffected (artifact cancels inside it).
- Fix: intercept column added (lambda 1.0); home stays ±1 external factor.
- Post-fix LAST3: SGA +9.55 (+7.0/+2.6), Jokic +9.46 (+8.4/+1.0),
  Wembanyama +6.67 (+1.1/+5.6 - defense-first, correct),
  White +6.37 (+2.9/+3.4 two-way). Post-fix CAREER: Jokic +10.61 (+8.9/+1.7),
  Garnett +9.29 (+2.4/+6.9 defense-first, correct), Duncan +8.18 (+2.7/+5.5),
  Curry +7.95 (+8.1/-0.1 offense-only, correct).
- Predictiveness unchanged within noise (intercept changes split convention,
  not signal): forward folds 0.378 / 0.364 / 0.353, MAE ~5.8-6.0.
- Lesson logged: lineup-design matrices without an intercept force the
  scoring constant into player coefficients; always carry one.

- Overnight queue complete (2026-08-22 morning, exit 0): BBRef games pages
  39/40 seasons 1957-96 (1973 missing - transient 502, top-up queued),
  2,848 box scores fetched night one of the multi-night grind (resumable
  manifest), verification sets (EPM/MAMBA/DARKO/xRAPM/Substack/DWRAPM)
  landed, stats.nba.com probe still unreachable (skipped as designed).
  Total 154 MB + Gabes 4.0 GB. Consolidated download root: data/downloads/
  (daemon ROOT resolution) - rapm-lab/data/downloads merged in and removed;
  manifests merged. Known follow-ups: (1) top-up NBA_1973 games page +
  re-crawl any fetch_error rows; (2) pre-1978 link-count audit vs official
  schedule; (3) old_data team-file dedup at lake build.

- Lambda grid complete (dev-tuned): optimum lam_off=500, lam_def=2000,
  forward mean corr 0.388 (vs symmetric 3000/3000 at 0.372). Surface is
  flat in lambda_def across 2000-4500; lighter offense shrinkage wins on
  event-grade data - supersedes the old foundry off-2000/def-4500 finding
  for this substrate. Heatmap: outputs/lambda_grid_last3.png.
- Bake-off launched (tune-bakeoff daemon, --fast): ridge_tuned vs lightgbm
  vs extra_trees vs bilinear embeddings on identical forward folds.
  First fold: ridge 0.395 vs LGBM 0.307 - linear ahead early, consistent
  with the 2026 ledger's LGBM loss. Summary lands in
  outputs/bakeoff_last3.csv on completion; process exit auto-notifies.
- BBRef top-up crawler running (bbref-topup daemon): monthly schedule pages
  for every sub-400-link season 1957-96 (old schedules are month-split),
  fetch_error retries incl. 1973, then continued resumable box crawl.
- Fixed latent --no-lgbm flag bug in tune_bakeoff.py (argparse rejected it
  before the sys.argv check could fire).

# 2026-08-24 - RAPM lab bake-off made sequential and resumable

- Activated validation suite v1 with one explicit role assignment:
  2024--2026 are development/selection folds, and Season 2027 is the untouched
  single-shot confirmation. The persistence comparison is a rejection gate,
  not confirmation.
- Refactored `rapm-lab/tune_bakeoff.py` to build and release one fold at a
  time instead of retaining three training frames and sparse designs. A
  standalone fold now always includes every available earlier canonical
  season, so running 2025 alone matches its training scope inside the default
  three-fold command.
- Added explicit model selection, fixed-lambda reuse, and output-path controls.
  Each model-fold row remains append-only and completed pairs skip the fold
  build entirely. `SuffStats` no longer retains an unused copy of the design.
- Exact 2024 ridge smoke check with frozen development penalties 500/2000:
  1,227 games, correlation 0.39522678134541767, MAE 5.749080145374502. This
  matches the saved pre-refactor row exactly. An immediate rerun read the
  checkpoint, built no fold, appended no duplicate, and exited in 0.53 seconds.
- The nonlinear run resumed only after the exact smoke check. The completed
  lambda surface remains selection evidence; it does not confirm 500/2000.

- GPT Pro critique handoff created: rapm-lab/GPT_PRO_CRITIQUE_HANDOFF.md (also copied to ~/Downloads). It includes full IDEAS ledger synthesis, data/source state, formulation, validation suite, historical results, and explicit critique questions.
- Completed all three development folds for ridge, LightGBM, and the fast
  bilinear embedding model. Mean game-margin correlation / MAE: ridge
  0.388/5.830, LightGBM 0.301/5.947, bilinear 0.296/6.771. Ridge wins both
  metrics against each completed challenger in aggregate; this remains
  development evidence only.
- Fold-first execution was operationally reordered after the 2024 Extra Trees
  fit ran for more than one hour without a checkpoint. Cheap models were
  completed first, bilinear second, and Extra Trees moved to one persistent
  fold per run. Model definitions, folds, and metrics did not change.
- The 2024 Extra Trees fold eventually completed after roughly three hours:
  correlation 0.270, MAE 5.929 over 1,227 games, versus ridge 0.395/5.749 on
  identical games. By principal decision, do not run its 2025 or 2026 folds.
  Preserve the one row as an incomplete diagnostic; never average it with the
  three-fold models or describe it as a completed challenger.
- Extra Trees is removed from default bake-off runs but remains an explicit
  reproducibility option. Ridge is the development winner among completed
  lab candidates. This does not replace the production 3000/3000/300 normal
  RAPM penalties or constitute confirmation.
- Persistence gate implemented in `rapm-lab/validate_persistence.py`. For each
  held-out season T, it fits the same zero-prior 500/2000 coefficient model on
  T-1 only, gives unseen players zero, and verifies exact game-count identity
  against the candidate before comparison.
- Ridge passes persistence in every development fold. Correlation, ridge versus
  persistence: 2024 0.395/0.358, 2025 0.390/0.332, 2026 0.379/0.319. Mean
  correlation is 0.388/0.336; mean MAE is 5.830/6.008. This rejects the
  persistence-null explanation but remains development evidence, not 2027
  confirmation. Output: `outputs/persistence_last3.csv`.

# 2026-08-24 - Repository and RAPM lab consolidation

- Merged the production package, modular web client, and research control plane
  into the active repository. Stale UI branches remain historical snapshots and
  do not enter the product.
- Moved the only maintained lab implementation to `research/rapm_lab/`.
  Canonical inputs remain in the main data contracts. Lab downloads, external
  mirrors, and outputs now stay below the lab and are ignored by Git.
- Preserved 1.3 GB of historical downloads, 5.1 GB of the Gabriel mirror, and
  1.1 MB of experiment results. The Basketball Reference crawler stopped at a
  resumable checkpoint after 28,974 box pages; no downloaded row was deleted.
- Added safe command-line confirmation to the overnight queue. Asking for help
  no longer starts a wait or a download. Fixed the Gabriel sparse checkout so
  adding one source directory does not remove a previously fetched directory.
- Imported the sequential bake-off and previous-season persistence gate. The
  recorded 2024-26 results remain development evidence. Season 2027 remains
  untouched confirmation.

# 2026-08-25 - Unified five-year rolling RAPM and lambda matrices

- Froze `rolling_5y_rapm_2014_2026_v1`: nine retrospective windows from
  2014-18 through 2022-26. The model is zero-prior terminal-lineup RAPM with
  3000 / 3000 / 300 offense / defense / home penalties. Each season's scoring
  environment is removed before the player fit.
- Used the explicit source transition already accepted by the annual research
  timeline: legacy terminal possession caches through 2023 and canonical event
  terminal lineups from 2024. The six unchanged windows ending 2018-23 exactly
  reproduce the pinned legacy rolling artifact for offense, defense, and net.
- The 2020-24 hybrid window remains close to the legacy-only 2024 reference:
  matched-player correlations are .9960 offense, .9939 defense, and .9958 net.
  Net RMSE is .1717 points per 100. This is source sensitivity, not model lift.
- Completed all nine windows: 8,620 player-window rows, 1,706 distinct players,
  and 1,167 eligible peak rows. Every rating is finite, names are complete,
  keys are unique, and maximum `offense + defense = net` error is `8.88e-16`.
- The first packaging pass correctly rejected 12 unresolved 2026 IDs. All 12
  resolved from the canonical `player_games.parquet` crosswalk. No coefficient
  or possession input changed.
- Runtime was 597 seconds with 3.48 GB maximum resident memory. Execution was
  sequential. The 2014-18 pilot took 65 seconds and peaked at 2.20 GB.
- Stored nine sparse lambda-training packages and eight next-season evaluation
  packages. Each contains `X'X`, centered `X'y`, player ordering, side exposure,
  and game-aggregated held-out design/targets. Raw possession-level `X` is not
  stored. Total run storage is 20 MB. Season 2027 was not loaded.
- Solving the default penalties from stored matrices reproduces fitted ratings
  within `1.22e-7` points per 100, inside the `1e-6` numerical tolerance implied
  by the production conjugate-gradient solve. Across 2019-26 held-out seasons,
  equal-season mean game-margin correlation is .3690, MAE 11.036 points, and
  RMSE 14.070 points. These are the baseline for future lambda research, not a
  selected penalty result.
- The 2022-26 qualified net leaders are Nikola Jokic +10.04, Shai
  Gilgeous-Alexander +7.62, Giannis Antetokounmpo +7.14, Derrick White +5.66,
  and Alex Caruso +5.03 points per 100.
- Local run: `rolling_5y_rapm_2014_2026_a7754bfb77` under
  `research/rapm_lab/outputs/rolling_5y_2014_2026/`. Status remains research.

# 2026-08-25 - Five-year rolling RAPM lambda grid is a null result

- Froze and scored 196 offense / defense / home penalty combinations on the
  stored rolling-five-year sufficient statistics. Selection used only
  next-season 2019-23 results; the chosen setting was then evaluated once on
  reused diagnostic Seasons 2024-26. Season 2027 was not loaded.
- The correlation-first selection rule chose offense 2000, defense 1000, and
  home 1000. On selection seasons it raised equal-season mean game-margin
  correlation from .37084 to .37298 (+.00214), but worsened MAE from 10.6209
  to 10.7366 (+.1157 points).
- On reused diagnostics, correlation rose from .36587 to .37522 (+.00935),
  while MAE worsened from 11.7277 to 11.7727 (+.0451) and RMSE worsened from
  14.9358 to 14.9657.
- A 2,000-draw paired whole-game bootstrap, stratified by season, estimated
  baseline-minus-candidate MSE at -.9103 with a 95% interval of
  [-2.1307, .3270]. Only 7.15% of draws favored the candidate on MSE.
- The candidate fails the preregistered diagnostic MAE and paired-bootstrap
  gates. Classification is `research_null`; retain 3000 / 3000 / 300 as the
  five-year rolling baseline. This does not contradict the earlier 500 / 2000
  lab result, which used a different training scope, source substrate, and
  2024-26 development contract.
- Local run: `lambda_grid_v1_ef9f6a7a5f` inside
  `rolling_5y_rapm_2014_2026_a7754bfb77`. It contains all 1,568 fold scores,
  3,681 diagnostic game predictions, and all bootstrap draws.

# 2026-08-25 - Expanded RAPM penalty families remain research nulls

- Froze `rolling_5y_lambda_frontier_v1` before execution. The search used
  next-season 2019-23 for selection, reused 2024-26 only for diagnostics, and
  never loaded Season 2027. Home remained fixed at 300 because the earlier
  four-value home search was numerically immaterial.
- Scored 169 distinct penalty configurations: 64 log-space Sobol scalar
  candidates spanning 30-30,000 per side, seven scalar anchors, 75 local scalar
  refinements, 18 full-covariance offense/defense candidates, and five
  empirical-Bayes adaptation strengths. This produced 845 selection-fold fits.
- The scalar correlation choice was offense 2317 / defense 1768. Its reused
  diagnostic correlation was .37190 versus .36587 for 3000 / 3000, but RMSE
  worsened from 14.93581 to 14.94091 and only 31.35% of 2,000 paired whole-game
  draws favored it on MSE.
- The scalar RMSE choice was offense 9043 / defense 6020. It had the strongest
  adjacent-window stability, but diagnostic correlation fell to .34256 and
  RMSE worsened to 15.01062. This is direct evidence that stability by itself
  rewards over-shrinkage.
- The bivariate correlation choice used offense 2317 / defense 1768 with a
  +.25 published OFF/DEF prior correlation. Diagnostic correlation rose to
  .37451, but MAE/RMSE worsened to 11.75905/14.95151. Only 16.3% of paired
  draws favored it on MSE and 2.55% favored it on MAE.
- The training-only GCV scalar choice was offense 2895 / defense 3816. It was
  nearly the existing baseline and did not improve diagnostics. An independent
  32-probe trace run reproduced the GCV ordering, so trace randomness does not
  explain the null.
- The empirical-Bayes implementation used a training-only 3000/3000 pilot,
  posterior second moments, one type-II moment update, and bounded per-player
  precision multipliers. Global side precisions were approximately 3260-3558
  across selection folds. Full player adaptation produced roughly 1900-9400
  10th-90th percentile precisions and won GCV reproducibly, but diagnostic
  MAE/RMSE worsened to 11.76002/14.95504; paired MSE improvement probability
  was only 13.4%.
- The unadapted empirical-Bayes variance-component model changed diagnostic MAE
  by only -0.00052, while correlation and RMSE worsened. Its paired improvement
  probabilities were 57.35% MAE and 17.1% MSE: a statistical null, not a win.
- Twelve heterogeneous synthetic recovery seeds favored full adaptive EB on
  average net recovery (correlation .4756 vs .4606; RMSE 2.539 vs 2.578 for the
  baseline), showing the mechanism can work when its heterogeneity assumptions
  are true. It did not transfer to real held-out seasons and expanded the
  latest sub-500-possession net tail from 4.25 to 6.43 points per 100.
- Player relabeling invariance passed to `8.88e-16`; all stored ratings satisfy
  offense + defense = net exactly; finalist keys and diagnostic game rows are
  unique. Classification: multi-family research null. Retain 3000 / 3000 / 300.
- Local run: `lambda_frontier_v1_45ccc734c0` inside
  `rolling_5y_rapm_2014_2026_a7754bfb77`, with `audit.json` recording the
  independent GCV and paired-game checks.

# 2026-08-25 - Home helps error; garbage, rubberband, and clock fatigue do not

- Froze six context ablations on five-year rolling RAPM. Each fold trained on
  the preceding five seasons and scored the identical next season: 2019-23 for
  selection and reused 2024-26 for diagnostics. Player penalties stayed at
  3000 / 3000, home and context penalties at 300, and Season 2027 was not
  loaded. No time decay was included.
- The primary score used only frozen player ratings plus the fitted home term.
  Rubberband and clock columns were zeroed at evaluation so they could not buy
  apparent generalization by directly using live game state. A secondary
  conditional score retained them and is explicitly not a pregame forecast.
- Home reduced reused-diagnostic RMSE from 15.0512 to 14.9358 and MAE from
  11.8529 to 11.7277 versus no-home. Correlation was effectively tied and
  slightly lower (.36587 versus .36595). Retain the global home term.
- Hard garbage filtering removed 920,959 of 9,358,559 training rows (9.84%).
  It worsened selection correlation/RMSE from .37084/13.5503 to
  .36888/13.6232 and diagnostic correlation/RMSE from .36587/14.9358 to
  .35827/15.0383. Reject it for this five-year estimand.
- Quarter-specific pre-possession offense margin raised primary selection
  correlation to .37430 and diagnostic correlation to .37596, but worsened
  RMSE to 13.7341 and 15.0155. The paired diagnostic baseline-minus-candidate
  MSE was -2.3443 with 95% interval [-3.7928, -.8895]; only .1% of 2,000
  whole-game bootstrap draws favored rubberband. Reject it: it changes ranking
  while degrading calibrated game-margin error.
- Including live quarter-margin terms in held-out predictions was catastrophic
  (selection/diagnostic correlation -.1845/-.1214), consistent with the prior
  endogeneity diagnosis: player quality creates the lead being controlled.
- The legacy clock-fatigue proxy was numerically inert. Diagnostic correlation,
  MAE, and RMSE were .36587, 11.72764, and 14.93577 versus .36587, 11.72766,
  and 14.93581 for baseline. This proxy measures game-clock state, not observed
  player fatigue; do not label it a fatigue adjustment.
- The old fast-optimizer claim that home + quarter rubberband + fatigue
  “worked” came from fitting and scoring the same random 30,000 possessions;
  its reported MSE gain was only .000206. It is not out-of-sample evidence.
- All 48 fold-variant rows are finite; variants scored identical games in each
  season; tests passed. Classification: `research_null`. Local run:
  `context_adjustments_v1_full_46dec665f1`.

# 2026-08-25 - Team-specific home effects do not beat one global effect

- **Question:** Does each NBA franchise have a repeatable home-court deviation
  that improves future-game predictions beyond one global home coefficient?
- **Data and design:** Added a verified official game-to-home-team map for
  2014--16 to the existing 2017--26 panel. Every regular season contains all 30
  teams and every modeled possession game resolves exactly once. Each fold uses
  the preceding five regular seasons, terminal lineups, zero prior, and fixed
  3000 / 3000 / 300 player/global-home penalties. Selection seasons are
  2019--23; reused diagnostics are 2024--26; Season 2027 was not loaded.
- **Algebra:** The raw global home column equals the sum of the 30 franchise
  columns. The first pilot exposed that post-fit centering would not separate
  the global and team penalties. That invalid full run was stopped. The final
  implementation solves a KKT system that constrains the exposure-weighted mean
  team deviation to zero during fitting. A synthetic test covers the constraint.
- **Selection:** Seven team-deviation penalties from 30 to 30,000 were frozen.
  Selection chose 30,000, the strongest shrinkage tested. Even there, mean
  2019--23 RMSE was 13.5622 versus 13.5503 for global home, and the candidate
  won only one of five folds.
- **Diagnostics:** On 2024--26, team home worsened equal-season mean RMSE from
  14.9358 to 14.9433 and MAE from 11.7277 to 11.7302. Correlation fell from
  .36587 to .36510. It won one of three diagnostic folds by RMSE, by only
  .0004 points in 2026.
- **Paired uncertainty:** Across 2,000 whole-game bootstrap draws stratified by
  season, baseline-minus-candidate MSE was -.2134 with a 95% interval of
  [-.6265, .2064]. Only 15.35% of draws favored team home.
- **Audit:** All variants scored identical unique games and actual margins; all
  46 fold rows are finite; every fold/variant has 30 teams. The global baseline
  reproduces the prior context run exactly for correlation, MAE, and RMSE.
- **Verdict:** `research_null`. Retain one global home effect. Do not interpret
  franchise deviations as altitude, travel, crowd, arena, or referee effects.
  Run `team_home_v1_full_0ade8d3301` is local under
  `research/rapm_lab/outputs/team_home/`.

# 2026-08-25 - Conserved scoring channels exactly recover five-year RAPM

- **Question:** Can the discrete possession outcome be decomposed without the
  cost and non-additivity of a full multinomial player model?
- **Design:** On the 2022--26 terminal-lineup regular-season panel, define three
  additive targets: `1 * I(points=1)`, `2 * I(points=2)`, and
  `points * I(points>=3)`. Fit all three with the same zero-prior
  3000 / 3000 / 300 ridge design and remove each target's season scoring mean.
  One sparse factorization solves all three right-hand sides.
- **Quality:** The run covers 1,229,744 possessions, 6,141 games, and 1,029
  players. Target and rating recomposition errors are zero at stored precision.
  Every channel satisfies offense plus defense equals net to `8.88e-16`. The
  summed rating matches canonical normal RAPM within `1.32e-7` points per 100;
  intercept error is `4.01e-11`.
- **Read:** Among players above 5,000 possessions per side, Nikola Jokic leads
  total net and the two-point channel. Giannis Antetokounmpo leads the one-point
  channel. Sam Hauser leads the three-plus channel. These are lineup-adjusted
  associations with where points appeared, not individual shot-credit claims.
- **Verdict:** The cheap additive decomposition works. It is more useful as the
  first factorized RAPM baseline than a true multinomial fit because it exactly
  conserves ordinary RAPM. A turnover / FT / 2P / 3P / OREB version needs an
  explicit value ledger whose row components sum to observed or expected
  possession value. Run `points_channel_rapm_v1_4507aab97c` is local under
  `research/rapm_lab/outputs/points_channel_rapm/`.

# 2026-08-25 - The rubber-band curve is real but does not improve this RAPM

- **Empirical check:** Reused the 2022--26 terminal-lineup panel and the exact
  five-year normal-RAPM fit. After removing fitted lineup, home, and season
  scoring effects, grouped possession residuals by pre-possession offense
  margin and quarter.
- **Shape:** A linear fit changes residual efficiency by -0.085, -0.166,
  -0.153, and -0.245 points per 100 for each point of offensive lead in
  Q1 through Q4. Per 10 points, that is -0.85, -1.66, -1.53, and -2.45.
  Quadratic coefficients are near zero in Q2 through Q4; the Q1 tail is sparse.
- **Interpretation:** This supports a quarter-specific linear score-effect
  description. It does not make live margin exogenous. Strong players help
  create the leads being controlled, and strategic response also changes with
  score.
- **Decision:** Do not run another polynomial or cutoff grid. The already frozen
  quarter-linear model worsened reused-diagnostic RMSE from 14.9358 to 15.0155,
  and only 0.1% of paired whole-game draws favored it on MSE. Keep the empirical
  curve as a descriptive team/game-state result, not a player-RAPM adjustment.

# 2026-08-25 - Aging helps forecasts; interaction, multinomial, and coach challengers do not

- **Contract:** Season 2027 was never loaded. Earlier seasons select model
  settings and 2026 is reused diagnostic evidence. Baselines and challengers
  score identical games. These experiments do not change production ratings.
- **Aging:** The annual 2014--26 panel contains 5,053 adjacent transitions for
  1,240 players. Stored age is integer-valued, so 0.1/0.5/1/2-year values are
  kernel bandwidths, not true subannual observation intervals. A one-year
  Gaussian smoother lowered net weighted RMSE from 2.0045 to 1.9904 for a
  one-season input, 1.7707 to 1.7442 for three seasons, and 1.7440 to 1.7079
  for five seasons. The first loses .0040 correlation; the latter two gain
  .0113 and .0165. Use aging only to translate ratings forward. Descriptive
  retrospective RAPM remains age-neutral.
- **2--5-player interactions:** Fit one regularized unit layer to residuals
  after player RAPM. Penalties were selected on 2025 and checked on 2026.
  Pair, trio, and four-player layers worsened RMSE by .0345, .0702, and .0362.
  The five-player layer improved RMSE by only .0039 and correlation by .0006,
  which is effectively a null. These are conditional lineup associations, not
  evidence of causal chemistry.
- **Six-sided factor RAPM:** Built shooting-eFG, turnover, and offensive-rebound
  surfaces for offense and defense on 743,946 2024--26 possessions. The ledger
  maps 98.43% of 1,176,393 relevant events, with 431,452 shots, 95,583 turnover
  possessions, and 302,567 resolved missed-shot rebound opportunities. The
  sides use different denominators and do not sum to points impact. Retain as
  descriptive skill surfaces, distinct from the exact conserved points-channel
  decomposition.
- **Multinomial RAPM:** Fit a 0/1/2/3-plus softmax lineup model. Alpha .001 won
  the 2025 selection fold. On 2026, multinomial margin RMSE/correlation were
  15.5084/.3326 versus 15.4732/.3344 for linear points RAPM. Its log loss
  1.1109 was also worse than constant class rates at 1.1096. Classification:
  `research_null` for prediction.
- **Win-probability RAPM:** Used prior-season-trained player-neutral WP surfaces
  to create a conserved possession-change target for 497,177 possessions and
  2,454 games in 2025--26. Maximum game conservation error is `1.11e-16` and
  terminal jumps are only .72% of absolute credit. Net correlates .738 with
  exact-row points RAPM, but 2025-to-2026 net stability is only .125. Keep as
  descriptive leverage credit, not player strength or a forecast.
- **Coach RAPM:** Parsed 325 Basketball Reference coach-season rows and assigned
  all 11,969 modeled 2017--26 games. Joint player/coach ridge selected the
  strongest coach penalty tested, 100,000. On 2026, coach columns worsened RMSE
  by .0147 and correlation by .0012. Coach, roster, franchise, assistants, and
  organization remain too confounded for a portable public coach rating.
- **Decision:** Retain terminal-lineup, zero-prior `3000 / 3000 / 300` RAPM as
  the reference. Promote none of these challengers. The durable result summary
  is `docs/impact/RAPM_FRONTIER_RESULTS_2026_08_25.md`; local runs are
  `aging_resolution_v1_540ec99a49`, `lineup_interactions_v1_958fa5d618`,
  `possession_outcome_rapm_v1_3b8d88046a`,
  `win_probability_rapm_v1_f679b24223`, and `coach_rapm_v1_32b9c1065e`.

# 2026-08-25 - Actual-clock rubber-band curve transfers to 2026

- **Question:** After removing lineup expectations out of fold, how does an
  offense's pre-possession lead change scoring as actual game time elapses?
- **Repair:** The prior context experiment used possession order inside a
  quarter as a clock proxy. The new contract uses exact possession-start elapsed
  seconds on 743,946 canonical 2024--26 possessions. Five whole-game folds per
  season produce lineup residuals with no held-game rows in their RAPM fit.
- **Split:** 2024 develops the curve, 2025 selects among one/four/eight time
  bins and margin caps, and exposed 2026 supplies reused diagnostics. Season
  2027 was not loaded.
- **Selection:** Eight actual six-minute bins with margin clipped at 15 won
  2025 MSE. The neighboring 10-point cap is statistically tied: winner-minus-
  runner-up MSE improvement `0.0000097`, 95% interval `[-0.0000283, 0.0000486]`.
  Do not interpret 15 as a known strategy threshold.
- **Curve:** The 2024--25 points-per-100 slope for each point of offense margin
  is `+0.003, -0.056, -0.199, -0.176, -0.044, -0.147, -0.173, -0.524` across
  the eight consecutive six-minute bins. The first six minutes show no effect;
  the final six are the clear maximum. A ten-point lead in the final six minutes
  corresponds to `-5.24` points per 100 versus the cross-fitted lineup baseline.
- **Diagnostic:** On 2026, residual RMSE changes from `1.191592` to `1.191427`.
  MSE improves 0.028%; all 2,000 fixed-prediction whole-game resamples favor the
  curve, with interval `[0.000180, 0.000602]`. Pairwise annual slope correlation
  is at least 0.691.
- **Decision:** The score association is real, small, and late-game-heavy. It is
  still endogenous game context, not causal effort or a garbage-time label.
  Keep it local. The next gate must refit adjusted player RAPM and compare
  future-game calibration and player stability on identical rows. Run
  `rubberband_adjustment_v1_34be1ee621`; full specification in
  `docs/impact/RUBBERBAND_ADJUSTMENT.md`.

# 2026-08-25 - Clock and possession-progress adjusted RAPM do not clear the gate

- **Question:** Does the empirical rubber-band curve change player RAPM in a
  useful way, and can fixed possession progress reproduce the actual-clock
  result on identical rows?
- **Design:** Use the same 743,946 canonical 2024--26 terminal-lineup rows. Fit
  the two eight-segment score curves on 2024--25 out-of-fold lineup residuals.
  Actual clock uses six-minute segments. The proxy counts completed regulation
  possessions before the current row in fixed 25-possession bins and never uses
  final game length. Subtract only the signed-margin slope from points; do not
  subtract the segment intercept. Fit zero-prior `3000 / 3000 / 300` player
  RAPM on 2024--25 and score the same 1,228 reused 2026 games.
- **Context result:** The eight clock and possession-progress slopes correlate
  0.971. Residual RMSE falls from 1.191592 to 1.191427 with clock and from
  1.191617 to 1.191461 with possession progress. The score association is real
  but explains little possession variance.
- **Player result:** Normal RAPM scores 15.473 RMSE and .334 correlation.
  Clock-adjusted scores 15.491/.344; possession-adjusted scores 15.499/.343.
  Paired RMSE-change intervals are `[-.043, +.079]` and `[-.036, +.087]`.
  Conditional observed-score-path addback fails badly at 17.933 and 18.020
  RMSE because score margin is endogenous.
- **Ratings:** On the full descriptive 2024--26 refit, each adjusted net rating
  correlates 0.991 with normal RAPM and moves by 0.239 points per 100 on
  average. Complete qualified player leaderboards are saved and exposed only
  in the local RAPM Lab.
- **Decision:** Keep the descriptive rubber-band curve. Reject clock-adjusted
  and possession-adjusted player ratings under the frozen gate. Season 2027 was
  not loaded. Run `rubberband_progress_rapm_v2_b72716c2fb`.

# 2026-08-25 - Corrected standalone pair through lineup RAPM

- **Correction:** The earlier `lineup_interactions_v1_958fa5d618` experiment
  fits residual unit layers after ordinary one-player RAPM. It does not answer
  the requested standalone unit estimand and is now labeled accordingly.
- **Design:** Pair RAPM contains only unordered two-player offense and defense
  unit columns. Trio, four-man, and lineup RAPM analogously contain only units
  of size three, four, or five. Each model also contains one signed home-offense
  column. There are no individual player columns, player-RAPM predictions,
  residual targets, or SPM priors.
- **Split:** Five-season 2020--24 training selects ridge penalties on 2025.
  Five-season 2021--25 training is refit and compared with one-player RAPM on
  the same 1,228 reused-diagnostic 2026 games. Season 2027 is not loaded.
- **Result:** One-player RAPM RMSE is 15.2962. Pair, trio, four-man, and lineup
  RMSE values are 15.7676, 16.0763, 16.3187, and 16.5222. Their paired RMSE
  deltas are +.471, +.780, +1.022, and +1.226. All 2,000 whole-game bootstrap
  draws lose for every challenger; the respective 95% intervals are
  `[.260, .688]`, `[.516, 1.043]`, `[.728, 1.329]`, and `[.884, 1.565]`.
- **Coverage:** Training-only exposure floors produce 2026 test-slot coverage
  of 41.4%, 21.7%, 10.5%, and 3.7% from pair through lineup. Unseen units get a
  zero coefficient. Sparsity is therefore part of the result, especially for
  higher orders.
- **Decision:** Reject these standalone unit models as replacements for player
  RAPM under the frozen specification. Keep them as a local descriptive
  research view. Do not interpret a unit coefficient as individual value or
  causal chemistry. Run `standalone_unit_rapm_v1_460b34a3b1`.

# 2026-08-25 - JE categorical score-state curve replicated; adjusted ratings rejected

- **Question:** Does JE's rubber-band pattern appear when exact pre-possession
  score-margin indicators are fitted inside our RAPM, and do the resulting
  player coefficients generalize better?
- **Design:** Use 3,080,228 audited 2014--26 terminal-lineup possessions. Fit
  offense, points-allowed defense, home, and 115 exact margin indicators
  jointly on 2014--25. Margins below -57 or above +57 are top-coded. Player
  penalties remain `3000 / 3000 / 300`; score indicators use alpha 1, matching
  the public reproduction. Season 2026 is reused diagnostics; 2027 is untouched.
- **Curve:** Relative to a tied score, an offense trailing by 10 scores `+4.80`
  points per 100 and one trailing by 20 scores `+9.26`. Leading by 10 is
  `-1.25`; leading by 20 is `-1.02`. The leading tail is noisier and weaker
  than JE's published historical figure, but the asymmetric rubber-band shape
  is present.
- **Held-out result:** Normal player-only RAPM scores 15.436 RMSE and .353
  correlation on the same 1,228 games. The score-state-controlled player
  coefficients score 15.579/.365. RMSE worsens by .143; the paired 95% interval
  is `[+.043, +.244]`. Adding the observed 2026 score path is invalid as a
  forecast and scores 19.617 RMSE.
- **Ratings:** The full descriptive adjusted net ratings correlate .985 with
  normal RAPM and move .376 points per 100 on average.
- **Decision:** Keep the JE curve as a local context diagnostic. Reject the
  adjusted player rating. Do not call the curve causal effort or garbage time.
  Run `rubberband_je_replication_v1_c8bdb4484d`.

# 2026-08-25 - Same-age RAPM separates age context from player standardization

- **Question:** Does controlling lineup age inside a 2014--26 RAPM improve
  held-out game estimates, and does the resulting age-27 player rating improve
  over ordinary RAPM?
- **Design:** Add separate offense and points-allowed defense counts for every
  integer age from 19 through 43, with age 27 omitted. Fit them jointly with
  terminal-lineup player and home coefficients. Player penalties remain
  `3000 / 3000 / 300`; age penalties 100, 1,000, and 10,000 are selected on
  2025. Age coverage is 99.958% of 30,802,280 player-possession slots.
- **Diagnostic:** On the same 1,228 reused 2026 games, normal RAPM scores
  15.436 RMSE and .353 correlation. Age-27 player coefficients alone score
  15.477/.341, a +.042 RMSE change with paired interval `[-.068, +.157]`.
  Player coefficients plus actual lineup ages score 15.258/.380, a -.177
  change with paired interval `[-.331, -.012]`; 98.0% of whole-game resamples
  improve.
- **Ratings:** The full age-27 net ratings correlate .962 with ordinary pooled
  RAPM and move .364 points per 100 on average. The categorical curve is not a
  smoothed biological aging curve and becomes noisy at sparse extreme ages.
- **Decision:** Keep actual-age controls as a research context challenger.
  Do not replace the reference leaderboard with age-27 ratings: they do not
  improve neutral held-out prediction. Run `age_adjusted_rapm_v1_99ad4ffb22`.

# 2026-08-25 - Three factor RAPMs reconstruct annual points RAPM

- **Question:** Can annual shooting-eFG, turnover, and offensive-rebound RAPMs
  estimate ordinary annual points RAPM after learning their scale on earlier
  seasons?
- **Data repair:** The 2024 V3 source has no explicit `shotValue`. Infer three
  points only when the official description contains `3PT`; otherwise treat a
  field goal as two points. This recovers 215,080 historical shot values. The
  final ledger contains 743,946 possessions, 646,532 shots, and 302,567
  resolved missed-shot rebound opportunities; 98.431% of relevant events map
  to a canonical possession.
- **Design:** Fit each factor separately by season with the same terminal-lineup
  `3000 / 3000 / 300` ridge. Fit offense and defense reconstruction maps on
  2024, select ridge alpha on 2025, refit on 2024--25, and diagnose qualified
  2026 players. Player-season weights are the square root of minimum-side
  target possessions and are not features.
- **Result:** On 387 reused 2026 players, all three factors reconstruct net
  RAPM with .964 correlation, .930 weighted R-squared, and .545 RMSE versus
  2.052 for a mean-only baseline. Shooting alone reaches .718 R-squared;
  turnover .174 and rebounding .222. All 2,000 player resamples favor the full
  model over the mean baseline. Standardized offense coefficients are 1.204
  shooting, .516 turnover, and .540 rebounding; defense values are .971, .495,
  and .381.
- **Decision:** The factors support a useful mechanistic representation of
  ordinary RAPM. This is reconstruction, not independent validation or causal
  attribution: the inputs and target share the same season and lineup design.
  Run `factor_rapm_reconstruction_v1_ed61c90a5a`.

# 2026-08-25 - Full RAPM Lab sweep and frozen production candidate

- **Validation spine:** Development uses earlier seasons, 2025 selects a frozen
  candidate, and 2026 is reused diagnostics on identical games. Comparisons use
  game-margin RMSE, correlation, and paired whole-game resampling. Season 2027
  remains untouched. Descriptive full fits never count as predictive evidence.
- **Five-point rubber band:** On 3,080,228 possessions, trailing offenses score
  `+1.90`, `+4.00`, `+5.11`, `+7.48`, and `+7.95` points per 100 in the -5
  through -25 buckets versus a tie. Leading effects are much smaller and
  negative. The selected differential-penalty candidate (`2317` offense,
  `1768` defense) loses to `3000 / 3000` on 2026 by `+.017` RMSE; paired 95%
  `[-.021, +.055]`. Keep the context curve and reject it from player RAPM.
- **Differential penalties:** The broader five-year frontier also selects the
  symmetric `3000 / 3000` baseline. Offense and defense penalties are allowed
  to differ in research, but they did not improve the frozen points target.
- **Age, 1997-2026:** The full fit uses 6,738,828 possessions and 35,532 games
  with 99.979% age-slot coverage. Actual lineup-age controls improve 2026 RMSE
  by `-.163`, paired 95% `[-.287, -.024]`. Age-27 player-only ratings worsen by
  `+.052`. Keep age as contextual prediction control, not the reference rating.
- **Coach, 1997-2026:** A joint player, age, coach, and home fit covers 191
  coaches. Coach terms worsen 2026 RMSE by `+.0109`, paired 95%
  `[+.0067, +.0151]`, so they fail prediction. Descriptive net coach ratings
  correlate `.802` with xRAPM across 188 matched coaches; association is not
  causal coaching value.
- **Win-probability RAPM:** `3000` offense and `10000` defense narrowly beat
  `3000 / 3000` by `.0057` game-total RMSE on reused 2026. Nine rolling
  five-year windows from 2014-18 through 2022-26 conserve game WP credit within
  `1.11e-15`. The gain is too small for a points-model claim; keep WP as a
  leverage-credit research metric.
- **TS factor reconstruction:** Annual true-shooting, turnover, and offensive-
  rebound RAPMs train a side-specific ridge map on 2024, select alpha `100` on
  2025, and diagnose on 387 qualified 2026 players. Net correlation is `.974`,
  weighted R-squared `.948`, and RMSE `.470` versus `2.052` for the mean. TS
  alone reaches `.730` R-squared. This reconstructs same-season points RAPM;
  shared lineups and outcomes mean it is not independent validation.
- **Pair bucketing:** Exposure buckets increase 2025 pair-slot coverage from
  43.0% to 48.7% but lose. The hard-floor pair winner then loses to one-player
  RAPM by `+.471` RMSE on 2026, paired 95% `[+.260, +.688]`. The preregistered
  stop rule prevents wasteful age-adjusted trio, quartet, and lineup expansions.
- **Target horizon:** On the same 2020-26 next seasons, equal-season mean RMSE
  is `14.140` for five years, `14.177` for six, `14.183` for three, and `14.376`
  for one. Use five-year RAPM as the stable/predictive SPM-target challenger;
  keep one-year RAPM as the retrospective public estimand. This builds target
  panels; it does not silently redefine the current annual SPM feature model.
- **Five-year intervals:** Nine `3000 / 3000 / 300` rolling windows produce
  8,620 player-window rows. Point estimates reproduce the validated sufficient-
  statistics fit within `1.22e-7`. Published Lab intervals are fixed-window
  homoskedastic ridge sampling intervals; they do not include game clustering,
  ridge bias, or peak-window selection.
- **Luck adjustment:** Replacing realized threes and free throws with leave-
  current-game-out player-season empirical-Bayes expectations improves 2026
  RMSE by `-.093`, but the paired 95% interval `[-.237, +.057]` crosses zero.
  Keep as a challenger. The teammate-eFG fit excludes all 2,225 shots whose
  shooter cannot be matched to the terminal offensive lineup and fits 644,307
  valid shooter-removed rows; it remains descriptive, not causal spacing.
- **External reproduction:** CourtSignal 2019 normal RAPM matches 527 Ryan
  Davis tutorial players. Pearson correlations are `.937` offense, `.911`
  defense, and `.934` net; net rank correlation is `.919`. Our net spread is
  1.394 times the tutorial scale because the parsers, game coverage, home term,
  regularization selection, recentering, and output rounding differ.
- **Decision:** The local production candidate is rolling five-year terminal-
  lineup, zero-prior `3000 / 3000 / 300` RAPM with fixed-window analytic
  intervals. Do not add rubber-band, coach, age-27, pair-unit, multinomial, or
  luck adjustments. Age context and luck adjustment remain challengers; TS
  factors, WP credit, teammate eFG, and coach ratings remain research views.

# 2026-08-25 - Teammate effects and observable play channels, 2024-2026

- **Question:** Which players are associated with changes in their teammates'
  scoring, turnovers, assists, steals, blocks, and rebounding after controlling
  lineup context? How does ordinary lineup impact split across observable shot
  and scoring outcomes?
- **Teammate design:** Expand each valid opportunity into five focal-player
  rows. Subtract the focal player's own event from the team target. Fit the
  focal coefficient alongside separate nuisance blocks for the other four
  same-side players and five opponents. The focal penalty is 3,000; teammate
  and opponent nuisance penalties are 12,000 and 15,000 to account for their
  four- and five-fold row exposure. Positive turnover values mean fewer
  teammate turnovers. These coefficients remain descriptive associations.
- **Data:** 743,946 regular-season possessions, 646,545 shot attempts, and
  379,584 classified rebound opportunities from 2024-2026. Eligible source
  event mapping is 98.443%. Named scorer attribution covers 98.651% of official
  points; the scoring target itself uses exact canonical possession points.
  Named assister, stealer, blocker, and rebounder lineup coverage is 99.704%,
  99.769%, 98.251%, and 98.640%.
- **Outcome channels:** Fit rim assists, transition points, three-point points,
  free-throw points, non-rim two-point attempt frequency, and rim points using
  a shared `3000 / 3000 / 300` possession design. A second table uses mutually
  exclusive observable finish labels: transition, putback, cut, drive,
  pull-up, post-like, jump shot, and other.
- **Limit:** Finish labels come from shot descriptions and the official
  fast-break qualifier. Basic play-by-play cannot identify pick-and-roll ball
  handler, roll man, isolation, handoff, or post-up possessions reliably.
  Therefore the result is not called Synergy playtype RAPM.
- **Decision:** Keep both leaderboards in the localhost RAPM Lab. Do not add
  these coefficients to production AIO or make causal spacing/teammate claims.

# 2026-08-25 - Five-year time decay plus actual-age controls fails transfer

- **Question:** Does a current-age-conditioned, time-decayed five-year RAPM
  predict the next season better than ordinary unweighted five-year RAPM?
- **Design:** Use 2020-2024 to select on 2025, then freeze and refit 2021-2025
  for a reused 2026 diagnostic. Select half-life from no decay, 0.5, 1, 1.5, 2,
  3, and 5 years; then age penalty from 1,000, 10,000, and 100,000; then shared
  player penalty from 1,500, 3,000, and 6,000. Age enters as actual offensive
  and defensive lineup-age controls. Published challenger ratings add each
  active 2026 player's observed-age effect rather than standardizing everyone
  to age 27.
- **Selection:** A 5-year half-life, age penalty 1,000, and player penalty 6,000
  improve 2025 game-margin RMSE from 14.855 to 14.472 (`-.383`) and correlation
  from .369 to .415.
- **Diagnostic:** On the same 1,228 reused 2026 games, ordinary five-year RAPM
  scores 15.296 RMSE and .365 correlation. The frozen candidate scores 15.408
  and .354: `+.112` RMSE, paired 95% `[-.038, +.265]`, with only 7.4% of game
  resamples favoring the candidate.
- **Decision:** Reject promotion. Keep ordinary unweighted, no-age five-year
  RAPM as the stable production reference. Display the failed current-age
  challenger locally so the negative result is auditable. Season 2027 remains
  untouched.

# 2026-08-25 - Joint actual-clock rubber-band columns do not improve RAPM

- **Question:** Does the selected rubber-band shape work better when encoded
  directly beside home and player columns instead of subtracted from points?
- **Design:** Keep the possession target unchanged. Add eight actual six-minute
  columns containing the offense's pre-possession margin clipped at 15 and
  scaled to `[-1, 1]`. Fit those columns jointly with terminal-lineup offense,
  defense, and home. Player penalties remain `3000 / 3000 / 300`; select only
  context shrinkage on 2025 after fitting 2024, then refit 2024-25 and diagnose
  on the same 1,228 reused 2026 games.
- **Selection:** All context candidates lose to normal RAPM on 2025. The least
  harmful candidate is the largest tested context penalty, 3,000: RMSE 15.067
  versus 15.054. This boundary result favors shrinking the new columns away.
- **Diagnostic:** Neutral player-only RMSE changes from 15.473 to 15.504
  (`+.031`) while correlation changes from .334 to .344. The paired 95% RMSE
  interval is `[-.035, +.100]`; 19.1% of resamples favor the candidate. The
  full 2024-26 adjusted ratings correlate .990 with normal RAPM and move .262
  points per 100 on average.
- **Decision:** Reject promotion and retain normal RAPM. The empirical score
  curve remains descriptive. Run `rubberband_joint_clock_v1_7f055889f9` is
  local; Season 2027 was not loaded.

# 2026-08-25 - Smooth age and signed score controls fail the blocked gate

- **Question:** Do player ratings improve when the unchanged possession-points
  model jointly adds smooth lineup age and pre-possession score state beside
  player offense, player defense, and home?
- **Design:** The score search compared ten signed buckets at 1-5, 6-10,
  11-15, 16-20, and 21-plus points, clipped linear terms, and cubic splines.
  The age search compared cubic offense and defense lineup-age splines with 4,
  6, or 8 knots. Every model used `3000 / 3000 / 300` player and home penalties,
  selected nuisance shrinkage after fitting 2020-24 and scoring 2025, then
  refit 2021-25 and scored the same 1,228 reused 2026 games. Score context was
  neutralized at prediction time; known lineup ages remained in the prediction.
- **Coverage:** 1,656,346 possessions and 8,279 games. Age was known for
  99.922% of player slots. Season 2027 was not loaded.
- **Selection:** The least harmful score shape was the requested ten signed
  buckets at the strongest tested penalty, 10,000. Age-only selected 4 knots
  and penalty 100. The joint model also selected score penalty 10,000.
- **Player-only diagnostic:** Normal RAPM scored 15.296 RMSE and .365
  correlation. Removing all nuisance terms at prediction time, age-only scored
  15.454, score-only 15.333, and age-plus-score 15.397. Their RMSE changes were
  +.158, +.036, and +.101. Age-only's paired 95% interval was
  `[+.011, +.304]`; the other two intervals crossed zero.
- **Pregame-context diagnostic:** Known lineup ages remain available before a
  game, while the future score path does not. Under that rule, age-only scored
  15.284, a -.012 change with paired 95% `[-.161, +.129]`. Age-plus-score
  scored 15.443, a +.147 change. Calibration slopes were .816 normal, .750
  age-only, .738 score-only, and .662 joint; the higher correlations do not
  compensate for poorer scale calibration.
- **Decision:** Retain normal RAPM. Every player-only adjusted rating loses.
  Known age is neutral in pregame prediction, and signed score buckets fail.
  Runs `rubberband_score_signal_v1_deac872ede` and
  `age_score_context_v1_7e8689fee8` remain local.

# 2026-08-25 - External RAPM reproduction and plus-minus comparators

- **Question:** Do CourtSignal's saved RAPM panels agree with independent RAPM
  implementations, and how do non-RAPM plus-minus variants relate to the same
  ratings?
- **Alignment policy:** Use NBA player ID plus exact season or rolling window
  whenever the reference provides IDs. Name-only comparisons use deterministic
  accent and punctuation normalization, exclude ambiguous duplicate names, and
  never use fuzzy matching. Every result is labeled `exact_key_scope`,
  `same_window_weight_mismatch`, `different_estimand`, or `invalid_direct`.
- **Exact RAPM checks:** Across 5,217 matched 2014-2023 player-seasons, Ryan
  Davis net RAPM has Pearson `.967`, rank correlation `.962`, and a CourtSignal
  scale slope of `1.391`. Ryan's labels use NBA season start years: `2018-23`
  means the five seasons ending 2019-2023. Across 5,513 exact five-year rows,
  net Pearson/rank correlation is `.957/.948`; across 5,869 exact three-year
  rows it is `.980/.970`.
  Ryan's annual luck-adjusted net RAPM is less aligned: Pearson `.777`.
- **Current xRAPM:** The 2024-2026 net table matches 687 NBA IDs with Pearson
  `.897`, rank correlation `.888`, and slope `.998`. This is not an exact model
  replication because xRAPM gives less weight to 2024 and 2025 while the
  CourtSignal three-year fit weights each possession equally.
- **Different-estimand checks:** Pooled net Pearson correlations are `.574` for
  official DARKO WOWY season averages (2017-2026), `.436` for RAPTOR on/off
  (2014-2022), and `.634` for the local legacy AuPM (2014-2024). RAPTOR's
  low-exposure tails explain its Pearson/rank divergence: requiring 1,000
  minutes raises net Pearson/rank correlation to `.917/.912` across 2,449
  player-seasons. The 2024 raw PBPStats on-court margin correlates `.711` with
  one-year RAPM. These are descriptive agreement checks, not validation targets.
- **AuPM reproduction:** The archived local formula reproduces the stored AuPM
  column to maximum absolute error `1.78e-15`. It is a local historical formula
  and is not labeled as canonical Ben Taylor AuPM.
- **Game-level PM:** A transparent GPM-style ridge uses final home margin per
  100 and signed player minute shares. Lambda `10` wins the 2024-to-2025
  selection fold. Refit on 2024-2025, it scores 15.413 RMSE and `.334`
  correlation on reused 2026 games. Its full 2024-2026 player ratings correlate
  `.621` with three-year possession RAPM. Home advantage is the unpenalized
  home-margin intercept, `+1.696` points per 100. It is not an exact WOWYR
  reproduction.
- **Long-span references:** The downloaded 1997-2024 file correlates `.944`
  with the older exact-scope CourtSignal export, but the comparison is marked
  invalid for current-model validation because the join is name-only and the
  CourtSignal artifact came from a legacy engine. The downloaded 2017-2025
  total-only file is not scored because no exact-window artifact exists; it is
  not compared dishonestly with the 2014-2026 fit.
- **Decision:** The independent same-key RAPM checks are strong enough to rule
  out a gross sign, join, or scale failure. They do not replace future-game
  validation. Keep all comparison data and leaderboards localhost-only under
  run `external_reproduction_benchmark_v1_0a95702214`.

# 2026-08-25 - DARKO WOWY and RAPTOR on/off reproductions

- **DARKO public aggregation:** Downloaded 1,478 complete public player-game
  Final Cut histories with zero failures. The simple unweighted season mean
  reproduces every public offense, defense, and net season average across 5,497
  player-seasons. All Pearson and rank correlations are 1.0. Maximum absolute
  error is below `7.55e-15`. This verifies the public season-average operation,
  not DARKO's private daily model or smoothing.
- **RAPTOR table identity:** The local modern RAPTOR player table matches the
  official FiveThirtyEight GitHub file across all 4,684 non-null on/off rows.
  Offense, defense, and net values match exactly.
- **RAPTOR-style reproduction:** Built the disclosed three-family courtmate
  chain from regular-season possession lineups. The shared offense/defense fit
  trains on 2014-2018 and is frozen for 2019-2022. For 1,000-minute seasons, the
  proxy reaches `.9658` net Pearson and `.9575` rank correlation against the
  published regular-season team-stint target across 1,036 player-seasons. This
  target is distinct from the player CSV identity check, which combines regular
  season and playoffs. Fitted coefficients are `+.5919` for opposition-adjusted own
  on-court rating, `-.5964` for direct courtmates without the player, and
  `+.2431` for second-order courtmate context. Courtmate context stays within
  team for traded players, and second-order context excludes the focal player.
- **Limit:** FiveThirtyEight did not publish its three coefficients, exact
  opposition adjustment, or exact second-order weighting. The fit is labeled
  `RAPTOR-on/off-inspired proxy`, never an exact algorithm reproduction. The
  feature families are correlated, so their fitted coefficients are not causal
  or individually identified player-value weights.
- **Decision:** The public data paths, aggregation, signs, and high-exposure
  behavior pass. Keep these checks local and diagnostic. They do not turn WOWY
  or RAPTOR on/off into predictive targets for CourtSignal RAPM.

# 2026-08-26 - Predictive SPM harness repaired; official defense source is a null

- **Harness defects:** The checked-in predictive SPM runner parsed its YAML
  contract as JSON, scored persistence on fewer player-seasons than raw SPM,
  reported unweighted correlation where the contract asked for weighted
  correlation, omitted held-out calibration slope, and did not enforce pinned
  sources, folds, or the Season 2027 exclusion. The runner now validates YAML,
  exact run IDs and folds before data access, scores every valid comparator on
  one row intersection, and reports weighted correlation, dispersion, and
  calibration. Five focused tests cover these contracts, including rejection
  of Season 2027 before any parquet read.
- **Phase A:** `spm_defense_source_transition_v1` compares the public feature
  artifact, the 2014--26 refresh input, and the latest official-defense input.
  All 50 selected non-official defense features are identical between the
  refresh and latest panels. Ten tracking features change on 2014--24 rows.
  The eight selected matchup fields change only in 2025--26. Old and new
  observed-source masks are reconstructed from the pinned raw sources rather
  than inferred from filled values. The no-fit lineage gate passes.
- **Phase B:** Keep the 68 selected defense fields, ridge alpha 3000, square-root
  possession weights, rows, targets, and frozen offense learner fixed. Compare
  the old source against the repaired official DFG/rim source in expanding
  2014-to-prior-season folds testing 2020--24. Missing source values remain
  missing until train-only median imputation with indicators.
- **Result:** Equal-season weighted defense RMSE is `0.919063` for the control
  and `0.919162` for the repaired-source challenger, a `+0.000099` change. The
  challenger wins two of five folds. Across 5,000 paired player-cluster draws,
  the probability of lower RMSE is `0.480`; the 95% interval for
  challenger-minus-control RMSE is `[-0.001004, +0.001516]`. Weighted
  correlation changes by `+0.000116`. Low- and high-exposure safety gates pass,
  but the primary, fold-consistency, and paired-evidence gates fail.
- **Decision:** Record a clean null and stop before AIO. The official sources
  improve provenance and 2026 coverage but do not improve historical defense
  prediction under the frozen model. Seasons 2025 and 2026 were not used in
  this comparison. Season 2027 was not loaded.

## 2026-08-26 - Corrected predictive SPM rescore

- **Method:** Rescored the frozen `predictive_spm_v1` after repairing contract
  parsing, exact artifact pins, common comparator rows, weighted correlation,
  held-out calibration reporting, row inclusion reasons, and fold-level
  checkpoint/resume. The features, learners, folds, and targets did not change.
  OpenMP and BLAS were limited to two threads after the first timed attempt
  exceeded the overnight CPU cap and was stopped before producing a result.
- **Result:** Across development folds 2019-24, equal-season weighted net RMSE
  is 1.9651 for prior-season RAPM persistence, 1.6094 for raw predictive SPM,
  and 1.6083 for calibrated SPM. Across the already-inspected 2025-26
  diagnostics, the values are 2.0826, 1.7546, and 1.7563. Every arm in every
  fold uses the same player-season rows.
- **Failure:** Defense dispersion remains far below the preregistered 0.85-1.15
  band. Raw defense ratios are 0.353 in 2025 and 0.334 in 2026; calibrated
  ratios are 0.322 and 0.309. The RMSE gain is therefore not sufficient for
  promotion.
- **Decision:** Retain `predictive_spm_v1_9392b98d58` as a research forecast and
  comparator. Do not promote it as the predictive SPM or production AIO prior.
  Season 2027 was rejected before data access and was not loaded.

## 2026-08-26 - Matched-window target-horizon pilot

- **Question:** Does the earlier five-year RAPM portability win survive after
  fitting an SPM on five-year statistical windows and using it as the fixed AIO
  center?
- **Method:** Froze a two-fold pilot on 2023 and 2024 future games. Compared
  one- and five-year windows using the same 126 offense and 50 defense feature
  names, matched-window RAPM labels, the frozen offense histogram-GBM and
  defense ridge learners, strictly earlier SPM training windows, and identical
  held-out game row sets. No 2025, 2026, or 2027 data was used.
- **Result:** One-year SPM-centered AIO improves zero-prior RAPM by 0.050 RMSE
  in 2023 and 0.158 in 2024. Five-year zero-prior RAPM is stronger than
  one-year zero-prior, but the five-year SPM center worsens it by 0.126 and
  0.020. Mean future-game RMSE is 13.6956 for one-year centered AIO, 13.6989
  for five-year zero-prior, 13.7718 for five-year centered AIO, and 13.7997 for
  one-year zero-prior.
- **Decision:** The pilot is valid and justifies adding three-year, six-year,
  and expanding-history arms. Do not promote five-year SPM. Easier label
  reconstruction and smoother RAPM do not establish a better downstream prior.

## 2026-08-26 - Full matched-window SPM target-horizon comparison

- **Method:** Rebuilt complete one-, three-, five-, six-year, and
  expanding-from-2014 RAPM targets for window ends 2014-23. Rolling windows use
  pre-2014 source seasons when needed so each label and statistical feature
  window has its named length. Fit the same 126 offense and 50 defense features
  with the frozen learners, trained only on earlier window ends, then scored
  zero-prior and full-SPM-centered RAPM on identical 2020-24 future games.
- **Result:** Five-year zero-prior RAPM wins mean future-game RMSE at 13.7681,
  ahead of six-year zero-prior at 13.8010 and three-year at 13.8189. In 10,000
  paired whole-game resamples, challenger-minus-five-year intervals are
  [0.0134, 0.0527] for six-year and [0.0058, 0.0951] for three-year. All other
  intervals also favor five-year.
- **SPM result:** Only the one-year SPM center improves its matching zero-prior
  model, by 0.0962 RMSE on average. Three-, five-, six-year, and expanding SPM
  centers all worsen downstream RMSE. The five-year center loses in every fold
  despite its stronger label correlation.
- **Decision:** Keep one-year RAPM as the retrospective SPM target. Freeze
  five-year zero-prior RAPM as the predictive history backbone. Do not force a
  full-strength SPM prior into that fit. Maximum loaded season was 2024;
  Seasons 2025, 2026, and 2027 were not used.

## 2026-08-26 - Predictive SPM trajectory ablation

- **Question:** Do age or lagged opportunity repair the frozen predictive SPM
  before it is used as a current-strength prior?
- **Method:** Compared the unchanged raw forecast with shared-age, separate
  offense/defense age, and side-age-plus-lagged-minutes/games residual ridge
  corrections. Every test season used only earlier target seasons. Selection
  used equal-season weighted net RMSE over 2020-24; 2025-26 were reused
  diagnostics only.
- **Result:** Raw SPM wins development RMSE at `1.604696`. Side-age scores
  `1.612733`, shared-age `1.612834`, and side-age plus opportunity `1.621506`.
- **Decision:** Freeze the raw predictive SPM. Age and opportunity are clean
  nulls for this model; neither enters the current-strength prior.

## 2026-08-26 - Predictive current-strength AIO

- **Method:** Fit terminal-lineup RAPM over five trailing seasons with fixed
  `3000/3000/300` penalties. Selected exponential possession decay from
  half-lives of `0.5`, `1`, `2`, `3`, `5`, and no decay on future-game margins
  from 2020-24. Compared zero-prior and raw predictive-SPM-centered fits on
  identical held-out games. No center-scale or player-penalty search was run.
- **Result:** A two-year half-life wins. The decayed SPM-prior AIO has mean
  development game-margin RMSE `13.7122`, versus `13.7429` for decay alone,
  `13.7550` for an undecayed SPM-prior fit, and `13.7681` for five-year
  zero-prior RAPM. It wins four of five folds. A 10,000-draw paired whole-game
  bootstrap favors it over each frozen comparator. Reused 2025 and 2026 RMSEs
  are `14.7719` and `15.1817`, versus `14.8551` and `15.2962` for the frozen
  five-year zero-prior reference.
- **Decision:** Keep `predictive_current_aio_2026_v1_c18e2472ec` as the
  research current-strength champion. It is not confirmed or public. Season
  2027 remains untouched.

## 2026-08-26 - CourtSignal 2026 research bundle

- **Artifact:** `courtsignal_2026_research_bundle_v1_3913f9efd6` joins two
  explicitly different 2026 estimands: retrospective annual SPM and predictive
  current-strength AIO.
- **Lineage:** The annual SPM row is the 2026 out-of-fold prediction from
  `single_season_spm_v1_47b3bd9b17`, trained on 2014-25. The predictive row
  uses 2021-25 possessions, two-year decay, and a raw predictive-SPM prior
  trained through 2025.
- **Coverage:** The bundle contains 582 active 2026 players. A predictive prior
  is available for 79.73%; rookies and nonconsecutive players receive the
  neutral RAPM center rather than fabricated forecasts.
- **Decision:** Local research use only pending untouched confirmation and a
  complete player-population policy. No raw NBA data enters the bundle.

## 2026-08-26 - Four-arm luck-adjusted RAPM target experiment

- **Data:** Used 743,946 canonical 2024-26 terminal-lineup possessions, 654,376
  field-goal attempts with location, complete FT events, and 2014-25 annual
  player shooting histories. Mapped conversion events account for 99.00% of
  possession points. No download was required.
- **Expectations:** Selected player-neutral logistic regularization and
  empirical-Bayes shooting half-lives/prior strengths through 2024 only.
  Expected-shot predictions exclude the current game; 2025 and 2026 use only
  earlier-season shots. Player skill add-backs use only seasons before the
  adjusted season.
- **Four arms:** Normal realized points; normal offense plus expected-conversion
  defense; expected-conversion RAPM plus an offense-only repeatable-shooter
  add-back; and fully player-neutral expected conversion.
- **Game result:** Normal RAPM wins both reused diagnostics. Its RMSE is
  `15.0541` in 2025 and `15.4732` in 2026. The three broad luck arms score
  `15.5132/15.5675`, `15.6971/15.5693`, and `15.7966/15.7941`.
- **Earlier result reproduced:** A separate player-skill FT/3P joint adjustment
  changes RMSE by `+0.0656` in 2025 and `-0.0834` in 2026. The 2026 result is
  close to the earlier documented `-0.093`, but its paired 95% interval is
  `[-0.2318, +0.0717]`, and the same frozen arm loses in 2025.
- **Future RAPM:** Expected-outcome arms lower future-normal-RAPM RMSE mainly by
  compressing dispersion, while losing net correlation to normal RAPM in both
  years. They therefore do not pass the downstream AIO gate.
- **SPM stop:** Complete expected-outcome labels begin in 2024. Only 2024 and
  2025 are legal training labels for a 2026 SPM, which is insufficient for the
  required chronological feature and learner selection. No luck SPM was
  forced or promoted.
- **Decision:** Retain normal realized-points RAPM. Record all luck variants as
  research nulls. Do not redesign on reused 2025/26 results. Season 2027 was
  not loaded.

## 2026-08-26 - Chronologically selected current player skills

- **Question:** Can the site display current underlying player skills without
  treating noisy current-season rates as ability or mixing them into impact?
- **Method:** Registered 34 shooting, creation, rebounding, and defense skills.
  Compared previous-season raw, career empirical Bayes, time-decayed empirical
  Bayes, and a time-decayed-plus-age residual model on six future-season folds
  from 2019 through 2024. Shooting proportions use grouped-binomial log loss;
  other skills use opportunity-weighted RMSE. Age must win at least four of six
  folds. Role conditioning was skipped because consistent frozen pre-season
  labels do not cover the folds. Parameters were selected through 2024, refit
  through 2025, and updated with observed 2026 data. Season 2027 was not loaded.
- **Result:** Run `predictive_player_skills_2026_v1_9271b4b024` contains 34
  selected skills, 235,212 player-skill-season rows, 558 current players, and
  303 players with raw 2026 observations for all skills. Selected arms are one
  career EB, 20 time-decayed EB, and 13 time-decayed EB plus age.
- **Decision:** Pass as a research current-skill surface. Keep it localhost-only
  until source rights, continuous-skill uncertainty, and an untouched season
  are resolved. Do not feed these estimates into RAPM, SPM, or AIO by default.

## 2026-08-26 - Independent audit corrections for current player skills

- **Audit:** Four independent read-only reviews covered statistical semantics,
  temporal leakage and lineage, basketball interpretation, and localhost UI.
- **Age correction:** The first artifact selected 13 age arms but served an
  EB-only posterior. Run `predictive_player_skills_2026_v1_a7eb0386fe` now
  applies the selected age residual to the preseason estimate and then updates
  it with current observations. The posterior identity error is zero. All 7,254
  comparable 2026 age-arm rows changed; median absolute change is `0.130` in
  each skill's native unit and the 95th percentile is `1.199`.
- **Rebounding correction:** Offensive and defensive recorded chances are not
  bounded binomial trials. Both skills now use rate RMSE/MAE. Selection still
  chooses a one-year half-life with prior strengths 100 and 250, respectively.
- **Game trajectory correction:** Playoffs are excluded. FT and three-point
  charts use the exact frozen preseason estimate and precision, stop at the
  regular-season prefix matching annual source totals, and must finish at the
  annual posterior. The build reconciles 490 FT and 511 three-point player
  series; unmatched series are withheld.
- **Lineage correction:** Predictive-SPM manifest paths are portable, target-
  horizon resume identities include statistical source hashes before any
  checkpoint reuse, and current AIO priors must match the registered source run
  with training cutoffs strictly before forecast seasons.
- **Decision:** Pass the corrected artifact for research use and the local UI.
  Keep 2025-26 labeled reused diagnostics, keep the AIO bootstrap conditional
  on its selected candidate set, and leave Season 2027 untouched.

## 2026-08-26 - Five-year SPM and one-season AIO

- **Question:** Does an SPM trained on matched five-year statistical and RAPM
  windows provide a better center for one-season RAPM than the annual-target
  SPM?
- **Design:** Forward-chained five-year window ends from 2018 through 2026;
  frozen histogram GBM offense, ridge defense, 127/68 feature contract, and
  square-root target-possession weights. The AIO likelihood remains only the
  rated season's terminal-lineup possessions with `3000 / 3000 / 300`
  penalties and center scale one.
- **Result:** Selected run `five_year_target_spm_v1_65550acb79` improves mean
  next-season game-margin RMSE from `14.4705` for annual-prior AIO and `14.5697`
  for zero-prior RAPM to `14.4005`. It wins every 2022--26 season. Mean
  correlation improves to `.3652` from `.3462` and `.3219`.
- **Uncertainty:** Development paired-game MSE difference is `-1.6763`, 95%
  interval `[-3.0445, -0.1203]`; reused 2025--26 difference is `-2.6385`,
  interval `[-4.6215, -0.7399]`.
- **Tradeoff:** Standalone next-year one-season-RAPM net RMSE worsens from
  `1.7543` to `1.9832`, while correlation rises from `.4106` to `.4232`.
- **Decision:** Replace the annual-target SPM as the research AIO prior. Do not
  change the public model until untouched 2027 confirmation.

## 2026-08-26 - Same-season Basketball Index and RAPTOR feature families

- **Leak boundary:** Annual empirical-Bayes features use only that season's
  league center and that player-season's opportunities. The defensive tracking
  builder no longer falls back to an all-season median. Missing source seasons
  are neutral zero. Five-year pooling occurs after annual estimates are frozen.
- **Candidates:** Tested eight grouped additions: shooting context, passing
  context, screening, playtype/transition, defensive hustle, defended-shot
  context, matchup 3PA volume, and opponent eFG outcome.
- **Selection:** Development was 2022--24. A group needed lower mean future-
  season RMSE, at least two fold wins, and less than `0.01` RMSE/correlation
  degradation among primary-team changers. Opponent shooting outcome was a
  falsification family and could not promote.
- **Selected:** Three passing features on offense; defended-2P value, rim
  matchup share, contested-3 share, and matchup 3PA share on defense.
- **AIO result:** Selected-minus-baseline game RMSE was `-0.0097`, `-0.0061`,
  `-0.0044`, `-0.0019`, and `+0.0126` in 2022--26. The final reused season
  reverses the small earlier gains.
- **Decision:** Keep run `five_year_spm_feature_research_v1_93c148510e` as a
  localhost-only challenger. Do not replace the five-year SPM reference and do
  not use 2027 before the untouched confirmation.

## 2026-08-26 - Matched public all-in-one comparison

- **Question:** How do the published website and five-year research AIO ratings agree with public
  all-in-one metrics, and how well does each aggregate to next-season team wins?
- **Contract:** Pairwise-complete offense, defense, and net correlations use
  2021--24 player-seasons with at least 250 minutes. The team test weights each
  year-Y rating by observed year-Y+1 minutes, replaces missing or sub-250-minute
  ratings at `-2.0`, multiplies the weighted mean by five, and correlates it with
  year-Y+1 win percentage. Replacement sensitivity covers `-3.0` through
  `-1.5`.
- **Result:** Corrected run `public_aio_benchmark_v1_e411f910ea` ranks MAMBA
  first at mean R-squared `0.6736`, then xRAPM `0.6439`, five-year research AIO
  `0.6405`, EPM `0.6270`, and Website AIO `0.6183`. The research and website AIO
  net ratings correlate `0.9596`. The prior comparison mislabeled a second
  five-year model as “Old AIO”; this run uses the exact website leaderboard rows.
- **Limits:** This is oracle-minutes retrodiction, not a preseason forecast.
  Only four folds are common. Supplied historical EPM may use today's model
  rather than archived season-end vintages. BoxPIPM-style is a transparent
  box-only baseline, not full PIPM.
- **Decision:** Add the benchmark, full correlation matrix, fold table, and
  metric definitions to the localhost SPM Lab. Do not promote the new AIO from
  these results. Add archived projected minutes before claiming forecast value.

## 2026-08-26 - Published annual SPM sample-weight ablation

- **Question:** Does `sqrt(min(Poss_Off, Poss_Def))` improve the exact published
  annual SPM relative to giving every player-season equal training weight?
- **Design:** Refit the website's 127-offense/68-defense feature contract on all
  2014--26 leave-one-season-out folds. Features, annual RAPM labels, learners,
  and held-out rows are identical; only the training sample weight changes.
  Score both possession-weighted and equal-player RMSE.
- **Result:** Under possession-weighted scoring, removing the weight changes
  offense RMSE from `1.01234` to `1.01180`, defense from `0.94474` to `0.98145`,
  and net from `1.38808` to `1.42511`. Under equal-player scoring, it improves
  offense from `0.93175` to `0.92403` but worsens defense from `0.89000` to
  `0.89502` and net from `1.28793` to `1.29651`.
- **Runtime caveat:** The current histogram-GBM runtime correlates `0.9961` with
  the saved website offense predictions but is not bitwise identical; defense
  reproduces exactly. Both ablation arms use the same current runtime.
- **Decision:** Keep weighting for the combined annual SPM because defense and
  net improve materially. Test unweighted offense plus weighted defense as the
  next frozen challenger; do not silently change the website fit.

## 2026-08-26 - Five-year SPM redundancy and next-season importance

- **Question:** Which saved five-year SPM feature families matter for
  next-season player RAPM, and do redundant raw/era-relative encodings improve
  chronological generalization?
- **Scope:** Run `five_year_spm_feature_audit_v1_4172cf5408` uses the exact
  persisted 126-offense/50-defense base matrix. The later 130/72 challenger did
  not persist its extended training matrix, so its added fields are not in the
  individual permutation table. Future feature-research runs now save that
  matrix.
- **Design:** Train on five-year windows ending before each rating season,
  predict the following season's one-year zero-prior RAPM in 2019--23, and
  score identical players with weighted MAE/RMSE, Pearson, and Spearman.
  Grouped permutation is primary; individual permutation is diagnostic.
- **Redundancy:** Sixteen selected pairs have absolute correlation at least
  `.95`; eight exceed `.98`; five exceed `.995`. Raw and era-relative at-rim
  frequency, assists, FTA, turnovers, and points are effectively duplicate
  five-year signals.
- **Importance:** Offense shooting/scoring/spacing is positive in all five
  folds (`+0.110` MAE when shuffled). Defense disruption, creation/role,
  foul-pressure, and rebounding families are positive in all five folds.
  Offense creation/passing and ball-security have negative mean MAE importance.
- **Pruning result:** Dropping redundant raw fields improves RMSE in all five
  folds but worsens Pearson and Spearman in all five. Dropping the redundant
  relative fields improves Pearson in four folds but worsens MAE and RMSE in
  three. No variant wins the multi-metric gate.
- **Decision:** Keep the frozen five-year model. Next test must refit compact
  mechanism-level offense and defense specifications under the same
  chronological player, team-changer, and downstream game gates. Do not select
  a model because lower dispersion improves RMSE.

## 2026-08-26 - Role-conditioned five-year SPM and zone shotmaking

- **Question:** Do annual offense/defense roles or a zone-adjusted shotmaking
  feature improve prediction of the following season's one-year RAPM?
- **Design:** Run `five_year_spm_role_research_v1_3edacae610` trains the exact
  persisted 126-offense/50-defense five-year SPM on prior window ends and tests
  2021--23. Every variant scores the same players. The primary score is the
  equal-fold mean next-season possession-weighted Pearson correlation;
  Spearman and MAE are guardrails, with RMSE diagnostic. Soft roles add annual
  role coordinates and indicators. Hard experts split the frozen model by
  role, with a global fallback below 100 training rows.
- **Role result:** Soft role context improves net Pearson by `.00446`, net
  Spearman by `.00237`, net MAE by `.01002`, and net RMSE by `.01720`. Offense
  is essentially tied; defense correlation rises `.00145` while defense MAE
  worsens `.01805`. Hard experts reduce net Pearson `.00323` and worsen net MAE
  `.04184`. The defense role map was developed through 2021, so only the
  2022-to-2023 fold is strictly post-map; soft-role net Pearson improves
  `.00690` on that fold.
- **Shotmaking:** Added a descriptive five-zone points-above-expectation metric
  using leave-one-player-out window baselines for rim, short-mid, long-mid,
  corner-three, and arc-three attempts, then `attempts / (attempts + 200)`
  shrinkage. Unlike the existing defender-distance metric, it does not reward
  an easy rim-heavy zone mix by itself. As an SPM input it lowers offense
  Pearson in all three folds (mean `-.00300`) and worsens offense MAE `.00700`.
- **Decision:** Keep soft role context as a research challenger. Do not replace
  the SPM from one clean post-map defense fold. Reject hard role experts. Keep
  zone shotmaking as a player-skill metric, not a current SPM feature.

## 2026-08-26 - Downstream team-win gate and tracking clarification

- **Correction:** The prior role decision used next-season one-year RAPM as its
  primary target. The requested deployment-adjacent target is next-season team
  wins after applying season-Y ratings to observed player-team minutes in Y+1.
- **Design:** Run `spm_role_team_win_benchmark_v1_21bdb974c8` evaluates the
  same role-known player cohort for 2020--22 ratings and 2021--23 outcomes.
  Team strength is five times the next-season-minute-weighted player rating;
  missing/sub-250-minute ratings receive -2.0. Player-team minute identity
  coverage is at least 99.9875%.
- **Result:** Baseline mean R-squared is `.5573`. Soft role context reaches
  `.5660` (`+.0088`); hard role experts reach `.5839` (`+.0266`); zone
  shotmaking falls to `.5489` (`-.0083`). Hard experts lose in the first fold
  and win the next two. In the only strictly post-role-development fold,
  2022-to-2023, hard experts improve R-squared from `.4509` to `.5197`.
- **Limits:** This is oracle-minutes retrodiction, not a forecast, and contains
  only 90 team-seasons. Qualifying rating coverage is 83.1--85.9% of outcome
  minutes. Retain hard and soft roles as challengers; neither is promoted.
- **Feature clarification:** The offense/defense permutation tables were not
  swapped, but they are model-dependence diagnostics rather than causal
  feature rankings. `rim_points_saved_p100` already exists in the extended
  defense contract. Added `event_stops_p100` to the feature builder using
  steals, recovered blocks, charges drawn, and offensive fouls drawn; it is not
  Dean Oliver Stop% and is not selected until it clears the downstream gate.
- **Shot-data boundary:** Exact modern PBP has location and lineups but no
  shot-level nearest defender. Aggregate tracking has defender distance,
  dribble/touch, catch/pull-up, and playtype context in separate marginals.
  Do not fabricate a joint row. Use the public 2014-15 shot log only for a
  historical expected-shot prototype until a permitted current row source is
  available.

## 2026-08-26 - Historical shot location and nearest-defender prototype

- **Question:** Does exact shot location plus nearest-defender and pre-shot
  context improve expected-make prediction on the one public row-level season?
- **Design:** Run `historical_shot_quality_2015_v1_70e64472e0` uses the 128,069
  public 2014-15 shot-log rows and fuzzy-matches 99.40% to local PBP coordinates
  and fast-break flags within five seconds. The first 675 games train a fixed
  histogram GBM; the later 229 games are untouched test rows.
- **Result:** Location-only log loss/AUC are `.6450/.6337`. Adding nearest
  defender distance improves them to `.6381/.6504`. Adding shot clock,
  dribbles, touch time, period, home, and fast-break context improves them to
  `.6340/.6576`.
- **Leak caught:** The first run included PBP `assisted` and reached AUC `.9118`.
  Assists are recorded only after made shots, so that feature reveals the
  outcome. Run `historical_shot_quality_2015_v1_25b7a61b78` is explicitly
  marked invalid. The valid run neither uses assists nor matches on make/miss;
  zero dribbles and short touch time proxy receiving a pass.
- **Decision:** The richer expected-shot design is valid as a historical
  prototype. It cannot enter current SPM because no permitted modern source
  supplies exact location and nearest-defender distance on the same shot row.

## 2026-08-26 - KOBE input completion and SelfORB/Rim Points Saved definitions

- **Source correction:** Gabriel's pinned `54b57cf` annual sheets contain the
  observed `SelfOReb` count. The resumable ingest now covers the previously
  absent 2014--19 and 2024 files, so all 2014--26 model seasons have the field.
- **SelfORB adjusted TS:** The feature builder now calculates
  `PTS / (2 * (FGA + 0.44 * FTA - SelfOReb))`. It also emits `SelfOReb` per 100.
  Run `statistical_features_v1_8df75d821e` validates 6,942 player-seasons with
  no duplicate keys. Among 5,913 rows with at least 250 offensive possessions,
  adjusted TS averages 1.02 percentage points above ordinary TS and correlates
  `.9922` with it. These are research candidates; the frozen SPM feature list
  is unchanged.
- **Rim Points Saved correction:** The existing `rim_points_saved_p100` is the
  empirical-Bayes stabilized field, using `DFGA / (DFGA + 100)`. The builder
  now also emits `rim_points_saved_p100_raw`, the literal
  `2 * rim_DFGA_p100 * (normal rim FG% - defended rim FG%)`. For 300 attempts,
  60% normal shooting, and 55% defended shooting, the raw total is 30 points.
- **KOBE-inspired rerun:** Run `historical_shot_quality_2015_v1_c5258e797c`
  adds period clock and shooter-minus-nearest-defender height to the prior
  context arm. Height-difference coverage is 99.83%. Test log loss improves
  from `.6340` to `.6336`, and AUC rises from `.6576` to `.6591`.
- **Boundary:** This is not a direct KOBE reproduction. Narsu used separate
  close- and long-shot logistic models. The CourtSignal prototype uses one
  histogram GBM and a later within-season test. Modern SPM still lacks a
  permitted row-level source joining exact shot context to nearest defender.

## 2026-08-26 - Sparse function-first five-year SPM

- **Question:** Can one feature per declared player function retain the full
  five-year SPM's downstream signal? The frozen challenger uses seven offense
  fields and five defense fields. Separate ridge models use alpha 3000,
  within-window feature z-scores, training-fold median imputation, and the
  existing square-root possession weight. Roles and player demographics are
  absent.
- **Data and split:** Run `sparse_function_spm_v1_4f1ecaa353` rebuilds 8,620
  complete 2014--26 five-year player windows from the pinned Gabriel sheets.
  Historical predictions for 2021--26 use only earlier five-year RAPM labels.
  The rebuilt 2018--23 inputs correlate at least `.9999999` with the stored
  reference inputs. Season 2027 is absent.
- **Primary result:** On identical 2021 and 2022 ratings applied to observed
  next-season minutes, mean team-win R-squared falls from `.5446` for the full
  five-year SPM to `.4547` for the sparse model. It loses both folds.
- **Secondary result:** Across five next-season one-year RAPM folds, sparse net
  RMSE improves from `1.9813` to `1.9027`, but Pearson falls from `.4241` to
  `.3519` and Spearman from `.3308` to `.2727`. The compact ridge mainly
  shrinks dispersion rather than preserving player ordering. Offense and
  defense show the same pattern.
- **Decision:** Null. Keep the full five-year SPM. Do not run the sparse AIO or
  tune these hand-picked inputs after reading the result. A user-authored
  function list will be a new frozen experiment, not a revision of this run.

## 2026-08-26 - Shooting-foul lineage correction and principal-selected SPM

- **Lineage correction:** Gabriel's `ShootingFouls` field is shooting fouls
  committed, not drawn. It correlates `.951` with total personal fouls in the
  pinned 2025 sheet. The source separately exposes
  `TwoPtShootingFoulsDrawn` and `ThreePtShootingFoulsDrawn`. The feature builder
  now emits their sum per 100 offensive possessions as
  `shooting_fouls_drawn_p100` and emits `ShootingFouls / DefPoss * 100` as
  `shooting_fouls_committed_p100`. A regression test fixes both denominators.
- **Prior run:** `sparse_function_spm_v1_4f1ecaa353` used the mislabeled field
  on offense and is now invalid for its declared contract, not a valid null.
  The full five-year SPM baseline also contains that field in its offense list;
  preserve its predictive numbers but refit before clean side interpretation.
- **Frozen challenger:** Run `hand_selected_sparse_spm_v1_f04379a684` uses the
  first twelve principal-named metrics: PTS/100, zTS, stabilized cTOV, Box
  Creation, OREB/100, spacing, offensive load, rim attempts/100, event
  stops/100, rim points saved/100, defensive-rebound contests/100 and shooting
  fouls committed/100. Live-ball turnovers are thirteenth and excluded.
- **Coverage:** zTS spans 2014--26. Defended-rim data span 2014--25 and match
  99.49% of source rows to player IDs; the 2026 five-year row pools observed
  2022--25 rim seasons. No 2027 data are loaded.
- **Result:** Mean next-season team-win R-squared is `.4720` versus `.5446` for
  the full model, with losses in both folds. Five-fold next-season one-year
  RAPM net Pearson is `.3986` versus `.4241` and Spearman is `.3108` versus
  `.3308`. Net RMSE improves to `1.8913` from `1.9813`, again indicating useful
  shrinkage without enough downstream ordering signal.
- **Decision:** Research null. Keep the full five-year SPM as baseline, correct
  its foul-field lineage before promotion, and do not run the hand-selected
  challenger's AIO update.

## 2026-08-26 - Sparse factor-target SPM and teammate context

- **Question:** Can small related feature sets estimate annual shooting-TS,
  turnover and offensive-rebound RAPMs, and does leave-one-player-out teammate
  context help? Run `factor_target_sparse_spm_v1_5b120e918f` trains on 2024,
  selects per-target ridge penalties on 2025, refits on 2024--25 and diagnoses
  387 qualified 2026 players. Season 2027 is untouched.
- **Features:** Each of six factor-side heads uses two to five directly related
  inputs. The context candidate adds two or three possession-weighted
  same-team fields after subtracting the focal player's contribution. Context
  covers spacing, creation, rim pressure, turnover burden, offensive load,
  OREB, DREB, contests, event stops, deflections and rim points saved. Roles,
  demographics, minutes, games, on/off and external metrics remain excluded.
- **Factor targets:** Context improves all six 2025 selection RMSEs and four of
  six reused 2026 RMSEs. On 2026 it raises turnover offense/defense R-squared
  from `.266/.367` to `.295/.413`, offensive-rebound offense/defense from
  `.279/.126` to `.432/.159`, and leaves shooting offense/defense flat at
  `.217/.035` versus `.218/.037`.
- **Normal RAPM reconstruction:** Oracle factor ratings reconstruct 2026 normal
  net RAPM at `.470` RMSE, `.974` correlation and `.948` R-squared. Predicted
  factors reach `1.835/.457/.200` without context and `1.781/.505/.247` with
  context. Direct related-feature SPM reaches `1.794/.486/.236`; adding context
  improves it to `1.710/.563/.305`. Mean-only RMSE is `2.052`.
- **Interpretation:** The factor decomposition is sound when factor ratings are
  observed. Sparse statistical estimation of the factor targets is the main
  bottleneck. Teammate context is worth retaining for research, mainly for
  turnover and rebounding; teammate spacing itself has little conditional
  weight in this run.
- **Caveats:** Same-season context can absorb team and scheme strength and is
  not player skill. Annual `TEAM_ID` makes traded-player context approximate.
  Defended-shot data end in 2025, so the 2026 shooting-defense test lacks the
  observed rim/DFG family. This is reused diagnostic evidence, not promotion.

## 2026-08-26 - Full-feature factor ceiling and overall SPM context

- **Full factor ceiling:** Run `factor_target_full_feature_spm_v1_69496cee37`
  gives each factor head the available frozen full SPM bank: all 127 offense
  fields and 60 of 68 defense fields. The eight missing defense fields are the
  scorer-matchup aggregates that stop before the 2024--26 panel. Ridge
  penalties are selected on 2025 after a 2024 fit, then refit through 2025 for
  one reused 2026 diagnostic. Season 2027 is absent.
- **Factor result:** Full inputs improve five of six sparse heads. With all six
  same-side teammate contexts, 2026 R-squared is `.307/.126` shooting
  offense/defense, `.392/.432` turnover offense/defense and `.502/.110`
  offensive-rebound offense/defense. Offensive-rebound defense remains better
  under the sparse model. Full predicted factors plus context reconstruct
  ordinary net RAPM at `1.703` RMSE, `.573` correlation and `.312` R-squared,
  within `.010` RMSE of the direct annual full-feature model plus context.
- **Defense interpretation:** The factor decomposition is not the main
  failure. Oracle factor ratings still reconstruct ordinary RAPM at `.948`
  R-squared. Shooting defense remains a measurement problem: the current panel
  lacks the eight scorer-matchup fields, 2026 DFG and rim-DFG observations, and
  shot-level defender responsibility. More generic features roughly triple
  shooting-defense R-squared but leave it weak at `.126`.
- **Actual five-year SPM context test:** Run
  `five_year_spm_teammate_context_v1_13d270986a` starts from the exact stored
  predictions in `five_year_target_spm_v1_65550acb79`. A chronological
  second-stage ridge predicts remaining side-specific five-year RAPM error from
  six pooled teammate-context fields per side. It selects penalties on the 2024
  rating and freezes them for 2025 and 2026.
- **Five-year result:** Same-window net RMSE improves by `.009`, `.012` and
  `.014` in 2024, 2025 and 2026. On matched next-season annual RAPM, the 2024
  rating improves 2025 RMSE from `1.471` to `1.469`; the 2025 rating improves
  2026 from `1.520` to `1.508`. Offense, defense and net improve on both folds.
  The correction is small and survives only reused evidence. Keep the context
  family for a future joint refit; do not replace the frozen five-year SPM.
- **Unrun primary benchmark:** The checkout lacks exact 2025--26 player-team
  minute stints, so the preferred next-season team-win retrodiction cannot be
  reproduced without assigning traded-player minutes to one annual team. Do
  not weaken that benchmark. The current downstream result is next-season
  annual RAPM on identical matched players.
- **Data limits:** Complete pooled-context coverage is 87.5 percent and falls to
  83.6 percent in the 2026 scored rows; the training-fold imputer handles the
  remainder. Same-team context may absorb team and scheme strength, and annual
  `TEAM_ID` makes traded-player context approximate. This is not causal
  teammate value or independent confirmation.

## 2026-08-26 - Five-year SPM cheating ladder

- **Question:** Does replacing the selected five-year SPM with a five-year
  BoxPIPM-style model help next-season team prediction, and what happens when
  the statistical-only boundary is relaxed with age, minutes, position,
  on/off, AuPM, RAPTOR on/off, or a BPM-style team adjustment?
- **Design:** Run `spm_cheating_ladder_v1_fff340f6b6` scores rating seasons
  2022--24 against 2023--25 wins with observed next-season minutes and a
  250-minute / -2.0 replacement rule. New residual corrections train only on
  earlier rating windows. The same 963-player matched population is used for
  the secondary five-year RAPM test. Season 2027 is untouched.
- **Box result:** Five-year BoxPIPM-style is tied but slightly worse on team
  wins (`.51811` versus `.51816` mean R-squared) and far worse on player net
  RAPM (`1.7838` versus `1.4541` weighted RMSE). Do not replace SPM.
- **Demographic and external result:** Age/minutes/position and legacy AuPM
  lower mean team-win R-squared. RAPTOR on/off raises it by `.0190`, but its
  five-season coverage falls from 54.7 percent in 2022 to 43.2 percent in 2024
  and its player-target correlation degrades. It is stale research context.
- **On/off result:** Derived raw on/off validates against the retained
  RAPTOR/WOWY table at `.8891` Pearson and `.9137` rank correlation on 3,181
  player-seasons with at least 1,000 possessions. Adding it raises team-win
  R-squared by `.0162`, wins all three folds, and lowers player net RMSE to
  `1.2869`. The paired 10,000-draw team bootstrap interval is
  `[-.0028, +.0337]`, so this is not promotion evidence.
- **Team adjustment:** Reconciling each team's player sum to observed team net
  raises mean team-win R-squared by `.0599` but worsens player RAPM RMSE to
  `1.4569`. This is persistent team context, not better individual attribution.
  Keep it in a future team forecast layer rather than SPM.
- **Data limit:** Retained annual totals assign traded players to one primary
  team. This raises the annual-SPM QA benchmark by `.0297` R-squared relative
  to the exact-stint run. Absolute values are not comparable to the public
  benchmark; all arms here still use identical rows. Full details are in
  `docs/impact/SPM_CHEATING_LADDER_V1.md`.

## 2026-08-26 - PIPM as the prior, not the standalone metric

- **Correction:** The cheating ladder did not answer whether PIPM is a better
  prior for AIO. It compared standalone ratings. Run
  `aio_prior_bakeoff_v1_0a3591a402` changes only the prior inside the same
  one-season terminal-lineup `3000 / 3000 / 300` RAPM update.
- **Design:** Five arms score identical 2022--24 future games: zero prior,
  frozen five-year SPM, selected five-year SPM, forward-chained BoxPIPM-style,
  and a PIPM-like box plus raw-on/off prior. All use center scale one. The box
  and PIPM-like models train only on earlier five-year target windows.
- **Result:** Mean game-margin RMSE is `13.8902` for selected five-year SPM,
  `13.8360` for BoxPIPM-style and `13.8312` for PIPM-like. BoxPIPM wins 2022
  and 2024 and loses 2023. Its mean gain is `0.0541` points per game. The paired
  MSE interval against selected SPM is `[-2.6854, -0.5279]` over 10,000
  season-stratified whole-game draws.
- **Decision:** BoxPIPM-style earns a canonical 2025--26 follow-up. It does not
  replace SPM from reused development evidence. The slightly better PIPM-like
  arm is not preferred because raw same-season on/off double counts lineup
  outcomes already present in the RAPM likelihood. Full details are in
  `docs/impact/AIO_PRIOR_BAKEOFF_V1.md`.

## 2026-08-26 - Canonical 2025--26 AIO prior follow-up

- **Question:** Does the BoxPIPM-style prior's 2022--24 development gain survive
  the frozen canonical 2025 and 2026 follow-up after the same one-season RAPM
  update?
- **Design:** Run `aio_prior_canonical_followup_v1_8c61405875` reconstructs the
  2024 and 2025 annual sufficient statistics from stored canonical five-year
  matrices, then fits the fixed terminal-lineup `3000 / 3000 / 300` update with
  center scale one. Every arm scores the same 1,226 games in 2025 and 1,228 in
  2026. Season 2027 is absent.
- **QA:** The first attempted run exposed a scope bug: a development helper
  filtered frozen SPM priors to 2021--23, turning the later SPM arms into zero
  priors. The final runner owns its 2024--25 scope explicitly. It reproduces
  the frozen AIO RMSE within `9.2e-11`; annual matrix recombination error is at
  floating-point precision and offense plus defense equals net exactly.
- **Result:** BoxPIPM-style beats selected five-year SPM in both seasons. Mean
  next-season game-margin RMSE falls from `15.1588` to `15.1012`; mean
  correlation rises from `.3636` to `.3652`. Paired mean MSE difference is
  `-1.7293`, with a 10,000-draw whole-game interval of
  `[-3.0402, -0.4367]`.
- **Decision:** Freeze BoxPIPM-style as the research AIO prior. Do not change
  the public zero-prior RAPM or production site. The raw-on/off PIPM-like arm
  remains ineligible because it double counts lineup outcomes. Season 2027 is
  the untouched confirmation gate.

## 2026-08-27 - Full statistical-prior validation suite

- **Question:** Does the BoxPIPM-style AIO-prior result survive tests of
  midseason updating and adjacent-season player impact, or did one future-game
  metric create a misleading winner?
- **Design:** Run `impact_validation_suite_v1_4f2ad7cdd8` orders five tests:
  next-season game MSE, midseason second-half game MSE, forward aging-adjusted
  annual-RAPM correlation, reverse aging-adjusted annual-RAPM correlation, and
  same-season annual-RAPM correlation. Declared weights are `.50 / .25 / .15 /
  .05 / .05`. The composite averages within-fold percentile ranks, not raw
  metrics. An equal-weight sensitivity is reported beside it.
- **Primary result:** Across the five saved 2022--26 future-game folds,
  BoxPIPM-style has mean MSE `206.476` versus `208.128` for selected five-year
  SPM. The paired whole-game interval for Box minus selected is `[-2.478,
  -0.836]`. Both beat zero prior.
- **Other tests:** Selected five-year SPM has the best 2022--24 midseason MSE,
  `184.335`, and beats the base five-year SPM by `0.244` MSE with interval
  `[0.069, 0.419]`. Five-year SPM has the best forward annual correlation,
  `.476`, while BoxPIPM has the lowest forward annual RMSE, `1.851`.
- **Composite:** Selected five-year SPM ranks first under the declared weights,
  but base five-year SPM ranks first with equal weights. The ordering is not
  weight-stable, so the composite cannot support promotion.
- **Decision:** Keep BoxPIPM-style as the frozen research AIO prior because it
  wins the primary future-game test. Keep the five-year SPM family for
  standalone statistical ratings and midseason adaptation. Do not claim one
  universal metric. Wait for 2027 before public promotion.
- **QA:** Initial artifact `impact_validation_suite_v1_021be06a12` is invalid.
  An all-or-nothing age join omitted both adjacent-season tests. The corrected
  run keeps valid age-matched rows and uses the smaller origin/target exposure
  weight. It includes all five tests, uses whole games for both split and
  uncertainty, and contains no 2027 rows. Intermediate artifact
  `impact_validation_suite_v1_07c7b85efc` is superseded by the exposure fix.

## 2026-08-27 - Current BoxPIPM-prior AIO leaderboard

- **Question:** What does the frozen BoxPIPM-style research prior produce when
  it is updated by the complete 2026 single-season possession likelihood?
- **Design:** Run `current_box_pipm_aio_v1_939ad77840` builds rolling five-year
  box inputs through 2026, selects side-specific ridge penalties on earlier
  windows (`100` offense, `30` defense), and applies the fixed one-season
  terminal-lineup `3000 / 3000 / 300` AIO update with center scale one.
- **QA:** The 2024--26 annual matrices are recovered exactly from canonical
  rolling matrices and legacy annual sufficient statistics; maximum matrix
  error is `1.14e-13`. All 582 active players have names, the prior covers 100
  percent of active possessions, offense plus defense equals net exactly, and
  Season 2027 is absent.
- **Leaderboard:** With at least 1,000 possessions on each side, the top five
  are Victor Wembanyama (`+8.65`), Nikola Jokic (`+8.19`), Shai
  Gilgeous-Alexander (`+7.97`), Kawhi Leonard (`+7.76`) and Jimmy Butler III
  (`+6.64`) points per 100. This is a retrospective research leaderboard, not
  a 2027 forecast.

## 2026-08-27 - External PIPM reconstruction comparison

- **Source boundary:** No verified original Goldstein PIPM release is present.
  Run `pipm_reference_comparison_v1_49a3c2c973` parses the public
  Basketball Database reconstruction for 2015--24 using exact NBA player IDs.
  Traded-player rows are possession-weighted across teams. The 2024 page has
  zero PIPM fields, so it is retained for QA and excluded from scoring.
- **Matched comparison:** On 1,106 player-seasons from 2021--23 with at least
  1,000 possessions in both sources, the BoxPIPM-style prior has pooled net
  Pearson/Spearman agreement of `.631/.583`. After the single-season RAPM
  update, agreement rises to `.808/.779`; net RMSE versus the reconstruction
  falls from `1.587` to `1.360` points per 100.
- **Interpretation:** The update moves the simple box prior substantially
  toward an external PIPM-like rating. This is agreement, not proof of better
  prediction, and the external values are a third-party reconstruction rather
  than source-of-truth PIPM.

## 2026-08-27 - Ryan Davis RAPM labels for SPM

- **Question:** Does training the same SPM architecture on Ryan Davis's exact
  five-year RAPM coefficients improve the downstream AIO? Ryan's released
  tables contain no player-window exposure weights. His public possession
  regression weights retained rows by possession count, which is one per row;
  it does not publish an alternative set of player-level SPM weights.
- **Design:** Run `ryan_target_spm_v1_31eccca595` holds the 126-offense and
  50-defense feature set, learner families, matched player-windows, AIO
  penalties, center scale and 2022--24 future games fixed. It changes only the
  five-year training labels. The weighted arm uses CourtSignal exposure weights
  to isolate the target; a uniform-weight sensitivity is reported separately.
- **Result:** Ryan-target SPM wins all three future-game folds. Mean RMSE falls
  from `13.8920` to `13.8267`; after scaling Ryan's side-specific target spread
  using training windows only, it falls to `13.8213` and mean correlation rises
  from `.3614` to `.3750`. The paired whole-game MSE difference is `-1.974`,
  with a 5,000-draw 95 percent interval of `[-2.648, -1.330]`.
- **Controls:** Uniform weighting wins only one of three seasons and its MSE
  interval crosses zero. Merely shrinking the CourtSignal center produces only
  a `0.0045` RMSE gain, so Ryan's advantage is not explained by its smaller
  coefficient variance alone.
- **Decision:** Ryan-target SPM is the next research challenger, not a public
  replacement. The evidence uses three reused folds and the stored base feature
  architecture; it needs a frozen later confirmation before promotion.

## 2026-08-27 - Direct-net SPM with residual defense

- **Question:** Does fitting five-year RAPM net directly, fitting offense
  separately, and defining defense as `net - offense` improve the SPM prior?
  Run `net_first_spm_v1_2029e38965` compares two versions: the 151-field union
  of the stored offense and defense banks for both heads, and that full bank
  for net with the 126-field offense bank for offense. Both heads use the same
  frozen histogram GBM. No model or feature tuning follows the results.
- **SPM target fit:** Direct net improves the component-first CourtSignal
  baseline on the same 2021--23 player-window rows. Mean weighted net RMSE
  falls from `1.5760` to `1.5092`, and correlation rises from `.7291` to
  `.7593`. The later selected five-year SPM remains better at `1.4430` RMSE and
  `.7838` correlation.
- **AIO result:** The target-fit gain does not transfer to future games. Mean
  2022--24 margin RMSE is `13.8957` for either net-first arm, versus `13.8920`
  for the matched component-first CourtSignal SPM, `13.8902` for selected SPM,
  `13.8360` for BoxPIPM-style, and `13.8213` for rescaled Ryan-target SPM.
- **Paired tests:** Against the matched component-first SPM, the net-first full
  arm wins two of three seasons but has mean MSE delta `+0.132` and a 95 percent
  whole-game interval of `[-0.450, +0.724]`. It loses to Ryan in all three
  seasons with delta `+2.106` and interval `[+1.286, +2.928]`, and loses to
  BoxPIPM with delta `+1.725` and interval `[+0.895, +2.594]`.
- **Feature-bank result:** Letting the offense head use the 25 defense-only
  fields changes mean future-game RMSE by less than `0.0001`. It marginally
  improves same-window offense fit but slightly worsens residual-defense fit.
- **Decision:** Reject the direct-net residual-defense construction as the AIO
  prior. It optimizes the intermediate five-year RAPM target without improving
  the downstream decision metric. Keep Ryan-target and BoxPIPM-style ahead of
  it. "Full" here means the 151 inputs in the stored matched panel; later zTS
  and matchup-defense additions were unavailable and were not fabricated.

## 2026-08-27 - PIPM breaker: target, features, learner and context

- **Question:** Does BoxPIPM's apparent advantage come from its small feature
  bank, its RAPM label, its learner, or context such as minutes, `(GS/GP)^2`,
  normal on/off, position-adjusted offensive rebounding and spacing?
- **Design:** Run `pipm_breaker_v1_d154ebea55` compares 15 candidates. The
  fixed 2-by-2 crosses 15 box versus 126-offense/50-defense full features with
  CourtSignal versus training-fold-rescaled Ryan five-year RAPM labels. Tuned
  arms search ridge, elastic net and histogram GBM inside earlier windows.
  Every prior receives the same one-season terminal-lineup `3000 / 3000 / 300`
  update and scores identical 2022--24 games. The attached PIPM database is an
  agreement check only because it mixes regular season and playoffs.
- **Result:** The 15-feature CourtSignal-targeted ridge control is first at
  `13.8644` mean game-margin RMSE and `.3635` correlation. Box/Ryan ridge is
  statistically tied at `13.8668`; paired MSE delta `+0.073`, 95% interval
  `[-0.030, +0.179]`. Minutes plus starter share scores `13.9029`, tuned box
  `13.9050`, tuned full CourtSignal `13.9198`, adjusted OREB plus spacing
  `13.9387`, ordinary on/off `13.9424`, and all context `13.9693`. None wins.
- **Correlation:** Nineteen feature pairs exceed `.95` absolute correlation.
  A training-only `.98` pruning arm drops nine offense and two defense fields
  per fold but worsens RMSE to `13.9457`; no silent deletion is made.
- **QA:** Basketball-Reference starts/minutes map 100% of source minutes for
  2014--23. PIPM identity coverage is 99.12% of source minutes. The Gabriel
  possession source retains 681,393 possessions across 3,442 games after
  quarantining 90 official-score mismatches. Candidates score identical games,
  offense plus defense equals net exactly, and Season 2027 is absent.
- **Decision:** Reject the new context and large-model candidates. Better
  five-year RAPM fit does not transfer to future games. Keep the small ridge
  control for research and require one mechanism-driven feature family at a
  time under the same downstream gate. Full report:
  `docs/impact/PIPM_BREAKER_V1.md`.

## 2026-08-27 - Stabilized rim assists in the five-year prior

- **Question:** Do rim assists add transferable creation information to the
  15-feature CourtSignal-targeted prior?
- **Design:** Run `rim_assist_spm_challenger_v1_23e599d812` builds annual rim
  assists per 100 offensive possessions. The estimator shrinks each player to
  the same-season league center with 500 prior possessions. The five-year
  feature pools frozen annual estimates with offensive-possession weights.
  The source observes 89.23% of matched player-window rows and 98.79% of their
  possessions. Missing fringe-player rows receive the same-window neutral
  center. The challenger changes only the offense prior. Both models receive
  the same one-season RAPM update and score identical 2022--24 games.
- **Result:** Rim assists lose all three folds. Mean game-margin RMSE rises from
  `13.8644` to `13.8764`; mean correlation falls from `.3635` to `.3620`.
  Paired MSE increases by `0.337` points squared per game. The 5,000-draw 95%
  interval is `[+0.172, +0.508]`.
- **Decision:** Reject rim assists as an SPM/AIO input. Retain the statistic in
  the descriptive player-skill layer. Full report:
  `docs/impact/RIM_ASSIST_SPM_CHALLENGER_V1.md`.

## 2026-08-27 - BoxPIPM versus PIPM before and after RAPM

- **Question:** Does the 15-feature BoxPIPM recreation beat a public PIPM
  reference before or after both metrics receive the same one-season RAPM
  update?
- **Source QA:** The attached PIPM file is not a complete 2020-21 season. Its
  latest rows have a maximum of 22 games and a median of 15. The run excludes
  that file from scoring. It uses the existing third-party regular-season PIPM
  reference for 2021-23 and replaces rows below 250 minutes with -1 offense and
  -1 defense.
- **Design:** Run `pipm_four_way_comparison_v1_0f1473b838` scores four fixed
  ratings on identical 2022-24 games. Both posterior ratings use the same
  terminal-lineup one-season RAPM, `3000 / 3000 / 300` penalties, and center
  scale 1. The bootstrap resamples whole games within season and gives each
  season equal weight.
- **Result:** BoxPIPM plus RAPM ranks first at `13.8644` mean fold RMSE and
  `.3635` mean correlation. PIPM plus RAPM scores `14.0578`, standalone BoxPIPM
  scores `14.0595`, and standalone PIPM scores `14.0835`.
- **Paired test:** BoxPIPM plus RAPM beats PIPM plus RAPM in all three folds.
  Equal-season MSE falls by `5.473`; the 5,000-draw interval is
  `[-7.108, -3.776]`. Standalone BoxPIPM and PIPM are tied. Their MSE difference
  is `-0.583` with interval `[-4.373, 3.126]`.
- **Decision:** Keep BoxPIPM as the research AIO prior. Do not claim that it
  beats original PIPM because the available full-season source is a third-party
  reconstruction. Full report: `docs/impact/PIPM_FOUR_WAY_COMPARISON_V1.md`.

## 2026-08-27 - Full frozen feature contract through 2026

- **Question:** Can the exact 127-offense and 68-defense full-SPM contract be
  built on one reproducible annual spine from 2014 through 2026?
- **Design:** Pin Gabriel player sheets through 2026, add 2012--13 only as lag
  history for 2014 engineering, rebuild playtype/zTS and observed defense, and
  ingest pinned Apache-2.0 matchup archives through 2026. Preserve same-season
  annual estimates. Pool only the finished annual estimates for five-year rows.
- **Fix:** Parquet player sheets repeat identical season-total defensive
  exposure rows. Summing those rows broke matchup exposure QA. The matchup
  builder now deduplicates identical `(PLAYER_ID, DefPoss)` rows before it sums
  distinct stints. The 2018 correlation rose from `.23` to `.99981`.
- **Result:** Run `full_spm_features_2014_2026_v1_4c77ae6acc` contains 6,942
  annual player-seasons for 2014--26 and 8,620 rolling five-year rows for
  2018--26. It has zero duplicate keys, zero infinite selected values, and zero
  2027 rows. The 2026 matchup source covers 1,230 games and 240,839 raw rows;
  point reconciliation has zero failures and exposure correlation is `.99994`.
- **Coverage:** Hustle and matchup assignments start in 2018. The source ledger
  records 2014--17 as unavailable. Low-opportunity player fields retain missing
  values for training-fold median imputation.
- **Decision:** The 2025--26 feature-input blocker is closed. Do not promote a
  model from this data refresh. Refit and validate the SPM/AIO as a separate
  experiment. Full report: `docs/impact/FULL_SPM_FEATURES_2014_2026.md`.

## 2026-08-27 - Full SPM refit and late-start feature ablation

- **Question:** Do the five hustle and eight matchup-defense fields that start
  in 2018 improve the five-year SPM and its one-season RAPM posterior?
- **Design:** Run `full_spm_history_ablation_v1_34725a86aa` compares the full
  127/68 contract, a fixed 127/55 history-complete ablation, and BoxPIPM-style.
  Each rating fold trains on earlier five-year windows. All priors receive the
  same `3000 / 3000 / 300` one-season RAPM update. Five test seasons cover
  2022--26. The paired test resamples whole games 5,000 times inside season.
- **Result:** Full SPM beats the reduced SPM standalone by `-2.820` MSE, with
  interval `[-4.095, -1.555]`. After RAPM, the difference shrinks to `-0.461`,
  with interval `[-1.026, +0.109]`. Do not remove the 13 fields.
- **Box comparison:** Full SPM and BoxPIPM-style are tied standalone. Their MSE
  difference is `-0.843`, with interval `[-2.800, +1.112]`. Full SPM plus RAPM
  loses to BoxPIPM-style plus RAPM by `+1.550` MSE, with interval
  `[+0.690, +2.399]`. BoxPIPM-style wins four of five seasons.
- **QA:** Candidates score identical games. Prior possession coverage is 100%.
  Offense plus defense equals net exactly. Matrix reconstruction error is below
  `1.14e-13`. Season 2027 is absent.
- **Decision:** Keep the full feature contract. Keep BoxPIPM-style as the
  research AIO prior. Do not promote from reused evidence. Full report:
  `docs/impact/FULL_SPM_HISTORY_ABLATION_V1.md`.

## 2026-08-27 - Same-season stabilization ablation

- **Question:** Does same-season empirical-Bayes stabilization improve the
  five-year SPM or its one-season RAPM posterior when raw and stabilized arms
  use the same concepts?
- **Design:** Run `spm_stabilization_ablation_v1_db618f06e8` removes the raw
  duplicates from the stabilized offense contract. Each arm uses 98 offense
  and 68 defense fields, including 37 offense and 10 defense value pairs. Both
  arms use the same frozen learners, five-year RAPM targets, chronological
  folds, square-root exposure weights, and `3000 / 3000 / 300` AIO update.
- **SPM result:** Stabilization lowers mean net target RMSE from `1.5281` to
  `1.4803` and raises correlation from `.6607` to `.6982`. It wins four of five
  future-game folds. Raw minus stabilized MSE is `+0.4252`, with interval
  `[-0.2914, +1.1672]` and 88.70% probability that stabilization is better.
- **AIO result:** The RAPM update weakens the difference. Raw minus stabilized
  MSE is `+0.1338`, with interval `[-0.2335, +0.5050]`. Stabilized wins three
  folds. The probability that stabilization is better is 76.82%.
- **QA:** Candidates score identical games. Prior possession coverage is 100%.
  Offense plus defense equals net exactly. Matrix reconstruction error is below
  `1.14e-13`. Season 2027 is absent.
- **Decision:** Retain same-season stabilization for standalone SPM research.
  Record a null for final AIO accuracy. Do not promote from reused evidence.
  Full report: `docs/impact/SPM_STABILIZATION_ABLATION_V1.md`.

## 2026-08-27 - Coverage-gated full-feature refit and prior-scale audit

- **Gate:** Run `full_feature_coverage_v1_3de4ec8954` audits all 170 selected
  features before imputation. The rebuild fills 1,012 missing possession cells
  for 506 player-seasons from the canonical RAPM target panel. It preserves all
  observed player-sheet exposures.
- **Coverage:** Seventy-five annual fields and 77 pooled five-year fields remain
  below 99% observed coverage. Every field has an explicit source, eligibility,
  opportunity, or missing-row cause. No field has an unexplained gap.
- **Corrected design:** Five-year rows now pool frozen annual feature estimates
  by possession exposure. They no longer re-engineer stabilized ratios from a
  raw five-year aggregate.
- **Refit:** Run `full_spm_history_ablation_v1_2eb5eb428c` finds that full SPM
  beats BoxPIPM-style standalone by `-2.750` paired MSE, with interval
  `[-4.821, -0.702]`. Full SPM plus RAPM trails BoxPIPM-style plus RAPM by
  `+0.681`, with interval `[-0.227, +1.587]`. The posterior comparison remains
  unresolved.
- **Feature history:** The full AIO beats the history-complete AIO by `-0.756`
  MSE, with interval `[-1.362, -0.151]`. Retain late-start hustle and matchup
  inputs with explicit missing-history indicators.
- **Prior scale:** Run `aio_prior_scale_audit_v1_aeca5715b3` selects scales from
  earlier folds only. Full SPM selects `.75`; BoxPIPM-style selects `1.00`.
  Scaling full SPM changes MSE by `-0.014`, with interval
  `[-0.543, +0.515]`. Prior scale does not explain the unresolved AIO result.
- **Forecast limitation:** Actual future lineups supply oracle exposure in this
  diagnostic. Unknown player slots rise from 8.28% in 2023 to 10.74% in 2026.
  Do not describe the result as a deployable forecast.
- **Decision:** Keep the public 2017–24 lineage unchanged. Keep full SPM and
  BoxPIPM-style as research AIO challengers. Reserve 2027 for frozen
  confirmation. Full report:
  `docs/impact/FULL_FEATURE_COVERAGE_AND_REFIT_2026_08_27.md`.

## 2026-08-27 - Semantic completion of every selected SPM input

- **Question:** Can the full 2014 through 2026 feature contract reach 100%
  model-ready coverage without treating every missing value as a real zero?
- **Design:** Run `semantically_complete_spm_features_v1_8be676bd0f` applies
  rules by unit. Missing event rates become zero after exposure repair. Raw
  ratios use same-season empirical-Bayes estimates. Missing level metrics use
  same-season medians. Source-specific defense fields use zero plus explicit
  availability fields. zTS uses low-sample playtype rows before it falls back
  to season-relative TS.
- **Coverage:** The annual panel has 6,942 rows. The five-year panel has 8,620
  rows. The expanded 175-feature contract has zero missing values in both.
  The run maps all 75 selected fields below 99% observed coverage to an exact
  reason and completion method.
- **Comparison:** Run `semantic_feature_completion_comparison_v1_235b4dea34`
  scores identical future games for five seasons. Completed SPM beats
  BoxPIPM-style standalone by `-3.104` MSE, with interval
  `[-5.219, -0.982]`. Completed and previous AIO are tied at `+0.030`, with
  interval `[-0.153, +0.212]`.
- **Matchups:** Removing the eight matchup fields worsens standalone SPM by
  `+2.757` MSE. Its interval is `[+1.534, +3.984]`. Keep the fields and their
  source flag.
- **Decision:** Use the finite panel for research. Keep BoxPIPM-style and the
  completed SPM as separate AIO challengers. Do not promote either from reused
  evidence. Full report: `docs/impact/SEMANTIC_FEATURE_COMPLETION_V1.md`.

## 2026-08-28 - Final BoxPIPM feature ladder and interpretability

- **Question:** Can a small, cumulative sequence of completed SPM feature
  families improve the BoxPIPM-style prior after the same one-season RAPM
  update?
- **Design:** Freeze eight ordered feature additions before the run. Train each
  statistical prior on earlier five-year RAPM windows. Tune ridge alpha inside
  each training fold. Apply the same `3000 / 3000 / 300` season-specific RAPM
  update. Score identical next-season games for outcome seasons 2022--26. Use
  equal-season MSE and 5,000 within-season whole-game bootstrap draws.
- **Result:** Box15 has the lowest point MSE at `207.421` and RMSE `14.402`.
  The closest cumulative matchup-defense arm has MSE `207.537` and interval for
  Box15 minus matchup `[-0.944, +0.737]`. The full completed ridge ceiling has
  MSE `208.443`; Box15 beats it by `-1.022`, with interval
  `[-2.007, -0.022]`. No added family passes the frozen selection rule.
- **Interpretation:** Fixed-model grouped permutation raises MSE by `7.063` for
  disruption/fouls, `6.076` for shooting/scoring, `3.059` for
  creation/security, and `1.361` for rebounding. Every group is positive in all
  five repeats. These values measure dependence, not causality.
- **QA:** The original generated leaderboard included five-year matrix players
  with zero 2026 exposure. Interpretation run
  `final_box_interpretability_v1_652799efb6` removes 894 such rows and reaches
  100% active-player name coverage. Fits and game predictions are unchanged.
- **Decision:** Keep Box15 as the research AIO prior. End the current feature
  search. Reserve Season 2027 for untouched confirmation. Full report:
  `docs/impact/FINAL_BOX_FEATURE_LADDER_V1.md`.
## 2026-08-30 — One-cutoff walk-forward SPM pilot

**Question:** Does the completed Full SPM or BoxSPM predict later game margins
better before and after the same one-season RAPM update?

**Method:** Run `walk_forward_spm_pilot_v1_afcc388e8d` trains each SPM mapping
on five-year windows ending before 2024, builds ratings with player inputs
through 2024, and scores 1,226 identical 2025 games. Each AIO uses only 2024
possessions for its `3000 / 3000 / 300` RAPM update. The comparison uses 1,000
paired whole-game bootstrap draws.

**Result:** Full SPM lowers standalone RMSE from `15.263` to `15.103`; its MSE
difference against BoxSPM is `-4.856` with interval `[-9.543, -0.160]`. After
the RAPM update, Full SPM records RMSE `14.894` versus `14.864` for BoxSPM. The
posterior MSE difference is `+0.884` with interval `[-0.897, +2.724]`. The
future design assigns zero impact to 9.80% of player-possession slots because
those players lack a 2024 rating.

**Decision:** The chronological scorer works. The standalone result favors
Full SPM in this reused fold. The posterior result remains unresolved. Freeze
both definitions and park the full validation contract for independent review.
