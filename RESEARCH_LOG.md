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
pricing. Created `RESEARCH_BACKLOG.md` as the active dependency-aware planning page.

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

**Promote?:** P0 in `RESEARCH_BACKLOG.md`: canonical data repair, independent simple
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
detail. Added tests and an operator guide in `NBA_IMPACT_BUILD.md`.

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
