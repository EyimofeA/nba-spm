# RAPM Validation Suite v1 (predeclared)

Status: ACTIVE — approved for development and selection on 2024–2026.
Applies to: every `research/rapm_lab` rating variant from now on.

## Metric being validated

Ratings produced by `research/rapm_lab/rapm_ridge.py` (or successors) for a given
window and lambda configuration. Ratings are per-100-possession offensive
and defensive coefficients; NET = OFF + DEF.

## Tests

### V1. Forward game-margin retrodiction  [PRIMARY]
For each test season T in {2024, 2025, 2026}: train strictly on seasons < T,
predict every game's per-100 margin from the ten on-court ratings plus home
term, report Pearson correlation and MAE against actual margins.
Score = mean corr across the three folds.
Already implemented (`--windows` forward folds).

### V2. Team win%/net-rating projection  [knarsu3 method]
ratings(T-1) x minutes(T), summed by team(T) -> predicted team net rating.
Correlate with actual T win% AND actual T net rating (Pearson + Spearman).
Rules: players < 250 minutes in T get replacement NET (-2.5);
players absent in T-1 use their most recent season with >= 1000 minutes.
Two variants reported: ACTUAL minutes(T) and PROJECTED minutes(T)
(projected requires a minutes model - marked partial until then).
Verification targets: public metrics scored identically from
verification/ files (EPM, MAMBA, DARKO DPM, LEBRON if obtained).

### V3. Persistence baseline  [gate]
For fold T, fit the same zero-prior coefficient specification on season T-1
only, using the candidate's frozen penalties. Unseen players receive zero.
Apply those frozen ratings to the exact V1 test possessions and games.
The primary gate is mean game-margin correlation across the development folds;
MAE is supporting evidence. A variant that does not beat persistence is a null.

### V4. Split-half reliability  [tiebreaker]
Within-window odd/even possession halves: per-player rating correlation,
exposure-bucketed. Used to break ties between configs with equal V1.

### V5. In-sample diagnostics  [sanity only, never selection]
Possession-level R^2, residual distribution, anchor checks (Jokic/LeBron/
Garnett era anchors top; no unexplained |NET| > 15 outliers).

### V6. Playoff transfer  [newly permitted by principal 2026-08-22]
Ratings trained on regular season only, scored on playoff game margins
per season where playoffs exist in the data. Reported separately, never
mixed into selection.

### V7. Lineup-age controls  [research diagnostic]
The frozen `age_adjusted_rapm_v1` contract fits categorical offense and defense
lineup-age controls jointly with player RAPM. Age shrinkage was selected on
2025 game-margin RMSE, then ordinary player-only, age-27 player-only, and
actual-lineup-age predictions were compared on the same reused 2026 games with
whole-game paired resampling. The RMSE winner was also the correlation winner,
so the experiment-specific selection metric did not change the selected
candidate. This diagnostic does not alter the v1 reference or the Season 2027
confirmation gate.

## Selection protocol

- Hyperparameter grids are declared before running; full grid surfaces are
  reported, not just winners.
- Primary = V1 mean corr. Secondary = V2 win% corr. Tiebreak = V4.
- Fold definitions frozen (2024/2025/2026 forward). No result-driven fold
  changes; any change requires a new suite version.
- Seasons 2024–2026 are development and selection only. The persistence gate
  can reject a model but cannot confirm one. Season 2027 is the untouched,
  single-shot confirmation.

## Target-length decision (separate from validation)

The harness produces three horizons from the same possession data:
single-season, rolling 3-year, career-span. The suite scores each horizon;
the SPM-target question (one-year vs overlapping multi-year windows,
JE-style long-span) is decided by these numbers when the SPM lane opens -
not by taste.

## Status labeling (advisory 2026-08-22)

Folds 2024/2025/2026 are DEVELOPMENT AND SELECTION ONLY. Any lambda or
config chosen on them is `dev-tuned`, never `final` or `validated`.
Promotion path: dev-tuned -> Season 2027 single-shot confirmation
(predeclared contract, scored exactly once) -> validated. Until then all
outputs carry a dev-tuned label.
