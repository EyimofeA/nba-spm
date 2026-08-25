# Feature Foundry — log · prove · measure · explore

Autonomous feature search for the same-window SPM prior. Fixed harness in
`rapm/src/feature_eval.py` and `features/prepare.py`. Agents edit candidates
only.

## Setup

1. Read `PROJECT.md`, last 3 `RESEARCH_LOG.md` entries, this file, `prepare.py`.
2. **Minutes baseline is frozen** in `prepare.GATE_BASELINES` (0.7335 / 0.6953). Do **not** re-run unless harness breaks — use `python3 src/feature_foundry.py --verify-harness` once.
3. **Compute budget = feature candidates.** Each gen needs `candidates/gen_NNN/build.py` that creates **new columns** (not re-slicing existing box stats).
4. Initialize or append to `results.tsv` (tab-separated).

## Autoresearch loop (what agents should do)

1. **Propose** new feature logic in `candidates/gen_NNN/build.py` (ratios, merges from staging, transforms).
2. **Build** — `build()` adds columns not already in `spm_features_windows.parquet`.
3. **Evaluate** — foundry runs splice on **new cols only** (`use_columns: new_only`).
4. **Log** features_chosen + improvement_log.md.
5. If beat minutes on f24+f23 → vault. Else discard and propose next `build.py`.

`type: subset` (gen 002–005) was a mistake — column ablations of existing box, not new research.

## What you CAN do

- Write feature functions under `features/candidates/gen_NNN/`.
- Submit hand-designed features to `features/user/`.
- Download from domains in `data_allowlist.yaml` into `features/staging/`.
- Profile staging data (join rate, blocklist scan) before proposing merges.
- Append every run to `results.tsv`, `experiments.csv`, and `RESEARCH_LOG.md`.

## What you CANNOT do

- Edit `prepare.py`, `feature_eval.py`, or gate/anchor logic.
- Use on/off ratings, plus-minus, or team ORtg/DRtg/wins as player features.
- Use features computed from the same possessions as RAPM labels.
- Skip OOF prior fitting or global-strength priors (per-player tau² only).
- Merge agent downloads into parquet without user approval after profiling.
- Select on possession RMSE or team L2 metrics (diagnostic only).

## Resource policy (tiny PC)

- **One RAPM possession fit at a time.** Use `run_lock.py`; if `outputs/rapm_run.lock` exists, wait or check `pgrep`.
- Never parallelize fold f24 + f23 + vault — sequential only.
- SPM ridge / GBM / feature Tier 0–1 are cheap; matrix build + CG solve is expensive (~3–7 min/fold).
- **Never re-run minutes as a foundry generation** — that's ~20 min wasted. Gen ≥2 only.
- Vault skipped automatically unless search folds beat frozen minutes.
- After every generation: `python3 src/log_run_greps.py foundry_gN` → `outputs/grep_digest.log`.
- Operator cheat sheet: `rapm/OPERATOR_GREPS.md`.

## Human vs agent viewers

- **Human:** open `outputs/viewer/human.html` (FM / xRAPM style panel + foundry tab). Rebuild: `python3 src/build_human_viewer.py`.
- **Agent:** open `outputs/viewer/agent.html` (dense results.tsv + registry). Same rebuild command.

## The goal

Beat the minutes prior on **all search folds** (f24 + f23), pass anchors, then
confirm on vault (2015–17 → 2018). Baselines in `prepare.py`:

- Minutes prior (c=2): 0.7335 / 0.6953
- Zero-prior champion: 0.6596 / 0.5939

**Ship fallback:** if nothing beats minutes on f24+f23+vault with anchors pass,
the minutes prior remains the product. High OOF R² alone does not ship.

## Output

Run the harness; record one TSV row per candidate with status
`keep`, `discard`, `crash`, or `leak_suspect`. Log failures — never delete.

## The experiment loop

LOOP until manually stopped:

1. Propose one candidate (foundry or user lane).
2. Blocklist-scan columns; reject leakage suspects before fitting.
3. Run two-pass splice via `feature_eval.run_splice` on f24, then f23.
4. If both folds beat minutes at best c and anchors pass → run vault (Tier 3).
5. Append row to `results.tsv` (include `features_chosen`); log to `experiments.csv` and `features/improvement_log.md`.
6. If improved → advance; else discard and try the next idea.

**NEVER STOP** once the loop starts. If stuck, retry failed gens with new
angles, combine near-misses, or request Data Curator refresh — do not pause for
permission.

**Simplicity:** prefer fewer features and lower complexity_score unless the
gate gain is meaningful on both folds.
