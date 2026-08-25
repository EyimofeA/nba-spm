# Operator greps — copy/paste status checks

Run after any long job. Or: `python3 src/log_run_greps.py <tag>` appends all of these to `outputs/grep_digest.log`.

## Keep Mac awake (overnight runs)

```bash
rapm/scripts/keep_awake.sh start    # caffeinate -dims until you stop it
rapm/scripts/keep_awake.sh status
rapm/scripts/keep_awake.sh stop

# Autoresearch (build.py candidates → gate → propose next)
nohup rapm/scripts/run_autoresearch.sh >> rapm/outputs/autoresearch.log 2>&1 &
tail -f rapm/outputs/autoresearch.log rapm/outputs/foundry_g6.log
python3 rapm/src/autoresearch_proposer.py pending   # gens waiting to run
```

```bash
pgrep -fl "feature_foundry|feature_eval|spm_v2|standard_rapm" || echo "idle"
cat outputs/rapm_run.lock 2>/dev/null || echo "no lock"
```

## Minutes / SPM gate (experiments.csv)

```bash
grep -E "mprior|spmv2|foundry" outputs/diagnostics/experiments.csv | tail -20
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/diagnostics/experiments.csv')
df = df.dropna(subset=['margin_corr'])
print(df.nlargest(8, 'margin_corr')[['name','margin_corr','margin_rmse','anchors_ok']])
"
```

## Foundry user lane (results.tsv)

```bash
column -t -s $'\t' features/results.tsv | tail -15
column -t -s $'\t' features/leaderboard.csv | head -15
```

## Run completion flags

```bash
grep -h -E "BASELINE_SUMMARY|FOUNDRY_GEN.*done|SPMV2_ALL_DONE|EXPERIMENT_FAILED" \
  outputs/feature_eval_baseline.log outputs/foundry_g0.log outputs/spmv2_run.log 2>/dev/null
test -f outputs/foundry_g0.done && echo "foundry g0 DONE"
```

## Harness repro check (must match before trusting foundry)

```bash
grep "BASELINE_REPRO\|c=2" outputs/feature_eval_baseline.log
# expect f24 ~0.7346, f23 ~0.6954 at c=2
```

## Resource policy (tiny PC)

- **One RAPM fit at a time** — `run_lock.py` / `outputs/rapm_run.lock`
- Do **not** parallelize fold f24 + f23
- SPM ridge/GBM is cheap; possession matrix build + CG solve is expensive
- Foundry Tier 2 budget: ~1 candidate at a time

## Digest (all greps logged)

```bash
tail -80 outputs/grep_digest.log
python3 src/log_run_greps.py nightly
```
