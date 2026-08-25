#!/bin/bash
# Run foundry one fold at a time (tiny PC): bash scripts/run_foundry_folds.sh 2 f24 f23
# Gen ≥2 = feature candidates in features/candidates/gen_NNN/. Never re-run minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="${1:-2}"
shift || true
FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then FOLDS=(f24 f23); fi
LOG="$ROOT/outputs/foundry_g${GEN}.log"
"$ROOT/scripts/keep_awake.sh" start
cd "$ROOT/src"
for f in "${FOLDS[@]}"; do
  echo "=== FOLD $f $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  caffeinate -dims -i python3 feature_foundry.py "$GEN" --folds "$f" 2>&1 | tee -a "$LOG" || {
    echo "FOLD_FAILED $f" | tee -a "$LOG"
    exit 1
  }
done
echo "ok" > "$ROOT/outputs/foundry_g${GEN}.done"
python3 log_run_greps.py "foundry_g${GEN}_done"
