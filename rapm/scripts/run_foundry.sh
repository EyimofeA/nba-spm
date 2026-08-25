#!/bin/bash
# Run foundry (or any rapm job) with caffeinate + single-process lock friendly logging.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="${1:-0}"
LOG="$ROOT/outputs/foundry_g${GEN}.log"
DONE="$ROOT/outputs/foundry_g${GEN}.done"

"$ROOT/scripts/keep_awake.sh" start

cd "$ROOT/src"
echo "FOUNDRY_LAUNCH g${GEN} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
caffeinate -dims -i python3 feature_foundry.py "$GEN" >> "$LOG" 2>&1
echo "ok" > "$DONE"
python3 log_run_greps.py "foundry_g${GEN}_done"
