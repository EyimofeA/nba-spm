#!/bin/bash
# Overnight autoresearch: pending build gens → propose → evaluate. One RAPM fit at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/outputs/autoresearch.log"
"$ROOT/scripts/keep_awake.sh" start
cd "$ROOT/src"
echo "AUTORESEARCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
export PYTHONUNBUFFERED=1
caffeinate -dims python3 autoresearch_loop.py --max-gens 20 2>&1 | tee -a "$LOG"
echo "AUTORESEARCH_END $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
