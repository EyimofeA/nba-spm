#!/bin/bash
# Keep Mac awake during overnight RAPM/foundry runs.
# Usage: ./scripts/keep_awake.sh start|stop|status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/outputs/caffeinate.pid"

start() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "caffeinate already running pid $(cat "$PIDFILE")"
    return 0
  fi
  caffeinate -dims &
  echo $! > "$PIDFILE"
  echo "caffeinate -dims started pid $(cat "$PIDFILE")"
}

stop() {
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "stopped caffeinate $pid"
    fi
    rm -f "$PIDFILE"
  else
    echo "no pidfile"
  fi
}

status() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "awake pid $(cat "$PIDFILE")"
  else
    echo "not running"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" ;;
esac
