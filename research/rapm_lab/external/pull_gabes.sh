#!/bin/bash
# Resumable downloader for gabriel1200/merged_playbyplay (3.28 GB).
# Stage 1: blobless sparse clone (metadata only, fast).
# Stage 2: checkout old_data (1997-2013).
# Stage 3: checkout pbp_data (2014-2026).
# Every stage retries with backoff; safe to re-run — completed stages are skipped.

set -u
DEST="$(dirname "$0")/external/merged_playbyplay"
LOG="$(dirname "$0")/external/pull.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$(dirname "$0")/external"
touch "$LOG"

if [ ! -d "$DEST/.git" ]; then
  log "STAGE1 clone start"
  for i in 1 2 3 4 5 6 7 8; do
    if git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/gabriel1200/merged_playbyplay "$DEST" >>"$LOG" 2>&1; then
      log "STAGE1 clone done"; break
    fi
    log "STAGE1 attempt $i failed; retrying in 30s"; sleep 30
    [ -d "$DEST/.git" ] && break
  done
fi
[ -d "$DEST/.git" ] || { log "FATAL clone failed"; exit 1; }

cd "$DEST" || exit 1
git config remote.origin.promote true 2>/dev/null || true

for STAGE_DIR in old_data pbp_data; do
  if [ -d "$DEST/$STAGE_DIR" ] && [ -n "$(ls -A "$DEST/$STAGE_DIR" 2>/dev/null)" ]; then
    log "STAGE for $STAGE_DIR already present, skipping"
    continue
  fi
  log "STAGE checkout $STAGE_DIR start"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if git sparse-checkout add "$STAGE_DIR" >>"$LOG" 2>&1 && [ -n "$(ls -A "$STAGE_DIR" 2>/dev/null)" ]; then
      log "STAGE checkout $STAGE_DIR done ($(ls "$STAGE_DIR" | wc -l | tr -d ' ') entries)"
      break
    fi
    log "STAGE $STAGE_DIR attempt $i failed; retrying in 45s"
    sleep 45
  done
  [ -n "$(ls -A "$DEST/$STAGE_DIR" 2>/dev/null)" ] || { log "FATAL $STAGE_DIR incomplete"; exit 1; }
done

log "PULL_COMPLETE size=$(du -sh "$DEST" | cut -f1)"
