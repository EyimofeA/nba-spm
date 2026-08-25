#!/usr/bin/env python3
"""Single-process lock for heavy RAPM fits (tiny-PC policy)."""
from __future__ import annotations

import fcntl
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from paths import OUTPUTS, ensure_dirs

ensure_dirs()
LOCK_PATH = OUTPUTS / "rapm_run.lock"


@contextmanager
def rapm_run_lock(name: str = "rapm"):
    # Clear stale lock from dead processes (common when nohup dies mid-fit).
    if LOCK_PATH.exists():
        text = LOCK_PATH.read_text().strip()
        pid = None
        for part in text.split():
            if part.startswith("pid="):
                try:
                    pid = int(part.split("=", 1)[1])
                except ValueError:
                    pass
        if pid is not None:
            try:
                os.kill(pid, 0)
            except OSError:
                LOCK_PATH.unlink(missing_ok=True)
                print(f"RAPM_LOCK_CLEARED stale pid={pid}", flush=True)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = LOCK_PATH.read_text().strip() if LOCK_PATH.stat().st_size else "unknown"
        print(f"RAPM_LOCK_BUSY holder={holder} — only one RAPM process at a time", flush=True)
        fh.close()
        sys.exit(2)
    fh.write(f"pid={os.getpid()} name={name}\n")
    fh.flush()
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
