#!/usr/bin/env python3
"""Autoresearch loop — run pending build candidates, propose new gens, repeat."""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

from autoresearch_proposer import pending_gens, propose_next, save_state, load_state
from paths import OUTPUTS, ensure_dirs
from run_lock import LOCK_PATH

ensure_dirs()
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent


def clear_stale_lock() -> None:
    if not LOCK_PATH.exists():
        return
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
            import os

            os.kill(pid, 0)
            print(f"LOCK_HELD pid={pid} — waiting", flush=True)
            return
        except OSError:
            pass
    LOCK_PATH.unlink(missing_ok=True)
    print(f"LOCK_CLEARED stale holder={text!r}", flush=True)


def run_gen(gen_id: int) -> bool:
    log = OUTPUTS / f"foundry_g{gen_id}.log"
    done = OUTPUTS / f"foundry_g{gen_id}.done"
    if done.exists():
        print(f"SKIP gen_{gen_id:03d} (done)", flush=True)
        return True

    clear_stale_lock()
    # Resume: if log shows FOUNDRY_GEN_N done, skip; else run full foundry (handles both folds)
    if log.exists() and f"FOUNDRY_GEN_{gen_id} done" in log.read_text():
        done.write_text("ok\n")
        print(f"SKIP gen_{gen_id:03d} (already in log)", flush=True)
        return True

    cmd = [sys.executable, str(SRC / "feature_foundry.py"), str(gen_id)]
    print(f"AUTORESEARCH_RUN gen_{gen_id:03d} ...", flush=True)
    t0 = time.time()
    try:
        with open(log, "a") as lf:
            lf.write(f"\n=== AUTORESEARCH {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
            proc = subprocess.run(cmd, cwd=SRC, stdout=lf, stderr=subprocess.STDOUT, check=True)
        done.write_text("ok\n")
        state = load_state()
        if gen_id not in state.setdefault("completed", []):
            state["completed"].append(gen_id)
            save_state(state)
        print(f"AUTORESEARCH_DONE gen_{gen_id:03d} elapsed={time.time()-t0:.0f}s", flush=True)
        return True
    except subprocess.CalledProcessError:
        print(f"AUTORESEARCH_FAIL gen_{gen_id:03d} see {log}", flush=True)
        return False
    except Exception:
        traceback.print_exc()
        return False


def run_loop(max_gens: int = 20, propose: bool = True) -> None:
    """Run all pending gens, then propose and run until max_gens or queue empty."""
    ran = 0
    # Pass 1: existing pending (6, 7, 8, 9...)
    for gen_id in pending_gens(OUTPUTS):
        if ran >= max_gens:
            break
        if run_gen(gen_id):
            ran += 1
        else:
            break

    # Pass 2: propose new
    while ran < max_gens and propose:
        gen_id = propose_next(materialize=True)
        if gen_id is None:
            print("AUTORESEARCH_QUEUE_EMPTY", flush=True)
            break
        if not run_gen(gen_id):
            break
        ran += 1

    subprocess.run([sys.executable, str(SRC / "build_human_viewer.py")], cwd=SRC, check=False)
    print(f"AUTORESEARCH_LOOP finished ran={ran}", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max-gens", type=int, default=20)
    p.add_argument("--no-propose", action="store_true")
    p.add_argument("--gen", type=int, default=None, help="run single gen only")
    args = p.parse_args()

    if args.gen is not None:
        ok = run_gen(args.gen)
        sys.exit(0 if ok else 1)
    run_loop(max_gens=args.max_gens, propose=not args.no_propose)


if __name__ == "__main__":
    main()
