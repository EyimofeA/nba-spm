#!/usr/bin/env python3
"""Append common grep snapshots to a digest — don't lose status in log noise."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from paths import DIAGNOSTICS_DIR, FEATURES_DIR, OUTPUTS, ensure_dirs

ensure_dirs()
DIGEST = OUTPUTS / "grep_digest.log"
ROOT = Path(__file__).resolve().parent.parent


def run_grep(label: str, cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=30)
        out = (r.stdout or r.stderr or "").strip()
        return f"### {label}\n```\n{out[:8000]}\n```\n"
    except Exception as e:
        return f"### {label}\n(error: {e})\n"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "manual"
    blocks = [
        run_grep(
            "experiments.csv last 8",
            ["tail", "-8", str(DIAGNOSTICS_DIR / "experiments.csv")],
        ),
        run_grep(
            "results.tsv last 8",
            ["tail", "-8", str(FEATURES_DIR / "results.tsv")],
        ),
        run_grep(
            "EXPERIMENT_DONE in logs",
            ["grep", "-h", "EXPERIMENT_DONE", str(OUTPUTS / "spmv2_run.log"),
             str(OUTPUTS / "feature_eval_baseline.log"), str(OUTPUTS / "foundry_g0.log")],
        ),
        run_grep(
            "BASELINE / FOUNDRY / SPMV2 done flags",
            ["grep", "-h", "-E", "BASELINE_SUMMARY|FOUNDRY_GEN|SPMV2_ALL_DONE|FOUNDRY_G0_DONE",
             str(OUTPUTS / "feature_eval_baseline.log"), str(OUTPUTS / "foundry_g0.log"),
             str(OUTPUTS / "spmv2_run.log")],
        ),
        run_grep(
            "best margin_corr from experiments",
            ["python3", "-c",
             "import pandas as pd; p='outputs/diagnostics/experiments.csv'; "
             "df=pd.read_csv(p); df=df.dropna(subset=['margin_corr']); "
             "print(df.nlargest(5,'margin_corr')[['name','margin_corr','margin_rmse']].to_string(index=False))"],
        ),
        run_grep(
            "chosen features / improvement",
            ["tail", "-20", str(FEATURES_DIR / "improvement_log.md")],
        ),
        run_grep(
            "CHOSEN_FEATURES in logs",
            ["grep", "-h", "CHOSEN_FEATURES", str(OUTPUTS / "foundry_g0.log"),
             str(OUTPUTS / "foundry_g1.log"), str(OUTPUTS / "grep_digest.log")],
        ),
    ]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = f"\n---\n## grep_digest {tag} @ {ts}\n\n" + "\n".join(blocks)
    with open(DIGEST, "a") as f:
        f.write(body)
    print(DIGEST, flush=True)


if __name__ == "__main__":
    main()
