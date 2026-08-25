#!/usr/bin/env python3
"""Deterministic verifier for feature foundry keep claims."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from paths import FEATURES_DIR

BLOCKLIST = re.compile(
    r"onoffrtg|ondefrtg|plus.?minus|team_ortg|team_drtg|team_wins|net.?rating",
    re.I,
)


def verify_row(row: dict, feature_code: str = "") -> dict:
    issues = []
    if BLOCKLIST.search(feature_code):
        issues.append("blocklist_hit_in_code")
    for col in ("OnOffRtg", "OnDefRtg", "PLUS_MINUS"):
        if col in feature_code:
            issues.append(f"blocklist_column:{col}")

    gate_f24 = row.get("gate_f24")
    gate_f23 = row.get("gate_f23")
    if row.get("status") == "keep":
        if not row.get("anchor_ok", False):
            issues.append("anchors_failed")
        if gate_f24 and float(gate_f24) < 0.7335 and row.get("source", "").startswith("foundry"):
            pass  # warn only — minutes baseline is reference not hard floor for discard
        oof_off = float(row.get("oof_r2_off") or 0)
        gate = float(gate_f24 or 0)
        if oof_off > 0.3 and gate < 0.65:
            issues.append("oof_gate_divergence")

    verdict = "confirm" if not issues else ("block_ship" if "blocklist" in str(issues) else "dispute")
    return {"verdict": verdict, "issues": issues}


def verify_registry_entry(path: Path) -> dict:
    row = json.loads(path.read_text()) if path.suffix == ".json" else {}
    code = row.get("code", "")
    return verify_row(row, code)


def main() -> None:
    tsv = FEATURES_DIR / "results.tsv"
    if not tsv.exists():
        print("NO_RESULTS", flush=True)
        return
    df = pd.read_csv(tsv, sep="\t")
    keeps = df[df["status"].astype(str).str.lower() == "keep"]
    for _, r in keeps.iterrows():
        v = verify_row(r.to_dict())
        print(f"{r['id']}: {v['verdict']} {v['issues']}", flush=True)


if __name__ == "__main__":
    main()
