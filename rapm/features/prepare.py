#!/usr/bin/env python3
"""Fixed constants for the Feature Foundry harness (read-only for agents).

Column catalog from spm_features_windows.parquet, leakage blocklist, fold
definitions, and baseline gate numbers. Import from feature_eval and candidate
scripts — do not modify during search runs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import DATA, FEATURES_DIR  # noqa: E402

FEATURES_PARQUET = DATA / "spm_features_windows.parquet"

FOLDS = {
    "f24": {"train": [2021, 2022, 2023], "test": 2024},
    "f23": {"train": [2020, 2021, 2022], "test": 2023},
    "vault": {"train": [2015, 2016, 2017], "test": 2018},
}
SEARCH_FOLDS = ("f24", "f23")
CONFIRM_FOLD = "vault"

# Reproduce these through feature_eval before trusting the harness.
GATE_BASELINES = {
    "minutes_prior_c2": {"f24": 0.7335, "f23": 0.6953},
    "champion_zero_prior": {"f24": 0.6596, "f23": 0.5939},
}

META_COLS = frozenset({"PLAYER_ID", "Window_End", "MIN", "GP"})
BLOCKLIST_EXACT = META_COLS | {"OnOffRtg", "OnDefRtg"}
BLOCKLIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"on.?off",
        r"plus.?minus",
        r"\bpm\b",
        r"team_.*(ortg|drtg|rating|wins)",
        r"\b(net_)?rtg\b",
        r"\bortg\b|\bdrtg\b",
    )
)


def is_blocked(name: str) -> bool:
    if name in BLOCKLIST_EXACT:
        return True
    return any(p.search(name) for p in BLOCKLIST_PATTERNS)


def load_catalog(path: Path | None = None) -> dict:
    """Load column catalog from the SPM features parquet."""
    path = path or FEATURES_PARQUET
    df = pd.read_parquet(path, columns=None)
    cols = []
    for c in df.columns:
        cols.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "blocked": is_blocked(c),
            "meta": c in META_COLS,
        })
    allowed = [x["name"] for x in cols if not x["blocked"]]
    return {
        "path": str(path),
        "n_rows": len(df),
        "columns": cols,
        "allowed_features": allowed,
        "n_allowed": len(allowed),
        "n_blocked": sum(x["blocked"] for x in cols),
    }


def allowed_feature_cols(path: Path | None = None) -> list[str]:
    return load_catalog(path)["allowed_features"]


def main() -> None:
    cat = load_catalog()
    print(f"catalog: {cat['path']}")
    print(f"rows={cat['n_rows']:,} allowed={cat['n_allowed']} blocked={cat['n_blocked']}")
    print("folds:", {k: f"{v['train']} -> {v['test']}" for k, v in FOLDS.items()})
    print("gate baselines:", GATE_BASELINES)
    blocked = [c["name"] for c in cat["columns"] if c["blocked"] and not c["meta"]]
    if blocked:
        print("blocked sample:", blocked[:12])


if __name__ == "__main__":
    main()
