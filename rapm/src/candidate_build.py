#!/usr/bin/env python3
"""Load candidate build.py — engineers NEW columns on top of base parquet."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from paths import DATA, FEATURES_DIR

CANDIDATES = FEATURES_DIR / "candidates"
BASE_PARQUET = DATA / "spm_features_windows.parquet"


def load_build_module(gen_id: int):
    path = CANDIDATES / f"gen_{gen_id:03d}" / "build.py"
    if not path.exists():
        raise FileNotFoundError(f"No build.py at {path}")
    spec = importlib.util.spec_from_file_location(f"candidate_gen_{gen_id:03d}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{path} must define build(feats) -> (feats, new_cols)")
    return mod


def apply_candidate_build(gen_id: int, base: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Run candidate build; return augmented frame + names of NEW columns only."""
    base = base if base is not None else pd.read_parquet(BASE_PARQUET)
    mod = load_build_module(gen_id)
    out = mod.build(base)
    if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError("build() must return (DataFrame, list[str] new column names)")
    feats, new_cols = out
    missing = [c for c in new_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"build() listed columns not in output: {missing}")
    return feats, list(new_cols)


def staging_dir_for(gen_id: int) -> Path | None:
    """Latest staging folder if manifest references staging."""
    cfg_path = CANDIDATES / f"gen_{gen_id:03d}" / "manifest.json"
    if not cfg_path.exists():
        return None
    import json

    cfg = json.loads(cfg_path.read_text())
    sid = cfg.get("staging_run")
    if sid:
        p = FEATURES_DIR / "staging" / sid
        return p if p.exists() else None
    # default: most recent curator run
    staging = FEATURES_DIR / "staging"
    runs = sorted(staging.glob("curator_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None
