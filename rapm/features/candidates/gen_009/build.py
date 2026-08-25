"""Gen 009 — blend NEW box-derived + tracking interaction features."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent


def _load_build(name: str):
    path = _HERE.parent / name / "build.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = feats
    all_new: list[str] = []
    for sub in ("gen_006", "gen_008"):
        mod = _load_build(sub)
        df, cols = mod.build(df)
        all_new.extend(cols)
    return df, list(dict.fromkeys(all_new))
