#!/usr/bin/env python3
"""User lane: evaluate feature functions from rapm/features/user/."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

from feature_eval import append_results_tsv, run_splice, splice_to_tsv_row
from paths import DATA, FEATURES_DIR, ensure_dirs

ensure_dirs()
FEAT_PARQUET = DATA / "spm_features_windows.parquet"


def load_user_module(user_name: str):
    path = FEATURES_DIR / "user" / f"{user_name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"user_{user_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_feature_funcs(mod) -> list[tuple[str, object]]:
    out = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        fn = getattr(mod, name)
        if callable(fn) and name.startswith("feat_"):
            out.append((name, fn))
    return out


def eval_user_features(user_name: str, fold: str = "f24", c: float = 2.0) -> None:
    """Tier-2 quick eval: minutes splice unchanged; logs user feature metadata."""
    mod = load_user_module(user_name)
    funcs = list_feature_funcs(mod)
    if not funcs:
        print(f"No feat_* functions in user/{user_name}.py", flush=True)
        return

    if FEAT_PARQUET.exists():
        sample = pd.read_parquet(FEAT_PARQUET).head(100)
    else:
        sample = pd.DataFrame()

    rows = []
    for fname, fn in funcs:
        t0 = time.time()
        try:
            if not sample.empty:
                vals = [fn(row.to_dict()) for _, row in sample.iterrows()]
                finite = sum(pd.notna(v) and abs(v) < 1e12 for v in vals)
                sanity = finite / len(vals) if vals else 0
            else:
                sanity = float("nan")
            status = "tier1_only"
        except Exception as e:
            sanity = 0
            status = "crash"
            fname = f"{fname}:{e}"

        # baseline splice for reference gate numbers
        splice_res = run_splice(fold, c_grid=(c,), name_prefix=f"user_ref_{user_name}")
        ref = splice_res[0]
        row = splice_to_tsv_row(ref, source=f"user:{user_name}")
        row["id"] = f"user_{user_name}_{fname}_{fold}"
        row["description"] = fname
        row["status"] = status
        row["tier_reached"] = 1 if status == "tier1_only" else 0
        row["complexity_score"] = len(str(fn.__code__.co_code))
        rows.append(row)
        print(f"USER_FEATURE {fname}: sanity={sanity:.2f} ref_gate={ref.margin_corr}", flush=True)

    append_results_tsv(rows)
    reg = FEATURES_DIR / "registry.jsonl"
    with open(reg, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"USER_SUBMIT_DONE {len(rows)} features -> results.tsv", flush=True)


def main() -> None:
    user = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv else "default"
    fold = sys.argv[sys.argv.index("--fold") + 1] if "--fold" in sys.argv else "f24"
    eval_user_features(user, fold=fold)


if __name__ == "__main__":
    main()
