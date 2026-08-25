#!/usr/bin/env python3
"""Feature foundry — evaluate feature candidates from features/candidates/gen_NNN/.

Minutes baseline is FROZEN (prepare.GATE_BASELINES). Do not re-run except
`python3 feature_foundry.py --verify-harness` (one-time harness check).
Compute budget goes to feature subsets in candidate manifests.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

from candidate_build import apply_candidate_build
from feature_eval import append_results_tsv, run_splice, splice_to_tsv_row
from feature_report import features_chosen_str, features_for_config, log_generation_features, resolve_feature_cols
from paths import DATA, FEATURES_DIR, RAPM_ALL_WINDOWS_CSV, ensure_dirs
from run_lock import rapm_run_lock

ensure_dirs()
REGISTRY = FEATURES_DIR / "registry.jsonl"
CANDIDATES = FEATURES_DIR / "candidates"

# Frozen reference — never burn RAPM fits re-proving this.
FROZEN_MINUTES = {"f24": 0.7335, "f23": 0.6953}


def log_registry(row: dict) -> None:
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def load_candidate(gen_id: int) -> dict:
    path = CANDIDATES / f"gen_{gen_id:03d}" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No candidate at {path} — add manifest.json (+ build.py for new features)")
    cfg = json.loads(path.read_text())
    if cfg.get("deprecated"):
        raise ValueError(f"gen_{gen_id:03d} is deprecated ({cfg.get('deprecated_reason', 'ablation only')})")
    cfg.setdefault("prior", "spmv2_subset")
    cfg.setdefault("type", "subset")
    cfg.setdefault("c_grid", [2.0])
    if isinstance(cfg["c_grid"], list):
        cfg["c_grid"] = tuple(float(x) for x in cfg["c_grid"])
    return cfg


def feature_cols_for_candidate(gen_id: int, cfg: dict) -> tuple[pd.DataFrame, list[str] | None]:
    """Return feature table and column list for this candidate."""
    base = pd.read_parquet(DATA / "spm_features_windows.parquet")
    if cfg.get("type") == "build":
        built, new_cols = apply_candidate_build(gen_id, base)
        if cfg.get("use_columns") == "new_only":
            return built, new_cols
        extra = resolve_feature_cols(cfg) or []
        return built, list(dict.fromkeys(new_cols + extra))
    # legacy: subset of existing parquet only (not autoresearch)
    return base, resolve_feature_cols(cfg)


def make_prior_fn(fold: str, cfg: dict, gen_id: int, feats_built: pd.DataFrame, fcols: list[str] | None):
    kind = cfg.get("prior", "spmv2_subset")
    if kind == "minutes":
        return None
    if kind in ("spmv2", "spmv2_subset"):
        from spm_v2 import build_spm_prior_pooled, merge_labels, pool_feats_for_fold

        panel = pd.read_csv(RAPM_ALL_WINDOWS_CSV)
        alpha = float(cfg.get("alpha", 1000.0))
        residual = bool(cfg.get("residual", False))
        pooled = pool_feats_for_fold(feats_built, fold)
        merged = merge_labels(pooled, panel)

        def prior_fn(dm, beta):
            return build_spm_prior_pooled(
                dm, beta, merged, alpha=alpha, residual=residual, feature_cols=fcols
            )

        return prior_fn
    raise ValueError(f"unknown prior kind: {kind}")


def _beats_minutes_on_search(rows: list[dict]) -> bool:
    best = {"f24": 0.0, "f23": 0.0}
    for r in rows:
        for fold in ("f24", "f23"):
            v = r.get(f"gate_{fold}")
            if v != "" and v is not None and pd.notna(v):
                best[fold] = max(best[fold], float(v))
    return best["f24"] > FROZEN_MINUTES["f24"] and best["f23"] > FROZEN_MINUTES["f23"]


def run_verify_harness() -> None:
    """One-shot minutes repro — run once, not as a foundry generation."""
    cfg = {"prior": "minutes", "code": "minutes_harness_verify", "description": "harness verify only"}
    print("VERIFY_HARNESS minutes (one-time — do not repeat)", flush=True)
    rows = []
    with rapm_run_lock("verify_harness"):
        for fold in ("f24", "f23"):
            for r in run_splice(fold, c_grid=(2.0,), name_prefix="harness_verify"):
                row = splice_to_tsv_row(r, source="harness:verify", features_chosen=features_chosen_str(features_for_config(cfg)))
                rows.append(row)
    append_results_tsv(rows)
    print("VERIFY_HARNESS done", flush=True)


def run_generation(gen_id: int, folds: tuple[str, ...] | None = None) -> None:
    cfg = load_candidate(gen_id)
    feats_built, fcols = feature_cols_for_candidate(gen_id, cfg)
    cfg_for_report = {**cfg, "feature_cols": fcols}
    feat_doc = features_for_config(cfg_for_report)
    if cfg.get("type") == "build" and fcols:
        feat_doc["summary"] = f"NEW built features ({len(fcols)}): " + ", ".join(fcols[:8])
        if len(fcols) > 8:
            feat_doc["summary"] += ", …"
        feat_doc["features"] = fcols
        feat_doc["n_features"] = len(fcols)
    feat_short = features_chosen_str(feat_doc)
    print(f"FOUNDRY_GEN_{gen_id} type={cfg.get('type')} candidate={cfg.get('code')}", flush=True)
    print(f"CHOSEN_FEATURES gen={gen_id} {feat_short}", flush=True)
    rows = []
    with rapm_run_lock(f"foundry_gen_{gen_id}"):
        for fold in folds or _default_folds(cfg):
            if fold == "vault" and rows and not _beats_minutes_on_search(rows):
                print("  skip vault — search folds did not beat frozen minutes", flush=True)
                break
            print(f"  fold {fold} ...", flush=True)
            try:
                prior_fn = make_prior_fn(fold, cfg, gen_id, feats_built, fcols)
                for r in run_splice(
                    fold,
                    prior_fn=prior_fn,
                    c_grid=cfg.get("c_grid", (2.0,)),
                    name_prefix=f"g{gen_id}_{cfg.get('code', 'cand')}",
                ):
                    row = splice_to_tsv_row(
                        r, source=f"foundry:gen{gen_id}", features_chosen=feat_short
                    )
                    row["description"] = cfg.get("description", row.get("description", ""))
                    row["tier_reached"] = 3 if fold == "vault" else 2
                    gate_ok = r.anchors_ok and r.margin_corr >= 0.69
                    row["status"] = "keep" if gate_ok else "discard"
                    rows.append(row)
                    log_registry({**row, "code": cfg["code"], "gen": gen_id, "features": feat_doc})
                    print(f"  {r.name} corr={r.margin_corr} status={row['status']}", flush=True)
            except Exception:
                print(f"  FOLD_FAIL {fold}", flush=True)
                traceback.print_exc()
                raise

    append_results_tsv(rows)
    log_generation_features(gen_id, {**cfg, "feature_cols": fcols}, rows)
    src = Path(__file__).parent
    subprocess.run([sys.executable, str(src / "fig_foundry_progress.py")], check=False)
    subprocess.run([sys.executable, str(src / "build_human_viewer.py")], check=False)
    subprocess.run([sys.executable, str(src / "verify_keep.py")], check=False)
    subprocess.run([sys.executable, str(src / "log_run_greps.py"), f"foundry_g{gen_id}"], check=False)
    print(f"FOUNDRY_GEN_{gen_id} done {len(rows)} rows", flush=True)


def _default_folds(cfg: dict) -> tuple[str, ...]:
    if cfg.get("search_folds_only"):
        return ("f24", "f23")
    return ("f24", "f23", "vault")


def update_leaderboard() -> None:
    tsv = FEATURES_DIR / "results.tsv"
    if not tsv.exists():
        return
    df = pd.read_csv(tsv, sep="\t")
    df["gate_best"] = df[["gate_f24", "gate_f23", "gate_vault"]].apply(
        lambda r: pd.to_numeric(r, errors="coerce").max(), axis=1
    )
    df = df.sort_values("gate_best", ascending=False)
    df.to_csv(FEATURES_DIR / "leaderboard.csv", index=False)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("gen", type=int, nargs="?", default=6, help="gen id → candidates/gen_NNN/ (use build.py for new features)")
    p.add_argument("--folds", default=None, help="comma-separated folds (default from manifest)")
    p.add_argument("--verify-harness", action="store_true", help="one-shot minutes check only")
    args = p.parse_args()
    if args.verify_harness:
        run_verify_harness()
        return
    folds = None
    if args.folds:
        folds = tuple(f.strip() for f in args.folds.split(",") if f.strip())
    run_generation(args.gen, folds=folds)
    update_leaderboard()


if __name__ == "__main__":
    main()
