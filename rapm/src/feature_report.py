#!/usr/bin/env python3
"""Document which features each prior / foundry generation uses."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paths import DATA, FEATURES_DIR, ensure_dirs

ensure_dirs()

IMPROVEMENT_LOG = FEATURES_DIR / "improvement_log.md"
CHOSEN_JSONL = FEATURES_DIR / "chosen_features.jsonl"

TIER1_NAMES = frozenset({
    "MIN", "GP", "OffPoss", "DefPoss", "PTS", "AST", "TOV", "STL", "BLK",
    "OREB", "DREB", "PF", "PFD", "FTA", "FTM", "FG2A", "FG2M", "FG3A", "FG3M",
    "USG_PCT", "TS_PCT", "AGE", "AtRimFrequency", "AtRimAccuracy",
    "Corner3Frequency", "Arc3Frequency", "Assisted2sPct", "Assisted3sPct",
    "LiveBallTurnoverPct", "ShootingFoulsDrawnPct",
})
TIER2_NAMES = frozenset({
    "DRIVES", "DRIVE_PTS", "DRIVE_AST", "DRIVE_TOV", "TOUCHES", "PAINT_TOUCHES",
    "POST_TOUCHES", "ELBOW_TOUCHES", "TIME_OF_POSS", "AVG_SEC_PER_TOUCH",
    "AVG_DRIB_PER_TOUCH", "PASSES_MADE", "POTENTIAL_AST", "SECONDARY_AST",
    "AST_PTS_CREATED", "CATCH_SHOOT_FGA", "PULL_UP_FGA",
    "wide_open_FG3A", "wide_open_FG3M", "tight_FGA", "very_tight_FGA",
    "ShotQualityAvg", "REB_CONTEST", "REB_CHANCES",
})

MINUTES_PRIOR_FEATURES = [
    "log1p(off_poss)",
    "log1p(off_poss)²",
    "log1p(def_poss)",
    "log1p(def_poss)²",
]


def spmv2_feature_cols() -> list[str]:
    from spm_v2 import feature_matrix

    feats = pd.read_parquet(DATA / "spm_features_windows.parquet")
    _, cols = feature_matrix(feats.head(200))
    return cols


def _base_col(name: str) -> str:
    return name.replace("_p100", "").replace("_sq", "").replace("log_", "")


def tier_breakdown(cols: list[str]) -> dict[str, list[str]]:
    tier1, tier2, derived = [], [], []
    for c in cols:
        base = _base_col(c)
        if c.startswith("log_"):
            derived.append(c)
        elif base in TIER2_NAMES or any(
            base.startswith(t)
            for t in ("DRIVE", "TOUCH", "PASS", "CATCH", "PULL", "wide", "tight", "REB", "TIME_OF", "AVG_", "ShotQuality", "PAINT", "POST", "ELBOW")
        ):
            tier2.append(c)
        elif base in TIER1_NAMES or c.endswith("_p100"):
            tier1.append(c)
        else:
            derived.append(c)
    return {"tier1_box": tier1, "tier2_tracking": tier2, "derived": derived}


def resolve_feature_cols(cfg: dict) -> list[str] | None:
    """None = full SPM v2 set; else explicit list from manifest."""
    if cfg.get("feature_cols"):
        return list(cfg["feature_cols"])
    tier = cfg.get("feature_tier")
    if not tier or tier == "all":
        return None
    full = spmv2_feature_cols()
    tiers = tier_breakdown(full)
    if tier == "tier1":
        return tiers["tier1_box"] + [c for c in tiers["derived"] if "log_" in c]
    if tier == "tier1_box":
        return tiers["tier1_box"]
    if tier == "derived":
        return tiers["derived"]
    if tier == "tier2":
        return tiers["tier2_tracking"]
    if tier == "tier1_plus_derived":
        return tiers["tier1_box"] + tiers["derived"]
    if tier == "tracking_core":
        want = {
            "DRIVES_p100", "TOUCHES_p100", "ShotQualityAvg", "PASSES_MADE_p100",
            "POTENTIAL_AST_p100", "AVG_SEC_PER_TOUCH", "AVG_DRIB_PER_TOUCH",
            "CATCH_SHOOT_FGA_p100", "PULL_UP_FGA_p100", "log_OffPoss", "log_DefPoss",
        }
        return [c for c in full if c in want]
    raise ValueError(f"unknown feature_tier: {tier}")


def features_for_config(cfg: dict) -> dict:
    kind = cfg.get("prior", "minutes")
    if kind in ("spmv2", "spmv2_subset"):
        cols = resolve_feature_cols(cfg) or spmv2_feature_cols()
        sub = {**cfg, "feature_cols": cols, "prior": "spmv2_subset"}
        tiers = tier_breakdown(cols)
        residual = bool(cfg.get("residual", False))
        summary = (
            f"SPM subset ({len(cols)} cols): "
            f"tier1={len(tiers['tier1_box'])}, tier2={len(tiers['tier2_tracking'])}, "
            f"derived={len(tiers['derived'])}"
        )
        if residual:
            summary += " · residual over minutes"
        return {
            "prior": "spmv2_subset",
            "alpha": cfg.get("alpha"),
            "residual": residual,
            "n_features": len(cols),
            "features": cols,
            "summary": summary,
            "tiers": tiers,
        }
    if kind == "minutes":
        return {
            "prior": "minutes",
            "n_features": len(MINUTES_PRIOR_FEATURES),
            "features": MINUTES_PRIOR_FEATURES,
            "summary": "Possession-minutes proxy only (log poss + quadratic per side). No box/tracking.",
            "tiers": {"minutes_shape": MINUTES_PRIOR_FEATURES},
        }
    if kind == "spmv2":
        cols = spmv2_feature_cols()
        tiers = tier_breakdown(cols)
        residual = bool(cfg.get("residual", False))
        summary = (
            f"Full SPM v2 pooled set ({len(cols)} cols): "
            f"tier1 box={len(tiers['tier1_box'])}, tier2 tracking={len(tiers['tier2_tracking'])}, "
            f"derived={len(tiers['derived'])}"
        )
        if residual:
            summary += " · residual = minutes prior + SPM on pass-1 residuals"
        return {
            "prior": "spmv2",
            "alpha": cfg.get("alpha"),
            "residual": residual,
            "n_features": len(cols),
            "features": cols,
            "summary": summary,
            "tiers": tiers,
        }
    if kind == "custom" and cfg.get("features"):
        feats = list(cfg["features"])
        return {
            "prior": "custom",
            "n_features": len(feats),
            "features": feats,
            "summary": cfg.get("feature_summary", ", ".join(feats[:8])),
            "tiers": {"custom": feats},
        }
    return {"prior": kind, "n_features": 0, "features": [], "summary": "unknown", "tiers": {}}


def features_chosen_str(feat: dict, max_list: int = 12) -> str:
    if feat["prior"] == "minutes":
        return "minutes: log_poss + log_poss² (per side)"
    cols = feat.get("features") or []
    if len(cols) <= max_list:
        return f"{feat['prior']} ({feat['n_features']}): " + ", ".join(cols)
    return f"{feat['prior']} ({feat['n_features']}): {', '.join(cols[:max_list])}, …"


def _gate_summary(rows: list[dict]) -> dict:
    out: dict[str, float | str] = {}
    for fold in ("f24", "f23", "vault"):
        col = f"gate_{fold}"
        for r in rows:
            v = r.get(col)
            if v != "" and v is not None and pd.notna(v):
                out[fold] = float(v)
    statuses = {r.get("status") for r in rows}
    out["status"] = "keep" if "keep" in statuses else "discard"
    return out


def _beats_minutes(gates: dict) -> bool:
    import sys

    feat_dir = Path(__file__).resolve().parent.parent / "features"
    if str(feat_dir) not in sys.path:
        sys.path.insert(0, str(feat_dir))
    from prepare import GATE_BASELINES

    bl = GATE_BASELINES["minutes_prior_c2"]
    for fold in ("f24", "f23"):
        if fold not in gates or gates[fold] <= bl[fold]:
            return False
    return True


def log_generation_features(gen_id: int, cfg: dict, rows: list[dict]) -> dict:
    feat = features_for_config(cfg)
    if cfg.get("feature_cols"):
        cols = list(cfg["feature_cols"])
        feat = {
            **feat,
            "features": cols,
            "n_features": len(cols),
            "summary": f"NEW built features ({len(cols)}): " + ", ".join(cols[:10]) + ("…" if len(cols) > 10 else ""),
        }
    gates = _gate_summary(rows)
    improved = _beats_minutes(gates)
    record = {
        "gen": gen_id,
        "code": cfg.get("code"),
        "description": cfg.get("description"),
        "ts": datetime.now(timezone.utc).isoformat(),
        "features_chosen": feat,
        "features_chosen_short": features_chosen_str(feat),
        "gates": gates,
        "beats_minutes_baseline": improved,
        "c_grid": list(cfg.get("c_grid", ())),
    }
    with open(CHOSEN_JSONL, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    tiers = feat.get("tiers") or {}
    tier_lines = []
    for tname, tcols in tiers.items():
        if not tcols:
            continue
        preview = ", ".join(tcols[:10])
        if len(tcols) > 10:
            preview += f", … (+{len(tcols) - 10} more)"
        tier_lines.append(f"- **{tname}** ({len(tcols)}): {preview}")

    gate_str = " · ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in gates.items()
    )
    verdict = "beats minutes on f24+f23" if improved else "does not beat minutes on both search folds"
    block = f"""
## Gen {gen_id} — {cfg.get('code', 'unknown')} ({record['ts'][:10]})

**Prior:** {feat.get('prior')} · {cfg.get('description', '')}

**Features chosen:** {feat.get('summary')}

{chr(10).join(tier_lines) if tier_lines else ''}

**Gates:** {gate_str}

**Improvement:** {verdict}

---
"""
    if not IMPROVEMENT_LOG.exists():
        IMPROVEMENT_LOG.write_text(
            "# Feature Foundry — chosen features by generation\n\n"
            "After each gen completes, which inputs were used and whether search folds improved.\n\n"
        )
    with open(IMPROVEMENT_LOG, "a") as f:
        f.write(block)

    print(f"CHOSEN_FEATURES gen={gen_id} {features_chosen_str(feat)}", flush=True)
    print(f"IMPROVEMENT_VERDICT gen={gen_id} {verdict} gates={gates}", flush=True)
    return record


def backfill_improvement_log() -> None:
    tsv = FEATURES_DIR / "results.tsv"
    if not tsv.exists():
        return
    df = pd.read_csv(tsv, sep="\t")
    if "features_chosen" not in df.columns:
        df["features_chosen"] = ""
    df.loc[df["id"].astype(str).str.startswith("mprior_"), "features_chosen"] = features_chosen_str(
        features_for_config({"prior": "minutes"})
    )
    df.to_csv(tsv, sep="\t", index=False)

    if IMPROVEMENT_LOG.exists() and IMPROVEMENT_LOG.stat().st_size > 200:
        print("improvement_log exists (tsv backfilled)", flush=True)
        return
    # Document archived gen 0 from results only
    sub = df[df["id"].astype(str).str.startswith("foundry_g0_")]
    if not sub.empty:
        log_generation_features(0, {"prior": "minutes", "code": "minutes_prior_baseline", "description": "archived harness smoke"}, sub.to_dict(orient="records"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill_improvement_log()
    elif len(sys.argv) > 1:
        kind = sys.argv[1]
        cfg: dict = {"prior": kind}
        if len(sys.argv) > 2:
            cfg["alpha"] = float(sys.argv[2])
        print(json.dumps(features_for_config(cfg), indent=2))
    else:
        print("usage: feature_report.py backfill | minutes | spmv2")
