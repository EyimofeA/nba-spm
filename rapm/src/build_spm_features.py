#!/usr/bin/env python3
"""Step 2: SPM feature table, one row per player-window, window-T info only.

Audits playersheet coverage against the production RAPM panel (>=95% of
possession-weighted player-windows must match by PLAYER_ID), then builds
two feature tiers:
  tier1 (box, 1997+): per-100 core box rates + shooting profile + usage + age
  tier2 (tracking, 2014+): drives, touches, defender-proximity shooting,
         shot quality, passing/creation, rebound detail

Window features = possession-weighted mean of the seasons in the window.
Output: data/spm_features_windows.parquet + an audit report printed and saved.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paths import DATA, DIAGNOSTICS_DIR, RAPM_RESULTS, ensure_dirs

ensure_dirs()
SHEETS = DATA.parent.parent / "data" / "raw" / "playersheets" / "year_totals"
PANEL = RAPM_RESULTS / "final_20260703_hl250" / "rapm_all_windows.csv"

# tier-1: available all eras (from box/pbp)
TIER1 = {
    "MIN": "sum", "GP": "sum", "OffPoss": "sum", "DefPoss": "sum",
    "PTS": "sum", "AST": "sum", "TOV": "sum", "STL": "sum", "BLK": "sum",
    "OREB": "sum", "DREB": "sum", "PF": "sum", "PFD": "sum", "FTA": "sum", "FTM": "sum",
    "FG2A": "sum", "FG2M": "sum", "FG3A": "sum", "FG3M": "sum",
    "USG_PCT": "wmean", "TS_PCT": "wmean", "AGE": "last",
    "AtRimFrequency": "wmean", "AtRimAccuracy": "wmean",
    "Corner3Frequency": "wmean", "Arc3Frequency": "wmean",
    "Assisted2sPct": "wmean", "Assisted3sPct": "wmean",
    "LiveBallTurnoverPct": "wmean", "ShootingFoulsDrawnPct": "wmean",
}
# tier-2: tracking era (~2014+)
TIER2 = {
    "DRIVES": "sum", "DRIVE_PTS": "sum", "DRIVE_AST": "sum", "DRIVE_TOV": "sum",
    "TOUCHES": "sum", "PAINT_TOUCHES": "sum", "POST_TOUCHES": "sum", "ELBOW_TOUCHES": "sum",
    "TIME_OF_POSS": "sum", "AVG_SEC_PER_TOUCH": "wmean", "AVG_DRIB_PER_TOUCH": "wmean",
    "PASSES_MADE": "sum", "POTENTIAL_AST": "sum", "SECONDARY_AST": "sum", "AST_PTS_CREATED": "sum",
    "CATCH_SHOOT_FGA": "sum", "PULL_UP_FGA": "sum",
    "wide_open_FG3A": "sum", "wide_open_FG3M": "sum",
    "tight_FGA": "sum", "very_tight_FGA": "sum",
    "ShotQualityAvg": "wmean", "REB_CONTEST": "sum", "REB_CHANCES": "sum",
    "OnDefRtg": "wmean", "OnOffRtg": "wmean",
}


def load_season(season: int) -> pd.DataFrame | None:
    f = SHEETS / f"{season}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, low_memory=False)
    df["season"] = season
    return df


def aggregate_window(frames: list[pd.DataFrame], spec: dict) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    cols = {k: v for k, v in spec.items() if k in df.columns}
    df["w_"] = df["MIN"].fillna(0).clip(lower=1)
    out = []
    for pid, g in df.groupby("PLAYER_ID"):
        rec = {"PLAYER_ID": pid}
        for c, how in cols.items():
            vals = pd.to_numeric(g[c], errors="coerce")
            if how == "sum":
                rec[c] = vals.sum()
            elif how == "wmean":
                m = vals.notna()
                rec[c] = np.average(vals[m], weights=g["w_"][m]) if m.any() else np.nan
            else:
                rec[c] = vals.dropna().iloc[-1] if vals.notna().any() else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def to_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sums to per-100-possession rates; keep pct/avg features as-is."""
    poss = df.get("OffPoss", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    rates = df.copy()
    for c in df.columns:
        if c in ("PLAYER_ID", "OffPoss", "DefPoss", "MIN", "GP", "AGE") or df[c].dtype == object:
            continue
        spec = TIER1.get(c) or TIER2.get(c)
        if spec == "sum":
            rates[c + "_p100"] = 100.0 * df[c] / poss
            rates.drop(columns=[c], inplace=True)
    return rates


def main() -> None:
    panel = pd.read_csv(PANEL)
    windows = sorted(panel["Window_End"].unique())
    audit = []
    feats_all = []
    for end in windows:
        seasons = [end - 2, end - 1, end]
        frames = [f for s in seasons if (f := load_season(s)) is not None]
        if not frames:
            audit.append({"window_end": end, "sheet_seasons": 0, "coverage": 0.0})
            continue
        f1 = aggregate_window(frames, {**TIER1, **TIER2})
        f1 = to_rates(f1)
        f1["Window_End"] = end
        pw = panel[panel["Window_End"] == end]
        merged = pw.merge(f1, on="PLAYER_ID", how="left", suffixes=("", "_f"))
        poss_w = merged["Poss_Off"].clip(lower=0)
        cov = float(np.average(merged["MIN"].notna(), weights=poss_w))
        tier2_cov = float(np.average(merged["TOUCHES_p100"].notna(), weights=poss_w)) if "TOUCHES_p100" in merged else 0.0
        audit.append({"window_end": end, "sheet_seasons": len(frames),
                      "coverage": round(cov, 4), "tier2_coverage": round(tier2_cov, 4),
                      "n_players": len(pw)})
        feats_all.append(f1)

    audit_df = pd.DataFrame(audit)
    feats = pd.concat(feats_all, ignore_index=True)
    out_path = DATA / "spm_features_windows.parquet"
    feats.to_parquet(out_path, index=False)
    audit_path = DIAGNOSTICS_DIR / "spm_feature_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    print(audit_df.to_string(index=False))
    bad = audit_df[(audit_df["coverage"] < 0.95) & (audit_df["sheet_seasons"] > 0)]
    print(f"\nwindows below 95% coverage: {len(bad)}")
    print(f"features: {feats.shape[1]-2} cols x {len(feats):,} player-windows")
    print(f"FEATURES_DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
