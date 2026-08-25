"""Gen 007 — playtype PPP features from staging (experimental; low join coverage).

Pivots playtype.csv → player-season PPP by type, merges to windows.
Only seasons with playtype data contribute; sparse by design until join fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[2] / "staging" / "curator_20260703_1459"
PLAYTYPE = STAGING / "playtype.csv"

# play types worth separate PPP signal
PLAY_TYPES = (
    "Isolation", "Transition", "PRBallHandler", "PRRollman", "Spotup",
    "OffScreen", "Cut", "Postup", "Handoff", "Misc",
)


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def build(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = feats.copy()
    new_cols: list[str] = []

    if not PLAYTYPE.exists():
        # stub cols so harness can run; all NaN → ridge handles via fillna in spm_v2
        for pt in PLAY_TYPES[:5]:
            c = f"new_pt_ppp_{_slug(pt)}"
            df[c] = np.nan
            new_cols.append(c)
        return df, new_cols

    pt = pd.read_csv(PLAYTYPE, low_memory=False)
    if "PLAYER_ID" not in pt.columns or "PPP" not in pt.columns:
        return df, new_cols

    pt["PLAYER_ID"] = pd.to_numeric(pt["PLAYER_ID"], errors="coerce")
    pt["year"] = pd.to_numeric(pt.get("year", pt.get("SEASON_ID", np.nan)), errors="coerce")
    pt["PPP"] = pd.to_numeric(pt["PPP"], errors="coerce")
    ptype = pt.get("playtype", pt.get("PLAY_TYPE", "")).astype(str)

    rows = []
    for pid, g in pt.groupby("PLAYER_ID"):
        if pd.isna(pid):
            continue
        for season, gs in g.groupby("year"):
            if pd.isna(season):
                continue
            rec = {"PLAYER_ID": int(pid), "season": int(season)}
            for pt_name in PLAY_TYPES:
                m = gs[ptype.str.contains(pt_name, case=False, na=False)]
                if len(m):
                    rec[f"new_pt_ppp_{_slug(pt_name)}"] = m["PPP"].mean()
            rows.append(rec)

    if not rows:
        return df, new_cols

    ppp = pd.DataFrame(rows)
    # window = mean across 3 seasons ending Window_End
    out_rows = []
    for end, wdf in df.groupby("Window_End"):
        seasons = [end - 2, end - 1, end]
        sub = ppp[ppp["season"].isin(seasons)]
        if sub.empty:
            continue
        agg = sub.groupby("PLAYER_ID").mean(numeric_only=True).reset_index()
        agg["Window_End"] = end
        out_rows.append(agg)

    if not out_rows:
        return df, new_cols

    merged = pd.concat(out_rows, ignore_index=True)
    ppp_cols = [c for c in merged.columns if c.startswith("new_pt_")]
    df = df.merge(merged[["PLAYER_ID", "Window_End"] + ppp_cols], on=["PLAYER_ID", "Window_End"], how="left")

    # volume-weighted blend: primary handler vs spotup spread
    if "new_pt_ppp_isolation" in df.columns and "new_pt_ppp_spotup" in df.columns:
        df["new_pt_iso_minus_spot"] = df["new_pt_ppp_isolation"] - df["new_pt_ppp_spotup"]
        ppp_cols.append("new_pt_iso_minus_spot")

    new_cols = [c for c in ppp_cols if c in df.columns]
    return df, new_cols
