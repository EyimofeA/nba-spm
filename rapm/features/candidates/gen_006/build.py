"""Gen 006 — NEW derived box features (not in base parquet).

Ratios, interactions, window z-scores. Autoresearch should add columns like these,
not re-filter existing USG_PCT / PTS_p100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def build(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = feats.copy()
    new_cols: list[str] = []

    def add(name: str, s: pd.Series) -> None:
        df[name] = s.astype(float)
        new_cols.append(name)

    # --- ratios (new information vs raw rates alone) ---
    if {"AST_p100", "TOV_p100"}.issubset(df.columns):
        add("new_ast_tov_ratio", _safe_div(df["AST_p100"], df["TOV_p100"].clip(lower=0.5)))
    if {"STL_p100", "BLK_p100"}.issubset(df.columns):
        add("new_def_event_rate", df["STL_p100"] + df["BLK_p100"])
    if {"FTA_p100", "FG2A_p100", "FG3A_p100"}.issubset(df.columns):
        fga = df["FG2A_p100"].fillna(0) + df["FG3A_p100"].fillna(0)
        add("new_fta_rate", _safe_div(df["FTA_p100"], fga))
    if {"AtRimFrequency", "Arc3Frequency", "Corner3Frequency"}.issubset(df.columns):
        threes = df["Arc3Frequency"].fillna(0) + df["Corner3Frequency"].fillna(0)
        add("new_rim_vs_three", _safe_div(df["AtRimFrequency"], threes))
    if {"Assisted2sPct", "USG_PCT"}.issubset(df.columns):
        add("new_self_create_usg", (1.0 - df["Assisted2sPct"].fillna(0.5)) * df["USG_PCT"])
    if {"TS_PCT", "USG_PCT"}.issubset(df.columns):
        add("new_efficiency_usage", df["TS_PCT"] * df["USG_PCT"])
    if {"LiveBallTurnoverPct", "USG_PCT"}.issubset(df.columns):
        add("new_live_tov_usg", df["LiveBallTurnoverPct"] * df["USG_PCT"])

    # --- window-relative z-scores (context-adjusted, new cols) ---
    for col, out in (
        ("USG_PCT", "new_usg_z"),
        ("TS_PCT", "new_ts_z"),
        ("AtRimFrequency", "new_rim_freq_z"),
    ):
        if col in df.columns and "Window_End" in df.columns:
            g = df.groupby("Window_End")[col]
            add(out, g.transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-9 else 1.0)))

    # --- age curve interaction ---
    if {"AGE", "TS_PCT"}.issubset(df.columns):
        add("new_age_ts", (df["AGE"] - 27.0) * df["TS_PCT"])

    return df, new_cols
