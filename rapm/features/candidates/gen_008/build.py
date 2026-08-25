"""Gen 008 — NEW tracking-era interaction features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _d(a, b, eps=1e-6):
    return a / (b + eps)


def build(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = feats.copy()
    new_cols: list[str] = []

    def add(name: str, s: pd.Series) -> None:
        df[name] = s.astype(float)
        new_cols.append(name)

    pairs = [
        ({"DRIVES_p100", "TOUCHES_p100"}, "new_drive_touch_ratio", lambda d: _d(d["DRIVES_p100"], d["TOUCHES_p100"])),
        ({"PASSES_MADE_p100", "POTENTIAL_AST_p100"}, "new_pass_ast_ratio", lambda d: _d(d["PASSES_MADE_p100"], d["POTENTIAL_AST_p100"])),
        ({"DRIVE_PTS_p100", "DRIVES_p100"}, "new_drive_ppp", lambda d: _d(d["DRIVE_PTS_p100"], d["DRIVES_p100"].clip(lower=0.5))),
        ({"CATCH_SHOOT_FGA_p100", "PULL_UP_FGA_p100"}, "new_cns_pull_ratio", lambda d: _d(d["CATCH_SHOOT_FGA_p100"], d["PULL_UP_FGA_p100"])),
        ({"wide_open_FG3M_p100", "wide_open_FG3A_p100"}, "new_wide_open_acc", lambda d: _d(d["wide_open_FG3M_p100"], d["wide_open_FG3A_p100"].clip(lower=0.5))),
        ({"REB_CONTEST_p100", "REB_CHANCES_p100"}, "new_reb_contest_rate", lambda d: _d(d["REB_CONTEST_p100"], d["REB_CHANCES_p100"])),
        ({"PAINT_TOUCHES_p100", "TOUCHES_p100"}, "new_paint_touch_share", lambda d: _d(d["PAINT_TOUCHES_p100"], d["TOUCHES_p100"])),
        ({"AST_PTS_CREATED_p100", "PASSES_MADE_p100"}, "new_ast_pts_per_pass", lambda d: _d(d["AST_PTS_CREATED_p100"], d["PASSES_MADE_p100"].clip(lower=0.5))),
    ]
    for need, name, fn in pairs:
        if need.issubset(df.columns):
            add(name, fn(df))

    if {"AVG_SEC_PER_TOUCH", "AVG_DRIB_PER_TOUCH"}.issubset(df.columns):
        add("new_pace_handle", df["AVG_SEC_PER_TOUCH"] * df["AVG_DRIB_PER_TOUCH"])

    if {"ShotQualityAvg", "TS_PCT"}.issubset(df.columns):
        add("new_sq_ts_resid", df["ShotQualityAvg"] - df["TS_PCT"])

    return df, new_cols
