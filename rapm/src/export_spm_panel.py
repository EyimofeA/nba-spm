#!/usr/bin/env python3
"""Build single-season SPM panel for the viewer.

Architecture (window span configurable; production = 3yr):
  - Target y: multi-year RAPM Off/Def from production panel (Window_End = W).
  - Features X: single-season box/tracking for season s in [W-span+1 .. W].
  - Sample weight: single-season possessions × exp decay to window end (hl250).

Viewer shows one row per player-season: predict from that season's stats via
walk-forward ridge (train on windows with Window_End < Y).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from build_spm_features import TIER1, TIER2, aggregate_window, load_season, to_rates
from paths import DATA, RAPM_ALL_WINDOWS_CSV, ensure_dirs
from spm_v2 import feature_matrix

ensure_dirs()

ALPHA = 1000.0
WINDOW_SPAN = 3
DECAY_HL = 250.0
MIN_TRAIN_WINDOWS = 5
EXCLUDE = {"PLAYER_ID", "Window_End", "Season", "MIN", "GP", "OnOffRtg", "OnDefRtg"}


def _load_run_config() -> dict:
    meta_path = RAPM_ALL_WINDOWS_CSV.parent / "run_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"window": WINDOW_SPAN, "decay_hl": DECAY_HL}


def _season_end_ts(season: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(season), month=6, day=30)


def decay_poss_weight(season: int, window_end: int, poss: float, half_life: float) -> float:
    """Single-season poss weighted by recency to window end (matches RAPM hl decay)."""
    age_days = max((_season_end_ts(window_end) - _season_end_ts(season)).days, 0)
    return float(poss) * (0.5 ** (age_days / half_life))


def build_season_features(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        frame = load_season(s)
        if frame is None:
            continue
        f1 = aggregate_window([frame], {**TIER1, **TIER2})
        f1 = to_rates(f1)
        f1["Season"] = int(s)
        poss = f1.get("OffPoss", 0).fillna(0) + f1.get("DefPoss", 0).fillna(0)
        f1["Poss_Season"] = poss
        rows.append(f1)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_training_frame(
    panel: pd.DataFrame,
    season_feats: pd.DataFrame,
    *,
    window_span: int,
    half_life: float,
) -> pd.DataFrame:
    """Expand to (player, season s) rows with 3yr RAPM label at window_end W."""
    feat_cols = [c for c in season_feats.columns if c not in ("PLAYER_ID", "Season", "Poss_Season")]
    rows: list[dict] = []

    for window_end, pw in panel.groupby("Window_End"):
        window_end = int(window_end)
        window_seasons = list(range(window_end - window_span + 1, window_end + 1))
        for season in window_seasons:
            sf = season_feats[season_feats["Season"] == season]
            if sf.empty:
                continue
            merged = pw.merge(sf, on="PLAYER_ID", how="inner")
            if merged.empty:
                continue
            for r in merged.itertuples(index=False):
                poss = float(getattr(r, "Poss_Season", 0) or 0)
                w = decay_poss_weight(season, window_end, max(poss, 1.0), half_life)
                row = {
                    "PLAYER_ID": int(r.PLAYER_ID),
                    "Name": str(r.Name),
                    "Season": int(season),
                    "Window_End": window_end,
                    "Off": float(r.Off),
                    "Def": float(r.Def),
                    "RAPM": float(r.RAPM),
                    "Poss_Season": int(poss),
                    "w": w,
                }
                for c in feat_cols:
                    if hasattr(r, c):
                        row[c] = getattr(r, c)
                rows.append(row)

    return pd.DataFrame(rows)


def _prep_matrix(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    sub = df[[c for c in feat_cols if c in df.columns]].copy()
    return feature_matrix(sub)


def _feature_matrix_train_test(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pf_tr, fcols = _prep_matrix(train, feat_cols)
    pf_te, _ = _prep_matrix(test, feat_cols)
    mu = pf_tr[fcols].mean()
    sd = pf_tr[fcols].std().replace(0, 1)
    X_tr = ((pf_tr[fcols] - mu) / sd).fillna(0.0).values
    X_te = ((pf_te.reindex(columns=fcols, fill_value=0.0) - mu) / sd).fillna(0.0).values
    return X_tr, X_te, fcols


def _ridge_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray, X_te: np.ndarray, alpha: float
) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X_tr)), X_tr])
    Ws = np.sqrt(np.clip(w_tr, 1e-9, None))
    A = (Xb * Ws[:, None]).T @ (Xb * Ws[:, None]) + alpha * np.eye(Xb.shape[1])
    b = (Xb * Ws[:, None]).T @ (y_tr * Ws)
    coef = np.linalg.solve(A, b)
    Xtb = np.column_stack([np.ones(len(X_te)), X_te])
    return Xtb @ coef


def build_spm_panel(
    alpha: float = ALPHA,
    *,
    window_span: int | None = None,
    half_life: float | None = None,
) -> pd.DataFrame:
    cfg = _load_run_config()
    window_span = int(window_span or cfg.get("window", WINDOW_SPAN))
    half_life = float(half_life or cfg.get("decay_hl", DECAY_HL))

    panel = pd.read_csv(RAPM_ALL_WINDOWS_CSV)
    panel["PLAYER_ID"] = panel["PLAYER_ID"].astype(int)
    panel["Window_End"] = panel["Window_End"].astype(int)

    seasons = list(range(int(panel["Window_End"].min()) - window_span + 1, int(panel["Window_End"].max()) + 1))
    season_feats = build_season_features(seasons)
    train_frame = build_training_frame(
        panel, season_feats, window_span=window_span, half_life=half_life
    )
    if train_frame.empty:
        raise RuntimeError("SPM training frame empty — check panel + playersheets")

    predict_years = sorted(season_feats["Season"].unique())
    feat_cols = [c for c in season_feats.columns if c not in ("PLAYER_ID", "Season", "Poss_Season")]
    out_rows: list[dict] = []

    for year in predict_years:
        usable_windows = [w for w in sorted(train_frame["Window_End"].unique()) if w < year]
        if len(usable_windows) < MIN_TRAIN_WINDOWS:
            continue

        tr = train_frame[train_frame["Window_End"].isin(usable_windows)].copy()
        te = season_feats[season_feats["Season"] == year].merge(
            panel[panel["Window_End"] == year][["PLAYER_ID", "Name"]],
            on="PLAYER_ID",
            how="inner",
        )
        if tr.empty or te.empty:
            continue

        X_tr, X_te, fcols = _feature_matrix_train_test(tr, te, feat_cols)
        w_tr = tr["w"].clip(lower=1e-9).values
        y_off_tr = pd.to_numeric(tr["Off"], errors="coerce").values
        y_def_tr = pd.to_numeric(tr["Def"], errors="coerce").values
        spm_off = _ridge_predict(X_tr, y_off_tr, w_tr, X_te, alpha)
        spm_def = _ridge_predict(X_tr, y_def_tr, w_tr, X_te, alpha)
        spm_net = spm_off - spm_def

        # Tier vs 3yr RAPM at window ending this season (if in panel)
        panel_y = panel[panel["Window_End"] == year][["PLAYER_ID", "Off", "Def"]].rename(
            columns={"Off": "Off_lbl", "Def": "Def_lbl"}
        )
        panel_y["PLAYER_ID"] = panel_y["PLAYER_ID"].astype(int)
        te_eval = te.merge(panel_y, on="PLAYER_ID", how="left")
        resid = np.full(len(te), np.nan)
        if te_eval["Off_lbl"].notna().any():
            resid = (spm_off - te_eval["Off_lbl"].fillna(0).values) ** 2 + (
                spm_def - te_eval["Def_lbl"].fillna(0).values
            ) ** 2
        try:
            tiers = pd.qcut(pd.Series(resid).rank(method="first"), q=3, labels=["A", "B", "C"])
        except ValueError:
            tiers = pd.Series(["B"] * len(te))

        for i, r in enumerate(te.itertuples(index=False)):
            poss = int(getattr(r, "Poss_Season", 0) or 0)
            out_rows.append(
                {
                    "Name": r.Name,
                    "PLAYER_ID": int(r.PLAYER_ID),
                    "Window_End": int(year),
                    "Season": str(year),
                    "Season_Type": "regular",
                    "Off": round(float(spm_off[i]), 3),
                    "Def": round(float(spm_def[i]), 3),
                    "RAPM": round(float(spm_net[i]), 3),
                    "Poss_Off": poss // 2,
                    "Poss_Def": poss - poss // 2,
                    "Prior_Mode": "spm",
                    "Tier": str(tiers.iloc[i]),
                }
            )

    return pd.DataFrame(out_rows)


def main() -> None:
    cfg = _load_run_config()
    ws = int(cfg.get("window", WINDOW_SPAN))
    hl = float(cfg.get("decay_hl", DECAY_HL))
    df = build_spm_panel(window_span=ws, half_life=hl)
    out = DATA / "spm_panel_windows.csv"
    df.to_csv(out, index=False)
    print(
        f"SPM_PANEL_DONE rows={len(df)} seasons={df.Window_End.nunique()} "
        f"target={ws}yr hl={hl} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
