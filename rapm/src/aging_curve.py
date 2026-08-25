#!/usr/bin/env python3
"""A1+A2: survivorship-safe aging curve from the 1-year RAPM panel.

A1 — delta method: within-player year-over-year rating changes, possession-
weighted, averaged per age, integrated into a curve. Survivorship sensitivity:
players who exit the league contribute an imputed exit delta (0 / -0.5 / -1.0).

A2 — variants: (a) additive (the A1 curve), (b) level-dependent (delta
regressed on age dummies + prior level: does decline scale with ability?),
(c) separate Off/Def curves.

Gate — player-level retrodiction: predict each player's 2024 1-yr RAPM from
his 2023 rating + aging translation; compare MAE/corr vs no translation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paths import AGING_DIR, CAREER_RAPM_CSV, ensure_dirs

ensure_dirs()

MIN_POSS = 1500          # per season, offense side, to enter the delta panel
AGE_RANGE = (19, 40)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(CAREER_RAPM_CSV)
    df = df.dropna(subset=["Age", "RAPM"])
    df["Age"] = df["Age"].round().astype(int)
    df["poss"] = df[["Poss_Off", "Poss_Def"]].min(axis=1)
    return df[(df["Age"] >= AGE_RANGE[0]) & (df["Age"] <= AGE_RANGE[1])]


def build_deltas(df: pd.DataFrame, exit_delta: float | None) -> pd.DataFrame:
    """Consecutive-season within-player deltas. If exit_delta is not None,
    players with a season but no next season contribute an imputed delta."""
    cur = df[df["poss"] >= MIN_POSS][["PLAYER_ID", "Season", "Age", "RAPM", "Off", "Def", "poss"]]
    nxt = cur.rename(columns={c: c + "_next" for c in ["Season", "Age", "RAPM", "Off", "Def", "poss"]})
    nxt["Season"] = nxt["Season_next"] - 1
    m = cur.merge(nxt, on=["PLAYER_ID", "Season"], how="left")
    observed = m[m["RAPM_next"].notna() & (m["Age_next"] == m["Age"] + 1)].copy()
    observed["d_rapm"] = observed["RAPM_next"] - observed["RAPM"]
    observed["d_off"] = observed["Off_next"] - observed["Off"]
    observed["d_def"] = observed["Def_next"] - observed["Def"]
    observed["w"] = 2.0 / (1.0 / observed["poss"] + 1.0 / observed["poss_next"])  # harmonic mean
    if exit_delta is None:
        return observed
    exits = m[m["RAPM_next"].isna()].copy()
    exits["d_rapm"] = exit_delta
    exits["d_off"] = exit_delta / 2
    exits["d_def"] = -exit_delta / 2
    exits["w"] = exits["poss"]
    return pd.concat([observed, exits], ignore_index=True)


def curve_from_deltas(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby("Age").apply(
        lambda x: pd.Series({
            "delta": np.average(x[col], weights=x["w"]),
            "n": len(x),
            "w_sum": x["w"].sum(),
        }), include_groups=False,
    ).reset_index()
    # integrate deltas: f(age+1) = f(age) + delta(age); anchor youngest age at 0
    g = g.sort_values("Age")
    f = np.concatenate([[0.0], np.cumsum(g["delta"].values)])
    ages = np.concatenate([[g["Age"].iloc[0]], g["Age"].values + 1])
    curve = pd.DataFrame({"Age": ages, "f_raw": f})
    curve["f"] = curve["f_raw"] - curve["f_raw"].max()          # peak = 0
    # light smoothing: 3-point weighted moving average
    curve["f_smooth"] = curve["f"].rolling(3, center=True, min_periods=1).mean()
    return curve.merge(g[["Age", "n", "w_sum"]], on="Age", how="left")


def level_dependence(d: pd.DataFrame) -> dict:
    """Regress delta on age dummies + prior level. b<0 => higher-rated players
    decline faster (mean reversion + proportional aging mixed together)."""
    X = pd.get_dummies(d["Age"], prefix="age", drop_first=True).astype(float)
    X["level"] = d["RAPM"].values
    y = d["d_rapm"].values
    w = d["w"].values
    Xm = np.column_stack([np.ones(len(X)), X.values])
    Ws = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xm * Ws[:, None], y * Ws, rcond=None)
    b_level = float(coef[-1])
    return {"b_level": round(b_level, 4),
            "interpretation": "delta_next = f(age) + b*current_level; b<0 = mean reversion/proportional decline"}


def player_gate(df: pd.DataFrame, curve: pd.DataFrame) -> dict:
    """Predict 2024 RAPM from 2023 RAPM (+/- aging translation)."""
    f = dict(zip(curve["Age"], curve["f_smooth"]))
    a = df[(df["Season"] == 2023) & (df["poss"] >= MIN_POSS)][["PLAYER_ID", "Age", "RAPM"]]
    b = df[(df["Season"] == 2024) & (df["poss"] >= MIN_POSS)][["PLAYER_ID", "RAPM"]]
    m = a.merge(b, on="PLAYER_ID", suffixes=("_23", "_24"))
    m["translate"] = m["Age"].map(lambda x: f.get(x + 1, np.nan)) - m["Age"].map(lambda x: f.get(x, np.nan))
    m = m.dropna(subset=["translate"])
    naive_mae = float(np.mean(np.abs(m["RAPM_24"] - m["RAPM_23"])))
    aged_pred = m["RAPM_23"] + m["translate"]
    aged_mae = float(np.mean(np.abs(m["RAPM_24"] - aged_pred)))
    return {"n_players": len(m),
            "mae_no_translation": round(naive_mae, 4),
            "mae_aging_translated": round(aged_mae, 4),
            "corr_no_translation": round(float(np.corrcoef(m['RAPM_23'], m['RAPM_24'])[0, 1]), 4),
            "corr_aging_translated": round(float(np.corrcoef(aged_pred, m['RAPM_24'])[0, 1]), 4)}


def main() -> None:
    df = load_panel()
    print(f"panel: {len(df):,} player-seasons, {df['PLAYER_ID'].nunique():,} players", flush=True)

    results = {}
    curves = {}
    for name, exit_d in (("no_exit", None), ("exit_-0.5", -0.5), ("exit_-1.0", -1.0)):
        d = build_deltas(df, exit_d)
        curves[name] = curve_from_deltas(d, "d_rapm")
        results[f"gate_{name}"] = player_gate(df, curves[name])

    d_obs = build_deltas(df, None)
    off_curve = curve_from_deltas(d_obs, "d_off")
    def_curve = curve_from_deltas(d_obs, "d_def")
    results["level_dependence"] = level_dependence(d_obs)

    out = curves["no_exit"].rename(columns={"f_smooth": "f_total"})[["Age", "f_total", "n"]]
    out = out.merge(curves["exit_-0.5"][["Age", "f_smooth"]].rename(columns={"f_smooth": "f_total_exit05"}), on="Age")
    out = out.merge(curves["exit_-1.0"][["Age", "f_smooth"]].rename(columns={"f_smooth": "f_total_exit10"}), on="Age")
    out = out.merge(off_curve[["Age", "f_smooth"]].rename(columns={"f_smooth": "f_off"}), on="Age")
    out = out.merge(def_curve[["Age", "f_smooth"]].rename(columns={"f_smooth": "f_def"}), on="Age")
    path = AGING_DIR / "aging_curve_delta.csv"
    out.to_csv(path, index=False)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(out["Age"], out["f_total"], lw=2, color="black", label="total (observed deltas)")
    ax.plot(out["Age"], out["f_total_exit10"], lw=1, ls="--", color="gray", label="w/ exit penalty -1.0")
    ax.plot(out["Age"], out["f_off"], lw=1.2, color="tab:red", label="offense")
    ax.plot(out["Age"], out["f_def"], lw=1.2, color="tab:blue", label="defense (lower=better ages worse)")
    ax.set_xlabel("Age")
    ax.set_ylabel("RAPM vs peak (per 100)")
    ax.set_title("NBA aging curve, delta method, 1997-2024 1-yr RAPM panel")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(AGING_DIR / "aging_curve_delta.png", dpi=150)

    (AGING_DIR / "aging_curve_delta_meta.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"AGING_DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
