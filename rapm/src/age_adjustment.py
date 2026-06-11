"""age_adjustment.py — aging curve + age-adjusted RAPM + peak per player.

Delta method:
  1. Pair each (player, season) with that same player's next season.
  2. Compute ΔOff, ΔDef, ΔRAPM weighted by min(Poss_t, Poss_{t+1}).
  3. LOWESS-smooth per integer age → league-wide aging curves c(a).
  4. Integrate to get cumulative aging offset A(a) (anchored at A(19) = 0).

Age adjustment — "project to reference age" formulation.

    β_adj_at_ref(player, season) = β_raw(player, season) + (A(ref_age) - A(a_t))

This is subtly different from the "subtract cumulative offset" version:
  - A peak-age player (a_t = ref_age) gets NO penalty — they already are at
    the reference point.
  - A 22-year-old gets a small upward shift (their projected age-27 value).
  - A 38-year-old gets a large upward shift (their projected age-27 value).

This yields "what would this player-season look like if observed at age 27?"
Peak is then the MAX of these projections across a player's career — i.e.,
the best single season translated to peak-age.

Outputs:
  aging_curve.csv           — smoothed Δ + cumulative offset per age
  age_adjusted_rapm.csv     — every player-season with age-adj columns
  peak_rapm.csv             — one row per player, peak age-adjusted value
  peak_raw_rapm.csv         — one row per player, peak RAW RAPM season
  career_summary.csv        — one row per player, poss-weighted career mean
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from paths import (
    AGE_ADJUSTED_CSV,
    AGING_CURVE_CSV,
    CAREER_RAPM_CSV,
    CAREER_SUMMARY_CSV,
    PEAK_RAPM_CSV,
    PEAK_RAW_RAPM_CSV,
    ensure_dirs,
)


# =============================================================================
# Aging curve (delta method)
# =============================================================================
def compute_deltas(career: pd.DataFrame, min_poss: int = 200) -> pd.DataFrame:
    """Return per-pair delta records: one row per (player, season → season+1)."""
    df = career.dropna(subset=["Age"]).copy()
    df["Age"] = df["Age"].astype(int)
    df = df.sort_values(["PLAYER_ID", "Season"])

    # Only pairs where (Poss_Off + Poss_Def) ≥ min_poss in BOTH seasons
    df["Poss"] = df["Poss_Off"].fillna(0) + df["Poss_Def"].fillna(0)

    deltas = []
    for pid, g in df.groupby("PLAYER_ID"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if b["Season"] != a["Season"] + 1:
                continue                                       # skip non-consecutive
            if a["Poss"] < min_poss or b["Poss"] < min_poss:
                continue
            deltas.append({
                "PLAYER_ID": pid,
                "season_from": int(a["Season"]),
                "age_from": int(a["Age"]),
                "age_to": int(b["Age"]),
                "age_mid": (a["Age"] + b["Age"]) / 2.0,
                "dOff": b["Off"] - a["Off"],
                "dDef": b["Def"] - a["Def"],
                "dRAPM": b["RAPM"] - a["RAPM"],
                "weight": float(min(a["Poss"], b["Poss"])),
            })
    return pd.DataFrame(deltas)


def _lowess_smooth(x: np.ndarray, y: np.ndarray, w: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Weighted local linear smoother — Gaussian kernel over `x` at each grid point.

    Light-weight; avoids a hard statsmodels dependency.
    """
    n = len(x)
    if n == 0:
        return np.array([])
    bandwidth = max(frac * (x.max() - x.min()), 1.0)
    yhat = np.zeros(n)
    for i, xi in enumerate(x):
        dist = (x - xi) / bandwidth
        kw = np.exp(-0.5 * dist ** 2) * w
        if kw.sum() <= 0:
            yhat[i] = np.average(y, weights=w) if w.sum() > 0 else 0.0
            continue
        X = np.column_stack([np.ones(n), x - xi])
        W = np.diag(kw)
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
            yhat[i] = beta[0]
        except np.linalg.LinAlgError:
            yhat[i] = np.average(y, weights=kw)
    return yhat


def build_aging_curve(deltas: pd.DataFrame, age_min: int = 19, age_max: int = 40) -> pd.DataFrame:
    """Aggregate deltas by integer age, smooth, and integrate → cumulative aging offset.

    Returns a dataframe with:
      age, dOff, dDef, dRAPM   (smoothed per-age average deltas)
      cumOff, cumDef, cumRAPM  (cumulative integrated starting at age_min = 0)
      n_pairs                  (raw sample count per age bin)
    """
    d = deltas[(deltas["age_from"] >= age_min) & (deltas["age_to"] <= age_max)].copy()

    agg = (
        d.groupby("age_from")
         .apply(lambda g: pd.Series({
             "dOff_raw": np.average(g["dOff"], weights=g["weight"]),
             "dDef_raw": np.average(g["dDef"], weights=g["weight"]),
             "dRAPM_raw": np.average(g["dRAPM"], weights=g["weight"]),
             "weight_sum": g["weight"].sum(),
             "n_pairs": len(g),
         }), include_groups=False)
         .reset_index()
         .rename(columns={"age_from": "age"})
    )

    ages = np.arange(age_min, age_max + 1)
    agg = agg.set_index("age").reindex(ages).reset_index()
    agg["dOff_raw"] = agg["dOff_raw"].fillna(0.0)
    agg["dDef_raw"] = agg["dDef_raw"].fillna(0.0)
    agg["dRAPM_raw"] = agg["dRAPM_raw"].fillna(0.0)
    agg["weight_sum"] = agg["weight_sum"].fillna(0.0)
    agg["n_pairs"] = agg["n_pairs"].fillna(0).astype(int)

    x = agg["age"].to_numpy(dtype=float)
    w = agg["weight_sum"].to_numpy()
    w = np.where(w > 0, w, w[w > 0].min() / 10.0 if (w > 0).any() else 1.0)

    agg["dOff"] = _lowess_smooth(x, agg["dOff_raw"].to_numpy(), w, frac=0.4)
    agg["dDef"] = _lowess_smooth(x, agg["dDef_raw"].to_numpy(), w, frac=0.4)
    agg["dRAPM"] = _lowess_smooth(x, agg["dRAPM_raw"].to_numpy(), w, frac=0.4)

    # Cumulative integral A(age) = Σ_{a <= age} Δ(a). A(age_min) := 0.
    agg["cumOff"] = np.concatenate([[0.0], np.cumsum(agg["dOff"].to_numpy()[:-1])])
    agg["cumDef"] = np.concatenate([[0.0], np.cumsum(agg["dDef"].to_numpy()[:-1])])
    agg["cumRAPM"] = np.concatenate([[0.0], np.cumsum(agg["dRAPM"].to_numpy()[:-1])])

    return agg[["age", "n_pairs", "weight_sum",
                "dOff_raw", "dDef_raw", "dRAPM_raw",
                "dOff", "dDef", "dRAPM",
                "cumOff", "cumDef", "cumRAPM"]]


# =============================================================================
# Age-adjusted RAPM + peak per player
# =============================================================================
def apply_age_adjustment(career: pd.DataFrame, curve: pd.DataFrame, ref_age: int = 27) -> pd.DataFrame:
    """Project every player-season to a common reference age (default 27).

    Formula per column:
        projected_at_ref = raw + (A(ref_age) - A(a_t))

    Interpretation: "what would this season look like if played at ref_age?"
    A player at ref_age is unchanged. A younger player gains the expected
    improvement they'd accrue aging to ref_age. An older player gains back the
    decline they've already suffered since ref_age.

    Unlike the prior "subtract cumulative aging" formulation, this does NOT
    penalize peak-age seasons — their ref_age projection IS the observed value.
    """
    ages = curve["age"].to_numpy(dtype=int)
    cum_off = dict(zip(ages, curve["cumOff"]))
    cum_def = dict(zip(ages, curve["cumDef"]))
    cum_rapm = dict(zip(ages, curve["cumRAPM"]))

    min_age, max_age = int(ages.min()), int(ages.max())
    ref_age = int(np.clip(ref_age, min_age, max_age))

    def _cum(cum_map: dict, age: int) -> float:
        a = int(np.clip(age, min_age, max_age))
        return cum_map[a]

    ref_cum_off = _cum(cum_off, ref_age)
    ref_cum_def = _cum(cum_def, ref_age)
    ref_cum_rapm = _cum(cum_rapm, ref_age)

    df = career.dropna(subset=["Age"]).copy()
    df["Age"] = df["Age"].astype(int)
    df["cumOff_at_age"] = df["Age"].map(lambda a: _cum(cum_off, a))
    df["cumDef_at_age"] = df["Age"].map(lambda a: _cum(cum_def, a))
    df["cumRAPM_at_age"] = df["Age"].map(lambda a: _cum(cum_rapm, a))

    df["Ref_Age"] = ref_age
    df["Off_at_ref"] = df["Off"] + (ref_cum_off - df["cumOff_at_age"])
    df["Def_at_ref"] = df["Def"] + (ref_cum_def - df["cumDef_at_age"])
    df["RAPM_at_ref"] = df["RAPM"] + (ref_cum_rapm - df["cumRAPM_at_age"])

    # Keep the legacy "above aging curve" view too — useful for analysts
    # asking "who beat the curve this season?"
    df["Off_above_curve"] = df["Off"] - df["cumOff_at_age"]
    df["Def_above_curve"] = df["Def"] - df["cumDef_at_age"]
    df["RAPM_above_curve"] = df["RAPM"] - df["cumRAPM_at_age"]

    return df


def compute_peak_raw(career: pd.DataFrame, min_poss: int = 1500) -> pd.DataFrame:
    """One row per player — their best single-season raw RAPM (no aging adjustment).

    This avoids the late-career-survivor bias of the age-adjusted peak and gives
    a cleaner "peak true-talent season" view.
    """
    d = career.copy()
    d["Poss"] = d["Poss_Off"].fillna(0) + d["Poss_Def"].fillna(0)
    d = d[d["Poss"] >= min_poss]
    if d.empty:
        return pd.DataFrame(columns=[
            "PLAYER_ID", "Name", "Peak_Raw_RAPM", "Peak_Season", "Peak_Age",
            "Peak_Off", "Peak_Def", "Peak_Poss", "Seasons_Played"])
    idx = d.groupby("PLAYER_ID")["RAPM"].idxmax()
    peak = d.loc[idx].copy()
    counts = career.groupby("PLAYER_ID")["Season"].nunique().rename("Seasons_Played")
    peak = peak.merge(counts, on="PLAYER_ID", how="left")
    return peak.rename(columns={
        "RAPM": "Peak_Raw_RAPM",
        "Season": "Peak_Season",
        "Age": "Peak_Age",
        "Off": "Peak_Off",
        "Def": "Peak_Def",
        "Poss": "Peak_Poss",
    })[["PLAYER_ID", "Name", "Peak_Raw_RAPM", "Peak_Season", "Peak_Age",
        "Peak_Off", "Peak_Def", "Peak_Poss", "Seasons_Played"]
        ].sort_values("Peak_Raw_RAPM", ascending=False)


def compute_career_summary(career: pd.DataFrame, min_seasons: int = 5, min_poss_per_season: int = 1500) -> pd.DataFrame:
    """Per-player possession-weighted career mean RAPM across qualifying seasons.

    This is the cleanest "best career" measure — the sabermetric equivalent of
    career WAR, but derived from RAPM instead of box-score VORP.
    """
    d = career.copy()
    d["Poss"] = d["Poss_Off"].fillna(0) + d["Poss_Def"].fillna(0)
    d = d[d["Poss"] >= min_poss_per_season]
    if d.empty:
        return pd.DataFrame(columns=[
            "PLAYER_ID", "Name", "N_Seasons", "Total_Poss",
            "Career_Mean_RAPM", "Career_Wmean_RAPM",
            "Career_Wmean_Off", "Career_Wmean_Def"])

    def _summary(g: pd.DataFrame) -> pd.Series:
        w = g["Poss"].to_numpy()
        return pd.Series({
            "N_Seasons": int(g["Season"].nunique()),
            "Total_Poss": float(w.sum()),
            "Career_Mean_RAPM": float(g["RAPM"].mean()),
            "Career_Wmean_RAPM": float((g["RAPM"] * w).sum() / w.sum()),
            "Career_Wmean_Off": float((g["Off"] * w).sum() / w.sum()),
            "Career_Wmean_Def": float((g["Def"] * w).sum() / w.sum()),
        })

    g = d.groupby(["PLAYER_ID", "Name"]).apply(_summary, include_groups=False).reset_index()
    g = g[g["N_Seasons"] >= min_seasons]
    return g.sort_values("Career_Wmean_RAPM", ascending=False)


def compute_peak(age_adj: pd.DataFrame, min_poss: int = 1500) -> pd.DataFrame:
    """One row per player — their best season translated to reference age.

    Uses `RAPM_at_ref` (project-to-ref-age). A peak-age season stays put, and
    seasons on either side get the aging-curve bump. Peak is the single best
    of those translated seasons.
    """
    d = age_adj.copy()
    d["Poss"] = d["Poss_Off"].fillna(0) + d["Poss_Def"].fillna(0)
    d = d[d["Poss"] >= min_poss]
    if d.empty:
        return pd.DataFrame(columns=[
            "PLAYER_ID", "Name", "Peak_RAPM_at_Ref", "Peak_Season", "Peak_Age",
            "Raw_RAPM_at_Peak", "Ref_Age", "Seasons_Played"])
    idx = d.groupby("PLAYER_ID")["RAPM_at_ref"].idxmax()
    peak = d.loc[idx].copy()
    counts = age_adj.groupby("PLAYER_ID")["Season"].nunique().rename("Seasons_Played")
    peak = peak.merge(counts, on="PLAYER_ID", how="left")
    out = peak.rename(columns={
        "RAPM_at_ref": "Peak_RAPM_at_Ref",
        "Season": "Peak_Season",
        "Age": "Peak_Age",
        "RAPM": "Raw_RAPM_at_Peak",
    })[["PLAYER_ID", "Name", "Peak_RAPM_at_Ref", "Peak_Season", "Peak_Age",
        "Raw_RAPM_at_Peak", "Off_at_ref", "Def_at_ref", "Ref_Age", "Seasons_Played"]]
    return out.sort_values("Peak_RAPM_at_Ref", ascending=False)


# =============================================================================
# Main
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Build aging curve + age-adjusted RAPM + peaks.")
    ap.add_argument("--career-csv", default=str(CAREER_RAPM_CSV),
                    help="Long career table from rapm_career.py.")
    ap.add_argument("--age-min", type=int, default=19)
    ap.add_argument("--age-max", type=int, default=40)
    ap.add_argument("--delta-min-poss", type=int, default=500,
                    help="Minimum total possessions in BOTH seasons of a delta pair.")
    ap.add_argument("--peak-min-poss", type=int, default=1500,
                    help="Minimum total possessions for a season to qualify as a peak candidate.")
    ap.add_argument("--top-k", type=int, default=25,
                    help="How many players to print in each leaderboard.")
    ap.add_argument("--min-seasons", type=int, default=5,
                    help="Minimum qualifying seasons for the career-summary leaderboard.")
    ap.add_argument("--ref-age", type=int, default=27,
                    help="Reference age all seasons get projected to (default peak age ≈ 27).")
    args = ap.parse_args(argv)

    print(f"Loading career table: {args.career_csv}")
    career = pd.read_csv(args.career_csv)
    print(f"  {len(career):,} rows · {career['PLAYER_ID'].nunique():,} unique players "
          f"· seasons {career['Season'].min()}–{career['Season'].max()}")

    print("\nComputing per-pair deltas (player-season → next season)...")
    deltas = compute_deltas(career, min_poss=args.delta_min_poss)
    print(f"  {len(deltas):,} delta pairs")

    print("\nBuilding smoothed league-wide aging curve...")
    curve = build_aging_curve(deltas, age_min=args.age_min, age_max=args.age_max)
    curve.to_csv(AGING_CURVE_CSV, index=False)
    print(f"  aging curve → {AGING_CURVE_CSV}")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(curve[["age", "n_pairs", "dRAPM_raw", "dRAPM", "cumRAPM"]].to_string(index=False))

    print(f"\nApplying age adjustment (project every season to age {args.ref_age})...")
    age_adj = apply_age_adjustment(career, curve, ref_age=args.ref_age)
    age_adj.to_csv(AGE_ADJUSTED_CSV, index=False)
    print(f"  age-adjusted table → {AGE_ADJUSTED_CSV}")

    print(f"\nComputing per-player peak RAPM (projected to age {args.ref_age})...")
    peak = compute_peak(age_adj, min_poss=args.peak_min_poss)
    peak.to_csv(PEAK_RAPM_CSV, index=False)
    print(f"  ref-age peak → {PEAK_RAPM_CSV}")

    print("\nComputing per-player peak RAPM (raw, no aging adjustment)...")
    peak_raw = compute_peak_raw(career, min_poss=args.peak_min_poss)
    peak_raw.to_csv(PEAK_RAW_RAPM_CSV, index=False)
    print(f"  raw peak → {PEAK_RAW_RAPM_CSV}")

    print("\nComputing career summary (possession-weighted mean RAPM)...")
    summary = compute_career_summary(career, min_seasons=args.min_seasons,
                                      min_poss_per_season=args.peak_min_poss)
    summary.to_csv(CAREER_SUMMARY_CSV, index=False)
    print(f"  career summary → {CAREER_SUMMARY_CSV}")

    print(f"\nTop {args.top_k} — CAREER WEIGHTED-MEAN RAPM (≥{args.min_seasons} qualifying seasons):")
    print(summary[["Name", "N_Seasons", "Total_Poss",
                   "Career_Wmean_RAPM", "Career_Wmean_Off", "Career_Wmean_Def"]]
          .head(args.top_k).to_string(index=False))

    print(f"\nTop {args.top_k} — PEAK RAW RAPM (single best season):")
    print(peak_raw[["Name", "Peak_Season", "Peak_Age", "Peak_Raw_RAPM",
                    "Peak_Off", "Peak_Def", "Peak_Poss", "Seasons_Played"]]
          .head(args.top_k).to_string(index=False))

    print(f"\nTop {args.top_k} — PEAK RAPM (projected to age {args.ref_age}):")
    cols = ["Name", "Peak_RAPM_at_Ref", "Peak_Season", "Peak_Age",
            "Raw_RAPM_at_Peak", "Off_at_ref", "Def_at_ref", "Seasons_Played"]
    print(peak[cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
