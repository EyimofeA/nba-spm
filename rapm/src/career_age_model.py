"""Selection-aware career RAPM aging and peak model.

This is a pragmatic first pass built on top of the existing 1-year career RAPM
table. It estimates an aging curve from within-player year-to-year changes and
adds low-weight "exit" deltas for players who disappear before the final season,
so survivor bias does not make late-career aging look too friendly.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from paths import (
    CAREER_AGE_MODEL_CURVE_CSV,
    CAREER_AGE_MODEL_META_JSON,
    CAREER_AGE_MODEL_PEAK_3YR_CSV,
    CAREER_AGE_MODEL_PEAK_CSV,
    CAREER_AGE_MODEL_SEASONS_CSV,
    CAREER_AGE_MODEL_SUMMARY_CSV,
    CAREER_RAPM_CSV,
    ensure_dirs,
)


def load_career(path: str) -> pd.DataFrame:
    career = pd.read_csv(path)
    career = career.dropna(subset=["Age", "PLAYER_ID", "Season"]).copy()
    career["PLAYER_ID"] = career["PLAYER_ID"].astype(int)
    career["Season"] = career["Season"].astype(int)
    career["Age"] = career["Age"].astype(int)
    career["Poss"] = career["Poss_Off"].fillna(0) + career["Poss_Def"].fillna(0)
    return career.sort_values(["PLAYER_ID", "Season"])


def compute_deltas(
    career: pd.DataFrame,
    min_poss: int,
    exit_weight: float,
    exit_target: float,
) -> pd.DataFrame:
    """Build observed adjacent-season deltas plus low-weight exit deltas.

    Exit deltas are not a claim that vanished players became exactly replacement
    level. They are a conservative missing-data correction: non-returning seasons
    carry some evidence of decline instead of being dropped entirely.
    """
    final_season = int(career["Season"].max())
    rows = []
    for player_id, group in career.groupby("PLAYER_ID"):
        group = group.sort_values("Season").reset_index(drop=True)
        for idx, current in group.iterrows():
            if current["Poss"] < min_poss:
                continue

            next_rows = group[group["Season"] == current["Season"] + 1]
            if not next_rows.empty:
                nxt = next_rows.iloc[0]
                if nxt["Poss"] >= min_poss:
                    weight = float(min(current["Poss"], nxt["Poss"]))
                    rows.append({
                        "PLAYER_ID": player_id,
                        "season_from": int(current["Season"]),
                        "age_from": int(current["Age"]),
                        "delta_type": "observed",
                        "dOff": float(nxt["Off"] - current["Off"]),
                        "dDef": float(nxt["Def"] - current["Def"]),
                        "dRAPM": float(nxt["RAPM"] - current["RAPM"]),
                        "weight": weight,
                    })
                continue

            if current["Season"] >= final_season:
                continue
            rows.append({
                "PLAYER_ID": player_id,
                "season_from": int(current["Season"]),
                "age_from": int(current["Age"]),
                "delta_type": "exit",
                "dOff": float(exit_target - current["Off"]),
                "dDef": float(exit_target - current["Def"]),
                "dRAPM": float(exit_target - current["RAPM"]),
                "weight": float(current["Poss"] * exit_weight),
            })
    return pd.DataFrame(rows)


def weighted_smooth(x: np.ndarray, y: np.ndarray, w: np.ndarray, frac: float) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    bandwidth = max(frac * (float(x.max()) - float(x.min())), 1.0)
    yhat = np.zeros(len(x), dtype=float)
    for idx, xi in enumerate(x):
        dist = (x - xi) / bandwidth
        kw = np.exp(-0.5 * dist**2) * w
        if kw.sum() <= 0:
            yhat[idx] = float(np.average(y, weights=w)) if w.sum() > 0 else 0.0
            continue
        design = np.column_stack([np.ones(len(x)), x - xi])
        weighted_design = design * kw[:, None]
        try:
            beta = np.linalg.solve(design.T @ weighted_design, weighted_design.T @ y)
            yhat[idx] = float(beta[0])
        except np.linalg.LinAlgError:
            yhat[idx] = float(np.average(y, weights=kw))
    return yhat


def build_curve(deltas: pd.DataFrame, age_min: int, age_max: int, smooth_frac: float) -> pd.DataFrame:
    d = deltas[(deltas["age_from"] >= age_min) & (deltas["age_from"] <= age_max)].copy()
    if d.empty:
        raise RuntimeError("No aging deltas after filters.")

    def summarize(group: pd.DataFrame) -> pd.Series:
        observed = group["delta_type"].eq("observed")
        exits = group["delta_type"].eq("exit")
        return pd.Series({
            "n_pairs": len(group),
            "n_observed_pairs": int(observed.sum()),
            "n_exit_pairs": int(exits.sum()),
            "weight_sum": float(group["weight"].sum()),
            "exit_weight_sum": float(group.loc[exits, "weight"].sum()),
            "dOff_raw": float(np.average(group["dOff"], weights=group["weight"])),
            "dDef_raw": float(np.average(group["dDef"], weights=group["weight"])),
            "dRAPM_raw": float(np.average(group["dRAPM"], weights=group["weight"])),
        })

    ages = np.arange(age_min, age_max + 1)
    agg = d.groupby("age_from").apply(summarize, include_groups=False).reindex(ages).reset_index()
    agg = agg.rename(columns={"age_from": "age"})
    for col in ["n_pairs", "n_observed_pairs", "n_exit_pairs", "weight_sum", "exit_weight_sum"]:
        agg[col] = agg[col].fillna(0)
    for col in ["dOff_raw", "dDef_raw", "dRAPM_raw"]:
        agg[col] = agg[col].fillna(0.0)

    x = agg["age"].to_numpy(dtype=float)
    w = agg["weight_sum"].to_numpy(dtype=float)
    fallback_weight = w[w > 0].min() / 10.0 if (w > 0).any() else 1.0
    w = np.where(w > 0, w, fallback_weight)
    agg["dOff"] = weighted_smooth(x, agg["dOff_raw"].to_numpy(dtype=float), w, frac=smooth_frac)
    agg["dDef"] = weighted_smooth(x, agg["dDef_raw"].to_numpy(dtype=float), w, frac=smooth_frac)
    agg["dRAPM"] = weighted_smooth(x, agg["dRAPM_raw"].to_numpy(dtype=float), w, frac=smooth_frac)
    agg["cumOff"] = np.concatenate([[0.0], np.cumsum(agg["dOff"].to_numpy(dtype=float)[:-1])])
    agg["cumDef"] = np.concatenate([[0.0], np.cumsum(agg["dDef"].to_numpy(dtype=float)[:-1])])
    agg["cumRAPM"] = np.concatenate([[0.0], np.cumsum(agg["dRAPM"].to_numpy(dtype=float)[:-1])])
    return agg


def apply_age_adjustment(career: pd.DataFrame, curve: pd.DataFrame, ref_age: int) -> pd.DataFrame:
    ages = curve["age"].astype(int).to_numpy()
    min_age, max_age = int(ages.min()), int(ages.max())
    ref_age = int(np.clip(ref_age, min_age, max_age))
    by_age = curve.set_index("age")

    def cum(col: str, age: int) -> float:
        clipped = int(np.clip(age, min_age, max_age))
        return float(by_age.loc[clipped, col])

    ref_off = cum("cumOff", ref_age)
    ref_def = cum("cumDef", ref_age)
    ref_rapm = cum("cumRAPM", ref_age)
    out = career.copy()
    out["Ref_Age"] = ref_age
    out["cumOff_at_age"] = out["Age"].map(lambda age: cum("cumOff", age))
    out["cumDef_at_age"] = out["Age"].map(lambda age: cum("cumDef", age))
    out["cumRAPM_at_age"] = out["Age"].map(lambda age: cum("cumRAPM", age))
    out["Off_at_ref"] = out["Off"] + ref_off - out["cumOff_at_age"]
    out["Def_at_ref"] = out["Def"] + ref_def - out["cumDef_at_age"]
    out["RAPM_at_ref"] = out["RAPM"] + ref_rapm - out["cumRAPM_at_age"]
    return out


def weighted_mean(group: pd.DataFrame, value: str) -> float:
    weights = group["Poss"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(group[value], weights=weights))


def compute_peak(age_adjusted: pd.DataFrame, min_poss: int, peak_age_min: int, peak_age_max: int) -> pd.DataFrame:
    eligible = age_adjusted[
        (age_adjusted["Poss"] >= min_poss)
        & (age_adjusted["Age"] >= peak_age_min)
        & (age_adjusted["Age"] <= peak_age_max)
    ].copy()
    idx = eligible.groupby("PLAYER_ID")["RAPM_at_ref"].idxmax()
    peak = eligible.loc[idx].copy()
    counts = age_adjusted.groupby("PLAYER_ID")["Season"].nunique().rename("Seasons_Played")
    peak = peak.merge(counts, on="PLAYER_ID", how="left")
    return (
        peak.rename(columns={
            "Season": "Peak_Season",
            "Age": "Peak_Age",
            "RAPM": "Raw_RAPM_at_Peak",
            "RAPM_at_ref": "Peak_RAPM_at_Ref",
            "Off_at_ref": "Peak_Off_at_Ref",
            "Def_at_ref": "Peak_Def_at_Ref",
        })[
            [
                "PLAYER_ID", "Name", "Peak_RAPM_at_Ref", "Peak_Season", "Peak_Age",
                "Raw_RAPM_at_Peak", "Peak_Off_at_Ref", "Peak_Def_at_Ref",
                "Poss", "Seasons_Played",
            ]
        ]
        .rename(columns={"Poss": "Peak_Poss"})
        .sort_values("Peak_RAPM_at_Ref", ascending=False)
    )


def compute_peak_window(
    age_adjusted: pd.DataFrame,
    window: int,
    min_total_poss: int,
    peak_age_min: int,
    peak_age_max: int,
) -> pd.DataFrame:
    rows = []
    for player_id, group in age_adjusted.groupby("PLAYER_ID"):
        group = group.sort_values("Season")
        seasons = group["Season"].to_numpy()
        if len(seasons) == 0:
            continue
        for start in range(int(seasons.min()), int(seasons.max()) - window + 2):
            w = group[(group["Season"] >= start) & (group["Season"] < start + window)]
            if w.empty or w["Poss"].sum() < min_total_poss:
                continue
            avg_age = weighted_mean(w, "Age")
            if avg_age < peak_age_min or avg_age > peak_age_max:
                continue
            rows.append({
                "PLAYER_ID": player_id,
                "Name": w["Name"].iloc[0],
                "Start_Season": start,
                "End_Season": start + window - 1,
                "N_Seasons_In_Window": int(w["Season"].nunique()),
                "Poss_Weighted_Age": avg_age,
                "Total_Poss": float(w["Poss"].sum()),
                "Peak_3yr_RAPM_at_Ref": weighted_mean(w, "RAPM_at_ref"),
                "Peak_3yr_Off_at_Ref": weighted_mean(w, "Off_at_ref"),
                "Peak_3yr_Def_at_Ref": weighted_mean(w, "Def_at_ref"),
                "Raw_3yr_RAPM": weighted_mean(w, "RAPM"),
            })
    if not rows:
        return pd.DataFrame()
    windows = pd.DataFrame(rows)
    idx = windows.groupby("PLAYER_ID")["Peak_3yr_RAPM_at_Ref"].idxmax()
    return windows.loc[idx].sort_values("Peak_3yr_RAPM_at_Ref", ascending=False)


def compute_summary(age_adjusted: pd.DataFrame, min_seasons: int, min_poss: int) -> pd.DataFrame:
    eligible = age_adjusted[age_adjusted["Poss"] >= min_poss].copy()
    rows = []
    for (player_id, name), group in eligible.groupby(["PLAYER_ID", "Name"]):
        if group["Season"].nunique() < min_seasons:
            continue
        rows.append({
            "PLAYER_ID": player_id,
            "Name": name,
            "N_Seasons": int(group["Season"].nunique()),
            "First_Season": int(group["Season"].min()),
            "Last_Season": int(group["Season"].max()),
            "First_Age": int(group["Age"].min()),
            "Last_Age": int(group["Age"].max()),
            "Has_Prime_Season": bool(((group["Age"] >= 23) & (group["Age"] <= 34)).any()),
            "Total_Poss": float(group["Poss"].sum()),
            "Career_Wmean_RAPM": weighted_mean(group, "RAPM"),
            "Career_Wmean_Off": weighted_mean(group, "Off"),
            "Career_Wmean_Def": weighted_mean(group, "Def"),
            "Career_Wmean_RAPM_at_Ref": weighted_mean(group, "RAPM_at_ref"),
            "Career_Wmean_Off_at_Ref": weighted_mean(group, "Off_at_ref"),
            "Career_Wmean_Def_at_Ref": weighted_mean(group, "Def_at_ref"),
        })
    return pd.DataFrame(rows).sort_values("Career_Wmean_RAPM", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a selection-aware career RAPM age model.")
    parser.add_argument("--career-csv", default=str(CAREER_RAPM_CSV))
    parser.add_argument("--age-min", type=int, default=19)
    parser.add_argument("--age-max", type=int, default=40)
    parser.add_argument("--ref-age", type=int, default=27)
    parser.add_argument("--delta-min-poss", type=int, default=500)
    parser.add_argument("--peak-min-poss", type=int, default=1500)
    parser.add_argument("--peak-window", type=int, default=3)
    parser.add_argument("--peak-window-min-poss", type=int, default=4500)
    parser.add_argument("--peak-age-min", type=int, default=23)
    parser.add_argument("--peak-age-max", type=int, default=34)
    parser.add_argument("--min-seasons", type=int, default=5)
    parser.add_argument("--exit-weight", type=float, default=0.25)
    parser.add_argument("--exit-target", type=float, default=0.0)
    parser.add_argument("--smooth-frac", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    start = time.time()

    career = load_career(args.career_csv)
    deltas = compute_deltas(
        career,
        min_poss=args.delta_min_poss,
        exit_weight=args.exit_weight,
        exit_target=args.exit_target,
    )
    curve = build_curve(deltas, age_min=args.age_min, age_max=args.age_max, smooth_frac=args.smooth_frac)
    age_adjusted = apply_age_adjustment(career, curve, ref_age=args.ref_age)
    peak = compute_peak(
        age_adjusted,
        min_poss=args.peak_min_poss,
        peak_age_min=args.peak_age_min,
        peak_age_max=args.peak_age_max,
    )
    peak_window = compute_peak_window(
        age_adjusted,
        window=args.peak_window,
        min_total_poss=args.peak_window_min_poss,
        peak_age_min=args.peak_age_min,
        peak_age_max=args.peak_age_max,
    )
    summary = compute_summary(age_adjusted, min_seasons=args.min_seasons, min_poss=args.peak_min_poss)

    curve.to_csv(CAREER_AGE_MODEL_CURVE_CSV, index=False)
    age_adjusted.to_csv(CAREER_AGE_MODEL_SEASONS_CSV, index=False)
    peak.to_csv(CAREER_AGE_MODEL_PEAK_CSV, index=False)
    peak_window.to_csv(CAREER_AGE_MODEL_PEAK_3YR_CSV, index=False)
    summary.to_csv(CAREER_AGE_MODEL_SUMMARY_CSV, index=False)
    CAREER_AGE_MODEL_META_JSON.write_text(json.dumps({
        "career_csv": args.career_csv,
        "seasons": [int(career["Season"].min()), int(career["Season"].max())],
        "n_player_seasons": int(len(career)),
        "n_players": int(career["PLAYER_ID"].nunique()),
        "n_deltas": int(len(deltas)),
        "n_observed_deltas": int(deltas["delta_type"].eq("observed").sum()),
        "n_exit_deltas": int(deltas["delta_type"].eq("exit").sum()),
        "age_min": args.age_min,
        "age_max": args.age_max,
        "ref_age": args.ref_age,
        "delta_min_poss": args.delta_min_poss,
        "peak_age_min": args.peak_age_min,
        "peak_age_max": args.peak_age_max,
        "exit_weight": args.exit_weight,
        "exit_target": args.exit_target,
        "smooth_frac": args.smooth_frac,
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "curve": str(CAREER_AGE_MODEL_CURVE_CSV),
            "player_seasons": str(CAREER_AGE_MODEL_SEASONS_CSV),
            "peak": str(CAREER_AGE_MODEL_PEAK_CSV),
            "peak_3yr": str(CAREER_AGE_MODEL_PEAK_3YR_CSV),
            "summary": str(CAREER_AGE_MODEL_SUMMARY_CSV),
        },
    }, indent=2))

    print(f"Aging curve -> {CAREER_AGE_MODEL_CURVE_CSV}")
    print(f"Player seasons -> {CAREER_AGE_MODEL_SEASONS_CSV}")
    print(f"Peak -> {CAREER_AGE_MODEL_PEAK_CSV}")
    print(f"Peak {args.peak_window}yr -> {CAREER_AGE_MODEL_PEAK_3YR_CSV}")
    print(f"Summary -> {CAREER_AGE_MODEL_SUMMARY_CSV}")
    print(f"Meta -> {CAREER_AGE_MODEL_META_JSON}")
    print("\nTop career weighted RAPM:")
    print(summary[["Name", "N_Seasons", "Total_Poss", "Career_Wmean_RAPM", "Career_Wmean_RAPM_at_Ref"]].head(args.top_k).to_string(index=False))
    print(f"\nTop {args.peak_window}-year age-adjusted peaks:")
    print(peak_window[[
        "Name", "Start_Season", "End_Season", "Poss_Weighted_Age",
        "Peak_3yr_RAPM_at_Ref", "Raw_3yr_RAPM",
    ]].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
