"""Age-adjusted rookie impact research.

Builds rookie cohorts from Basketball Reference debut year, joins the impact
metrics that are consistently available back to 1997, and graphs rookie impact
over time.

Run from repo root:
    rapm/venv/bin/python "random research/age_adjusted_rookie_impact.py"
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing matplotlib. Run with the project environment, e.g. "
        "'rapm/venv/bin/python' 'random research/age_adjusted_rookie_impact.py', "
        "or install dependencies with 'pip install -r requirements.txt'."
    ) from exc

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "random research" / "outputs" / "age_adjusted_rookie_impact"

BREF_TOTALS = ROOT / "data" / "raw" / "site_Data" / "totals.csv"
BREF_ADVANCED = ROOT.parent / "mof's spm project" / "archive" / "Advanced.csv"
AGE_ADJUSTED_RAPM = ROOT / "rapm" / "outputs" / "aging" / "age_adjusted_rapm.csv"
CAREER_AGE_MODEL = ROOT / "rapm" / "outputs" / "aging" / "career_age_model_player_seasons.csv"

def normalize_name(value: object) -> str:
    """Return a stable ASCII key for joining Basketball Reference style names."""
    if pd.isna(value):
        return ""
    text = str(value).replace("*", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"[missing] {display_path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_bref_debuts() -> pd.DataFrame:
    """Basketball Reference saved totals lack rookie flags, so first saved year is debut."""
    totals = read_csv_if_exists(BREF_TOTALS, usecols=["Player", "year", "MP"])
    if totals.empty:
        raise FileNotFoundError(f"Missing Basketball Reference totals at {BREF_TOTALS}")

    totals["name_key"] = totals["Player"].map(normalize_name)
    totals["year"] = numeric(totals["year"])
    totals["MP"] = numeric(totals["MP"])
    totals = totals.dropna(subset=["name_key", "year"])

    debuts = (
        totals.sort_values(["name_key", "year", "MP"], ascending=[True, True, False])
        .groupby("name_key", as_index=False)
        .agg(
            bref_debut_year=("year", "min"),
            bref_name=("Player", "first"),
        )
    )
    debuts["bref_debut_year"] = debuts["bref_debut_year"].astype(int)
    return debuts


def load_bref_advanced() -> pd.DataFrame:
    advanced = read_csv_if_exists(BREF_ADVANCED)
    if advanced.empty:
        return pd.DataFrame()

    advanced["Season"] = numeric(advanced["season"]).astype("Int64")
    advanced["name_key"] = advanced["player"].map(normalize_name)
    advanced["BRef_MP"] = numeric(advanced["mp"])

    # Basketball Reference/Stathead includes team split rows plus a TOT row for
    # traded players. Keep TOT where present, otherwise the highest-minute row.
    advanced["_is_tot"] = advanced["tm"].astype(str).eq("TOT")
    advanced = advanced.sort_values(["Season", "name_key", "_is_tot", "BRef_MP"], ascending=[True, True, False, False])
    advanced = advanced.drop_duplicates(["Season", "name_key"], keep="first")

    rename = {
        "player_id": "BRef_Player_ID",
        "player": "BRef_Player",
        "pos": "BRef_Pos",
        "age": "BRef_Age",
        "experience": "BRef_Experience",
        "g": "BRef_G",
        "per": "BRef_PER",
        "ows": "BRef_OWS",
        "dws": "BRef_DWS",
        "ws": "BRef_WS",
        "ws_48": "BRef_WS48",
        "obpm": "BRef_OBPM",
        "dbpm": "BRef_DBPM",
        "bpm": "BRef_BPM",
        "vorp": "BRef_VORP",
    }
    advanced = advanced.rename(columns=rename)
    advanced["BRef_MPG"] = numeric(advanced["BRef_MP"]) / numeric(advanced["BRef_G"]).replace(0, np.nan)
    cols = [
        "Season",
        "name_key",
        "BRef_Player_ID",
        "BRef_Player",
        "BRef_Pos",
        "BRef_Age",
        "BRef_Experience",
        "BRef_G",
        "BRef_MP",
        "BRef_MPG",
        "BRef_PER",
        "BRef_OWS",
        "BRef_DWS",
        "BRef_WS",
        "BRef_WS48",
        "BRef_OBPM",
        "BRef_DBPM",
        "BRef_BPM",
        "BRef_VORP",
    ]
    return advanced[[c for c in cols if c in advanced.columns]]


def join_bref_advanced(rookies: pd.DataFrame) -> pd.DataFrame:
    advanced = load_bref_advanced()
    if advanced.empty:
        return rookies
    return rookies.merge(advanced, on=["Season", "name_key"], how="left")


def load_rookie_base(
    min_poss: int,
    min_age: float | None,
    max_age: float | None,
    start_season: int | None,
    end_season: int | None,
) -> pd.DataFrame:
    source = AGE_ADJUSTED_RAPM if AGE_ADJUSTED_RAPM.exists() else CAREER_AGE_MODEL
    career = read_csv_if_exists(source)
    if career.empty:
        raise FileNotFoundError("Missing age-adjusted RAPM player-season table.")

    career["Season"] = numeric(career["Season"]).astype("Int64")
    career["Age"] = numeric(career["Age"])
    career["PLAYER_ID"] = numeric(career["PLAYER_ID"]).astype("Int64")
    career["name_key"] = career["Name"].map(normalize_name)
    if "Poss" not in career.columns:
        career["Poss"] = numeric(career["Poss_Off"]).fillna(0) + numeric(career["Poss_Def"]).fillna(0)
    else:
        career["Poss"] = numeric(career["Poss"])

    debuts = load_bref_debuts()
    rookies = career.merge(debuts, on="name_key", how="left")
    rookies = rookies[rookies["bref_debut_year"].notna()].copy()
    rookies = rookies[rookies["Season"].astype(float).eq(rookies["bref_debut_year"].astype(float))]

    if start_season is not None:
        rookies = rookies[rookies["Season"] >= start_season]
    if end_season is not None:
        rookies = rookies[rookies["Season"] <= end_season]
    if min_age is not None:
        rookies = rookies[rookies["Age"] >= min_age]
    if max_age is not None:
        rookies = rookies[rookies["Age"] <= max_age]
    rookies = rookies[rookies["Poss"] >= min_poss].copy()

    rookies["Season"] = rookies["Season"].astype(int)
    rookies["PLAYER_ID"] = rookies["PLAYER_ID"].astype(int)
    rookies["Rookie_Age"] = rookies["Age"]
    rookies["Rookie_Age_Bin"] = rookies["Rookie_Age"].round().astype(int)
    rookies["Rookie_Poss"] = rookies["Poss"]
    rookies["Cohort_Definition"] = (
        "BRef debut season"
        + (f", age >= {min_age:g}" if min_age is not None else "")
        + (f", age <= {max_age:g}" if max_age is not None else "")
        + f", possessions >= {min_poss}"
    )

    keep = [
        "Season",
        "PLAYER_ID",
        "Name",
        "bref_name",
        "bref_debut_year",
        "Rookie_Age",
        "Rookie_Age_Bin",
        "Rookie_Poss",
        "Cohort_Definition",
        "MPG",
        "Tier",
        "Poss_Off",
        "Poss_Def",
        "Poss",
        "Off",
        "Def",
        "RAPM",
        "VORP_Off",
        "VORP_Def",
        "VORP",
        "Off_at_ref",
        "Def_at_ref",
        "RAPM_at_ref",
        "Off_above_curve",
        "Def_above_curve",
        "RAPM_above_curve",
        "name_key",
    ]
    return rookies[[c for c in keep if c in rookies.columns]].sort_values(["Season", "Name"])


METRICS_1997 = {
    "Raw RAPM": ("RAPM", "Poss"),
    "Raw Off": ("Off", "Poss"),
    "Raw Def (lower better)": ("Def", "Poss"),
    "Age-Adjusted RAPM": ("RAPM_at_ref", "Poss"),
    "Off At Ref Age": ("Off_at_ref", "Poss"),
    "Def At Ref Age (lower better)": ("Def_at_ref", "Poss"),
    "RAPM Above Aging Curve": ("RAPM_above_curve", "Poss"),
    "Off Above Aging Curve": ("Off_above_curve", "Poss"),
    "Def Above Aging Curve (lower better)": ("Def_above_curve", "Poss"),
    "RAPM VORP": ("VORP", "Poss"),
    "Off VORP": ("VORP_Off", "Poss"),
    "Def VORP": ("VORP_Def", "Poss"),
    "BRef BPM": ("BRef_BPM", "BRef_MP"),
    "BRef OBPM": ("BRef_OBPM", "BRef_MP"),
    "BRef DBPM": ("BRef_DBPM", "BRef_MP"),
    "BRef VORP": ("BRef_VORP", "BRef_MP"),
    "BRef PER": ("BRef_PER", "BRef_MP"),
    "BRef WS": ("BRef_WS", "BRef_MP"),
    "BRef WS/48": ("BRef_WS48", "BRef_MP"),
}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def reference_age_distribution(df: pd.DataFrame) -> pd.Series:
    """Possession-weighted rookie age mix used to age-standardize seasons."""
    if "Rookie_Age_Bin" not in df.columns:
        return pd.Series(dtype=float)
    weights = numeric(df["Poss"]) if "Poss" in df.columns else pd.Series(1.0, index=df.index)
    age_weights = weights.groupby(df["Rookie_Age_Bin"]).sum()
    age_weights = age_weights[age_weights > 0]
    return age_weights / age_weights.sum()


def age_standardized_mean(
    group: pd.DataFrame,
    value_col: str,
    weight_col: str,
    ref_age_weights: pd.Series,
) -> float:
    """Combine within-age means using a fixed reference age mix.

    If a season is missing an age bin, reference weights are re-normalized over
    the bins present that year rather than imputing a value.
    """
    if ref_age_weights.empty or "Rookie_Age_Bin" not in group.columns:
        return float("nan")

    age_means = {}
    for age, age_group in group.groupby("Rookie_Age_Bin"):
        values = numeric(age_group[value_col])
        weights = numeric(age_group[weight_col]) if weight_col in age_group.columns else pd.Series(1.0, index=age_group.index)
        value = weighted_mean(values, weights)
        if not np.isnan(value):
            age_means[int(age)] = value

    present_weights = ref_age_weights[ref_age_weights.index.isin(age_means)]
    if present_weights.empty:
        return float("nan")
    present_weights = present_weights / present_weights.sum()
    return float(sum(age_means[int(age)] * weight for age, weight in present_weights.items()))


def summarize_metrics(df: pd.DataFrame, metrics: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    ref_age_weights = reference_age_distribution(df)
    for label, (value_col, weight_col) in metrics.items():
        if value_col not in df.columns:
            continue
        cols = ["Season", "PLAYER_ID", "Rookie_Age_Bin", value_col]
        if weight_col in df.columns:
            cols.append(weight_col)
        work = df[[c for c in cols if c in df.columns]].copy()
        work[value_col] = numeric(work[value_col])
        if weight_col not in work.columns:
            work[weight_col] = 1.0
        else:
            work[weight_col] = numeric(work[weight_col])
        work = work.dropna(subset=[value_col])
        if work.empty:
            continue
        for season, group in work.groupby("Season"):
            rows.append(
                {
                    "Season": int(season),
                    "metric": label,
                    "value_column": value_col,
                    "weight_column": weight_col,
                    "n_players": int(group["PLAYER_ID"].nunique()),
                    "mean": float(group[value_col].mean()),
                    "median": float(group[value_col].median()),
                    "p25": float(group[value_col].quantile(0.25)),
                    "p75": float(group[value_col].quantile(0.75)),
                    "weighted_mean": weighted_mean(group[value_col], group[weight_col]),
                    "age_standardized_weighted_mean": age_standardized_mean(
                        group,
                        value_col,
                        weight_col,
                        ref_age_weights,
                    ),
                    "weight_sum": float(group[weight_col].fillna(0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "Season"])


def summarize_eras(df: pd.DataFrame, metrics: dict[str, tuple[str, str]]) -> pd.DataFrame:
    bins = [1996, 2004, 2014, 2024, 2030]
    labels = ["1997-2004", "2005-2014", "2015-2024", "2025+"]
    work = df.copy()
    work["era"] = pd.cut(work["Season"], bins=bins, labels=labels)
    rows = []
    for label, (value_col, weight_col) in metrics.items():
        if value_col not in work.columns:
            continue
        for era, group in work.dropna(subset=["era"]).groupby("era", observed=False):
            values = numeric(group[value_col])
            if values.notna().sum() == 0:
                continue
            weights = numeric(group[weight_col]) if weight_col in group.columns else pd.Series(1.0, index=group.index)
            rows.append(
                {
                    "era": str(era),
                    "metric": label,
                    "n_players": int(group.loc[values.notna(), "PLAYER_ID"].nunique()),
                    "mean": float(values.mean()),
                    "weighted_mean": weighted_mean(values, weights),
                    "median": float(values.median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "era"])


def summarize_age_mix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in df.groupby("Season"):
        total_players = group["PLAYER_ID"].nunique()
        total_poss = group["Rookie_Poss"].sum()
        for age, age_group in group.groupby("Rookie_Age_Bin"):
            rows.append(
                {
                    "Season": int(season),
                    "Rookie_Age_Bin": int(age),
                    "n_players": int(age_group["PLAYER_ID"].nunique()),
                    "player_share": float(age_group["PLAYER_ID"].nunique() / total_players) if total_players else np.nan,
                    "poss": float(age_group["Rookie_Poss"].sum()),
                    "poss_share": float(age_group["Rookie_Poss"].sum() / total_poss) if total_poss else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["Season", "Rookie_Age_Bin"])


def add_metric_trends(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    trend_col = "age_standardized_weighted_mean"
    if trend_col not in summary.columns:
        trend_col = "weighted_mean"
    for metric, group in summary.dropna(subset=[trend_col]).groupby("metric"):
        if len(group) < 3:
            continue
        x = group["Season"].astype(float).to_numpy()
        y = group[trend_col].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "metric": metric,
                "trend_column": trend_col,
                "first_season": int(group["Season"].min()),
                "last_season": int(group["Season"].max()),
                "n_seasons": int(len(group)),
                "slope_per_season": float(slope),
                "slope_per_decade": float(slope * 10),
                "trendline_start": float(intercept + slope * x.min()),
                "trendline_end": float(intercept + slope * x.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("metric")


def clear_output_dir(out_dir: Path) -> None:
    """Remove stale research outputs so old partial-era plots do not linger."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".csv", ".png"}:
            path.unlink()


def plot_metric_grid(
    summary: pd.DataFrame,
    metric_names: list[str],
    title: str,
    output_path: Path,
    rolling_window: int,
) -> None:
    plot_data = summary[summary["metric"].isin(metric_names)].copy()
    plot_col = "age_standardized_weighted_mean"
    if plot_col not in plot_data.columns:
        plot_col = "weighted_mean"
    plot_data = plot_data.dropna(subset=[plot_col])
    if plot_data.empty:
        print(f"[skip plot] no data for {output_path.name}")
        return

    metrics_present = [m for m in metric_names if m in set(plot_data["metric"])]
    ncols = 2
    nrows = int(np.ceil(len(metrics_present) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4 * nrows, 4)), squeeze=False)
    fig.suptitle(title, fontsize=16, y=0.995)

    for ax, metric in zip(axes.ravel(), metrics_present):
        group = plot_data[plot_data["metric"] == metric].sort_values("Season")
        ax.plot(group["Season"], group[plot_col], marker="o", linewidth=1.2, alpha=0.45, label="annual age-std")
        smooth = group[plot_col].rolling(rolling_window, min_periods=1, center=True).mean()
        ax.plot(group["Season"], smooth, linewidth=2.8, label=f"centered {rolling_window}-yr")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(metric)
        ax.set_xlabel("Rookie season")
        ax.set_ylabel("Age-standardized weighted mean")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[len(metrics_present) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rookie_counts(rookies: pd.DataFrame, output_path: Path) -> None:
    counts = (
        rookies.groupby("Season")
        .agg(n_rookies=("PLAYER_ID", "nunique"), avg_age=("Rookie_Age", "mean"), total_poss=("Rookie_Poss", "sum"))
        .reset_index()
    )
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    axes[0].bar(counts["Season"], counts["n_rookies"], color="#4C78A8")
    axes[0].set_ylabel("Rookies")
    axes[0].set_title("Rookie sample over time")
    axes[1].plot(counts["Season"], counts["avg_age"], marker="o", color="#F58518")
    axes[1].set_ylabel("Avg age")
    axes[2].plot(counts["Season"], counts["total_poss"], marker="o", color="#54A24B")
    axes[2].set_ylabel("Total poss")
    axes[2].set_xlabel("Rookie season")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_age_mix(age_mix: pd.DataFrame, output_path: Path) -> None:
    if age_mix.empty:
        return
    pivot = age_mix.pivot(index="Season", columns="Rookie_Age_Bin", values="poss_share").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(pivot.index, [pivot[col] for col in pivot.columns], labels=[f"Age {col}" for col in pivot.columns])
    ax.set_title("Rookie Age Mix Over Time")
    ax.set_xlabel("Rookie season")
    ax.set_ylabel("Possession share")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncols=min(len(pivot.columns), 5), fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gaussian_smooth(years: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray:
    """Smooth annual values with a Gaussian kernel over season."""
    out = np.full_like(values, np.nan, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return out
    x = years.astype(float)
    for idx, year in enumerate(x):
        dist = (x[valid] - year) / bandwidth
        weights = np.exp(-0.5 * dist**2)
        if weights.sum() > 0:
            out[idx] = float(np.average(values[valid], weights=weights))
    return out


def zscore(values: pd.Series) -> pd.Series:
    values = numeric(values)
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def build_standardized_context_series(
    cohort: pd.DataFrame,
    annual: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    frames = []
    for metric in metric_names:
        rows = annual[annual["metric"] == metric].copy()
        if rows.empty:
            continue
        value_col = "age_standardized_weighted_mean"
        if value_col not in rows.columns:
            value_col = "weighted_mean"
        rows = rows[["Season", value_col]].rename(columns={value_col: "value"})
        rows["series"] = metric
        rows["series_type"] = "impact"
        frames.append(rows)

    context_rows = []
    for season, group in cohort.groupby("Season"):
        if "BRef_MP" in group.columns:
            minute_weights = numeric(group["BRef_MP"])
        else:
            minute_weights = numeric(group["Rookie_Poss"])
        context_rows.append(
            {
                "Season": int(season),
                "Avg Rookie Age": weighted_mean(numeric(group["Rookie_Age"]), minute_weights),
                "Avg MPG": float(numeric(group["BRef_MPG"]).mean()) if "BRef_MPG" in group.columns else float(numeric(group["MPG"]).mean()),
            }
        )
    context = pd.DataFrame(context_rows)
    for column in ["Avg Rookie Age", "Avg MPG"]:
        rows = context[["Season", column]].rename(columns={column: "value"})
        rows["series"] = column
        rows["series_type"] = "context"
        frames.append(rows)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["series", "Season"])
    out["z_score"] = out.groupby("series")["value"].transform(zscore)
    return out.sort_values(["series_type", "series", "Season"])


def plot_standardized_context_trends(
    trend_data: pd.DataFrame,
    output_path: Path,
    title: str,
    bandwidth: float,
) -> None:
    if trend_data.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    for series, group in trend_data.groupby("series", sort=False):
        group = group.sort_values("Season")
        years = group["Season"].to_numpy(dtype=float)
        z_values = group["z_score"].to_numpy(dtype=float)
        smooth = gaussian_smooth(years, z_values, bandwidth)
        linestyle = "--" if group["series_type"].iloc[0] == "context" else "-"
        linewidth = 3.0 if group["series_type"].iloc[0] == "context" else 2.4
        ax.plot(years, smooth, label=series, linewidth=linewidth, linestyle=linestyle)
        ax.scatter(years, z_values, alpha=0.15, s=16)

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Rookie season")
    ax.set_ylabel("Standardized value (z-score)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_combined_standardized_context_trends(
    full_trends: pd.DataFrame,
    top_trends: pd.DataFrame,
    output_path: Path,
    title: str,
    bandwidth: float,
) -> pd.DataFrame:
    """Plot all standardized context series for both cohorts on one chart."""
    full = full_trends.copy()
    full["cohort"] = "All qualifying"
    top = top_trends.copy()
    top["cohort"] = "Top 20 minutes"
    combined = pd.concat([full, top], ignore_index=True)
    combined["combined_z_score"] = combined.groupby("series")["value"].transform(zscore)
    combined.to_csv(output_path.with_suffix(".csv"), index=False)

    if combined.empty:
        return combined

    fig, ax = plt.subplots(figsize=(15, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_order = list(dict.fromkeys(combined["series"].tolist()))
    color_map = {series: colors[idx % len(colors)] for idx, series in enumerate(series_order)}
    linestyle_map = {"All qualifying": "--", "Top 20 minutes": "-"}

    for (series, cohort), group in combined.groupby(["series", "cohort"], sort=False):
        group = group.sort_values("Season")
        years = group["Season"].to_numpy(dtype=float)
        z_values = group["combined_z_score"].to_numpy(dtype=float)
        smooth = gaussian_smooth(years, z_values, bandwidth)
        label = f"{series} - {cohort}"
        linewidth = 2.8 if cohort == "Top 20 minutes" else 2.0
        ax.plot(
            years,
            smooth,
            color=color_map[series],
            linestyle=linestyle_map[cohort],
            linewidth=linewidth,
            label=label,
        )

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Rookie season")
    ax.set_ylabel("Combined standardized value (z-score)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return combined


AGE_GROUP_SERIES = {
    "Age-Adjusted RAPM": ("RAPM_at_ref", "Poss", "weighted_mean"),
    "BRef BPM": ("BRef_BPM", "BRef_MP", "weighted_mean"),
    "BRef VORP": ("BRef_VORP", "BRef_MP", "weighted_mean"),
    "RAPM VORP": ("VORP", "Poss", "weighted_mean"),
    "Avg MPG": ("BRef_MPG", "BRef_MP", "mean"),
}


def summarize_age_group_series(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    era_rows = []
    bins = [1996, 2004, 2014, 2024]
    labels = ["1997-2004", "2005-2014", "2015-2024"]

    for cohort_name, cohort in cohorts.items():
        work = cohort.copy()
        work["era"] = pd.cut(work["Season"], bins=bins, labels=labels)
        for age, age_group in work.groupby("Rookie_Age_Bin"):
            for series, (value_col, weight_col, method) in AGE_GROUP_SERIES.items():
                if value_col not in age_group.columns:
                    continue
                for season, season_group in age_group.groupby("Season"):
                    values = numeric(season_group[value_col])
                    weights = numeric(season_group[weight_col]) if weight_col in season_group.columns else pd.Series(1.0, index=season_group.index)
                    if method == "mean":
                        value = float(values.mean())
                    else:
                        value = weighted_mean(values, weights)
                    annual_rows.append(
                        {
                            "cohort": cohort_name,
                            "Rookie_Age_Bin": int(age),
                            "Season": int(season),
                            "series": series,
                            "value": value,
                            "n_players": int(season_group["PLAYER_ID"].nunique()),
                            "weight_sum": float(weights.fillna(0).sum()),
                        }
                    )
                for era, era_group in age_group.dropna(subset=["era"]).groupby("era", observed=False):
                    values = numeric(era_group[value_col])
                    weights = numeric(era_group[weight_col]) if weight_col in era_group.columns else pd.Series(1.0, index=era_group.index)
                    if method == "mean":
                        value = float(values.mean())
                    else:
                        value = weighted_mean(values, weights)
                    era_rows.append(
                        {
                            "cohort": cohort_name,
                            "Rookie_Age_Bin": int(age),
                            "era": str(era),
                            "series": series,
                            "value": value,
                            "n_players": int(era_group["PLAYER_ID"].nunique()),
                            "weight_sum": float(weights.fillna(0).sum()),
                        }
                    )

    annual = pd.DataFrame(annual_rows).sort_values(["cohort", "series", "Rookie_Age_Bin", "Season"])
    trend_rows = []
    for (cohort_name, age, series), group in annual.dropna(subset=["value"]).groupby(["cohort", "Rookie_Age_Bin", "series"]):
        if len(group) < 3:
            continue
        x = group["Season"].astype(float).to_numpy()
        y = group["value"].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        trend_rows.append(
            {
                "cohort": cohort_name,
                "Rookie_Age_Bin": int(age),
                "series": series,
                "first_season": int(group["Season"].min()),
                "last_season": int(group["Season"].max()),
                "n_seasons": int(len(group)),
                "slope_per_season": float(slope),
                "slope_per_decade": float(slope * 10),
                "trendline_start": float(intercept + slope * x.min()),
                "trendline_end": float(intercept + slope * x.max()),
            }
        )
    trends = pd.DataFrame(trend_rows).sort_values(["cohort", "series", "Rookie_Age_Bin"])
    eras = pd.DataFrame(era_rows).sort_values(["cohort", "series", "Rookie_Age_Bin", "era"])
    return annual, trends, eras


def plot_age_group_series(annual: pd.DataFrame, output_path: Path, bandwidth: float) -> None:
    if annual.empty:
        return

    series_order = [s for s in AGE_GROUP_SERIES if s in set(annual["series"])]
    ncols = 2
    nrows = int(np.ceil(len(series_order) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, max(4 * nrows, 5)), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ages = sorted(annual["Rookie_Age_Bin"].dropna().unique())
    age_colors = {age: colors[idx % len(colors)] for idx, age in enumerate(ages)}
    cohort_styles = {"All qualifying": "--", "Top 20 minutes": "-"}

    for ax, series in zip(axes.ravel(), series_order):
        panel = annual[annual["series"] == series]
        for (cohort_name, age), group in panel.groupby(["cohort", "Rookie_Age_Bin"], sort=False):
            group = group.dropna(subset=["value"]).sort_values("Season")
            if group.empty:
                continue
            years = group["Season"].to_numpy(dtype=float)
            values = group["value"].to_numpy(dtype=float)
            smooth = gaussian_smooth(years, values, bandwidth)
            ax.plot(
                years,
                smooth,
                color=age_colors[age],
                linestyle=cohort_styles.get(cohort_name, "-"),
                linewidth=2.4 if cohort_name == "Top 20 minutes" else 1.8,
                label=f"Age {int(age)} - {cohort_name}",
            )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_title(series)
        ax.set_xlabel("Rookie season")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(series_order) :]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle("Rookie Trends Within Age Groups", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def select_top_minutes_rookies(rookies: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep the top N rookies in each season by Basketball Reference minutes."""
    if top_n <= 0 or rookies.empty:
        return pd.DataFrame()

    work = rookies.copy()
    if "BRef_MP" in work.columns:
        work["Minutes_For_Rank"] = numeric(work["BRef_MP"])
    else:
        work["Minutes_For_Rank"] = np.nan
    work["Minutes_For_Rank"] = work["Minutes_For_Rank"].fillna(numeric(work["Rookie_Poss"]))
    work = work.sort_values(["Season", "Minutes_For_Rank", "Name"], ascending=[True, False, True])
    work["Minutes_Rank"] = work.groupby("Season").cumcount() + 1
    work = work[work["Minutes_Rank"] <= top_n].copy()
    work["Cohort_Definition"] = work["Cohort_Definition"] + f", top {top_n} by BRef minutes"
    return work


def write_cohort_outputs(
    cohort: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    title_suffix: str,
    rolling_window: int,
    smooth_bandwidth: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_path = out_dir / f"{prefix}_player_rookie_metrics.csv"
    cohort.to_csv(player_path, index=False)

    annual = summarize_metrics(cohort, METRICS_1997)
    annual.to_csv(out_dir / f"{prefix}_annual_metric_summary.csv", index=False)

    eras = summarize_eras(cohort, METRICS_1997)
    eras.to_csv(out_dir / f"{prefix}_era_metric_summary.csv", index=False)

    age_mix = summarize_age_mix(cohort)
    age_mix.to_csv(out_dir / f"{prefix}_rookie_age_mix.csv", index=False)

    trends = add_metric_trends(annual)
    trends.to_csv(out_dir / f"{prefix}_metric_trends.csv", index=False)

    plot_rookie_counts(cohort, out_dir / f"{prefix}_rookie_sample_over_time.png")
    plot_age_mix(age_mix, out_dir / f"{prefix}_rookie_age_mix_over_time.png")
    plot_metric_grid(
        annual,
        [
            "Raw RAPM",
            "Age-Adjusted RAPM",
            "RAPM Above Aging Curve",
            "RAPM VORP",
            "BRef BPM",
            "BRef VORP",
        ],
        f"Rookie Impact Over Time - RAPM and Basketball Reference ({title_suffix})",
        out_dir / f"{prefix}_rookie_total_impact_over_time.png",
        rolling_window,
    )
    plot_metric_grid(
        annual,
        [
            "Off At Ref Age",
            "Def At Ref Age (lower better)",
            "Off Above Aging Curve",
            "Def Above Aging Curve (lower better)",
            "Off VORP",
            "Def VORP",
        ],
        f"Rookie Impact Over Time - Offense and Defense Splits ({title_suffix})",
        out_dir / f"{prefix}_rookie_off_def_impact_over_time.png",
        rolling_window,
    )
    plot_metric_grid(
        annual,
        [
            "BRef BPM",
            "BRef OBPM",
            "BRef DBPM",
            "BRef VORP",
            "BRef PER",
            "BRef WS/48",
        ],
        f"Rookie Impact Over Time - Basketball Reference Advanced ({title_suffix})",
        out_dir / f"{prefix}_rookie_bref_advanced_over_time.png",
        rolling_window,
    )
    standardized = build_standardized_context_series(
        cohort,
        annual,
        [
            "Age-Adjusted RAPM",
            "RAPM VORP",
            "BRef BPM",
            "BRef VORP",
        ],
    )
    standardized.to_csv(out_dir / f"{prefix}_standardized_context_trends.csv", index=False)
    plot_standardized_context_trends(
        standardized,
        out_dir / f"{prefix}_standardized_context_trends_over_time.png",
        f"Standardized Rookie Impact, Age, and MPG ({title_suffix})",
        smooth_bandwidth,
    )
    return trends, standardized


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    clear_output_dir(out_dir)

    rookies = load_rookie_base(
        args.min_poss,
        args.min_rookie_age,
        args.max_rookie_age,
        args.start_season,
        args.end_season,
    )
    rookies = join_bref_advanced(rookies)
    if args.require_bref_experience_one and "BRef_Experience" in rookies.columns:
        experience = numeric(rookies["BRef_Experience"])
        rookies = rookies[experience.isna() | experience.eq(1)].copy()

    top_minutes_rookies = select_top_minutes_rookies(rookies, args.top_n_minutes)

    player_path = out_dir / "player_rookie_metrics.csv"
    rookies.to_csv(player_path, index=False)

    annual = summarize_metrics(rookies, METRICS_1997)
    annual_path = out_dir / "annual_metric_summary.csv"
    annual.to_csv(annual_path, index=False)

    eras = summarize_eras(rookies, METRICS_1997)
    eras_path = out_dir / "era_metric_summary.csv"
    eras.to_csv(eras_path, index=False)

    age_mix = summarize_age_mix(rookies)
    age_mix_path = out_dir / "rookie_age_mix.csv"
    age_mix.to_csv(age_mix_path, index=False)

    trends = add_metric_trends(annual)
    trends_path = out_dir / "metric_trends.csv"
    trends.to_csv(trends_path, index=False)

    plot_rookie_counts(rookies, out_dir / "rookie_sample_over_time.png")
    plot_age_mix(age_mix, out_dir / "rookie_age_mix_over_time.png")
    plot_metric_grid(
        annual,
        [
            "Raw RAPM",
            "Age-Adjusted RAPM",
            "RAPM Above Aging Curve",
            "RAPM VORP",
            "BRef BPM",
            "BRef VORP",
        ],
        "Rookie Impact Over Time - RAPM and Basketball Reference",
        out_dir / "rookie_total_impact_over_time.png",
        args.rolling_window,
    )
    plot_metric_grid(
        annual,
        [
            "Off At Ref Age",
            "Def At Ref Age (lower better)",
            "Off Above Aging Curve",
            "Def Above Aging Curve (lower better)",
            "Off VORP",
            "Def VORP",
        ],
        "Rookie Impact Over Time - 1997-Backed Offense and Defense Splits",
        out_dir / "rookie_off_def_impact_over_time.png",
        args.rolling_window,
    )
    plot_metric_grid(
        annual,
        [
            "BRef BPM",
            "BRef OBPM",
            "BRef DBPM",
            "BRef VORP",
            "BRef PER",
            "BRef WS/48",
        ],
        "Rookie Impact Over Time - Basketball Reference Advanced",
        out_dir / "rookie_bref_advanced_over_time.png",
        args.rolling_window,
    )
    standardized = build_standardized_context_series(
        rookies,
        annual,
        [
            "Age-Adjusted RAPM",
            "RAPM VORP",
            "BRef BPM",
            "BRef VORP",
        ],
    )
    standardized_path = out_dir / "standardized_context_trends.csv"
    standardized.to_csv(standardized_path, index=False)
    plot_standardized_context_trends(
        standardized,
        out_dir / "standardized_context_trends_over_time.png",
        "Standardized Rookie Impact, Age, and MPG (All Qualifying Rookies)",
        args.smooth_bandwidth,
    )
    if not top_minutes_rookies.empty:
        top_trends, top_standardized = write_cohort_outputs(
            top_minutes_rookies,
            out_dir,
            f"top{args.top_n_minutes}_minutes",
            f"Top {args.top_n_minutes} by Minutes",
            args.rolling_window,
            args.smooth_bandwidth,
        )
        combined_standardized_path = out_dir / "combined_standardized_context_trends_over_time.png"
        plot_combined_standardized_context_trends(
            standardized,
            top_standardized,
            combined_standardized_path,
            "Standardized Rookie Impact, Age, and MPG - All vs Top 20 Minutes",
            args.smooth_bandwidth,
        )
    else:
        top_trends = pd.DataFrame()
        combined_standardized_path = None

    age_group_annual, age_group_trends, age_group_eras = summarize_age_group_series(
        {
            "All qualifying": rookies,
            f"Top {args.top_n_minutes} minutes": top_minutes_rookies,
        }
    )
    age_group_annual_path = out_dir / "within_age_annual_summary.csv"
    age_group_trends_path = out_dir / "within_age_metric_trends.csv"
    age_group_eras_path = out_dir / "within_age_era_summary.csv"
    age_group_annual.to_csv(age_group_annual_path, index=False)
    age_group_trends.to_csv(age_group_trends_path, index=False)
    age_group_eras.to_csv(age_group_eras_path, index=False)
    plot_age_group_series(
        age_group_annual,
        out_dir / "within_age_trends_over_time.png",
        args.smooth_bandwidth,
    )

    print("\nAge-adjusted rookie impact research complete")
    print(f"  rookies: {len(rookies):,}")
    if not top_minutes_rookies.empty:
        print(f"  top-{args.top_n_minutes} minutes rows: {len(top_minutes_rookies):,}")
    print(f"  seasons: {rookies['Season'].min()}-{rookies['Season'].max()}")
    print(f"  cohort: first BRef season, age {args.min_rookie_age:g}-{args.max_rookie_age:g}, poss >= {args.min_poss}")
    if "BRef_BPM" in rookies.columns:
        print(f"  BRef BPM matched rows: {rookies['BRef_BPM'].notna().sum():,}")
    print(f"  output: {out_dir.relative_to(ROOT)}")
    print(f"  player rows: {player_path.relative_to(ROOT)}")
    print(f"  annual summary: {annual_path.relative_to(ROOT)}")
    print(f"  era summary: {eras_path.relative_to(ROOT)}")
    print(f"  trends: {trends_path.relative_to(ROOT)}")
    print(f"  standardized trends: {standardized_path.relative_to(ROOT)}")
    if combined_standardized_path is not None:
        print(f"  combined standardized chart: {combined_standardized_path.relative_to(ROOT)}")
    if not top_trends.empty:
        print(f"  top-{args.top_n_minutes} trends: {(out_dir / f'top{args.top_n_minutes}_minutes_metric_trends.csv').relative_to(ROOT)}")
    print(f"  within-age trends: {age_group_trends_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph age-adjusted rookie impact across available metrics.")
    parser.add_argument("--min-poss", type=int, default=1000, help="Minimum rookie possessions in RAPM table.")
    parser.add_argument("--min-rookie-age", type=float, default=18, help="Minimum age for rookie cohort.")
    parser.add_argument("--max-rookie-age", type=float, default=22, help="Maximum age for rookie cohort.")
    parser.add_argument("--start-season", type=int, default=1997)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--rolling-window", type=int, default=3, help="Centered rolling average window for plots.")
    parser.add_argument("--smooth-bandwidth", type=float, default=2.0, help="Gaussian smoothing bandwidth in seasons.")
    parser.add_argument("--top-n-minutes", type=int, default=20, help="Also generate a top-N-by-minutes rookie cohort.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-require-bref-experience-one",
        dest="require_bref_experience_one",
        action="store_false",
        help="Do not enforce Basketball Reference experience == 1 where the advanced table is matched.",
    )
    parser.set_defaults(require_bref_experience_one=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
