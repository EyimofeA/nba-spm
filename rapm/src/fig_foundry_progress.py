#!/usr/bin/env python3
"""Foundry progress dashboard — autoresearch progress pattern, dark viewer theme."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import FEATURES_DIR, OUTPUTS, ensure_dirs

ensure_dirs()
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(exist_ok=True)
TSV = FEATURES_DIR / "results.tsv"
BASELINE_F24 = 0.7335
BASELINE_F23 = 0.6953

# Match outputs/viewer/human.html CSS tokens
THEME = {
    "bg": "#1a1d23",
    "surface": "#23272f",
    "surface2": "#2c313c",
    "border": "#3a404d",
    "text": "#e8eaed",
    "muted": "#9aa0a8",
    "accent": "#5b8def",
    "keep": "#4caf82",
    "discard": "#666666",
    "user": "#5b8def",
    "warn": "#ffb74d",
    "danger": "#e57373",
    "baseline_f24": "#ffb74d",
    "baseline_f23": "#ff8a65",
}


def apply_dark_theme(fig: plt.Figure, axes: np.ndarray) -> None:
    fig.patch.set_facecolor(THEME["bg"])
    for ax in axes.ravel():
        ax.set_facecolor(THEME["surface"])
        ax.tick_params(colors=THEME["muted"], labelsize=8)
        ax.xaxis.label.set_color(THEME["text"])
        ax.yaxis.label.set_color(THEME["text"])
        ax.title.set_color(THEME["text"])
        for spine in ax.spines.values():
            spine.set_color(THEME["border"])
        ax.grid(True, color=THEME["border"], alpha=0.35, linewidth=0.6)


def load_df() -> pd.DataFrame | None:
    if not TSV.exists() or TSV.stat().st_size < 10:
        return None
    df = pd.read_csv(TSV, sep="\t")
    df = df.reset_index(drop=True)
    for col in ("gate_f24", "gate_f23", "oof_r2_off", "oof_r2_def"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gate_plot"] = df["gate_f24"].fillna(df["gate_f23"])
    df["baseline"] = np.where(df["gate_f24"].notna(), BASELINE_F24, BASELINE_F23)
    df["gate_delta"] = df["gate_plot"] - df["baseline"]
    df["oof_r2_mean"] = df[["oof_r2_off", "oof_r2_def"]].mean(axis=1, skipna=True)
    return df.dropna(subset=["gate_plot"])


def plot_progress_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    status = df["status"].astype(str).str.lower()
    user = df["source"].astype(str).str.startswith("user")
    foundry = ~user

    ax.scatter(
        df.index[status == "discard"],
        df.loc[status == "discard", "gate_plot"],
        c=THEME["discard"],
        s=22,
        alpha=0.55,
        label="Discarded",
        zorder=2,
    )
    ax.scatter(
        df.index[foundry & (status == "keep")],
        df.loc[foundry & (status == "keep"), "gate_plot"],
        c=THEME["keep"],
        s=58,
        marker="o",
        edgecolors=THEME["border"],
        linewidths=0.5,
        label="Kept (foundry)",
        zorder=4,
    )
    ax.scatter(
        df.index[user],
        df.loc[user, "gate_plot"],
        c=THEME["user"],
        s=58,
        marker="s",
        edgecolors=THEME["border"],
        linewidths=0.5,
        label="User",
        zorder=4,
    )

    keep = df[status == "keep"]
    if not keep.empty:
        ax.step(
            keep.index,
            keep["gate_plot"].cummax(),
            where="post",
            color=THEME["keep"],
            linewidth=2,
            alpha=0.75,
            label="Running best",
        )

    ax.axhline(
        BASELINE_F24,
        color=THEME["baseline_f24"],
        ls="--",
        lw=0.9,
        alpha=0.85,
        label=f"minutes f24 ({BASELINE_F24})",
    )
    ax.axhline(
        BASELINE_F23,
        color=THEME["baseline_f23"],
        ls=":",
        lw=0.9,
        alpha=0.85,
        label=f"minutes f23 ({BASELINE_F23})",
    )
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Gate correlation")
    ax.set_title("Progress scatter")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, facecolor=THEME["surface2"], edgecolor=THEME["border"])


def plot_status_counts(ax: plt.Axes, df: pd.DataFrame) -> None:
    status = df["status"].astype(str).str.lower()
    counts = status.value_counts()
    order = ["keep", "discard", "crash", "leak_suspect"]
    labels = [s for s in order if s in counts.index] + [s for s in counts.index if s not in order]
    colors_map = {
        "keep": THEME["keep"],
        "discard": THEME["discard"],
        "crash": THEME["danger"],
        "leak_suspect": THEME["warn"],
    }
    vals = [counts.get(l, 0) for l in labels]
    bars = ax.bar(
        labels,
        vals,
        color=[colors_map.get(l, THEME["muted"]) for l in labels],
        edgecolor=THEME["border"],
        linewidth=0.6,
    )
    ax.set_ylabel("Count")
    ax.set_title(f"Status counts (n={len(df)})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, str(v), ha="center", va="bottom", color=THEME["text"], fontsize=8)
    ax.set_ylim(0, max(vals, default=1) * 1.15)


def plot_oof_vs_gate(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["oof_r2_mean"])
    if sub.empty:
        ax.text(0.5, 0.5, "No OOF R² logged yet", ha="center", va="center", transform=ax.transAxes, color=THEME["muted"])
        ax.set_title("OOF R² vs gate (reward-hack detector)")
        return

    status = sub["status"].astype(str).str.lower()
    colors = np.where(status == "keep", THEME["keep"], THEME["discard"])
    ax.scatter(sub["oof_r2_mean"], sub["gate_plot"], c=colors, s=42, alpha=0.85, edgecolors=THEME["border"], linewidths=0.4)
    for _, row in sub.iterrows():
        ax.annotate(
            str(row.get("id", ""))[:18],
            (row["oof_r2_mean"], row["gate_plot"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color=THEME["muted"],
            alpha=0.9,
        )
    ax.axhline(BASELINE_F24, color=THEME["baseline_f24"], ls="--", lw=0.7, alpha=0.6)
    ax.axhline(BASELINE_F23, color=THEME["baseline_f23"], ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("Mean OOF R² (off + def) / 2")
    ax.set_ylabel("Gate correlation")
    ax.set_title("OOF R² vs gate (reward-hack detector)")


def plot_top_gate_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df.nlargest(10, "gate_delta").sort_values("gate_delta", ascending=True)
    if sub.empty:
        ax.set_title("Top 10 gate Δ vs minutes baseline")
        return

    colors = [THEME["keep"] if s == "keep" else THEME["muted"] for s in sub["status"].astype(str).str.lower()]
    ylabels = [str(i)[:22] for i in sub["id"]]
    bars = ax.barh(ylabels, sub["gate_delta"], color=colors, edgecolor=THEME["border"], linewidth=0.5)
    ax.axvline(0, color=THEME["muted"], lw=0.8)
    ax.set_xlabel("Δ gate vs minutes baseline")
    ax.set_title("Top 10 gate Δ vs minutes baseline")
    for bar, v in zip(bars, sub["gate_delta"]):
        ax.text(v + (0.001 if v >= 0 else -0.001), bar.get_y() + bar.get_height() / 2, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=THEME["text"])


def main() -> None:
    df = load_df()
    if df is None:
        print("NO_RESULTS_YET", flush=True)
        return
    if df.empty:
        return

    n_keep = int((df["status"].astype(str).str.lower() == "keep").sum())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    apply_dark_theme(fig, axes)

    plot_progress_scatter(axes[0, 0], df)
    plot_status_counts(axes[0, 1], df)
    plot_oof_vs_gate(axes[1, 0], df)
    plot_top_gate_delta(axes[1, 1], df)

    fig.suptitle(
        f"Feature Foundry Progress ({len(df)} experiments, {n_keep} kept)",
        color=THEME["text"],
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "fig_foundry_progress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
    print(out, flush=True)


if __name__ == "__main__":
    main()
