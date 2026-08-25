#!/usr/bin/env python3
"""Gate comparison dashboard: minutes c=2 vs SPM v2 pooled vs champion zero-prior."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DIAGNOSTICS_DIR, OUTPUTS, ensure_dirs

# prepare.py holds reproduced champion f23 (not logged under a fold suffix in experiments.csv)
_FEATURES = Path(__file__).resolve().parent.parent / "features"
if str(_FEATURES) not in sys.path:
    sys.path.insert(0, str(_FEATURES))
from prepare import GATE_BASELINES  # noqa: E402

ensure_dirs()
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(exist_ok=True)
EXPERIMENTS_CSV = DIAGNOSTICS_DIR / "experiments.csv"

THEME = {
    "bg": "#1a1d23",
    "surface": "#23272f",
    "border": "#3a404d",
    "text": "#e8eaed",
    "muted": "#9aa0a8",
}

# experiment name in experiments.csv per model × fold
MODEL_ROWS = {
    "minutes c=2": {"f24": "mprior_repro_c2_f24", "f23": "mprior_repro_c2_f23"},
    "SPM v2 pooled": {"f24": "spmv2_pooled_a1000_c2_f24", "f23": "spmv2_pooled_a1000_c2_f23"},
    "champion zero-prior": {"f24": "decay_hl250", "f23": None},
}

MODEL_COLORS = {
    "minutes c=2": "#5b8def",
    "SPM v2 pooled": "#4caf82",
    "champion zero-prior": "#9aa0a8",
}


def gate_value(df: pd.DataFrame, name: str | None, fold: str) -> float:
    if name is None:
        return float(GATE_BASELINES["champion_zero_prior"][fold])
    row = df.loc[df["name"] == name, "margin_corr"]
    if row.empty:
        raise KeyError(f"Missing experiment row: {name}")
    return float(row.iloc[0])


def main() -> None:
    if not EXPERIMENTS_CSV.exists():
        print("NO_EXPERIMENTS_CSV", flush=True)
        return

    df = pd.read_csv(EXPERIMENTS_CSV)
    folds = ("f24", "f23")
    models = list(MODEL_ROWS.keys())

    values = {
        model: {fold: gate_value(df, MODEL_ROWS[model][fold], fold) for fold in folds}
        for model in models
    }

    x = np.arange(len(folds))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["surface"])

    for i, model in enumerate(models):
        offset = (i - 1) * width
        heights = [values[model][fold] for fold in folds]
        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor=THEME["border"],
            linewidth=0.6,
        )
        for bar, v in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.004,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=THEME["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["f24 (2021-23 → 2024)", "f23 (2020-22 → 2023)"], color=THEME["text"])
    ax.set_ylabel("Next-season game-margin correlation", color=THEME["text"])
    ax.set_title(
        "Gate metric by prior type (c=2 where applicable)\n"
        "source: outputs/diagnostics/experiments.csv",
        color=THEME["text"],
    )
    ax.set_ylim(0.55, 0.78)
    ax.tick_params(colors=THEME["muted"])
    for spine in ax.spines.values():
        spine.set_color(THEME["border"])
    ax.grid(axis="y", color=THEME["border"], alpha=0.35, linewidth=0.6)
    ax.legend(frameon=True, facecolor=THEME["surface"], edgecolor=THEME["border"], labelcolor=THEME["text"])

    fig.tight_layout()
    out = FIG_DIR / "fig_gate_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=THEME["bg"])
    print(out, flush=True)


if __name__ == "__main__":
    main()
