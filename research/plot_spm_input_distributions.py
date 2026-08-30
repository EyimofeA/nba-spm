"""Plot the exact five-year inputs used by Box15 and five rich-SPM examples."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BOX_SOURCE = ROOT / "artifacts/research/historical_box15_extension/historical_box15_extension_v1_08ff4c34ff/five_year_box15_features.parquet"
RICH_SOURCE = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_40e72f25d2/five_year_features.parquet"
WINDOW_END = 2026
MINIMUM_DISPLAY_EXPOSURE = 500

BOX_FEATURES = {
    "PTS_p100": "Points",
    "AST_p100": "Assists",
    "TOV_p100": "Turnovers",
    "STL_p100": "Steals",
    "BLK_p100": "Blocks",
    "OREB_p100": "Off. rebounds",
    "DREB_p100": "Def. rebounds",
    "PF_p100": "Fouls",
    "PFD_p100": "Fouls drawn",
    "FTA_p100": "FT attempts",
    "FTM_p100": "FT made",
    "FG2A_p100": "2P attempts",
    "FG2M_p100": "2P made",
    "FG3A_p100": "3P attempts",
    "FG3M_p100": "3P made",
}

RICH_FEATURES = {
    "zts_pct_points": "zTS",
    "box_creation_2017_eb_p100": "Box Creation",
    "offensive_load_2017_eb_p100": "Offensive Load",
    "crafted_spacing_stable_v1": "Spacing",
    "rim_points_saved_p100": "Rim points saved",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(group: str, frame: pd.DataFrame, features: dict[str, str]) -> pd.DataFrame:
    rows = []
    exposure = frame[["OffPoss", "DefPoss"]].min(axis=1)
    for feature, label in features.items():
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values[np.isfinite(values)]
        displayed = values[np.isfinite(values) & exposure.ge(MINIMUM_DISPLAY_EXPOSURE)]
        low, high = displayed.quantile([0.01, 0.99])
        note = ""
        if feature == "zts_pct_points" and (displayed.abs() > 100).any():
            note = "BLOCK: impossible fallback values"
        if feature == "rim_points_saved_p100" and not (displayed.gt(0).any() and displayed.lt(0).any()):
            note = "BLOCK: two-sided metric has one sign"
        rows.append(
            {
                "group": group,
                "feature": feature,
                "label": label,
                "rows": int(len(frame)),
                "finite_rows": int(len(finite)),
                "display_rows": int(len(displayed)),
                "zero_fraction": float(displayed.eq(0).mean()),
                "raw_minimum": float(finite.min()),
                "minimum": float(displayed.min()),
                "p01": float(low),
                "p25": float(displayed.quantile(0.25)),
                "median": float(displayed.median()),
                "p75": float(displayed.quantile(0.75)),
                "p99": float(high),
                "maximum": float(displayed.max()),
                "raw_maximum": float(finite.max()),
                "outside_display_rows": int(((displayed < low) | (displayed > high)).sum()),
                "quality_note": note,
            }
        )
    return pd.DataFrame(rows)


def plot_panel(
    frame: pd.DataFrame,
    features: dict[str, str],
    summary: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    path: Path,
    columns: int,
) -> None:
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.1, rows * 3.7))
    axes = np.atleast_1d(axes).ravel()
    weights = np.sqrt(frame[["OffPoss", "DefPoss"]].min(axis=1).clip(lower=0))
    exposure = frame[["OffPoss", "DefPoss"]].min(axis=1)
    for axis, (feature, label) in zip(axes, features.items(), strict=False):
        record = summary.loc[summary["feature"].eq(feature)].iloc[0]
        values = pd.to_numeric(frame[feature], errors="coerce")
        visible = (
            values.between(record.p01, record.p99)
            & np.isfinite(values)
            & exposure.ge(MINIMUM_DISPLAY_EXPOSURE)
        )
        bins = np.linspace(record.p01, record.p99, 27)
        axis.hist(values[visible], bins=bins, density=True, color="#8D99AE", alpha=0.42, label="Players")
        axis.hist(
            values[visible], bins=bins, weights=weights[visible], density=True,
            histtype="step", linewidth=2.2, color="#F59E0B", label="Exposure-weighted",
        )
        flagged = bool(record.quality_note)
        axis.axvline(
            record["median"], color="#E5E7EB" if flagged else "#64748B",
            linewidth=1.3, linestyle="--",
        )
        axis.set_title(label, loc="left", fontsize=13, fontweight="bold", color="#F8FAFC")
        axis.text(
            0.01, 0.96,
            f"median {record['median']:.2f}  |  500+ poss range {record['minimum']:.2f} to {record['maximum']:.2f}",
            transform=axis.transAxes, va="top", fontsize=8.7,
            color="#CBD5E1" if flagged else "#475569",
        )
        if flagged:
            axis.text(
                0.01, 0.82, record.quality_note, transform=axis.transAxes,
                va="top", fontsize=9, fontweight="bold", color="#FCA5A5",
            )
            axis.set_facecolor("#351A23")
        axis.grid(axis="y", color="#334155", alpha=0.45, linewidth=0.7)
        axis.tick_params(colors="#CBD5E1", labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes[len(features):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.975), frameon=False, labelcolor="#E5E7EB")
    fig.suptitle(title, x=0.03, y=0.99, ha="left", color="#F8FAFC", fontsize=21, fontweight="bold")
    fig.text(0.03, 0.952, subtitle, ha="left", color="#94A3B8", fontsize=11)
    fig.text(
        0.03, 0.012,
        "The panels use players with at least 500 possessions and display their 1st–99th percentile. The audit table retains every raw tail.",
        ha="left", color="#94A3B8", fontsize=9,
    )
    fig.patch.set_facecolor("#0B1120")
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.91), h_pad=2.0, w_pad=1.2)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> Path:
    box = pd.read_parquet(BOX_SOURCE).query("Window_End == @WINDOW_END").copy()
    rich = pd.read_parquet(RICH_SOURCE).query("Window_End == @WINDOW_END").copy()
    for name, frame in (("box", box), ("rich", rich)):
        if frame.empty or frame.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError(f"Invalid {name} plotting grain.")
    config = {
        "window": "2022-2026",
        "window_end": WINDOW_END,
        "box_source_sha256": sha256(BOX_SOURCE),
        "rich_source_sha256": sha256(RICH_SOURCE),
        "box_features": list(BOX_FEATURES),
        "rich_features": list(RICH_FEATURES),
        "display_quantiles": [0.01, 0.99],
        "minimum_display_exposure": MINIMUM_DISPLAY_EXPOSURE,
        "weight_overlay": "sqrt_min_offposs_defposs",
        "plot_code_sha256": sha256(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/spm_input_distribution_audit" / f"spm_input_distribution_audit_v1_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    box_summary = summarize("Box15", box, BOX_FEATURES)
    rich_summary = summarize("Rich SPM examples", rich, RICH_FEATURES)
    summary = pd.concat([box_summary, rich_summary], ignore_index=True)
    summary.to_parquet(output / "feature_summary.parquet", index=False)
    plot_panel(
        box, BOX_FEATURES, box_summary,
        title="Box15 input distributions",
        subtitle="2022–26 pooled window · 1,029 total players · 757 shown with 500+ possessions",
        path=output / "box15_inputs_2022_2026.png", columns=3,
    )
    plot_panel(
        rich, RICH_FEATURES, rich_summary,
        title="Five rich-SPM inputs",
        subtitle="Corrected 2022–26 pooled inputs · 757 players shown with 500+ possessions",
        path=output / "rich_spm_inputs_2022_2026.png", columns=2,
    )
    files = {}
    for path in sorted(output.iterdir()):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    run = {
        "run_id": output.name,
        "status": "data_quality_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "box_rows": int(len(box)),
            "rich_rows": int(len(rich)),
            "duplicate_keys": 0,
            "blocking_feature_failures": rich_summary.loc[rich_summary.quality_note.ne(""), "feature"].tolist(),
            "season_2027_loaded": False,
        },
        "files": files,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    print(main())
