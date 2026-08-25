#!/usr/bin/env python3
"""fig_experiment_scorecard: every experiment's retrodiction corr, one dot each."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import DIAGNOSTICS_DIR, OUTPUTS, ensure_dirs

ensure_dirs()
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DIAGNOSTICS_DIR / "experiments.csv")
df = df.dropna(subset=["margin_corr"])

CATEGORY = {
    "baseline": ("baseline_home",),
    "decay": ("decay_hl365", "decay_hl730", "decay_hl250", "decay_hl500"),
    "combo": ("combo_decay365_asym2000_4500", "combo_decay365_asym1500_6000", "combo_decay500_asym2000_4500"),
    "weights/pooling": ("soft_gt", "pool_250", "pool_500", "pool_1000"),
    "lambdas": ("asym_off1500_def6000", "asym_off6000_def1500", "asym_off2000_def4500", "asym_off4500_def2000"),
    "sparsity": ("enet_a1e-06_l10.15", "enet_a1e-05_l10.5"),
    "priors": ("prior_prev_s0.5", "prior_prev_s1.0", "prior_prev_s2.0", "prior_chain_depth3"),
    "window": ("win1_2023", "win5_2019_23"),
}
cat_of = {n: c for c, names in CATEGORY.items() for n in names}
df["cat"] = df["name"].map(cat_of).fillna("other")
df = df[df["margin_corr"] > 0.4]  # drop the broken enet fits, noted in caption
df = df.sort_values("margin_corr")

colors = {"baseline": "black", "decay": "tab:red", "combo": "tab:orange",
          "weights/pooling": "tab:gray", "lambdas": "tab:blue",
          "priors": "tab:purple", "window": "tab:green", "sparsity": "tab:brown", "other": "lightgray"}

fig, ax = plt.subplots(figsize=(8, 7))
baseline = float(df.loc[df["name"] == "baseline_home", "margin_corr"].iloc[0])
ax.axvline(baseline, color="black", lw=0.8, ls=":", label=f"baseline ({baseline:.3f})")
for _, r in df.iterrows():
    marker = "o" if r["anchors_ok"] in (True, "True") else "x"
    ax.scatter(r["margin_corr"], r["name"], color=colors.get(r["cat"], "gray"),
               s=45, marker=marker, zorder=3)
ax.set_xlabel("Next-season game-margin correlation (train 2021-23 -> test 2024, n=1,201 games)")
ax.set_title("Experiment scorecard: every variant tested, 2026-07-03\n"
             "(x marker = failed sign-anchor test; elastic-net runs corr<0.4 omitted)")
handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=k) for k, c in colors.items()
           if k in df["cat"].unique()]
ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
out = FIG_DIR / "fig_experiment_scorecard.png"
fig.savefig(out, dpi=150)
print(out)
