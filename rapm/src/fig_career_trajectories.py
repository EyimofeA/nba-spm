#!/usr/bin/env python3
"""fig_career_trajectories: small-multiple RAPM careers with 95% CI bands,
from the final 26-window production panel."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import OUTPUTS, RAPM_RESULTS, ensure_dirs

ensure_dirs()
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

panel = pd.read_csv(RAPM_RESULTS / "final_20260703" / "rapm_all_windows.csv")

PLAYERS = ["LeBron James", "Nikola Jokic", "Tim Duncan", "Stephen Curry",
           "Kevin Garnett", "Chris Paul", "Dirk Nowitzki", "Rudy Gobert", "James Harden"]

fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharey=True)
for ax, name in zip(axes.ravel(), PLAYERS):
    d = panel[panel["Name"] == name].sort_values("Window_End")
    d = d[(d["Poss_Off"] > 2000)]
    ax.axhline(0, color="gray", lw=0.5)
    ax.fill_between(d["Window_End"], d["RAPM_CI_lo"], d["RAPM_CI_hi"],
                    alpha=0.2, color="tab:blue", lw=0)
    ax.plot(d["Window_End"], d["RAPM"], color="tab:blue", lw=1.6)
    ax.set_title(name, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
fig.suptitle("Career RAPM trajectories, 3-year rolling windows 1997-2024 (shaded = 95% CI)\n"
             "final_20260703 panel; windows with <2000 offensive possessions omitted", fontsize=11)
fig.supylabel("RAPM per 100 possessions", fontsize=10)
fig.supxlabel("Window end season", fontsize=10)
fig.tight_layout()
out = FIG_DIR / "fig_career_trajectories.png"
fig.savefig(out, dpi=150)
print(out)
