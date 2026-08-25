#!/usr/bin/env python3
"""fig_decay_shape: learned model-free bucket weights vs exponential decay fits."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import DIAGNOSTICS_DIR, OUTPUTS, ensure_dirs

ensure_dirs()
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

d0 = json.loads((DIAGNOSTICS_DIR / "decay_buckets_best.json").read_text())
mids = np.array(d0["bucket_mid_days"])
wts = np.array(d0["best_weights"])

t = np.linspace(0, mids.max() + 60, 300)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, 0.5 ** (t / 250), color="black", lw=2, label="exponential hl=250d (champion)")
ax.plot(t, 0.5 ** (t / 365), color="gray", lw=1.2, ls="--", label="exponential hl=365d")
ax.scatter(mids, wts, s=70, color="tab:red", zorder=3, label="learned free weights (6 buckets)")
for x, y in zip(mids, wts):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="tab:red")
ax.set_xlabel("Possession age (days before end of training window)")
ax.set_ylabel("Possession weight")
ax.set_title("Decay function: model-free bucket weights vs exponential fits\n"
             "(train 2021-23, tuned on 2024 game-margin retrodiction, n=620,236 possessions)")
ax.set_ylim(0, 1.35)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = FIG_DIR / "fig_decay_shape.png"
fig.savefig(out, dpi=150)
print(out)
