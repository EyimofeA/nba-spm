# Feature Foundry — chosen features by generation

After each gen completes, which inputs were used and whether search folds improved.


## Gen 0 — minutes_prior_baseline (2026-07-03)

**Prior:** minutes · minutes prior c=2 harness repro

**Features chosen:** Possession-minutes proxy only (log poss + quadratic per side). No box/tracking.

- **minutes_shape** (4): log1p(off_poss), log1p(off_poss)², log1p(def_poss), log1p(def_poss)²

**Gates:** f24=0.7346 · f23=0.6954 · vault=0.6481 · status=keep

**Improvement:** beats minutes on f24+f23

---

## Policy (2026-07-03)

Minutes baseline is **frozen** — do not re-run as foundry gens. **Autoresearch = `candidates/gen_NNN/build.py`** (new columns). Gen 002–005 were mistaken column ablations, not new features.

---

## Gen 6 — derived_box_v1 (2026-07-03)

**Prior:** spmv2_subset · NEW engineered: ratios, interactions, window z-scores (not base parquet cols)

**Features chosen:** SPM subset (56 cols): tier1=28, tier2=24, derived=4

- **tier1_box** (28): OffPoss, DefPoss, USG_PCT, TS_PCT, AGE, AtRimFrequency, AtRimAccuracy, Corner3Frequency, Arc3Frequency, Assisted2sPct, … (+18 more)
- **tier2_tracking** (24): AVG_SEC_PER_TOUCH, AVG_DRIB_PER_TOUCH, ShotQualityAvg, DRIVES_p100, DRIVE_PTS_p100, DRIVE_AST_p100, DRIVE_TOV_p100, TOUCHES_p100, PAINT_TOUCHES_p100, POST_TOUCHES_p100, … (+14 more)
- **derived** (4): log_OffPoss, log_OffPoss_sq, log_DefPoss, log_DefPoss_sq

**Gates:** f24=0.7456 · status=keep

**Improvement:** does not beat minutes on both search folds

---

## Gen 6 — derived_box_v1 (2026-07-03)

**Prior:** spmv2_subset · NEW engineered: ratios, interactions, window z-scores (not base parquet cols)

**Features chosen:** NEW built features (11): new_ast_tov_ratio, new_def_event_rate, new_fta_rate, new_rim_vs_three, new_self_create_usg, new_efficiency_usage, new_live_tov_usg, new_usg_z, new_ts_z, new_rim_freq_z…

- **derived** (11): new_ast_tov_ratio, new_def_event_rate, new_fta_rate, new_rim_vs_three, new_self_create_usg, new_efficiency_usage, new_live_tov_usg, new_usg_z, new_ts_z, new_rim_freq_z, … (+1 more)

**Gates:** f23=0.7275 · status=keep

**Improvement:** does not beat minutes on both search folds

---

## Gen 6 — derived_box_v1 CORRECTED

**Features chosen:** 11 NEW engineered (ratios, z-scores) — see results.tsv

**Best gates:** f24=0.7456 (c=4) · f23=0.7275 (c=4)

**Improvement:** BEATS frozen minutes on both search folds (0.7335 / 0.6953).

**Vault (c=4):** 0.7051 — keep (minutes vault was 0.6481 discard). First new-feature candidate to clear all three folds.

---