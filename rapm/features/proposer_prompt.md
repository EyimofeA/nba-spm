# LLM proposer hints

Target: beat frozen minutes prior on f24/f23 gate scores (currently ~0.7335 / ~0.6953 at c=4).

**Do:**
- Ratios and interactions among per-100 box stats (USG, AST%, REB%, TS%, etc.)
- Role composites (playmaker, rim protector, spacer)
- Window z-scores within Window_End cohort
- Age / experience curves when Age or GP available
- Tracking-derived rates when columns exist (drives, touches, passing)

**Don't:**
- Leak labels (Off, Def, RAPM, OnOffRtg, OnDefRtg, plus-minus)
- Re-use gen_006 columns verbatim — mutate or combine differently
- More than 15 features

Recent winner: gen_006 `derived_box_v1` — 11 ratio/z-score features beat minutes on all folds.
