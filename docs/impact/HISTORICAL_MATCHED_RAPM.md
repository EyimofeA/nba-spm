# Matched historical RAPM comparison

This is a research-only comparison of the historical V3 possession and
terminal-lineup candidate against the legacy terminal-lineup possession cache.
It does not replace the canonical RAPM input or create a public rating.

## Frozen design

The experiment uses the exact intersection of regular-season NBA game IDs in
each source.  Regular season is enforced by the `002` game-ID prefix.  The
candidate uses the last ordinal lineup segment for each possession.  The
legacy source already stores one terminal lineup per possession.

Each source is fit separately with the same zero-prior ridge model:

* five offense player indicators;
* five defense player indicators;
* one home-court indicator;
* points scored on the possession as the target;
* offense penalty 3000;
* defense penalty 3000;
* home-court penalty 300.

The fit is centered after estimation by the possession-weighted offense and
defense means.  Ratings are reported in points per 100 possessions.  Positive
defense means fewer points allowed.

## Reproducible run

The run is produced by
`src/nba_impact/models/historical_matched_rapm.py`:

```python
from pathlib import Path
from nba_impact.models.historical_matched_rapm import run_matched_comparison

roots = {
    season: Path(f"data/lake/silver/candidates/historical_v3_{season}_regular_main")
    for season in range(2017, 2024)
}
run_matched_comparison(roots, "rapm/data/possession_cache", "artifacts")
```

The checked run is version `historical_matched_rapm_v1_5cdea91ec419`.
It writes `season_coverage.parquet`, `rating_comparison.parquet`,
`game_margin_comparison.parquet`, and `run.json` under the matching directory
in `artifacts/research/historical_matched_rapm/`.

## Results

All candidate games were present in the legacy cache after regular-season
filtering.  The candidate is a strict subset of the available legacy cache
because V3 has not passed every historical game through its lineup gates.

| season | matched games | V3 possessions | legacy possessions | points delta (V3 - legacy) |
|---:|---:|---:|---:|---:|
| 2017 | 941 | 183,171 | 181,044 | +1,809 |
| 2018 | 1,014 | 199,696 | 197,616 | +1,634 |
| 2019 | 1,139 | 230,514 | 228,515 | +1,982 |
| 2020 | 949 | 192,896 | 190,767 | +1,926 |
| 2021 | 892 | 179,181 | 177,240 | +1,591 |
| 2022 | 1,077 | 214,138 | 212,130 | +1,843 |
| 2023 | 1,124 | 225,784 | 223,211 | +2,156 |

Matched-player agreement is high but not exact.  Across seasons, net-rating
Pearson correlation is 0.971–0.981 and net-rating RMSE is 0.298–0.350 points
per 100.  Net Spearman correlation is 0.964–0.975.  Defense agreement is
slightly weaker than offense in 2020 and 2023.

The artifact contains the complete per-season offense, defense, and net
correlations, RMSE values, and rank correlations.  Do not summarize these as
validation of V3 accuracy.  They measure source sensitivity under different
possession boundaries and lineup assignments.

## Held-out game metric

The same source-specific design supports a chronological within-season split.
The last 20% of matched games are held out.  A source is fit on its earlier
rows and predicts the held-out games using their observed lineups.  This is a
lineup-conditioned retrodiction, not a forecast of a game before lineups are
known.  The artifact reports RMSE, MAE, margin correlation, calibration, and
unknown-player exposure for both sources.

This metric is useful for checking whether a source changes the fit's
retrodictive behavior.  It is not a promotion gate.  The V3 and legacy
possession totals differ on the same games, so a difference in game RMSE cannot
be attributed to lineup quality alone.

The V3 candidate had lower held-out margin RMSE in one of seven seasons. Its
mean RMSE was 0.118 points per game higher, its mean MAE was 0.112 points per
game higher, and its mean margin correlation was 0.011 lower. These are small
source-sensitivity differences, but they provide no evidence that the V3
candidate is a more predictive RAPM input.

## Boundary and next step

The result supports a narrow conclusion: the two terminal-lineup sources give
similar player ordering on their matched historical games, with meaningful but
small rating differences. It does not prove that either source has the right
within-possession lineup timing, and it does not support a predictive promotion.
Keep V3 as a versioned research candidate while the full official player-game
cache is completed. Use the legacy source as the public historical reference
until the official-only rebuild reproduces these gates. Keep this comparison
and its hashes as a provenance record.
