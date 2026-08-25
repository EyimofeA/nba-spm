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
    season: Path(
        "data/lake/silver/candidates/historical_official_preferred_2017_2023/"
        f"attached/project_season={season}"
    )
    for season in range(2017, 2024)
}
run_matched_comparison(
    roots,
    "rapm/data/possession_cache",
    "data/lake/bronze/official_game_scores/official_game_scores.parquet",
    "artifacts",
)
```

The checked run is version `historical_matched_rapm_v1_1888b9164f9d`.
It writes `season_coverage.parquet`, `rating_comparison.parquet`,
`rating_comparison_by_exposure.parquet`, `game_margin_comparison.parquet`,
`game_splits.parquet`, and `run.json` under the matching directory in
`artifacts/research/historical_matched_rapm/`. The run hashes its comparison
code, RAPM code, official-score input, candidate possession and segment inputs,
and both candidate QA ledgers.

## Results

All candidate games were present in the legacy cache after regular-season
filtering.  The candidate is a strict subset of the available legacy cache
because V3 has not passed every historical game through its lineup gates.

| season | matched games | V3 possessions | legacy possessions | points delta (V3 - legacy) |
|---:|---:|---:|---:|---:|
| 2017 | 996 | 194,364 | 192,164 | +1,843 |
| 2018 | 1,051 | 206,981 | 204,839 | +1,718 |
| 2019 | 1,135 | 229,686 | 227,708 | +1,961 |
| 2020 | 949 | 192,877 | 190,789 | +1,897 |
| 2021 | 892 | 179,181 | 177,240 | +1,591 |
| 2022 | 1,084 | 215,543 | 213,533 | +1,848 |
| 2023 | 1,143 | 229,514 | 226,895 | +2,192 |

Matched-player agreement is high but not exact.  Across seasons, net-rating
Pearson correlation is 0.971–0.981 and net-rating RMSE is 0.299–0.355 points
per 100.  Net Spearman correlation is 0.965–0.974.  Defense agreement is
slightly weaker than offense in 2020 and 2023.

The artifact contains the complete per-season offense, defense, and net
correlations, RMSE values, and rank correlations.  Do not summarize these as
validation of V3 accuracy.  They measure source sensitivity under different
possession boundaries and lineup assignments.

The agreement is not an artifact of low-exposure shrinkage. Requiring at least
2,000 possessions on offense and defense in both sources leaves 210--262
players per season. Net Pearson correlation remains 0.969--0.982 and net
Spearman correlation remains 0.963--0.977. Net RMSE is 0.360--0.418 points per
100 at that threshold.

## Held-out game metric

The same source-specific design supports one persisted chronological within-
season split. The last 20% of matched games are held out. Both sources use the
same game IDs and are scored against the same official final-score margin. A
source is fit on its earlier rows and predicts the held-out games using their
observed lineups. This is a lineup-conditioned retrodiction, not a forecast of
a game before lineups are known. The artifact reports RMSE, MAE, margin
correlation, calibration, unknown-player exposure, and each source's margin-
reconstruction error.

This metric is useful for checking whether a source changes the fit's
retrodictive behavior.  It is not a promotion gate.  The V3 and legacy
possession totals differ on the same games, so a difference in game RMSE cannot
be attributed to lineup quality alone.

V3 reconstructs official margins more accurately: held-out reconstruction RMSE
is 0.57--0.97 points per game, versus 1.29--1.97 for legacy. That supports the
V3 event parser's score contract. It does not make V3 RAPM more predictive. V3
has lower held-out prediction RMSE in two of seven seasons. Its mean prediction
RMSE is 0.080 points per game higher, mean MAE is 0.050 higher, and mean margin
correlation is 0.012 lower.

## Boundary and next step

The result supports a narrow conclusion: the two terminal-lineup sources give
similar player ordering on their matched historical games, with meaningful but
small rating differences. It does not prove that either source has the right
within-possession lineup timing, and it does not support a predictive promotion.
Keep V3 as a versioned research candidate. The official player-game cache is
complete, but the matched prediction gate does not support a source promotion.
Use the legacy source as the public historical reference and keep this
comparison and its hashes as a provenance record.
