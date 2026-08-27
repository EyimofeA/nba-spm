# Full SPM Features, 2014–2026

## Result

Run `full_spm_features_2014_2026_v1_4c77ae6acc` supplies the exact frozen
127-offense and 68-defense feature contract for every season from 2014 through
2026. It contains 6,942 annual player-seasons and 8,620 rolling five-year rows.
Complete five-year windows end from 2018 through 2026.

This is a validated research input. It does not promote or overwrite a public
SPM or AIO rating.

## Sources

- Player sheets: pinned Gabriel `player_sheets` revision `54b57cf`, 2012–26.
  Seasons 2012–13 provide only the lag history needed to engineer 2014 without
  future data.
- Playtype and zTS: Gabriel playtype files, 2014–26.
- DFG and rim defense: observed player-sheet fields, 2014–26.
- Hustle: Gabriel hustle source, 2018–26.
- Matchup defense: pinned `shufinskiy/nba_data` revision
  `e829d4678be1e075f99e5d41a1c5f97089be446b`, 2018–26.

The matchup source contains all 1,230 regular-season games for 2026. It has
240,839 raw matchup rows, no duplicate source keys, no point-reconstruction
failures, and a 0.99994 correlation with player-sheet defensive exposure.

## Coverage

Coverage is the share of annual player-sheet rows with an observed source row.
Playtype applies its source eligibility rules. DFG and rim sources omit players
without an observed defended-shot row.

| Season | Player sheet | Playtype | DFG | Rim DFG | Hustle | Matchup |
|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 1.000 | .759 | .996 | .977 | — | — |
| 2015 | 1.000 | .787 | .996 | .986 | — | — |
| 2016 | 1.000 | .803 | .994 | .987 | — | — |
| 2017 | 1.000 | .767 | .996 | .990 | — | — |
| 2018 | 1.000 | .713 | .991 | .970 | 1.000 | 1.000 |
| 2019 | 1.000 | .747 | .987 | .975 | 1.000 | .998 |
| 2020 | 1.000 | .743 | .996 | .991 | 1.000 | .972 |
| 2021 | 1.000 | .739 | .994 | .991 | 1.000 | 1.000 |
| 2022 | 1.000 | .698 | .983 | .960 | 1.000 | 1.000 |
| 2023 | 1.000 | .746 | .991 | .981 | 1.000 | 1.000 |
| 2024 | 1.000 | .689 | .995 | .983 | 1.000 | 1.000 |
| 2025 | 1.000 | .740 | .998 | .988 | .996 | 1.000 |
| 2026 | 1.000 | .722 | .998 | .993 | .998 | 1.000 |

Hustle and matchup assignments are unavailable before 2018. The builder does
not label those seasons as observed. Player-level low-opportunity fields retain
missing values. The fitted pipelines impute medians from their training folds.

## QA

- Annual seasons: 2014–26, with no missing season.
- Five-year window ends: 2018–26, with no missing window.
- Duplicate player-season keys: 0.
- Infinite selected values: 0.
- Season 2027 rows: 0.
- Offense features: 127 exact contract fields.
- Defense features: 68 exact contract fields.

The matchup exposure audit first exposed duplicate season-total rows in the
Parquet player sheets. The builder now removes identical `(PLAYER_ID, DefPoss)`
rows before it sums distinct stints. The 2018 exposure correlation increased
from 0.23 to 0.99981 after this correction.

## Rebuild

Run:

```bash
uv run python research/build_full_spm_features_2014_2026.py
```

The result stores relative artifact paths and SHA-256 hashes for every direct
input. Raw source rows do not enter a website release bundle.
