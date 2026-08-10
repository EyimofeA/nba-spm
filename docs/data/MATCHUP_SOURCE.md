# Player Matchup Source

Updated 2026-08-10.

## Decision

Use the pinned `shufinskiy/nba_data` matchup archives as the new research source
for opponent-adjusted individual defense features. Do not use the empty Gabriel
defensive playtype files.

## Provenance

- Repository: https://github.com/shufinskiy/nba_data
- Pinned revision: `e829d4678be1e075f99e5d41a1c5f97089be446b`
- License: Apache-2.0
- Upstream description: NBA Stats player matchup data
- Manifest: `configs/ingest/shufinskiy_matchups_2017_2024.json`

The project season label is the season end. Upstream archive `matchups_2017`
therefore maps to project season 2018 (2017–18).

## Local coverage

Eight regular-season archives cover project seasons 2018–25. Total compressed
size is 30.38 MB. Validation finds 1,769,658 rows:

| Project season | Rows |
|---:|---:|
| 2018 | 231,888 |
| 2019 | 232,985 |
| 2020 | 181,840 |
| 2021 | 202,195 |
| 2022 | 229,901 |
| 2023 | 229,185 |
| 2024 | 230,703 |
| 2025 | 231,961 |

Every archive passes expected-byte, archive-member, minimum-row, CSV-width, and
required-column checks. Each local file has a sidecar manifest with its SHA-256,
retrieval time, row count, and full schema.

## Grain and fields

The grain is game, offensive player, and primary matchup defender. `person_id`
is the offensive player. `matchups_person_id` is the defender. Useful fields
include partial possessions, matchup minutes, player points, field goals,
three-pointers, turnovers, assists, shooting fouls, and help activity.

This is assignment data, not optical player coordinates. A matchup statistic
does not prove sole causal responsibility. Features must adjust for offensive
opponent quality and shrink small matchup samples.

## Download behavior

The generic ingest now validates `.tar.xz` members without extracting them. It
resumes `.partial` downloads, retries transient failures up to 20 times, writes
atomically, and skips checksum-valid existing files. A repeat run is read-only
apart from refreshed ingest summaries.

## Validated feature layer

Run `matchup_defense_features_v1_86d13d7357` converts all eight archives into
4,409 unique defender-seasons. It reconstructs every recorded point exactly.
It has no duplicate keys, negative exposures, null feature values, or non-finite
feature values. Exposure-weighted NBA player-ID coverage is 99.46%–99.95% by
season. Matchup exposure correlates 0.987–1.000 with the player-sheet defensive
possession field among matched IDs.

The primary research feature compares points allowed in each scorer-defender
pair with that scorer's leave-one-defender-out points rate. It then centers the
result within season and shrinks it toward zero using 500 assigned matchup
possessions. The panel also contains shot, turnover, assist, and shooting-foul
context. `matchup_possessions` is an audit field, not a model input.

Do not describe the result as causal points prevented. The assignment can encode
scheme, help, offensive role, and shot mix. The raw leaderboards put several rim
centers near the bottom, which is direct evidence that role remains in the
statistic after scorer-quality adjustment.
