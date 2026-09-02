# Matchup source audit

The official NBA Stats 2026 matchup feed contains 240,750 unique game-scorer-defender rows across all 1,230 regular-season games. Its file hash is `93c60e478a768662e7e6e720e2d064b1c1ba2746b04afcad4451cde61f0802f9`.

The archived 2026 feed contains 240,839 rows. The sources match on 240,693 keys. The archive has 146 unmatched rows. The official feed has 57 unmatched rows.

| Field | Exact match rate | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Partial possessions | 98.453% | 0.0057 | 7.5 |
| Player points | 99.889% | 0.0023 | 8.0 |
| Field-goal attempts | 99.745% | 0.0026 | 3.0 |
| Turnovers | 99.962% | 0.0004 | 1.0 |
| Assists | 99.952% | 0.0005 | 2.0 |

CourtSignal uses the official feed for 2026. It keeps the archive as a reproducibility check. Both sources contain aggregated scorer-listed-defender assignments. Neither source identifies the defender for each shot event. CourtSignal does not join shooter-only shot quality to an assigned defender.
