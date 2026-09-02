# Replication registry

CourtSignal separates exact output checks from methodology-aligned reconstructions and proxies.

| Metric | Status | What passed | What did not pass |
| --- | --- | --- | --- |
| BPM 2.0 | Methodology-aligned reconstruction | The public benchmark contains offense, defense, and net values on the intended scale. | The build does not reproduce every hidden source transformation. |
| Box PIPM | Methodology-aligned reconstruction | The local formula uses the documented compact box inputs and separate offense and defense mappings. | Box PIPM is not full PIPM. |
| Full PIPM | Published-output comparison | The reference comparison has 1,106 matched player-seasons across 2021–2023. | CourtSignal does not claim an exact full-PIPM recreation. |
| RAPTOR | Official output identity plus local proxy | The official public CSV reproduces exactly across 4,684 rows and all three components. | The local on/off model remains a proxy for unpublished RAPTOR coefficients. |
| DARKO WOWY | Exact public aggregation on the downloaded sample | Season averages reproduce the downloaded player-game rows to floating-point tolerance. | The downloaded sample covers 52 matched rows, about 0.95% of the published table. It does not reproduce DARKO or DPM. |
| xRAPM | Published-output comparison | The comparison uses the saved xRAPM table with player identity checks. | Correlation does not establish identical methodology or predictive validity. |

The public comparison panel uses the external benchmark run `external_all_in_one_benchmark_v2_e6817bf1fa`. It includes strict common coverage, maximal coverage, matched counts, player-season correlations, and 5,000 paired game bootstrap draws.

## Naming rule

CourtSignal may use `exact replication` only when a saved output matches its public source at the stated grain. It uses `methodology-aligned reconstruction` when the documented formula is available but the complete original pipeline is not. It uses `proxy` when the model follows a broad idea without reproducing the original estimator.
