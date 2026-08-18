from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.annual_rating_benchmark import build_annual_rating_benchmark


def test_annual_rating_benchmark_scores_identical_loso_rows(tmp_path) -> None:
    rows = []
    pipm_rows = []
    for season in (2021, 2022, 2023):
        for player in range(1, 7):
            off = player * 0.2 + season * 0.01
            defense = player * -0.1 + season * 0.005
            rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "Poss_Off": 1500,
                    "Poss_Def": 1600,
                    "sample_weight": 10.0,
                    "target_offense": off,
                    "target_defense": defense,
                    "target_net": off + defense,
                    "spm_offense": off,
                    "spm_defense": defense,
                    "spm_net": off + defense,
                    "bpm_offense": off * 2 + 1,
                    "bpm_defense": defense * 2 + 1,
                    "bpm_net": (off + defense) * 2 + 1,
                    "xrapm_offense": off * 3 - 2,
                    "xrapm_defense": defense * 3 - 2,
                    "xrapm_net": (off + defense) * 3 - 2,
                }
            )
            pipm_rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "box_pipm_style_offense": off * 0.9,
                    "box_pipm_style_defense": defense * 0.9,
                    "box_pipm_style_net": (off + defense) * 0.9,
                }
            )
    external_path = tmp_path / "external.parquet"
    pipm_path = tmp_path / "pipm.parquet"
    pd.DataFrame(rows).to_parquet(external_path, index=False)
    pd.DataFrame(pipm_rows).to_parquet(pipm_path, index=False)
    run = build_annual_rating_benchmark(
        external_path, pipm_path, artifact_root=tmp_path / "artifacts"
    )
    output = (
        tmp_path / "artifacts" / "models" / "annual_rating_benchmark" / run["run_id"]
    )
    metrics = pd.read_parquet(output / "fold_metrics.parquet")
    assert len(metrics) == 3 * 3 * 4
    assert set(metrics["candidate"]) == {"spm", "box_pipm_style", "bpm", "xrapm"}
    assert np.isclose(metrics.query("candidate == 'bpm'")["weighted_rmse"], 0.0).all()
    assert run["quality"]["matched_rows"] == 18
