from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.time_decayed_trajectory import (
    build_filtered_time_decay,
    build_time_decayed_trajectory,
)


def _targets() -> pd.DataFrame:
    rows = []
    for season in range(2014, 2021):
        for player_id in (1, 2, 3):
            offense = 0.25 * (season - 2014) + player_id
            defense = -0.1 * (season - 2014) + 0.5 * player_id
            rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Season": season,
                    "target_offense": offense,
                    "target_defense": defense,
                    "target_net": offense + defense,
                    "Poss_Off": 1000.0 + 100.0 * player_id,
                    "Poss_Def": 1100.0 + 100.0 * player_id,
                }
            )
    return pd.DataFrame(rows)


def test_filtered_time_decay_uses_no_future_information_and_preserves_components() -> None:
    targets = _targets()
    original = build_filtered_time_decay(targets, decay=0.5, exposure_power=0.0)
    changed = targets.copy()
    changed.loc[changed["Season"].eq(2020), "target_offense"] += 1000.0
    changed["target_net"] = changed["target_offense"] + changed["target_defense"]
    revised = build_filtered_time_decay(changed, decay=0.5, exposure_power=0.0)
    before = original.loc[original["Season"].le(2019)].reset_index(drop=True)
    after = revised.loc[revised["Season"].le(2019)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    assert np.allclose(
        original["filtered_net"],
        original["filtered_offense"] + original["filtered_defense"],
    )
    player_one = original.loc[original["PLAYER_ID"].eq(1)].set_index("Season")
    assert player_one.loc[2014, "filtered_offense"] == 1.0
    assert player_one.loc[2015, "filtered_offense"] == 1.0 + (0.25 / 1.5)


def test_time_decayed_trajectory_builds_forward_evaluation_artifacts(tmp_path: Path) -> None:
    targets = _targets()
    targets_path = tmp_path / "targets.parquet"
    names_path = tmp_path / "names.csv"
    targets.to_parquet(targets_path, index=False)
    pd.DataFrame(
        {"PLAYER_ID": [1, 2, 3], "PLAYER_NAME": ["One", "Two", "Three"]}
    ).to_csv(names_path, index=False)
    run = build_time_decayed_trajectory(
        targets_path,
        names_path,
        artifact_root=tmp_path,
        candidate_decays=(0.5, 0.8),
        candidate_exposure_powers=(0.0, 1.0),
        selection_origins=(2016, 2017),
        diagnostic_origins=(2018, 2019),
        minimum_side_possessions=1000.0,
    )
    output = Path(run["artifact_path"])
    trajectory = pd.read_parquet(output / "trajectories.parquet")
    metrics = pd.read_parquet(output / "forward_metrics.parquet")
    assert (output / "selection_candidates.parquet").exists()
    assert json.loads((output / "run.json").read_text())["run_id"] == run["run_id"]
    assert trajectory["PLAYER_NAME"].notna().all()
    assert not trajectory.duplicated(["PLAYER_ID", "Season"]).any()
    assert np.allclose(
        trajectory["filtered_net"],
        trajectory["filtered_offense"] + trajectory["filtered_defense"],
    )
    assert {"selection", "diagnostic"} == set(metrics["scope"])
    assert run["metrics"]["maximum_component_identity_error"] < 1e-12
