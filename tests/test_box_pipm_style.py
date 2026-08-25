from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    build_box_pipm_style_baseline,
)


def test_box_pipm_style_is_loso_and_component_additive(tmp_path) -> None:
    feature_rows = []
    target_rows = []
    for season in range(2017, 2021):
        for player in range(1, 9):
            row = {"PLAYER_ID": player, "Season": season}
            for index, feature in enumerate(BOX_PIPM_STYLE_FEATURES):
                row[feature] = player + season * 0.01 + index * 0.001
            feature_rows.append(row)
            offense = 0.12 * player + 0.02 * (season - 2017)
            defense = -0.08 * player + 0.01 * (season - 2017)
            target_rows.append(
                {
                    "PLAYER_ID": player,
                    "Season": season,
                    "target_offense": offense,
                    "target_defense": defense,
                    "target_net": offense + defense,
                    "Poss_Off": 1000 + player,
                    "Poss_Def": 1000 + player,
                }
            )
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.parquet"
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(target_rows).to_parquet(targets_path, index=False)

    run = build_box_pipm_style_baseline(
        features_path,
        targets_path,
        artifact_root=tmp_path / "artifacts",
        output_seasons=(2017, 2018, 2019, 2020),
        alpha_grid=(1.0, 10.0),
    )

    output = tmp_path / "artifacts" / "models" / "box_pipm_style" / run["run_id"]
    predictions = pd.read_parquet(output / "oof_predictions.parquet")
    assert len(predictions) == 32
    assert not predictions.duplicated(["PLAYER_ID", "Season"]).any()
    assert np.allclose(
        predictions["box_pipm_style_offense"] + predictions["box_pipm_style_defense"],
        predictions["box_pipm_style_net"],
    )
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["config"]["features"] == list(BOX_PIPM_STYLE_FEATURES)
    assert "not a replication" in manifest["caveats"][0].lower()
