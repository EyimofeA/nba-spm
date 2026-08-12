from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.annual_spm_priors import (
    build_forward_chained_annual_spm_priors,
)


def test_forward_priors_do_not_use_future_targets(tmp_path: Path, monkeypatch) -> None:
    seasons = range(2014, 2019)
    features = pd.DataFrame(
        [
            {
                "PLAYER_ID": player,
                "Window_End": season,
                "box_feature": float(player + season % 3),
                "zts_pct_points": float(player) / 10,
                "def_feature": float(player - season % 2),
                "dfg_attempts_p100": float(player),
            }
            for season in seasons
            for player in range(1, 9)
        ]
    )
    targets = features[["PLAYER_ID", "Window_End"]].rename(
        columns={"Window_End": "Season"}
    )
    targets["target_offense"] = targets["PLAYER_ID"] * 0.2
    targets["target_defense"] = targets["PLAYER_ID"] * -0.1
    targets["target_net"] = targets["target_offense"] + targets["target_defense"]
    targets["Poss_Off"] = 500
    targets["Poss_Def"] = 500
    feature_path = tmp_path / "features.parquet"
    target_path = tmp_path / "targets.parquet"
    features.to_parquet(feature_path, index=False)
    targets.to_parquet(target_path, index=False)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "run.json").write_text("{}")
    contract = {
        "contract_version": "test",
        "status": "frozen_research_contract",
        "estimand": "test",
        "components": {
            "offense": {"learner": "ridge", "additional_features": ["zts_pct_points"]},
            "defense": {"learner": "ridge", "additional_features": ["dfg_attempts_p100"]},
        },
        "validation_rules": {"rapm_scale_search_allowed": False},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(
        "nba_impact.models.annual_spm_priors._selected_single_season_features",
        lambda _: {"offense": ("box_feature",), "defense": ("def_feature",)},
    )

    first = build_forward_chained_annual_spm_priors(
        feature_path,
        target_path,
        reference,
        contract_path,
        artifact_root=tmp_path / "a",
        output_seasons=(2017,),
    )
    changed = targets.copy()
    changed.loc[changed["Season"].eq(2018), ["target_offense", "target_defense"]] = 9999
    changed["target_net"] = changed["target_offense"] + changed["target_defense"]
    changed_path = tmp_path / "changed_targets.parquet"
    changed.to_parquet(changed_path, index=False)
    second = build_forward_chained_annual_spm_priors(
        feature_path,
        changed_path,
        reference,
        contract_path,
        artifact_root=tmp_path / "b",
        output_seasons=(2017,),
    )

    left = pd.read_parquet(first["priors_path"])
    right = pd.read_parquet(second["priors_path"])
    pd.testing.assert_frame_equal(left, right)
    assert left["spm_training_end"].eq(2016).all()
    assert json.loads(Path(first["artifact_path"], "run.json").read_text())["run_id"] == first["run_id"]
