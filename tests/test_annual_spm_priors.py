from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.annual_spm_priors import (
    build_forward_chained_annual_spm_priors,
    build_leave_one_season_out_annual_spm_priors,
)


def test_oof_spm_predictions_become_annual_rapm_centers(tmp_path: Path) -> None:
    source = tmp_path / "single_season_spm" / "source"
    source.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 1, 2],
            "Season": [2023, 2023, 2024, 2024],
            "spm_offense": [1.0, -1.0, 2.0, -2.0],
            "spm_defense": [0.5, -0.5, 0.25, -0.25],
            "spm_net": [1.5, -1.5, 2.25, -2.25],
        }
    )
    predictions.to_parquet(source / "oof_predictions.parquet", index=False)
    (source / "run.json").write_text(
        json.dumps(
            {
                "run_id": "source",
                "estimand": "annual impact",
                "config": {"training_seasons": [2022, 2023, 2024]},
            }
        )
    )

    run = build_leave_one_season_out_annual_spm_priors(
        source, artifact_root=tmp_path / "artifacts"
    )
    priors = pd.read_parquet(run["priors_path"])
    assert priors["Window_End"].tolist() == [2023, 2023, 2024, 2024]
    assert priors["spm_training_season_count"].eq(2).all()
    assert priors["spm_training_rule"].eq("leave_one_season_out").all()
    np.testing.assert_allclose(
        priors["prior_net_per_100"],
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"],
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


def test_forward_priors_can_use_only_the_recent_training_window(tmp_path: Path, monkeypatch) -> None:
    seasons = range(2014, 2020)
    features = pd.DataFrame(
        {
            "PLAYER_ID": [player for season in seasons for player in range(1, 5)],
            "Window_End": [season for season in seasons for _ in range(1, 5)],
            "box_feature": [float(player) for _ in seasons for player in range(1, 5)],
            "def_feature": [float(player) for _ in seasons for player in range(1, 5)],
        }
    )
    targets = features[["PLAYER_ID", "Window_End"]].rename(columns={"Window_End": "Season"})
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
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({
        "contract_version": "test", "status": "frozen_research_contract", "estimand": "test",
        "components": {"offense": {"learner": "ridge", "additional_features": []}, "defense": {"learner": "ridge", "additional_features": []}},
        "validation_rules": {"rapm_scale_search_allowed": False},
    }))
    monkeypatch.setattr(
        "nba_impact.models.annual_spm_priors._selected_single_season_features",
        lambda _: {"offense": ("box_feature",), "defense": ("def_feature",)},
    )
    run = build_forward_chained_annual_spm_priors(
        feature_path, target_path, reference, contract_path,
        artifact_root=tmp_path / "artifacts", output_seasons=(2019,), train_window_seasons=3,
    )
    priors = pd.read_parquet(run["priors_path"])
    assert priors["spm_training_start"].eq(2016).all()
    assert priors["spm_training_end"].eq(2018).all()
    assert json.loads(Path(run["artifact_path"], "run.json").read_text())["config"]["train_window_seasons"] == 3
