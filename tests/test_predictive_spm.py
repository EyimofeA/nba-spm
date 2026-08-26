"""Contract and scoring checks for the predictive SPM experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.predictive_spm import (
    _load_contract,
    _predictive_metrics,
    _score_fold,
    _validate_pinned_inputs,
    build_predictive_spm,
)


def _contract_text(
    *,
    diagnostic: tuple[int, ...] = (2019, 2020),
    confirmation: tuple[int, ...] = (2021,),
    untouched: tuple[int, ...] = (2027,),
) -> str:
    return f"""
schema_version: experiment_preregistration_v1
experiment_id: predictive_spm_v1
status: preregistered
estimand_id: next_season_annual_impact_v1
data_contract:
  features: feature_run
  targets: target_run
  feature_reference_run: reference_run
  diagnostic_folds: {list(diagnostic)}
  confirmation_folds: {list(confirmation)}
  untouched_confirmation_seasons: {list(untouched)}
"""


def test_contract_parses_yaml_and_enforces_exact_folds(tmp_path) -> None:
    path = tmp_path / "contract.yml"
    path.write_text(_contract_text())

    contract = _load_contract(path, (2019, 2020, 2021))

    assert contract["estimand_id"] == "next_season_annual_impact_v1"
    with pytest.raises(ValueError, match="exactly match"):
        _load_contract(path, (2019, 2020))


def test_reserved_season_is_rejected_before_data_access(tmp_path, monkeypatch) -> None:
    path = tmp_path / "contract.yml"
    path.write_text(
        _contract_text(diagnostic=(2019,), confirmation=(2027,), untouched=(2027,))
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("data was read before the season guard"),
    )

    with pytest.raises(ValueError, match="must remain untouched"):
        build_predictive_spm(
            tmp_path / "feature_run" / "features.parquet",
            tmp_path / "target_run" / "targets.parquet",
            tmp_path / "reference_run",
            path,
            artifact_root=tmp_path / "artifacts",
            output_seasons=(2019, 2027),
        )


def test_pinned_inputs_must_match_manifests(tmp_path) -> None:
    path = tmp_path / "contract.yml"
    path.write_text(_contract_text())
    contract = _load_contract(path, (2019, 2020, 2021))
    for run_id in ("feature_run", "target_run", "reference_run"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({"run_id": run_id}))

    _validate_pinned_inputs(
        contract,
        tmp_path / "feature_run" / "features.parquet",
        tmp_path / "target_run" / "targets.parquet",
        tmp_path / "reference_run",
    )
    (tmp_path / "feature_run" / "run.json").write_text(
        json.dumps({"run_id": "wrong_feature_run"})
    )
    with pytest.raises(ValueError, match="features must be"):
        _validate_pinned_inputs(
            contract,
            tmp_path / "feature_run" / "features.parquet",
            tmp_path / "target_run" / "targets.parquet",
            tmp_path / "reference_run",
        )


def test_repository_contract_uses_exact_artifact_ids() -> None:
    contract = _load_contract(
        "research/experiments/predictive_spm_v1.yml",
        tuple(range(2019, 2027)),
    )

    assert contract["data_contract"]["features"] == "statistical_features_v2_b808fc1bf1"
    assert contract["data_contract"]["targets"] == "canonical_annual_target_panel_v1_2d9ff74ca3"


def test_saved_predictive_spm_manifests_use_portable_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "artifacts/models/predictive_spm").glob("*/run.json")):
        run = json.loads(path.read_text())
        assert str(root) not in json.dumps(run)
        for field in ("artifact_path", "predictions_path"):
            assert not Path(run[field]).is_absolute()
        if "checkpoint_path" in run:
            assert not Path(run["checkpoint_path"]).is_absolute()


def test_every_arm_scores_the_same_player_seasons() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "sample_weight": [1.0, 2.0, 3.0],
            "target_offense": [1.0, 2.0, 3.0],
            "target_defense": [0.5, 0.0, -0.5],
            "target_net": [1.5, 2.0, 2.5],
            "persistence_net": [1.0, np.nan, 3.0],
            "raw_offense": [0.8, 2.1, 2.7],
            "raw_defense": [0.4, 0.1, -0.4],
            "raw_net": [1.2, 2.2, 2.3],
            "calibrated_offense": [0.9, 2.0, 2.8],
            "calibrated_defense": [0.5, 0.0, -0.3],
            "calibrated_net": [1.4, 2.0, 2.5],
        }
    )

    scored, metrics = _score_fold(
        frame, season=2021, train_seasons=(2015, 2016, 2017), calibration_ok=True
    )

    assert scored["scored_common"].tolist() == [True, False, True]
    assert scored["evaluation_status"].tolist() == ["included", "excluded", "included"]
    assert scored["exclusion_reason"].tolist() == [
        "",
        "missing_prior_season_persistence",
        "",
    ]
    assert {row["rows"] for row in metrics} == {2}
    assert {row["evaluation_rows_before_common_filter"] for row in metrics} == {3}


def test_predictive_metrics_use_weights_and_report_calibration() -> None:
    actual = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.array([0.0, 2.0, 1.0, 4.0])
    weight = np.array([100.0, 1.0, 1.0, 1.0])

    metrics = _predictive_metrics(actual, prediction, weight)

    assert metrics["weighted_correlation"] != pytest.approx(metrics["correlation"])
    assert np.isfinite(metrics["calibration_slope"])
    assert np.isfinite(metrics["calibration_intercept"])
