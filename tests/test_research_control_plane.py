from __future__ import annotations

import json

from nba_impact.research.control_plane import (
    validate_pinned_artifacts,
    validate_release_manifest,
)


def _contract(path, **overrides):
    artifact = {
        "api_field": "current_rapm_run_id",
        "artifact_id": "rapm_test",
        "artifact_relative_dir": "models/rapm/rapm_test",
        "estimand_id": "trailing_observed_lineup_rapm_v1",
        "evidence_status": "production_reference_method",
        "season_scope": "2024-2026",
        "season_completeness": "test",
        "uncertainty_status": "not_estimated",
        "config_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "data_hashes_status": "embedded_in_run_config",
        "forbidden_interpretation": "forecast",
    }
    artifact.update(overrides)
    path.write_text(
        json.dumps({"schema_version": "pinned_artifact_contracts_v1", "artifacts": [artifact]})
    )


def test_pinned_artifact_control_passes_complete_relative_lineage(tmp_path) -> None:
    contract = tmp_path / "contracts.json"
    _contract(contract)
    output = tmp_path / "models" / "rapm" / "rapm_test"
    output.mkdir(parents=True)
    (output / "run.json").write_text(
        json.dumps({"run_id": "rapm_test", "status": "research_frozen_baseline"})
    )
    assert validate_pinned_artifacts(contract, tmp_path) == []


def test_pinned_artifact_control_rejects_missing_hash_and_research_production(tmp_path) -> None:
    contract = tmp_path / "contracts.json"
    _contract(contract, config_sha256="missing")
    output = tmp_path / "models" / "rapm" / "rapm_test"
    output.mkdir(parents=True)
    (output / "run.json").write_text(
        json.dumps({"run_id": "rapm_test", "status": "research_only"})
    )
    codes = {item.code for item in validate_pinned_artifacts(contract, tmp_path)}
    assert {"invalid_hash", "research_exposed_as_production"} <= codes


def test_release_manifest_rejects_absolute_path_and_reserved_season(tmp_path) -> None:
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "nba_impact_release_v1",
                "created_at": "2026-08-12T00:00:00+00:00",
                "row_set_sha256": "c" * 64,
                "artifacts": [{"season_scope": "2027", "relative_path": "/Users/test/rating.parquet"}],
            }
        )
    )
    codes = {item.code for item in validate_release_manifest(path)}
    assert {"absolute_release_path", "reserved_season_leakage"} <= codes
