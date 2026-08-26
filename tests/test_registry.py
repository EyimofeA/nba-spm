"""Registry compatibility checks."""

from __future__ import annotations

import duckdb
import pytest

from nba_impact.registry import register_model_run


def _run() -> dict:
    return {
        "run_id": "run_v1",
        "model_family": "test",
        "estimand_id": "test_estimand_v1",
        "status": "research",
        "created_at": "2026-08-26T00:00:00+00:00",
        "config": {},
        "metrics": {},
        "artifact_path": "/tmp/run_v1",
    }


def test_register_model_run_accepts_estimand_id(tmp_path) -> None:
    path = tmp_path / "registry.duckdb"

    register_model_run(path, _run())

    with duckdb.connect(str(path), read_only=True) as connection:
        row = connection.execute(
            "SELECT estimand FROM model_runs WHERE run_id = 'run_v1'"
        ).fetchone()
    assert row == ("test_estimand_v1",)


def test_register_model_run_rejects_missing_estimand(tmp_path) -> None:
    run = _run()
    run.pop("estimand_id")

    with pytest.raises(ValueError, match="requires estimand"):
        register_model_run(tmp_path / "registry.duckdb", run)
