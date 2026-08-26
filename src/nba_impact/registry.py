"""Small DuckDB registry for immutable datasets and model runs."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


def initialize_registry(path: str | Path) -> Path:
    registry = Path(path)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(registry)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                snapshot_id VARCHAR PRIMARY KEY,
                dataset_name VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                passed BOOLEAN NOT NULL,
                manifest_json JSON NOT NULL
            );
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                run_id VARCHAR PRIMARY KEY,
                model_family VARCHAR NOT NULL,
                estimand VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                dataset_snapshot_id VARCHAR,
                config_json JSON NOT NULL,
                metrics_json JSON NOT NULL,
                artifact_path VARCHAR NOT NULL
            );
            """
        )
    return registry


def register_snapshot(path: str | Path, snapshot: dict) -> None:
    registry = initialize_registry(path)
    with duckdb.connect(str(registry)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO dataset_snapshots
            VALUES (?, ?, CAST(? AS TIMESTAMP), ?, CAST(? AS JSON))
            """,
            [
                snapshot["snapshot_id"],
                snapshot["dataset"],
                snapshot["created_at"],
                snapshot["passed"],
                json.dumps(snapshot, sort_keys=True),
            ],
        )

def register_model_run(path: str | Path, run: dict) -> None:
    estimand = run.get("estimand", run.get("estimand_id"))
    if not isinstance(estimand, str) or not estimand:
        raise ValueError("Model run requires estimand or estimand_id.")
    registry = initialize_registry(path)
    with duckdb.connect(str(registry)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO model_runs
            VALUES (?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, CAST(? AS JSON), CAST(? AS JSON), ?)
            """,
            [
                run["run_id"],
                run["model_family"],
                estimand,
                run["status"],
                run["created_at"],
                run.get("dataset_snapshot_id"),
                json.dumps(run["config"], sort_keys=True),
                json.dumps(run["metrics"], sort_keys=True),
                run["artifact_path"],
            ],
        )
