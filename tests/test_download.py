from __future__ import annotations

import io
import tarfile

import pandas as pd

from nba_impact.data.download import (
    DownloadTask,
    TransientDownloadError,
    _download_with_retries,
    _validate_file,
    ingest_task,
    load_tasks,
    plan_ingest_manifest,
)


def test_validate_csv_contract(tmp_path) -> None:
    path = tmp_path / "players.csv"
    path.write_text("PLAYER_ID,year,value\n1,2026,3.2\n2,2026,4.1\n")
    task = DownloadTask(
        name="players",
        url="https://example.invalid/players.csv",
        destination="players.csv",
        provider="fixture",
        license="fixture",
        expected_min_rows=2,
        required_columns=("PLAYER_ID", "year"),
    )
    result = _validate_file(path, task)
    assert result["rows"] == 2
    assert result["columns"] == ["PLAYER_ID", "year", "value"]


def test_validate_tar_xz_member_contract(tmp_path) -> None:
    path = tmp_path / "matchups.tar.xz"
    payload = b"game_id,person_id,matchups_person_id\n1,10,20\n2,11,21\n"
    info = tarfile.TarInfo("matchups.csv")
    info.size = len(payload)
    with tarfile.open(path, mode="w:xz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    task = DownloadTask(
        name="matchups",
        url="https://example.invalid/matchups.tar.xz",
        destination="matchups.tar.xz",
        provider="fixture",
        license="fixture",
        archive_member="matchups.csv",
        expected_min_rows=2,
        required_columns=("game_id", "person_id", "matchups_person_id"),
    )

    result = _validate_file(path, task)

    assert result["rows"] == 2
    assert result["archive_member"] == "matchups.csv"


def test_ingest_promotes_complete_partial_without_network(tmp_path) -> None:
    destination = tmp_path / "source" / "sample.parquet"
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.parent.mkdir(parents=True)
    pd.DataFrame({"game_id": [1], "value": [2]}).to_parquet(partial, index=False)
    task = DownloadTask(
        name="complete_partial",
        url="https://unused.invalid/sample.parquet",
        destination="source/sample.parquet",
        provider="test",
        license="test",
        required_columns=("game_id",),
    )

    result = ingest_task(task, root=tmp_path)

    assert result["status"] == "downloaded"
    assert destination.exists()
    assert not partial.exists()


def test_validate_enforces_content_identity(tmp_path) -> None:
    path = tmp_path / "players.csv"
    path.write_text("PLAYER_ID,year\n1,2026\n")
    task = DownloadTask(
        name="players",
        url="https://example.invalid/players.csv",
        destination="players.csv",
        provider="fixture",
        license="fixture",
        expected_bytes=path.stat().st_size,
        expected_sha256="0" * 64,
        required_columns=("PLAYER_ID",),
    )

    try:
        _validate_file(path, task)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("wrong content hash should fail validation")


def test_dry_run_reports_remaining_bytes(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"provider":"fixture","tasks":[{"name":"one","url":"https://example.invalid/one.csv",'
        '"destination":"one.csv","provider":"fixture","license":"fixture","expected_bytes":42}]}'
    )

    plan = plan_ingest_manifest(manifest, root=tmp_path / "bronze")

    assert plan["tasks"] == 1
    assert plan["verified"] == 0
    assert plan["remaining_bytes"] == 42
    assert plan["results"][0]["status"] == "missing"


def test_manifest_task_defaults_are_applied_and_can_be_overridden(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"provider":"fixture","task_defaults":{"provider":"shared",'
        '"license":"Apache-2.0","required_columns":["gameId"],"max_attempts":9},'
        '"tasks":[{"name":"one","url":"https://example.invalid/one.parquet",'
        '"destination":"one.parquet"},{"name":"two",'
        '"url":"https://example.invalid/two.parquet","destination":"two.parquet",'
        '"max_attempts":3}]}'
    )

    _, tasks = load_tasks(manifest)

    assert tasks[0].provider == "shared"
    assert tasks[0].required_columns == ("gameId",)
    assert tasks[0].max_attempts == 9
    assert tasks[1].max_attempts == 3


def test_retry_budget_recovers_and_resumes_partial(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "sample.csv"
    attempts = 0

    def flaky_download(session, task, target):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            target.with_suffix(target.suffix + ".partial").write_bytes(b"PLAYER_ID,year\n")
            raise TransientDownloadError("connection dropped")
        target.write_text("PLAYER_ID,year\n1,2026\n")
        return _validate_file(target, task)

    monkeypatch.setattr("nba_impact.data.download._download_once", flaky_download)
    task = DownloadTask(
        name="flaky",
        url="https://example.invalid/sample.csv",
        destination="sample.csv",
        provider="fixture",
        license="fixture",
        required_columns=("PLAYER_ID",),
        max_attempts=4,
        retry_initial_seconds=0,
        retry_max_seconds=0,
    )

    result = _download_with_retries(object(), task, destination)

    assert attempts == 4
    assert result["rows"] == 1
