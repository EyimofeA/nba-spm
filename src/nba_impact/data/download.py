"""Manifest-driven, resumable HTTP ingestion with atomic Parquet validation."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .manifest import sha256_file, write_json_atomic


@dataclass(frozen=True)
class DownloadTask:
    name: str
    url: str
    destination: str
    provider: str
    license: str
    season: int | None = None
    season_type: str | None = None
    source_revision: str | None = None
    expected_min_rows: int = 1
    required_columns: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict) -> "DownloadTask":
        data = dict(payload)
        data["required_columns"] = tuple(data.get("required_columns", ()))
        return cls(**data)


class TransientDownloadError(RuntimeError):
    pass


def load_tasks(path: str | Path) -> tuple[dict, list[DownloadTask]]:
    manifest = json.loads(Path(path).read_text())
    return manifest, [DownloadTask.from_dict(item) for item in manifest["tasks"]]


def _validate_parquet(path: Path, task: DownloadTask) -> dict:
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    columns = parquet.schema_arrow.names
    if rows < task.expected_min_rows:
        raise ValueError(f"{task.name}: expected at least {task.expected_min_rows} rows, found {rows}")
    missing = sorted(set(task.required_columns) - set(columns))
    if missing:
        raise ValueError(f"{task.name}: missing required columns {missing}")
    return {"rows": rows, "columns": columns, "row_groups": parquet.num_row_groups}


def _validate_csv(path: Path, task: DownloadTask) -> dict:
    rows = 0
    malformed_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{task.name}: CSV is empty") from exc
        if not columns or any(not column.strip() for column in columns):
            raise ValueError(f"{task.name}: CSV header contains empty column names")
        width = len(columns)
        for row in reader:
            if not row:
                continue
            rows += 1
            malformed_rows += int(len(row) != width)
    if rows < task.expected_min_rows:
        raise ValueError(f"{task.name}: expected at least {task.expected_min_rows} rows, found {rows}")
    missing = sorted(set(task.required_columns) - set(columns))
    if missing:
        raise ValueError(f"{task.name}: missing required columns {missing}")
    if malformed_rows:
        raise ValueError(f"{task.name}: {malformed_rows} CSV rows do not match the header width")
    return {"rows": rows, "columns": columns, "row_groups": None}


def _validate_file(path: Path, task: DownloadTask) -> dict:
    suffix = Path(task.destination).suffix.lower()
    if suffix == ".parquet":
        return _validate_parquet(path, task)
    if suffix == ".csv":
        return _validate_csv(path, task)
    raise ValueError(f"{task.name}: unsupported destination format {suffix!r}")


@retry(
    retry=retry_if_exception_type((requests.RequestException, TransientDownloadError)),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _download_once(session: requests.Session, task: DownloadTask, destination: Path) -> dict:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        try:
            validation = _validate_file(partial, task)
        except Exception:
            pass
        else:
            partial.replace(destination)
            return validation
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "nba-impact-lab/0.1 (+research; resumable downloader)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with session.get(task.url, headers=headers, stream=True, timeout=(20, 120)) as response:
        if response.status_code in {429, 500, 502, 503, 504}:
            raise TransientDownloadError(f"{task.name}: transient HTTP {response.status_code}")
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if existing and not append:
            existing = 0
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    validation = _validate_file(partial, task)
    partial.replace(destination)
    return validation


def ingest_task(
    task: DownloadTask,
    *,
    root: str | Path,
    session: requests.Session | None = None,
) -> dict:
    root_path = Path(root)
    destination = root_path / task.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if destination.exists():
        validation = _validate_file(destination, task)
        status = "verified_existing"
    else:
        validation = _download_once(session or requests.Session(), task, destination)
        status = "downloaded"
    result = {
        **asdict(task),
        "required_columns": list(task.required_columns),
        "status": status,
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **validation,
    }
    write_json_atomic(result, destination.with_suffix(destination.suffix + ".manifest.json"))
    return result


def run_ingest_manifest(manifest_path: str | Path, *, root: str | Path) -> dict:
    manifest, tasks = load_tasks(manifest_path)
    results: list[dict] = []
    failures: list[dict] = []
    session = requests.Session()
    for task in tasks:
        try:
            result = ingest_task(task, root=root, session=session)
            results.append(result)
            print(
                f"{result['status']:>17} {task.name:<34} "
                f"{result['bytes'] / 1_000_000:7.2f} MB {result['rows']:>9,} rows"
            )
        except Exception as exc:
            failure = {"name": task.name, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"{'failed':>17} {task.name:<34} {failure['error']}")
    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "provider": manifest.get("provider"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path = Path(root) / "_ingest_runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json_atomic(summary, summary_path)
    if failures:
        raise RuntimeError(f"{len(failures)} download task(s) failed; see {summary_path}")
    return summary
