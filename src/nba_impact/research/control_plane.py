"""Validate pinned rating lineage before a release or model run.

This module deliberately uses JSON for the executable contract. The YAML files in
``research/`` remain human-readable policy records. Keeping the runtime contract
small avoids a second, implicit YAML parser and makes release validation portable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"(^/|[A-Za-z]:[\\/])")
_PRODUCTION_STATUSES = {"production", "production_reference", "production_reference_method"}


@dataclass(frozen=True)
class ControlPlaneIssue:
    code: str
    message: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pinned_contracts(path: str | Path) -> dict[str, Any]:
    contract = _read_json(Path(path))
    if contract.get("schema_version") != "pinned_artifact_contracts_v1":
        raise ValueError("Unsupported pinned artifact contract schema.")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Pinned artifact contract requires at least one artifact.")
    return contract


def validate_pinned_artifacts(
    contract_path: str | Path,
    artifact_root: str | Path,
) -> list[ControlPlaneIssue]:
    """Return all lineage issues. An empty result is the model-work gate."""
    contract = load_pinned_contracts(contract_path)
    root = Path(artifact_root)
    issues: list[ControlPlaneIssue] = []
    seen_fields: set[str] = set()
    for item in contract["artifacts"]:
        identifier = str(item.get("artifact_id", "<missing>"))
        required = (
            "api_field",
            "artifact_id",
            "artifact_relative_dir",
            "estimand_id",
            "evidence_status",
            "season_scope",
            "season_completeness",
            "uncertainty_status",
            "config_sha256",
            "code_sha256",
            "data_hashes_status",
            "forbidden_interpretation",
        )
        for field in required:
            if not item.get(field):
                issues.append(ControlPlaneIssue("missing_lineage", f"{identifier}: missing {field}"))
        api_field = item.get("api_field")
        if api_field in seen_fields:
            issues.append(ControlPlaneIssue("duplicate_api_field", f"Duplicate API field {api_field}."))
        seen_fields.add(api_field)
        for field in ("config_sha256", "code_sha256"):
            value = str(item.get(field, ""))
            if not _HEX_256.fullmatch(value):
                issues.append(ControlPlaneIssue("invalid_hash", f"{identifier}: {field} is not SHA-256."))
        relative = str(item.get("artifact_relative_dir", ""))
        if _ABSOLUTE_PATH.search(relative) or ".." in Path(relative).parts:
            issues.append(ControlPlaneIssue("unsafe_artifact_path", f"{identifier}: artifact path must be relative."))
            continue
        manifest = root / relative / "run.json"
        if not manifest.exists():
            issues.append(ControlPlaneIssue("missing_run_manifest", f"{identifier}: {manifest} is absent."))
            continue
        run = _read_json(manifest)
        if run.get("run_id") != item.get("artifact_id"):
            issues.append(ControlPlaneIssue("run_id_mismatch", f"{identifier}: run manifest ID does not match pin."))
        status = str(item.get("evidence_status", ""))
        run_status = str(run.get("status", "")).lower()
        if status in _PRODUCTION_STATUSES and run_status in {
            "research_only",
            "research_challenger",
            "research_lineup_sensitivity",
            "research_diagnostic_unverified",
        }:
            issues.append(ControlPlaneIssue("research_exposed_as_production", f"{identifier}: research run cannot be production."))
        if 2027 in _declared_seasons(run):
            issues.append(ControlPlaneIssue("reserved_season_leakage", f"{identifier}: run declares reserved Season 2027."))
    return issues


def validate_release_manifest(path: str | Path) -> list[ControlPlaneIssue]:
    """Reject non-portable release manifests before a local bundle is shared."""
    manifest_path = Path(path)
    release = _read_json(manifest_path)
    issues: list[ControlPlaneIssue] = []
    required = ("schema_version", "artifacts", "row_set_sha256", "created_at")
    for field in required:
        if not release.get(field):
            issues.append(ControlPlaneIssue("missing_release_field", f"Missing release field {field}."))
    if release.get("schema_version") != "nba_impact_release_v1":
        issues.append(ControlPlaneIssue("unsupported_release_schema", "Unsupported release schema."))
    if not _HEX_256.fullmatch(str(release.get("row_set_sha256", ""))):
        issues.append(ControlPlaneIssue("invalid_hash", "row_set_sha256 is not SHA-256."))
    for value in _walk_strings(release):
        if _ABSOLUTE_PATH.search(value):
            issues.append(ControlPlaneIssue("absolute_release_path", f"Release manifest contains absolute path: {value}"))
            break
    for artifact in release.get("artifacts", []):
        if artifact.get("evidence_status") in _PRODUCTION_STATUSES and str(artifact.get("run_status", "")).startswith("research"):
            issues.append(ControlPlaneIssue("research_exposed_as_production", "Research artifact exposed as production."))
        if 2027 in _declared_seasons(artifact):
            issues.append(ControlPlaneIssue("reserved_season_leakage", "Release artifact declares Season 2027."))
    return issues


def _declared_seasons(value: Any) -> set[int]:
    """Extract only explicit season fields; hashes and paths are not evidence."""
    years: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"season", "seasons", "season_range", "season_scope", "season_end"}:
                years.update(_declared_seasons(item))
            elif isinstance(item, (dict, list)):
                years.update(_declared_seasons(item))
    elif isinstance(value, list):
        for item in value:
            years.update(_declared_seasons(item))
    elif isinstance(value, int) and 1900 <= value <= 2100:
        years.add(value)
    elif isinstance(value, str):
        years.update(int(token) for token in re.findall(r"(?<!\d)(20\d{2})(?!\d)", value))
    return years


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def assert_control_plane(contract_path: str | Path, artifact_root: str | Path) -> None:
    issues = validate_pinned_artifacts(contract_path, artifact_root)
    if issues:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"Research control plane failed: {details}")
