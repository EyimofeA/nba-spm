"""Build a local derived-ratings release bundle without raw NBA rows."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.api.ratings import RatingsApiConfig, RatingsStore
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.research.control_plane import (
    assert_control_plane,
    load_pinned_contracts,
    validate_release_manifest,
)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(f"{path.suffix}.partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def build_local_release_bundle(
    api_config_path: str | Path,
    artifact_root: str | Path,
    contract_path: str | Path,
    *,
    release_root: str | Path,
) -> dict:
    """Package schemas, derived ratings, and checksums only.

    No bronze/silver source table, event row, credential, or absolute path enters
    this bundle. The result remains local until source-rights review authorizes
    an external release.
    """
    assert_control_plane(contract_path, artifact_root)
    api_config = RatingsApiConfig.from_json(api_config_path)
    store = RatingsStore(api_config, Path(artifact_root) / "models")
    contracts = load_pinned_contracts(contract_path)
    identity = hashlib.sha256(
        json.dumps(
            {
                "api_config": sha256_file(api_config_path),
                "contracts": sha256_file(contract_path),
                "annual": store._row_set_hash(store.annual, ["PLAYER_ID", "Season"]),
                "peaks": store._row_set_hash(
                    store.peaks, ["PLAYER_ID", "window_seasons", "peak_component"]
                ),
                "current": store._row_set_hash(store.current, ["player_id"]),
                "normal_rapm_uncertainty": {
                    scope: store._row_set_hash(frame, ["player_id"])
                    for scope, frame in store.normal_rapm_uncertainty.items()
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    output = Path(release_root) / f"nba_impact_local_v2_{identity}"
    if output.exists():
        manifest = output / "release_manifest.json"
        if not validate_release_manifest(manifest):
            return json.loads(manifest.read_text())
        raise ValueError(f"Existing release bundle is invalid: {output}")
    output.mkdir(parents=True)
    ratings = output / "derived_ratings"
    ratings.mkdir()
    _write_parquet_atomic(store.annual, ratings / "annual_aio.parquet")
    _write_parquet_atomic(store.rolling, ratings / "rolling_normal_rapm.parquet")
    _write_parquet_atomic(store.peaks, ratings / "rolling_peaks.parquet")
    _write_parquet_atomic(store.current, ratings / "current_normal_rapm.parquet")
    for scope, frame in store.normal_rapm_uncertainty.items():
        _write_parquet_atomic(
            frame, ratings / f"normal_rapm_uncertainty_{scope}.parquet"
        )
    if store.matchup is not None:
        _write_parquet_atomic(store.matchup, ratings / "matchup_defense_research.parquet")
    schema = {
        "schema_version": "ratings_api_v2",
        "metadata_example": store.v2_metadata(),
        "routes": [
            "/v2/meta",
            "/v2/leaderboards/annual",
            "/v2/leaderboards/current",
            "/v2/leaderboards/peaks",
            "/v2/leaderboards/normal-rapm-uncertainty?scope={scope}",
            "/v2/players/{player_id}",
        ],
    }
    write_json_atomic(schema, output / "ratings_api_v2_schema.json")
    write_json_atomic(contracts, output / "pinned_artifact_contracts.json")
    model_cards = output / "model_cards"
    model_cards.mkdir()
    for artifact in contracts["artifacts"]:
        (model_cards / f"{artifact['artifact_id']}.md").write_text(
            "\n".join(
                [
                    f"# {artifact['artifact_id']}",
                    "",
                    f"- Estimand: {artifact['estimand_id']}",
                    f"- Evidence: {artifact['evidence_status']}",
                    f"- Scope: {artifact['season_scope']}",
                    f"- Completeness: {artifact['season_completeness']}",
                    f"- Uncertainty: {artifact['uncertainty_status']}",
                    f"- Do not interpret as: {artifact['forbidden_interpretation']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    fixture = output / "synthetic_verification_fixture"
    fixture.mkdir()
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "offense_estimate": [1.0, -1.0],
            "defense_estimate": [0.5, 0.25],
            "net_estimate": [1.5, -0.75],
        }
    ).to_parquet(fixture / "ratings_fixture.parquet", index=False)
    (fixture / "README.md").write_text(
        "Synthetic only. Verify net_estimate equals offense_estimate plus defense_estimate.\n",
        encoding="utf-8",
    )
    row_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*.parquet"))
    }
    row_set_hash = hashlib.sha256(json.dumps(row_hashes, sort_keys=True).encode()).hexdigest()
    artifacts = []
    for item in contracts["artifacts"]:
        run = json.loads((Path(artifact_root) / item["artifact_relative_dir"] / "run.json").read_text())
        artifacts.append(
            {
                "artifact_id": item["artifact_id"],
                "relative_artifact_path": item["artifact_relative_dir"],
                "estimand_id": item["estimand_id"],
                "evidence_status": item["evidence_status"],
                "uncertainty_status": item["uncertainty_status"],
                "run_status": run.get("status"),
            }
        )
    release = {
        "schema_version": "nba_impact_release_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_scope": "local_only_derived_ratings",
        "row_set_sha256": row_set_hash,
        "artifacts": artifacts,
        "files": row_hashes,
        "source_rights_boundary": "This bundle excludes raw NBA source rows. Do not distribute it before source-rights review.",
    }
    write_json_atomic(release, output / "release_manifest.json")
    issues = validate_release_manifest(output / "release_manifest.json")
    if issues:
        shutil.rmtree(output)
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"Release bundle failed validation: {details}")
    return release
