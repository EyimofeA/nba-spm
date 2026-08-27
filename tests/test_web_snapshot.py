"""Focused checks for the static web snapshot contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_impact.api.web_snapshot import (
    EXTERNAL_BENCHMARK,
    MODEL_CATALOG,
    build_web_snapshot,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the smallest artifact set the snapshot builder can read."""
    artifact_root = tmp_path / "models"
    annual_dir = artifact_root / "annual_aio_ratings" / "annual_test"
    rolling_dir = artifact_root / "rolling_rapm_peaks" / "rolling_test"
    current_dir = artifact_root / "rapm" / "current_test"
    roles_dir = tmp_path / "features" / "side_roles" / "roles_test"
    stable_dir = tmp_path / "features" / "role_stabilization" / "stable_test"
    for directory in (annual_dir, rolling_dir, current_dir, roles_dir, stable_dir):
        directory.mkdir(parents=True)

    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["Alpha Guard", "Beta Wing"],
            "Season": [2024, 2024],
            "TEAM_ABBREVIATION": ["HIST", "HST2"],
            "Poss_Off": [2000, 900],
            "Poss_Def": [1990, 910],
            "aio_net": [5.0, 6.0],
            "aio_offense": [3.0, 4.0],
            "aio_defense": [2.0, 2.0],
            "normal_rapm_net": [4.0, 5.0],
            "normal_rapm_offense": [2.0, 3.0],
            "normal_rapm_defense": [2.0, 2.0],
            "spm_center_net": [4.5, 5.5],
            "spm_center_offense": [2.5, 3.5],
            "spm_center_defense": [2.0, 2.0],
            "spm_raw_net": [4.4, 5.4],
            "spm_raw_offense": [2.4, 3.4],
            "spm_raw_defense": [2.0, 2.0],
            "rapm_update_net": [0.5, 0.5],
            "rapm_update_offense": [0.5, 0.5],
            "rapm_update_defense": [0.0, 0.0],
        }
    )
    rolling = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "PLAYER_NAME": ["Alpha Guard"],
            "window_start": [2022],
            "window_end": [2024],
            "window_seasons": [3],
            "Poss_Off": [6000],
            "Poss_Def": [5990],
            "offense": [3.0],
            "defense": [2.0],
            "net": [5.0],
        }
    )
    peaks = rolling.assign(peak_component="net", peak_value=5.0, all_time_rank=1)
    current = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Alpha Guard"],
            "offense_per_100": [4.0],
            "defense_per_100": [1.0],
            "net_per_100": [5.0],
            "off_possessions": [6000],
            "def_possessions": [5990],
        }
    )
    annual.to_parquet(annual_dir / "ratings.parquet", index=False)
    rolling.to_parquet(rolling_dir / "rolling_ratings.parquet", index=False)
    peaks.to_parquet(rolling_dir / "player_peaks.parquet", index=False)
    current.to_parquet(current_dir / "ratings.parquet", index=False)

    offense_roles = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Season": [2024, 2024],
            "off_role_cluster": ["off_role_0", "off_role_2"],
            "off_role_confidence": [0.6, 0.5],
            "off_role_axis_1": [0.4, -0.3],
            "off_role_axis_2": [-0.2, 0.1],
            **{
                f"off_role_affinity_{index}": [value, value]
                for index, value in enumerate([0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
            },
        }
    )
    defense_roles = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Season": [2024, 2024],
            "def_role_cluster": ["def_role_4", "def_role_0"],
            "def_role_confidence": [0.7, 0.4],
            "def_role_axis_1": [0.2, -0.5],
            "def_role_axis_2": [0.3, 0.4],
            **{
                f"def_role_affinity_{index}": [value, value]
                for index, value in enumerate([0.05, 0.1, 0.05, 0.1, 0.7])
            },
        }
    )
    offense_roles.to_parquet(roles_dir / "offense_assignments.parquet", index=False)
    defense_roles.to_parquet(roles_dir / "defense_assignments.parquet", index=False)
    stable_offense = offense_roles.assign(
        off_role_stable_cluster="off_role_1", off_role_stable_confidence=0.55
    )
    stable_defense = defense_roles.assign(
        def_role_stable_cluster="def_role_3", def_role_stable_confidence=0.6
    )
    for index in range(6):
        stable_offense[f"off_role_stable_affinity_{index}"] = 0.1
    for index in range(5):
        stable_defense[f"def_role_stable_affinity_{index}"] = 0.2
    stable_offense.to_parquet(stable_dir / "offense_assignments.parquet", index=False)
    stable_defense.to_parquet(stable_dir / "defense_assignments.parquet", index=False)

    for directory, manifest in (
        (annual_dir, {"status": "research", "estimand": "annual", "caveats": []}),
        (rolling_dir, {"status": "research", "estimand": "rolling", "caveats": []}),
        (
            current_dir,
            {
                "status": "research_frozen_baseline",
                "estimand": "current",
                "config": {"seasons": [2024]},
                "caveats": [],
            },
        ),
        (roles_dir, {"run_id": "roles_test", "status": "validated_research_input"}),
        (stable_dir, {"run_id": "stable_test", "status": "validated_descriptive"}),
    ):
        (directory / "run.json").write_text(json.dumps(manifest))

    config_path = tmp_path / "ratings.json"
    config_path.write_text(
        json.dumps(
            {
                "contract_version": "ratings_api_v1",
                "annual_run_id": "annual_test",
                "rolling_run_id": "rolling_test",
                "current_rapm_run_id": "current_test",
                "side_roles_run_id": "roles_test",
                "role_stabilization_run_id": "stable_test",
            }
        )
    )
    aging_path = tmp_path / "aging_curve.csv"
    pd.DataFrame(
        {
            "Age": [24, 25, 26],
            "f_total": [-1.0, -0.5, 0.0],
            "f_off": [-0.6, -0.3, 0.0],
            "f_def": [-0.4, -0.2, 0.0],
        }
    ).to_csv(aging_path, index=False)
    return config_path, artifact_root, aging_path


def _build(tmp_path: Path) -> Path:
    config_path, artifact_root, aging_path = _artifacts(tmp_path)
    output_dir = tmp_path / "web"
    build_web_snapshot(
        config_path, artifact_root, aging_path, output_dir, shards=2
    )
    return output_dir


def _read(output_dir: Path, name: str) -> object:
    return json.loads((output_dir / name).read_text())


def test_snapshot_exports_every_selectable_model(tmp_path: Path) -> None:
    output_dir = _build(tmp_path)
    catalog = _read(output_dir, "catalog.json")
    assert [model["id"] for model in catalog["catalog"]["models"]] == [
        "aio",
        "rapm",
        "spm",
    ]

    leaderboard = _read(output_dir, "leaderboard-2024.json")
    alpha = next(row for row in leaderboard if row["PLAYER_ID"] == 1)
    for model in MODEL_CATALOG:
        for component in ("offense", "defense", "net"):
            assert isinstance(alpha[f"{model['prefix']}{component}"], float)
    # SPM is the possession-centered prior, not the raw statistical value.
    assert alpha["spm_net"] == 4.5
    assert alpha["normal_rapm_net"] == 4.0
    # The published decomposition stays exact: AIO minus SPM is the RAPM update.
    assert round(alpha["aio_net"] - alpha["spm_net"], 4) == 0.5

    shard = _read(output_dir, "ratings-01.json")
    annual = shard["1"]["annual"][0]
    assert annual["aio_net"] == 5.0
    assert annual["spm_net"] == 4.5
    assert annual["normal_rapm_net"] == 4.0


def test_snapshot_extends_only_normal_rapm_from_the_pinned_current_target_run(
    tmp_path: Path,
) -> None:
    config_path, artifact_root, aging_path = _artifacts(tmp_path)
    current_dir = tmp_path / "current_targets"
    current_dir.mkdir()
    pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 1, 3, 1, 3],
            "Season": [2024, 2024, 2025, 2025, 2026, 2026],
            "target_offense": [8.0, 7.0, 2.0, 1.0, 3.0, 2.0],
            "target_defense": [2.0, 1.0, 1.0, 0.5, 1.5, 1.0],
            "target_net": [10.0, 8.0, 3.0, 1.5, 4.5, 3.0],
            "Poss_Off": [4000, 3900, 2100, 800, 2200, 900],
            "Poss_Def": [3990, 3890, 2090, 790, 2190, 890],
        }
    ).to_parquet(current_dir / "targets.parquet", index=False)
    (current_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "current_targets_test",
                "model_family": "canonical_current_single_season_zero_prior_normal_rapm_targets",
                "config": {
                    "lineup_policy": "terminal", "prior": "zero", "game_types": ["regular"],
                    "lambda_off": 3000.0, "lambda_def": 3000.0, "lambda_home": 300.0,
                },
            }
        )
    )
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    for year in (2025, 2026):
        pd.DataFrame(
            {
                "PLAYER_ID": [1, 3],
                "PLAYER_NAME": ["Alpha Guard", "Current Rookie"],
                "TEAM_ABBREVIATION": ["AAA", "CCC"],
                "AGE": [26, 20],
            }
        ).to_csv(sheets / f"{year}.csv", index=False)

    output = tmp_path / "web-current"
    build_web_snapshot(
        config_path,
        artifact_root,
        aging_path,
        output,
        player_sheets_dir=sheets,
        current_normal_rapm_run_path=current_dir,
        shards=2,
    )
    catalog = _read(output, "catalog.json")
    availability = {row["id"]: row["seasons"] for row in catalog["catalog"]["models"]}
    assert catalog["catalog"]["seasons"] == [2024, 2025, 2026]
    assert availability == {"aio": [2024], "rapm": [2024, 2025, 2026], "spm": [2024]}
    assert catalog["lineage"]["current_normal_rapm_run_id"] == "current_targets_test"

    overlap_leaderboard = _read(output, "leaderboard-2024.json")
    overlap_row = next(row for row in overlap_leaderboard if row["PLAYER_ID"] == 1)
    assert overlap_row["normal_rapm_net"] == 10.0
    assert (overlap_row["Poss_Off"], overlap_row["Poss_Def"]) == (4000, 3990)
    assert overlap_row["aio_net"] == 5.0
    assert overlap_row["spm_net"] == 4.5
    assert overlap_row["PLAYER_NAME"] == "Alpha Guard"
    assert overlap_row["TEAM_ABBREVIATION"] == "HIST"

    current_leaderboard = _read(output, "leaderboard-2026.json")
    current_row = next(row for row in current_leaderboard if row["PLAYER_ID"] == 3)
    assert current_row["normal_rapm_net"] == 3.0
    assert "aio_net" not in current_row and "spm_net" not in current_row
    shard = _read(output, "ratings-01.json")
    current_player = shard["3"]
    assert [row["Season"] for row in current_player["annual"]] == [2025, 2026]
    assert all("normal_rapm_net" in row for row in current_player["annual"])
    assert all("aio_net" not in row and "spm_net" not in row for row in current_player["annual"])

    targets = pd.read_parquet(current_dir / "targets.parquet")
    targets.loc[
        ~((targets["PLAYER_ID"] == 2) & (targets["Season"] == 2024))
    ].to_parquet(current_dir / "targets.parquet", index=False)
    with pytest.raises(ValueError, match="overlap keys do not exactly match"):
        build_web_snapshot(
            config_path,
            artifact_root,
            aging_path,
            tmp_path / "web-key-mismatch",
            player_sheets_dir=sheets,
            current_normal_rapm_run_path=current_dir,
            shards=2,
        )

    targets.loc[(targets["PLAYER_ID"] == 1) & (targets["Season"] == 2024), "target_net"] = float("nan")
    targets.to_parquet(current_dir / "targets.parquet", index=False)
    with pytest.raises(ValueError, match="non-finite values"):
        build_web_snapshot(
            config_path,
            artifact_root,
            aging_path,
            tmp_path / "web-missing-value",
            player_sheets_dir=sheets,
            current_normal_rapm_run_path=current_dir,
            shards=2,
        )


def test_snapshot_is_raw_role_only_and_has_no_win_probability(tmp_path: Path) -> None:
    output_dir = _build(tmp_path)
    payload = "".join(path.read_text() for path in sorted(output_dir.glob("*.json")))
    assert "win_probability" not in payload
    assert "brier" not in payload
    assert "stabilized" not in payload
    assert "stable_role" not in payload

    role_map = _read(output_dir, "roles-offense-2024.json")
    assert set(role_map[0]) == {
        "PLAYER_ID",
        "PLAYER_NAME",
        "Season",
        "TEAM_ABBREVIATION",
        "x",
        "y",
        "raw_role",
    }
    catalog = _read(output_dir, "catalog.json")
    assert "win_probability" not in catalog["validation"]
    assert "role_stabilization" not in catalog["methods"]
    roles = _read(output_dir, "ratings-01.json")["1"]["roles"][0]
    assert set(roles["offense"]) == {"primary_role", "confidence", "memberships"}


def test_snapshot_external_benchmark_matches_verified_runs(tmp_path: Path) -> None:
    output_dir = _build(tmp_path)
    benchmark = _read(output_dir, "catalog.json")["validation"]["external_benchmark"]
    assert benchmark == json.loads(json.dumps(EXTERNAL_BENCHMARK))
    assert benchmark == {}


def test_snapshot_writes_a_portable_release_manifest(tmp_path: Path) -> None:
    output_dir = _build(tmp_path)
    manifest = _read(output_dir, "snapshot-manifest.json")
    assert manifest["schema_version"] == "nba_impact_release_v1"
    assert len(manifest["row_set_sha256"]) == 64
    assert all(not row["relative_path"].startswith("/") for row in manifest["artifacts"])
