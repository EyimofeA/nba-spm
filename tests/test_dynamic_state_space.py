from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nba_impact.models import dynamic_state_space
from nba_impact.models.dynamic_state_space import (
    build_annual_observation_variance,
    build_causal_state_space_filter,
    build_state_space_trajectory,
    _paired_forward_comparison,
)
from nba_impact import cli


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2, 2],
            "Season": [2018, 2019, 2020, 2018, 2019, 2020],
            "target_offense": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            "target_defense": [0.5, 0.0, -0.5, -0.5, 0.0, 0.5],
            "Poss_Off": [1000.0] * 6,
            "Poss_Def": [1000.0] * 6,
        }
    )
    targets["target_net"] = targets["target_offense"] + targets["target_defense"]
    variance = targets[["PLAYER_ID", "Season"]].copy()
    variance["observation_variance_offense"] = 0.25
    variance["observation_variance_defense"] = 0.25
    return targets, variance


def test_state_space_is_causal_and_preserves_component_identity() -> None:
    targets, variance = _inputs()
    first = build_causal_state_space_filter(targets, variance, phi=0.8, process_sd=0.5)
    changed = targets.copy()
    changed.loc[changed["Season"].eq(2020), "target_offense"] += 1000.0
    changed["target_net"] = changed["target_offense"] + changed["target_defense"]
    revised = build_causal_state_space_filter(changed, variance, phi=0.8, process_sd=0.5)
    pd.testing.assert_frame_equal(
        first.loc[first["Season"].lt(2020)].reset_index(drop=True),
        revised.loc[revised["Season"].lt(2020)].reset_index(drop=True),
    )
    assert np.allclose(
        first["filtered_net"], first["filtered_offense"] + first["filtered_defense"]
    )
    assert first[["filtered_variance_offense", "filtered_variance_defense"]].gt(0).all().all()


def test_state_space_rejects_missing_observation_variance() -> None:
    targets, variance = _inputs()
    with np.testing.assert_raises_regex(ValueError, "Every annual RAPM target"):
        build_causal_state_space_filter(targets, variance.iloc[:-1], phi=0.8, process_sd=0.5)


def test_paired_comparison_requires_identical_scored_rows() -> None:
    targets, variance = _inputs()
    state_space = build_causal_state_space_filter(targets, variance, phi=0.8, process_sd=0.5)
    baseline = state_space.copy()
    comparison = _paired_forward_comparison(
        state_space,
        baseline,
        targets,
        origins=(2018, 2019),
        minimum_side_possessions=1000.0,
    )
    assert comparison["state_space_minus_time_decay_rmse"].eq(0.0).all()
    with np.testing.assert_raises_regex(ValueError, "identical player-season rows"):
        _paired_forward_comparison(
            state_space,
            baseline.loc[~((baseline["PLAYER_ID"] == 2) & (baseline["Season"] == 2019))],
            targets,
            origins=(2018, 2019),
            minimum_side_possessions=1000.0,
        )


def test_state_space_builds_matched_time_decay_comparison(tmp_path) -> None:
    targets, variance = _inputs()
    targets_path = tmp_path / "targets.parquet"
    variance_path = tmp_path / "variance.parquet"
    names_path = tmp_path / "names.csv"
    baseline_path = tmp_path / "time_decay.parquet"
    targets.to_parquet(targets_path, index=False)
    variance.to_parquet(variance_path, index=False)
    pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["One", "Two"]}).to_csv(names_path, index=False)
    baseline = build_causal_state_space_filter(targets, variance, phi=0.8, process_sd=0.5)
    baseline.to_parquet(baseline_path, index=False)

    run = build_state_space_trajectory(
        targets_path,
        variance_path,
        names_path,
        baseline_path,
        artifact_root=tmp_path / "artifacts",
        candidate_phis=(0.8,),
        candidate_process_sds=(0.5,),
        selection_origins=(2018,),
        diagnostic_origins=(2019,),
    )

    assert Path(run["paired_time_decay_comparison_path"]).exists()
    assert run["metrics"]["selection_state_space_minus_time_decay_rmse"] == 0.0


def test_annual_variance_honors_requested_legacy_seasons_and_skips_current(
    tmp_path, monkeypatch
) -> None:
    """A narrowly requested legacy batch must not perform hidden current-data work."""
    players = np.arange(1, 11, dtype=int)
    targets = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Season": 2016,
            "target_offense": 0.0,
            "target_defense": 0.0,
            "target_net": 0.0,
        }
    )
    targets_path = tmp_path / "targets.parquet"
    targets.to_parquet(targets_path, index=False)
    current_possessions = tmp_path / "current_possessions.parquet"
    current_segments = tmp_path / "current_segments.parquet"
    current_possessions.write_bytes(b"placeholder")
    current_segments.write_bytes(b"placeholder")
    calls: list[int] = []

    def fake_legacy(_cache_dir, seasons, *, game_types):
        calls.extend(seasons)
        return pd.DataFrame({"gameid": ["0021600001"]})

    def fail_current(*_args, **_kwargs):
        raise AssertionError("A legacy-only request must not load current possessions.")

    monkeypatch.setattr(dynamic_state_space, "load_legacy_possessions", fake_legacy)
    monkeypatch.setattr(dynamic_state_space, "load_current_possessions", fail_current)
    monkeypatch.setattr(
        dynamic_state_space,
        "build_design",
        lambda _frame: SimpleNamespace(players=players),
    )
    monkeypatch.setattr(
        dynamic_state_space,
        "game_cluster_sandwich",
        lambda _design, _config: (np.eye(21) * 0.01, np.zeros(21), None),
    )

    run = build_annual_observation_variance(
        targets_path,
        tmp_path / "legacy",
        current_possessions,
        current_segments,
        artifact_root=tmp_path / "artifacts",
        seasons=(2016,),
    )

    assert calls == [2016]
    assert run["status"] == "research_measurement_input_complete"
    assert Path(run["observation_variance_path"]).exists()


def test_annual_variance_cli_forwards_explicit_season_scope(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "ensure_owned_dirs", lambda: None)
    monkeypatch.setattr(cli, "register_model_run", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "build_annual_observation_variance",
        lambda *_args, **kwargs: captured.update(kwargs) or {"run_id": "test"},
    )
    args = SimpleNamespace(
        targets=tmp_path / "targets.parquet",
        cache_dir=tmp_path / "cache",
        possessions=tmp_path / "possessions.parquet",
        segments=tmp_path / "segments.parquet",
        artifact_root=tmp_path / "artifacts",
        transition_season=2024,
        seasons=(2016,),
        registry=tmp_path / "registry.jsonl",
    )

    assert cli.command_build_annual_observation_variance(args) == 0
    assert captured["seasons"] == (2016,)
