from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import pytest

from nba_impact.api.ratings import RatingsApiConfig, RatingsStore
from nba_impact.api.server import dispatch


def _store(tmp_path: Path) -> RatingsStore:
    annual_dir = tmp_path / "annual_aio_ratings" / "annual_test"
    rolling_dir = tmp_path / "rolling_rapm_peaks" / "rolling_test"
    current_dir = tmp_path / "rapm" / "current_test"
    matchup_dir = tmp_path / "matchup_defense" / "matchup_test"
    uncertainty_dir = tmp_path / "rapm_uncertainty" / "uncertainty_test"
    role_run_id = f"roles_{tmp_path.name}"
    roles_dir = tmp_path.parent / "features" / "side_roles" / role_run_id
    stabilization_run_id = f"stable_{tmp_path.name}"
    stabilization_dir = (
        tmp_path.parent / "features" / "role_stabilization" / stabilization_run_id
    )
    annual_dir.mkdir(parents=True)
    rolling_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)
    matchup_dir.mkdir(parents=True)
    uncertainty_dir.mkdir(parents=True)
    roles_dir.mkdir(parents=True)
    stabilization_dir.mkdir(parents=True)
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 1],
            "PLAYER_NAME": ["Alpha Guard", "Beta Wing", "Alpha Guard"],
            "Season": [2024, 2024, 2023],
            "Poss_Off": [2000, 900, 1800],
            "Poss_Def": [1990, 910, 1810],
            "aio_net": [5.0, 6.0, 4.0],
            "aio_offense": [3.0, 4.0, 2.5],
            "aio_defense": [2.0, 2.0, 1.5],
            "normal_rapm_net": [4.0, 5.0, 3.0],
            "normal_rapm_offense": [2.0, 3.0, 1.5],
            "normal_rapm_defense": [2.0, 2.0, 1.5],
            "spm_center_net": [4.5, 5.5, 3.5],
            "spm_center_offense": [2.5, 3.5, 2.0],
            "spm_center_defense": [2.0, 2.0, 1.5],
            "spm_raw_net": [4.5, 5.5, 3.5],
            "spm_raw_offense": [2.5, 3.5, 2.0],
            "spm_raw_defense": [2.0, 2.0, 1.5],
            "rapm_update_net": [0.5, 0.5, 0.5],
            "rapm_update_offense": [0.5, 0.5, 0.5],
            "rapm_update_defense": [0.0, 0.0, 0.0],
        }
    )
    rolling = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["Alpha Guard", "Beta Wing"],
            "window_start": [2022, 2020],
            "window_end": [2024, 2024],
            "window_seasons": [3, 5],
            "Poss_Off": [6000, 8000],
            "Poss_Def": [5990, 8010],
            "offense": [3.0, 2.0],
            "defense": [2.0, 1.0],
            "net": [5.0, 3.0],
        }
    )
    peaks = pd.concat(
        [
            rolling.assign(peak_component="net", peak_value=rolling["net"]),
            rolling.assign(
                peak_component="offense", peak_value=rolling["offense"]
            ),
            rolling.assign(
                peak_component="defense", peak_value=rolling["defense"]
            ),
        ],
        ignore_index=True,
    )
    peaks["all_time_rank"] = peaks.groupby(
        ["window_seasons", "peak_component"]
    ).cumcount() + 1
    current = pd.DataFrame(
        {
            "player_id": [1, 3],
            "player_name": ["Alpha Guard", "Current Rookie"],
            "offense_per_100": [4.0, 1.0],
            "defense_per_100": [1.0, 2.0],
            "net_per_100": [5.0, 3.0],
            "off_possessions": [6000, 800],
            "def_possessions": [5990, 810],
        }
    )
    matchup = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Season": [2024, 2024],
            "matchup_possessions": [2000.0, 500.0],
            "matchup_fga_suppressed_vs_scorer_p100_eb": [0.1, -0.1],
            "matchup_shotmaking_points_saved_vs_scorer_p100_eb": [2.0, -1.0],
            "matchup_three_pa_suppressed_vs_scorer_p100_eb": [0.2, -0.2],
            "matchup_turnovers_forced_vs_scorer_p100_eb": [0.3, -0.3],
            "matchup_assists_suppressed_vs_scorer_p100_eb": [0.4, -0.4],
            "matchup_shooting_fouls_prevented_vs_scorer_p100_eb": [0.5, -0.5],
        }
    )
    uncertainty = pd.DataFrame(
        {
            "player_id": [1, 3],
            "player_name": ["Alpha Guard", "Current Rookie"],
            "off_possessions": [5000, 800],
            "def_possessions": [4990, 810],
            "uncertainty_method": ["whole_game_bootstrap", "whole_game_bootstrap"],
            "uncertainty_status": ["complete", "complete"],
        }
    )
    for component, estimates in {
        "offense": [4.0, 1.0],
        "defense": [1.0, 2.0],
        "net": [5.0, 3.0],
    }.items():
        uncertainty[f"{component}_estimate"] = estimates
        uncertainty[f"{component}_analytic_se"] = [0.5, 0.8]
        uncertainty[f"{component}_bootstrap_se"] = [0.6, 0.9]
        uncertainty[f"{component}_ci80_low"] = [value - 1 for value in estimates]
        uncertainty[f"{component}_ci80_high"] = [value + 1 for value in estimates]
        uncertainty[f"{component}_ci95_low"] = [value - 2 for value in estimates]
        uncertainty[f"{component}_ci95_high"] = [value + 2 for value in estimates]
        uncertainty[f"{component}_probability_above_zero"] = [1.0, 0.9]
        uncertainty[f"{component}_draw_coverage"] = [1000, 1000]
    annual.to_parquet(annual_dir / "ratings.parquet", index=False)
    rolling.to_parquet(rolling_dir / "rolling_ratings.parquet", index=False)
    peaks.to_parquet(rolling_dir / "player_peaks.parquet", index=False)
    current.to_parquet(current_dir / "ratings.parquet", index=False)
    matchup.to_parquet(matchup_dir / "features.parquet", index=False)
    uncertainty.to_parquet(uncertainty_dir / "ratings_uncertainty.parquet", index=False)
    offense_roles = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "Season": [2024],
            "off_role_cluster": ["off_role_0"],
            "off_role_confidence": [0.6],
            **{f"off_role_affinity_{index}": [[0.6, 0.2, 0.1, 0.05, 0.03, 0.02][index]] for index in range(6)},
        }
    )
    defense_roles = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "Season": [2024],
            "def_role_cluster": ["def_role_4"],
            "def_role_confidence": [0.7],
            **{f"def_role_affinity_{index}": [[0.05, 0.1, 0.05, 0.1, 0.7][index]] for index in range(5)},
        }
    )
    offense_roles.to_parquet(roles_dir / "offense_assignments.parquet", index=False)
    defense_roles.to_parquet(roles_dir / "defense_assignments.parquet", index=False)
    stable_offense = offense_roles.copy()
    stable_offense["off_role_stable_cluster"] = "off_role_1"
    stable_offense["off_role_stable_confidence"] = 0.55
    stable_defense = defense_roles.copy()
    stable_defense["def_role_stable_cluster"] = "def_role_3"
    stable_defense["def_role_stable_confidence"] = 0.6
    for index in range(6):
        stable_offense[f"off_role_stable_affinity_{index}"] = [
            0.1,
            0.55,
            0.1,
            0.1,
            0.1,
            0.05,
        ][index]
    for index in range(5):
        stable_defense[f"def_role_stable_affinity_{index}"] = [
            0.1,
            0.1,
            0.1,
            0.6,
            0.1,
        ][index]
    stable_offense.to_parquet(
        stabilization_dir / "offense_assignments.parquet", index=False
    )
    stable_defense.to_parquet(
        stabilization_dir / "defense_assignments.parquet", index=False
    )
    (roles_dir / "run.json").write_text(
        json.dumps({"run_id": role_run_id, "status": "validated_research_input"})
    )
    (stabilization_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": stabilization_run_id,
                "status": "validated_descriptive_stabilization",
            }
        )
    )
    (annual_dir / "run.json").write_text(
        json.dumps(
            {"status": "research", "estimand": "annual", "caveats": ["annual caveat"]}
        )
    )
    (rolling_dir / "run.json").write_text(
        json.dumps(
            {"status": "research", "estimand": "rolling", "caveats": ["rolling caveat"]}
        )
    )
    (current_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "research_frozen_baseline",
                "estimand": "current",
                "config": {"seasons": [2024, 2025, 2026]},
                "caveats": ["current caveat"],
            }
        )
    )
    (matchup_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "validated",
                "caveat": "Research assignment data is not causal defense.",
            }
        )
    )
    (uncertainty_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "uncertainty_test",
                "status": "research_uncertainty_complete",
                "estimand_id": "trailing_observed_lineup_rapm_v1",
                "config": {"seasons": [2025]},
                "caveats": ["bootstrap caveat"],
            }
        )
    )
    config = RatingsApiConfig(
        "ratings_api_v1",
        "annual_test",
        "rolling_test",
        "current_test",
        2,
        10,
        matchup_defense_run_id="matchup_test",
        normal_rapm_uncertainty_run_ids={"single_season_2025": "uncertainty_test"},
        side_roles_run_id=role_run_id,
        role_stabilization_run_id=stabilization_run_id,
    )
    return RatingsStore(config, tmp_path)


def test_annual_leaderboard_filters_and_ranks_after_filter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.annual_leaderboard(2024, minimum_possessions=1000)
    assert result["total"] == 1
    assert result["results"][0]["PLAYER_NAME"] == "Alpha Guard"
    assert result["results"][0]["rank"] == 1


def test_player_payload_contains_annual_rolling_and_peaks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.player(1)
    assert result is not None
    assert [row["Season"] for row in result["annual"]] == [2023, 2024]
    assert result["rolling"][0]["net"] == 5.0
    assert {row["peak_component"] for row in result["peaks"]} == {
        "offense",
        "defense",
        "net",
    }
    assert result["current_normal_rapm"]["net_per_100"] == 5.0
    assert result["roles"][0]["offense"]["primary_role"] == "Primary creator"
    assert result["roles"][0]["defense"]["memberships"][0]["label"] == "Interior / rim"
    assert (
        result["roles"][0]["offense"]["stabilized_primary_role"]
        == "Secondary handler"
    )


def test_dispatch_exposes_contract_and_rejects_invalid_metric(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status, payload = dispatch(store, "/v1/leaderboards/peaks?window=3&component=net")
    assert status == HTTPStatus.OK
    assert payload["results"][0]["PLAYER_NAME"] == "Alpha Guard"
    with pytest.raises(ValueError, match="unsupported annual metric"):
        dispatch(store, "/v1/leaderboards/annual?season=2024&metric=made_up")
    with pytest.raises(ValueError, match="unsupported peak window"):
        dispatch(store, "/v1/leaderboards/peaks?window=7&component=net")


def test_search_prefers_exact_and_prefix_matches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.search_players("Alpha")
    assert result["results"] == [{"PLAYER_ID": 1, "PLAYER_NAME": "Alpha Guard"}]


def test_current_leaderboard_uses_minimum_side_possessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.current_leaderboard("net", minimum_possessions=1000)
    assert result["seasons"] == [2024, 2025, 2026]
    assert result["total"] == 1
    assert result["results"][0]["player_name"] == "Alpha Guard"


def test_matchup_factor_route_is_filtered_and_research_labeled(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status, result = dispatch(
        store,
        "/v1/leaderboards/matchup-defense?season=2024&minimum_matchup_possessions=1000",
    )
    assert status == HTTPStatus.OK
    assert result["status"] == "research_only"
    assert result["total"] == 1
    assert result["results"][0]["PLAYER_NAME"] == "Alpha Guard"
    assert result["results"][0][
        "matchup_shotmaking_points_saved_vs_scorer_p100_eb"
    ] == 2.0
    assert store.player(1)["matchup_defense_factors"][0]["Season"] == 2024


def test_v2_wrap_adds_lineage_without_changing_v1_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    v1_status, v1 = dispatch(store, "/v1/leaderboards/current?metric=net")
    v2_status, v2 = dispatch(store, "/v2/leaderboards/current?metric=net")
    assert v1_status == v2_status == HTTPStatus.OK
    assert v2["contract_version"] == "ratings_api_v2"
    assert v2["data"]["metric"] == v1["metric"]
    assert v2["data"]["results"][0]["net_per_100"] == v1["results"][0]["net_per_100"]
    assert "uncertainty" in v2["data"]["results"][0]
    assert v2["lineage"]["estimand_id"] == "trailing_observed_lineup_rapm_v1"
    assert len(v2["lineage"]["row_set_sha256"]) == 64


def test_v2_scoped_uncertainty_never_reuses_current_rating_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status, result = dispatch(
        store,
        "/v2/leaderboards/normal-rapm-uncertainty?scope=single_season_2025&minimum_possessions=1000",
    )
    assert status == HTTPStatus.OK
    assert result["contract_version"] == "ratings_api_v2"
    assert result["data"]["seasons"] == [2025]
    assert result["data"]["total"] == 1
    row = result["data"]["results"][0]
    assert row["uncertainty"]["components"]["net"]["interval_95"] == {
        "low": 3.0,
        "high": 7.0,
    }
    player = store.player(1)
    assert player is not None
    assert player["normal_rapm_uncertainty"]["single_season_2025"]["seasons"] == [2025]
