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
    annual_dir.mkdir(parents=True)
    rolling_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)
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
    annual.to_parquet(annual_dir / "ratings.parquet", index=False)
    rolling.to_parquet(rolling_dir / "rolling_ratings.parquet", index=False)
    peaks.to_parquet(rolling_dir / "player_peaks.parquet", index=False)
    current.to_parquet(current_dir / "ratings.parquet", index=False)
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
    config = RatingsApiConfig(
        "ratings_api_v1", "annual_test", "rolling_test", "current_test", 2, 10
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
