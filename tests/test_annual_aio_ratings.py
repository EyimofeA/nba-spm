from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.annual_aio_ratings import (
    build_current_annual_aio_ratings,
    build_unified_annual_aio_ratings,
    fit_annual_aio_season,
)
from nba_impact.models.rapm import RapmConfig
from tests.test_current_single_season_rapm import _write_current_inputs


def test_annual_aio_ratings_preserve_decomposition() -> None:
    rng = np.random.default_rng(11)
    players = np.arange(1, 13)
    rows = []
    for possession in range(120):
        lineup = rng.choice(players, size=10, replace=False)
        rows.append(
            {
                "home_poss": bool(possession % 2),
                "pts": float(1 + (lineup[5:10].sum() - lineup[:5].sum()) * 0.002),
                **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                "season": 2024,
                "date": "2023-11-01",
                "period": 1,
                "num": possession + 1,
                "gameid": f"00223000{possession // 20}",
            }
        )
    priors = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Window_End": 2024,
            "prior_offense_per_100": (players - players.mean()) * 0.2,
            "prior_defense_per_100": (players.mean() - players) * 0.1,
        }
    )
    priors["prior_net_per_100"] = (
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    )
    ratings, quality = fit_annual_aio_season(
        pd.DataFrame(rows),
        priors,
        RapmConfig(seasons=(2024,), lambda_off=50, lambda_def=50, lambda_home=10),
        season=2024,
    )
    assert len(ratings) == len(players)
    assert ratings["prior_available"].all()
    np.testing.assert_allclose(
        ratings["aio_net"], ratings["aio_offense"] + ratings["aio_defense"]
    )
    np.testing.assert_allclose(
        ratings["aio_net"],
        ratings["spm_center_net"] + ratings["rapm_update_net"],
    )
    assert quality["max_component_identity_error"] < 1e-12


def test_current_annual_aio_uses_canonical_terminal_lineups(tmp_path) -> None:
    possessions, segments, names, player_games = _write_current_inputs(tmp_path)
    player_ids = list(range(1, 36))
    priors = pd.DataFrame(
        {
            "PLAYER_ID": player_ids * 2,
            "Window_End": [2024] * len(player_ids) + [2025] * len(player_ids),
            "prior_offense_per_100": np.tile(np.linspace(-1, 1, len(player_ids)), 2),
            "prior_defense_per_100": np.tile(np.linspace(1, -1, len(player_ids)), 2),
        }
    )
    priors["prior_net_per_100"] = (
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    )
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    priors_path = prior_dir / "priors.parquet"
    priors.to_parquet(priors_path, index=False)
    (prior_dir / "run.json").write_text('{"run_id":"test-priors","config":{"training_rule":"leave_one_season_out"}}')
    run = build_current_annual_aio_ratings(
        possessions,
        segments,
        priors_path,
        names,
        player_games,
        artifact_root=tmp_path,
        seasons=(2024, 2025),
        lambda_off=10.0,
        lambda_def=10.0,
        lambda_home=1.0,
    )
    ratings = pd.read_parquet(Path(run["artifact_path"]) / "ratings.parquet")
    assert set(ratings["Season"]) == {2024, 2025}
    np.testing.assert_allclose(ratings["aio_net"], ratings["aio_offense"] + ratings["aio_defense"])
    assert run["config"]["lineup_policy"] == "terminal"


def test_unified_annual_aio_preserves_source_provenance(tmp_path) -> None:
    possessions, segments, names, player_games = _write_current_inputs(tmp_path)
    player_ids = list(range(1, 36))
    priors = pd.DataFrame(
        {
            "PLAYER_ID": player_ids * 2,
            "Window_End": [2024] * len(player_ids) + [2025] * len(player_ids),
            "prior_offense_per_100": np.tile(np.linspace(-1, 1, len(player_ids)), 2),
            "prior_defense_per_100": np.tile(np.linspace(1, -1, len(player_ids)), 2),
        }
    )
    priors["prior_net_per_100"] = priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    priors_path = prior_dir / "priors.parquet"
    priors.to_parquet(priors_path, index=False)
    (prior_dir / "run.json").write_text('{"run_id":"test-priors","config":{"training_rule":"leave_one_season_out"}}')
    run = build_unified_annual_aio_ratings(
        tmp_path / "unused_legacy_cache",
        possessions,
        segments,
        priors_path,
        names,
        player_games,
        artifact_root=tmp_path,
        seasons=(2024, 2025),
        lambda_off=10.0,
        lambda_def=10.0,
        lambda_home=1.0,
    )
    ratings = pd.read_parquet(Path(run["artifact_path"]) / "ratings.parquet")
    quality = pd.read_parquet(Path(run["artifact_path"]) / "season_quality.parquet")
    assert set(ratings["Season"]) == {2024, 2025}
    assert set(quality["rapm_input_source"]) == {"canonical_event_terminal"}
    assert run["config"]["source_by_season"] == {
        "2024": "canonical_event_terminal",
        "2025": "canonical_event_terminal",
    }
