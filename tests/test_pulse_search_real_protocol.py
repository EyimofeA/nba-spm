from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nba_impact.models.rapm import RapmConfig
from nba_impact.models.statistical_impact import BOX_FEATURES
from nba_impact.models.stint_rapm import StintRapmDesign, build_stint_design
from research.pulse_search_protocol import (
    add_z_increment,
    fit_box15_prior,
    fit_constrained,
    fit_many_centers,
    game_split_mask,
    hustle_rates,
    segments_to_stints_mode,
)


def test_classic_score_string_is_away_then_home() -> None:
    from research.run_pulse_search_fetch_support import classic_event_scores

    path = Path("/workspace/data/lake/bronze/canonical_historical_events/season=2014/regular.parquet")
    if not path.exists():
        return
    scores = classic_event_scores(2014)
    opener = scores.loc[scores["game_id"].eq("0021300001")]
    if opener.empty:
        return
    assert int(opener["away_score"].iloc[0]) == 87
    assert int(opener["home_score"].iloc[0]) == 97


def _panel(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    rng = np.random.default_rng(7)
    for season in range(2010, 2018):
        for player in range(1, n + 1):
            rates = rng.normal(size=len(BOX_FEATURES))
            rows.append({"PLAYER_ID": player, "Window_End": season, **dict(zip(BOX_FEATURES, rates))})
            labels.append({
                "PLAYER_ID": player,
                "Window_End": season,
                "target_offense": float(rates[0] - rates[2]),
                "target_defense": float(rates[3] + rates[4]),
                "Poss_Off": 800,
                "Poss_Def": 800,
            })
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_box15_prior_is_past_only() -> None:
    features, targets = _panel()
    priors, quality = fit_box15_prior(features, targets, training_before=2016)
    assert quality["training_end"] == 2015
    assert quality["training_start"] == 2010
    assert set(priors["Window_End"].unique()) == {2016}
    assert quality["training_rows"] == 40 * 6


def test_fractional_possessions_sum_to_one_per_possession() -> None:
    possessions = pd.DataFrame({
        "possession_id": ["a", "b"],
        "points": [2, 3],
        "points_excl_tech": [2, 3],
        "technical_points": [0, 0],
        "offense_is_home": [True, False],
        "season_end": [2024, 2024],
        "game_id": ["0022300001", "0022300001"],
    })
    segments = pd.DataFrame({
        "possession_id": ["a", "a", "b"],
        "segment_number": [1, 2, 1],
        "home_player_1": [1, 1, 6],
        "home_player_2": [2, 2, 7],
        "home_player_3": [3, 3, 8],
        "home_player_4": [4, 4, 9],
        "home_player_5": [5, 11, 10],
        "away_player_1": [21, 21, 31],
        "away_player_2": [22, 22, 32],
        "away_player_3": [23, 23, 33],
        "away_player_4": [24, 24, 34],
        "away_player_5": [25, 25, 35],
    })
    stints = segments_to_stints_mode(possessions, segments, mode="fractional")
    assert abs(float(stints["home_possessions"].sum()) - 1.0) < 1e-9
    assert abs(float(stints["away_possessions"].sum()) - 1.0) < 1e-9
    assert abs(float(stints["home_points"].sum()) - 2.0) < 1e-9
    assert abs(float(stints["away_points"].sum()) - 3.0) < 1e-9


def test_contest_increment_moves_defense_center() -> None:
    players = np.array([1, 2, 3], dtype=int)
    design = StintRapmDesign(
        X=csr_matrix((2, 7)),
        points=np.array([1.0, 1.0]),
        possessions=np.array([1.0, 1.0]),
        players=players,
        game_ids=np.array(["a", "a"]),
        seasons=np.array([2025, 2025]),
        home_offense=np.array([True, False]),
        off_possessions=np.array([10.0, 10.0, 10.0]),
        def_possessions=np.array([400.0, 400.0, 400.0]),
    )
    center = np.zeros(7)
    features = pd.DataFrame({
        "PLAYER_ID": [1, 2, 3],
        "gabriel_contested_2pt_p100": [10.0, 0.0, -10.0],
    })
    shifted = add_z_increment(center, design, features, ("gabriel_contested_2pt_p100",), 1.0)
    assert shifted[3] != 0.0
    assert np.sign(shifted[3]) == -np.sign(shifted[5])


def test_hustle_rates_require_200_possessions() -> None:
    hustle = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "year": [2024, 2024],
        "CONTESTED_SHOTS_2PT": [40, 40],
        "CONTESTED_SHOTS_3PT": [0, 0],
        "DEFLECTIONS": [0, 0],
        "CHARGES_DRAWN": [0, 0],
        "DEF_BOXOUTS": [0, 0],
        "DEF_LOOSE_BALLS_RECOVERED": [0, 0],
    })
    rates = hustle_rates(hustle, 2024, np.array([1, 2]), np.array([500.0, 50.0]))
    assert rates.loc[rates["PLAYER_ID"].eq(1), "gabriel_contested_2pt_p100"].iloc[0] == 8.0
    assert rates.loc[rates["PLAYER_ID"].eq(2), "gabriel_contested_2pt_p100"].iloc[0] == 0.0


def _tiny_design() -> StintRapmDesign:
    frame = pd.DataFrame({
        "season": [2024, 2024],
        "game_id": ["0022300001", "0022300002"],
        "home_possessions": [10, 12],
        "away_possessions": [10, 11],
        "home_points": [12, 14],
        "away_points": [10, 11],
        "home_player_1": [1, 1],
        "home_player_2": [2, 2],
        "home_player_3": [3, 3],
        "home_player_4": [4, 4],
        "home_player_5": [5, 6],
        "away_player_1": [11, 11],
        "away_player_2": [12, 12],
        "away_player_3": [13, 13],
        "away_player_4": [14, 14],
        "away_player_5": [15, 16],
    })
    return build_stint_design(frame)


def test_constrained_fit_zeros_weighted_means() -> None:
    design = _tiny_design()
    config = RapmConfig((2024,), lambda_off=10.0, lambda_def=10.0, lambda_home=10.0, data_scope="test")
    center = np.zeros(design.X.shape[1])
    beta, _intercept = fit_constrained(design, config, center)
    n = len(design.players)
    off = float(np.average(beta[:n], weights=design.off_possessions))
    defense = float(np.average(beta[n:2 * n], weights=design.def_possessions))
    assert abs(off) < 1e-6
    assert abs(defense) < 1e-6


def test_many_centers_and_half_mask() -> None:
    design = _tiny_design()
    config = RapmConfig((2024,), lambda_off=50.0, lambda_def=50.0, lambda_home=10.0, data_scope="test")
    zero = np.zeros(design.X.shape[1])
    fits = fit_many_centers(design, config, {"rapm": zero, "prior": zero + 0.01})
    assert set(fits) == {"rapm", "prior"}
    mask = game_split_mask(design, second_half=False)
    assert mask.sum() == 2
    assert game_split_mask(design, second_half=True).sum() == 2
