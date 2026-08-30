from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.blocked_game_box_features import (
    BOX_COUNTS,
    aggregate_box15_features,
    audit_legacy_game_target_parity,
    player_game_exposure,
)


AWAY = (1, 2, 3, 4, 5)
HOME = (6, 7, 8, 9, 10)


def _possessions() -> pd.DataFrame:
    rows = []
    for game_id, home_poss in (("g1", 1), ("g1", 0), ("g2", 1)):
        rows.append(
            {
                "gameid": game_id,
                "home_poss": home_poss,
                **{f"a{index}": player for index, player in enumerate(AWAY, 1)},
                **{f"h{index}": player for index, player in enumerate(HOME, 1)},
            }
        )
    return pd.DataFrame(rows)


def _ledger() -> pd.DataFrame:
    exposure = player_game_exposure(_possessions())
    for field in BOX_COUNTS:
        exposure[field] = 0.0
    exposure.loc[
        exposure["game_id"].eq("g1") & exposure["PLAYER_ID"].eq(6), "PTS"
    ] = 2.0
    exposure.loc[
        exposure["game_id"].eq("g2") & exposure["PLAYER_ID"].eq(6), "AST"
    ] = 1.0
    return exposure


def test_player_game_exposure_assigns_both_sides() -> None:
    exposure = player_game_exposure(_possessions())
    home_g1 = exposure.loc[
        exposure["game_id"].eq("g1") & exposure["PLAYER_ID"].eq(6)
    ].iloc[0]
    away_g1 = exposure.loc[
        exposure["game_id"].eq("g1") & exposure["PLAYER_ID"].eq(1)
    ].iloc[0]
    assert (home_g1["OffPoss"], home_g1["DefPoss"]) == (1, 1)
    assert (away_g1["OffPoss"], away_g1["DefPoss"]) == (1, 1)
    assert not exposure.duplicated(["game_id", "PLAYER_ID"]).any()


def test_held_games_contribute_no_counts_or_possessions() -> None:
    ledger = _ledger()
    all_games = aggregate_box15_features(ledger, season=2021)
    training = aggregate_box15_features(
        ledger, season=2021, excluded_game_ids=("g2",)
    )
    all_player = all_games.set_index("PLAYER_ID").loc[6]
    train_player = training.set_index("PLAYER_ID").loc[6]
    assert all_player["OffPoss"] == 2
    assert train_player["OffPoss"] == 1
    assert all_player["AST_p100"] == 50.0
    assert train_player["AST_p100"] == 0.0


def test_all_minus_held_equals_training_aggregate() -> None:
    ledger = _ledger()
    raw = ledger.groupby("PLAYER_ID")[["OffPoss", "DefPoss", *BOX_COUNTS]].sum()
    held = (
        ledger.loc[ledger["game_id"].eq("g2")]
        .groupby("PLAYER_ID")[["OffPoss", "DefPoss", *BOX_COUNTS]]
        .sum()
    )
    training = (
        ledger.loc[~ledger["game_id"].eq("g2")]
        .groupby("PLAYER_ID")[["OffPoss", "DefPoss", *BOX_COUNTS]]
        .sum()
    )
    expected = raw.subtract(held, fill_value=0).sort_index()
    pd.testing.assert_frame_equal(expected, training.sort_index(), check_dtype=False)


def test_players_without_events_keep_finite_zero_rates() -> None:
    features = aggregate_box15_features(_ledger(), season=2021)
    player = features.set_index("PLAYER_ID").loc[1]
    values = player[[column for column in features if column.endswith("_p100")]]
    assert np.isfinite(values.astype(float)).all()
    assert values.astype(float).eq(0.0).all()


def test_game_target_parity_keeps_only_score_conserved_regulation() -> None:
    possessions = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "period": [1, 4, 1, 4, 1, 4],
            "home_poss": [1, 0, 1, 0, 1, 0],
            "pts": [100, 90, 100, 90, 100, 90],
        }
    )
    finals = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "scoreHome": [100, 101, 100],
            "scoreAway": [90, 90, 90],
            "official_max_period": [4, 4, 5],
        }
    )

    parity, quality = audit_legacy_game_target_parity(possessions, finals)

    assert parity.set_index("game_id")["strict_eligible"].to_dict() == {
        "g1": True,
        "g2": False,
        "g3": False,
    }
    assert quality == {
        "cache_games": 3,
        "complete_regulation_games": 2,
        "score_conserved_games": 2,
        "strict_eligible_games": 1,
        "overtime_or_incomplete_games": 1,
        "complete_regulation_score_mismatch_games": 1,
        "strict_eligible_fraction": 1 / 3,
    }
