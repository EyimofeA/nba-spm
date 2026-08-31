from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "research/rapm_lab/run_scorer_owned_current_rapm.py"
SPEC = importlib.util.spec_from_file_location("score_ledger_current_rapm", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame([{
        "game_id": "0000000001", "season": 2024, "home_team_id": 1,
        "away_team_id": 2, "home_score": 4, "away_score": 3,
    }])
    actions = pd.DataFrame([
        {"game_id": "0000000001", "season": 2024, "orderNumber": 10, "scoreHome": 0, "scoreAway": 0, "actionType": "jumpball", "description": "start"},
        {"game_id": "0000000001", "season": 2024, "orderNumber": 20, "scoreHome": 2, "scoreAway": 0, "actionType": "2pt", "description": "home score"},
        # The home technical does not consume the listed away possession.
        {"game_id": "0000000001", "season": 2024, "orderNumber": 30, "scoreHome": 3, "scoreAway": 0, "actionType": "freethrow", "description": "home technical Free Throw 1 of 1"},
        # The listed away possession also contains an unresolved nontechnical home score.
        {"game_id": "0000000001", "season": 2024, "orderNumber": 40, "scoreHome": 4, "scoreAway": 0, "actionType": "2pt", "description": "cross owner"},
        {"game_id": "0000000001", "season": 2024, "orderNumber": 50, "scoreHome": 4, "scoreAway": 3, "actionType": "3pt", "description": "away score"},
    ])
    possessions = pd.DataFrame([
        {"possession_id": "p1", "game_id": "0000000001", "possession_number": 1, "season_end": 2024, "offense_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_order_number": 10, "end_order_number": 20},
        {"possession_id": "p2", "game_id": "0000000001", "possession_number": 2, "season_end": 2024, "offense_team_id": 2, "home_team_id": 1, "away_team_id": 2, "start_order_number": 21, "end_order_number": 50},
    ])
    return actions, games, possessions


def test_adjusted_targets_keep_one_canonical_row_and_reconcile_score_ledger() -> None:
    actions, games, possessions = _inputs()
    targets, ledger = MODULE.build_adjusted_targets(actions, games, possessions)

    assert targets["possession_number"].tolist() == [1, 2]
    assert targets["retained_points"].tolist() == [2.0, 3.0]
    row = ledger.iloc[0]
    assert row["official_points"] == 7.0
    assert row["retained_own_possession_points"] == 5.0
    assert row["technical_ft_points"] == 1.0
    assert row["cross_owner_nontechnical_points"] == 1.0
    assert row["unmapped_points"] == 0.0
    assert row["raw_score_conserved"] and row["decomposition_conserved"]
    assert row["adjusted_margin"] == -1.0


def test_adjusted_targets_flag_when_raw_score_does_not_match_official_score() -> None:
    actions, games, possessions = _inputs()
    games.loc[0, "away_score"] = 4
    _, ledger = MODULE.build_adjusted_targets(actions, games, possessions)
    assert not ledger.loc[0, "raw_score_conserved"]
    assert not ledger.loc[0, "decomposition_conserved"]
