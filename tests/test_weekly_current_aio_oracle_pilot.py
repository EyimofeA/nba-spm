from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "research/run_weekly_current_aio_oracle_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("weekly_current_aio_oracle", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {SCRIPT}.")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_weekly_oracle_expansion_uses_ledger_games_only() -> None:
    predictions = pd.DataFrame(
        {
            "test_season": [2026, 2026, 2026],
            "game_id": ["a", "b", "c"],
            "actual_margin": [1.0, 2.0, 3.0],
            "predicted_margin": [0.0, 0.0, 0.0],
            "unknown_player_slots": [0, 0, 0],
            "arm": [PILOT.SELECTED_ARM] * 3,
            "squared_error": [1.0, 4.0, 9.0],
        }
    )
    game_dates = pd.DataFrame(
        {
            "season_end": [2026, 2026, 2026],
            "game_id": ["a", "b", "c"],
            "game_date": pd.to_datetime(
                ["2025-11-03", "2025-11-16", "2025-11-17"]
            ),
        }
    )
    ledger_source = game_dates.rename(
        columns={"season_end": "season", "game_date": "date", "game_id": "gameid"}
    )
    ledger = PILOT.build_weekly_cutoff_ledger(
        ledger_source, target_season=2026
    ).head(1)

    output = PILOT.expand_weekly_oracle_predictions(
        predictions, game_dates, ledger
    )

    assert output["game_id"].tolist() == ["a", "b"]
    assert output["cutoff_date"].eq(pd.Timestamp("2025-11-03")).all()
    assert output["rating_information_end_season"].eq(2025).all()


def test_rating_population_excludes_only_players_without_model_evidence() -> None:
    ratings = pd.DataFrame(
        {
            "target_season": [2024, 2024, 2024],
            "PLAYER_ID": [1, 2, 3],
            "PLAYER_NAME": ["History", "Prior", "Neither"],
            "Poss_Off": [10.0, 0.0, 0.0],
            "Poss_Def": [10.0, 0.0, 0.0],
            "offense": [1.0, 2.0, 0.0],
            "defense": [0.0, 0.0, 0.0],
            "net": [1.0, 2.0, 0.0],
            "arm": [PILOT.SELECTED_ARM] * 3,
        }
    )
    player_games = pd.DataFrame(
        {
            "season_end": [2024, 2024, 2024],
            "season_type": ["regular"] * 3,
            "played": [True] * 3,
            "player_id": [1, 2, 3],
        }
    )
    priors = pd.DataFrame({"Target_Season": [2024], "PLAYER_ID": [2]})

    eligible, audit = PILOT.evidence_backed_ratings(
        ratings, player_games, priors
    )

    assert eligible["PLAYER_ID"].tolist() == [2, 1]
    assert audit.iloc[0]["excluded_no_history_or_prior"] == 1
