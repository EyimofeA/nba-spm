import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from research.run_annual_rich_forward_aio import (
    MODEL_ORDER,
    _annual_rich_priors,
    _assert_identical_games,
)


def test_annual_rich_priors_join_frozen_sides() -> None:
    rows = []
    folds = (
        (2021, "learner_selection"),
        (2022, "diagnostic"),
        (2023, "diagnostic"),
        (2024, "diagnostic"),
        (2025, "diagnostic"),
    )
    for season, phase in folds:
        rows.extend(
            [
                {
                    "PLAYER_ID": 1,
                    "Season": season,
                    "phase": phase,
                    "arm": "audited_all",
                    "side": "offense",
                    "learner": "elastic_net",
                    "prediction": 2.0,
                },
                {
                    "PLAYER_ID": 1,
                    "Season": season,
                    "phase": phase,
                    "arm": "audited_all",
                    "side": "defense",
                    "learner": "ridge",
                    "prediction": 1.0,
                },
            ]
        )
    priors = _annual_rich_priors(pd.DataFrame(rows))
    assert len(priors) == 5
    assert priors["prior_net_per_100"].eq(3.0).all()
    assert priors["candidate"].eq("annual_rich_spm").all()


def test_identical_games_accepts_aligned_candidates() -> None:
    rows = []
    for candidate in MODEL_ORDER:
        for game_id, margin in ((1, 5.0), (2, -3.0)):
            rows.append(
                {
                    "candidate": candidate,
                    "test_season": 2023,
                    "game_id": game_id,
                    "actual_margin": margin,
                }
            )
    _assert_identical_games(pd.DataFrame(rows))
