from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_pipm_four_way_comparison import MODEL_ORDER, paired_game_bootstrap  # noqa: E402


def test_paired_game_bootstrap_compares_all_four_models_on_same_games() -> None:
    rows = []
    actual = [1.0, -2.0, 3.0, -4.0]
    offsets = {
        "box_prior": 2.0,
        "pipm_reference": 1.5,
        "box_prior_plus_rapm": 1.0,
        "pipm_reference_plus_rapm": 0.5,
    }
    for candidate in MODEL_ORDER:
        for game, value in enumerate(actual):
            rows.append(
                {
                    "game_id": str(game),
                    "test_season": 2022,
                    "candidate": candidate,
                    "actual_margin": value,
                    "predicted_margin": value + offsets[candidate],
                }
            )
    models, pairs = paired_game_bootstrap(pd.DataFrame(rows), draws=200, seed=7)

    assert len(models) == 4
    assert len(pairs) == 6
    assert models.iloc[0]["candidate"] == "pipm_reference_plus_rapm"
    assert models.iloc[0]["probability_best"] == 1.0
    assert int(pairs["primary_comparison"].sum()) == 4


def test_paired_game_bootstrap_is_deterministic() -> None:
    rows = []
    for candidate_index, candidate in enumerate(MODEL_ORDER):
        for game in range(6):
            rows.append(
                {
                    "game_id": str(game),
                    "test_season": 2022,
                    "candidate": candidate,
                    "actual_margin": float(game),
                    "predicted_margin": float(game + candidate_index),
                }
            )
    games = pd.DataFrame(rows)
    first = paired_game_bootstrap(games, draws=50, seed=11)
    second = paired_game_bootstrap(games, draws=50, seed=11)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
