from __future__ import annotations

from nba_impact.models.win_probability_mlp import SEEDS, build_mlp


def test_mlp_architecture_and_seeds_are_frozen() -> None:
    assert SEEDS == (7, 17, 29, 43, 71)
    model = build_mlp(seed=7).named_steps["mlp"]
    assert model.hidden_layer_sizes == (64, 64)
    assert model.batch_size == 1024
    assert model.max_iter == 100
    assert model.early_stopping is True
    assert model.random_state == 7
