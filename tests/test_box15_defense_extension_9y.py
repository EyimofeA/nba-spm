from pathlib import Path

import pandas as pd

from research.run_box15_defense_extension_9y import _load_contract, _select_alpha


ROOT = Path(__file__).resolve().parents[1]


def test_contract_keeps_defense_only_and_consecutive_folds() -> None:
    contract = _load_contract()
    assert contract["residual_model"]["side"] == "defense"
    assert contract["test_seasons"] == [season + 1 for season in contract["rating_seasons"]]
    assert contract["candidates"]["box15_9y_normal"] == []


def test_followup_contract_name_matches_experiment_id() -> None:
    path = ROOT / "research/experiments/box15_defense_mechanism_followup_9y_v1.yml"
    contract = _load_contract(path)
    assert contract["experiment_id"] == path.stem


def test_alpha_selection_uses_only_supplied_training_seasons() -> None:
    frame = pd.DataFrame(
        {
            "Window_End": [2014, 2014, 2015, 2015],
            "feature": [0.0, 1.0, 0.0, 1.0],
            "residual_target": [0.0, 1.0, 0.0, 1.0],
            "sample_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    selected, scores = _select_alpha(frame, ("feature",), (1.0, 10.0))
    assert selected in {1.0, 10.0}
    assert set(scores["training_seasons"]) == {2}
