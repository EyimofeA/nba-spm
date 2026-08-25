from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).parents[1]


def _module():
    path = REPO_ROOT / "research" / "rapm_lab" / "run_lambda_frontier.py"
    spec = importlib.util.spec_from_file_location("run_lambda_frontier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_scalar_stage_is_broad_and_deterministic() -> None:
    module = _module()
    contract = json.loads(
        (
            REPO_ROOT
            / "research"
            / "experiments"
            / "rolling_5y_lambda_frontier_v1.json"
        ).read_text()
    )
    first = module._scalar_stage_one(contract)
    second = module._scalar_stage_one(contract)
    assert first == second
    assert len(first) == 71
    assert min(candidate["lambda_off"] for candidate in first) < 100.0
    assert max(candidate["lambda_off"] for candidate in first) >= 30_000.0
    assert min(candidate["lambda_def"] for candidate in first) < 100.0
    assert max(candidate["lambda_def"] for candidate in first) >= 30_000.0


def test_frontier_selection_rules_remain_distinct() -> None:
    module = _module()
    summary = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "mean_correlation": [0.40, 0.3997, 0.38],
            "mean_rmse": [14.0, 13.9, 13.0],
            "mean_log_gcv": [0.30, 0.29, 0.31],
            "penalty_scale": [1000.0, 2000.0, 5000.0],
        }
    )
    assert module._select(summary, "selection_correlation")["candidate_id"] == "b"
    assert module._select(summary, "selection_rmse")["candidate_id"] == "c"
    assert module._select(summary, "training_gcv")["candidate_id"] == "b"
