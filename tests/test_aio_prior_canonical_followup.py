from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
SPEC = importlib.util.spec_from_file_location(
    "run_aio_prior_canonical_followup",
    RESEARCH / "run_aio_prior_canonical_followup.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_followup_spm_prior_loader_owns_2024_2025_scope(tmp_path: Path) -> None:
    rows = []
    for season in (2021, 2022, 2023, 2024, 2025):
        for variant in ("baseline", "selected_combined"):
            rows.append(
                {
                    "PLAYER_ID": 1,
                    "Window_End": season,
                    "variant": variant,
                    "prior_offense_per_100": 1.0,
                    "prior_defense_per_100": 2.0,
                    "prior_net_per_100": 3.0,
                }
            )
    path = tmp_path / "priors.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    loaded = MODULE._existing_spm_priors(path)

    assert set(loaded["Window_End"]) == {2024, 2025}
    assert set(loaded["candidate"]) == {
        "five_year_spm",
        "selected_five_year_spm",
    }


def test_followup_spm_prior_loader_rejects_missing_season(tmp_path: Path) -> None:
    path = tmp_path / "priors.parquet"
    pd.DataFrame(
        {
            "PLAYER_ID": [1, 1],
            "Window_End": [2024, 2024],
            "variant": ["baseline", "selected_combined"],
            "prior_offense_per_100": [1.0, 1.0],
            "prior_defense_per_100": [2.0, 2.0],
            "prior_net_per_100": [3.0, 3.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing seasons: \\[2025\\]"):
        MODULE._existing_spm_priors(path)
