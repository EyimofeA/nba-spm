from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nba_impact.models.five_year_spm_context import pool_five_year_context
from nba_impact.models.full_feature_factor_spm import (
    DEFENSE_CONTEXT,
    OFFENSE_CONTEXT,
    _feature_banks,
)


def test_full_feature_banks_record_missing_fields(tmp_path) -> None:
    manifest = tmp_path / "run.json"
    manifest.write_text(
        json.dumps(
            {
                "features": {
                    "offense": ["a", "b"],
                    "defense": ["c", "d"],
                }
            }
        )
    )
    available, missing = _feature_banks(manifest, {"a", "b", "c"})
    assert available == {"offense": ("a", "b"), "defense": ("c",)}
    assert missing == {"offense": (), "defense": ("d",)}
    assert len(OFFENSE_CONTEXT) == len(DEFENSE_CONTEXT) == 6


def test_five_year_context_uses_focal_player_possession_weights() -> None:
    rows = []
    for season, weight, value in ((2020, 100.0, 1.0), (2021, 300.0, 3.0)):
        row = {
            "PLAYER_ID": 1,
            "Season": season,
            "OffPoss": weight,
            "DefPoss": weight,
        }
        for field in (*OFFENSE_CONTEXT, *DEFENSE_CONTEXT):
            row[field] = value
        rows.append(row)
    output = pool_five_year_context(pd.DataFrame(rows), window_ends=(2021,))
    assert len(output) == 1
    for field in (*OFFENSE_CONTEXT, *DEFENSE_CONTEXT):
        assert np.isclose(output.loc[0, field], 2.5)
