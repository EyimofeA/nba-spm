from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_pulse_factor_ledger_reconciles_exactly() -> None:
    runs = sorted((ROOT / "artifacts/models/pulse_decomposition").glob("pulse_decomposition_v1_*"))
    if not runs:
        return
    frame = pd.read_parquet(runs[-1] / "factor_ledger.parquet")
    for prefix in ("rapm", "pulse_prior", "lineup_update", "pulse"):
        assert np.allclose(frame[f"{prefix}_offense"] + frame[f"{prefix}_defense"], frame[f"{prefix}_net"])
        for side in ("offense", "defense"):
            contributions = [
                column for column in frame
                if column.startswith(f"{prefix}_{side}_") and column.endswith("_contribution")
            ]
            reconstructed = frame[contributions].sum(axis=1) + frame[f"{prefix}_{side}_residual"]
            assert np.allclose(reconstructed, frame[f"{prefix}_{side}"])
    assert np.allclose(frame["pulse_prior_net"] + frame["lineup_update_net"], frame["pulse_net"])
