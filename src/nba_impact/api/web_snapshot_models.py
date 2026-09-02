"""Optional rating extensions for the public web snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def attach_rich_spm(
    historical: pd.DataFrame, path: str | Path | None
) -> pd.DataFrame:
    """Attach a validated Rich SPM player-season panel."""
    if path is None:
        return historical
    rich = pd.read_parquet(path).rename(
        columns={
            "rating_season": "Season",
            "spm_impact_offense": "rich_spm_offense",
            "spm_impact_defense": "rich_spm_defense",
            "spm_impact_net": "rich_spm_net",
        }
    )
    required = {
        "PLAYER_ID", "Season", "rich_spm_offense", "rich_spm_defense", "rich_spm_net",
    }
    if missing := sorted(required - set(rich.columns)):
        raise ValueError(f"Rich SPM ratings are missing columns: {missing}")
    rich = rich[[*sorted(required)]].copy()
    if rich.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Rich SPM ratings contain duplicate player-seasons.")
    if not np.allclose(
        rich["rich_spm_net"],
        rich["rich_spm_offense"] + rich["rich_spm_defense"],
        atol=1e-10,
    ):
        raise ValueError("Rich SPM ratings violate the side identity.")
    return historical.merge(
        rich, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    )
