from __future__ import annotations

import numpy as np
import pandas as pd

from research.run_pulse_model_screen import _represent


def _panel() -> pd.DataFrame:
    rows = []
    for season, shift in ((2020, 0.0), (2021, 100.0)):
        for player, weight in ((1, 100.0), (2, 400.0), (3, 900.0)):
            row = {
                "PLAYER_ID": player,
                "Window_End": season,
                "OffPoss": weight,
                "DefPoss": weight,
            }
            for index in range(15):
                row[f"feature_{index}"] = shift + player + index
            rows.append(row)
    return pd.DataFrame(rows)


def test_era_representations_are_season_local(monkeypatch) -> None:
    import research.run_pulse_model_screen as screen

    monkeypatch.setattr(
        screen, "BOX_PIPM_STYLE_FEATURES", tuple(f"feature_{i}" for i in range(15))
    )
    panel = _panel()
    relative = _represent(panel, "season_relative")
    standardized = _represent(panel, "season_standardized")
    weights = np.sqrt(panel["OffPoss"])
    for season in (2020, 2021):
        mask = panel["Window_End"].eq(season)
        assert np.isclose(
            np.average(relative.loc[mask, "feature_0"], weights=weights.loc[mask]),
            0.0,
        )
        assert np.isclose(
            np.average(standardized.loc[mask, "feature_0"], weights=weights.loc[mask]),
            0.0,
        )
        assert np.isclose(
            np.average(
                standardized.loc[mask, "feature_0"] ** 2,
                weights=weights.loc[mask],
            ),
            1.0,
        )


def test_raw_representation_does_not_mutate_panel(monkeypatch) -> None:
    import research.run_pulse_model_screen as screen

    monkeypatch.setattr(
        screen, "BOX_PIPM_STYLE_FEATURES", tuple(f"feature_{i}" for i in range(15))
    )
    panel = _panel()
    raw = _represent(panel, "raw")
    raw.loc[0, "feature_0"] = -999
    assert panel.loc[0, "feature_0"] != -999
