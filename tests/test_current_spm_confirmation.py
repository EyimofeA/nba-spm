import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models import current_spm_confirmation as confirmation


class _FeatureModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["feature"].to_numpy()


def test_current_confirmation_records_historical_envelope(monkeypatch, tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {"PLAYER_ID": [1, 2], "Window_End": [2025, 2025], "feature": [1.0, -1.0]}
    ).to_parquet(features_path, index=False)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "run.json").write_text(
        json.dumps(
            {
                "run_id": "frozen_v1",
                "models": {
                    "offense": {"features": ["feature"]},
                    "defense": {"features": ["feature"]},
                },
            }
        )
    )
    for component in ("offense", "defense"):
        (frozen / f"model_{component}.joblib").write_bytes(component.encode())
    pd.DataFrame(
        {
            "test_season": [2023, 2024] * 3,
            "component": ["offense"] * 2 + ["defense"] * 2 + ["net"] * 2,
            "weighted_rmse": [0.5] * 6,
            "correlation": [0.5] * 6,
        }
    ).to_parquet(frozen / "fold_metrics.parquet", index=False)
    possessions_path = tmp_path / "possessions.parquet"
    segments_path = tmp_path / "segments.parquet"
    names_path = tmp_path / "names.csv"
    player_games_path = tmp_path / "player_games.parquet"
    for path in (possessions_path, segments_path, names_path, player_games_path):
        path.write_bytes(path.name.encode())

    monkeypatch.setattr(
        confirmation,
        "load_current_possessions",
        lambda *args, **kwargs: pd.DataFrame(
            {"season": [2025, 2025], "gameid": ["a", "b"]}
        ),
    )
    monkeypatch.setattr(confirmation, "build_design", lambda frame: object())
    monkeypatch.setattr(
        confirmation, "fit_coefficients", lambda design, config: (np.array([0.0]), 1.1)
    )
    monkeypatch.setattr(confirmation, "load_current_player_names", lambda *args: {})
    monkeypatch.setattr(
        confirmation,
        "ratings_table",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["One", "Two"],
                "offense_per_100": [0.0, 0.0],
                "defense_per_100": [0.0, 0.0],
                "net_per_100": [0.0, 0.0],
                "off_possessions": [100, 100],
                "def_possessions": [100, 100],
            }
        ),
    )
    monkeypatch.setattr(confirmation.joblib, "load", lambda path: _FeatureModel())

    run = confirmation.run_current_spm_confirmation(
        features_path,
        frozen,
        possessions_path,
        segments_path,
        names_path,
        player_games_path,
        artifact_root=tmp_path / "artifacts",
    )

    assert run["quality"]["match_rate"] == 1.0
    assert run["decision"]["promotion"] == "do_not_promote"
    assert set(run["decision"]["components_outside_historical_range"]) == {
        "offense",
        "defense",
        "net",
    }
    assert Path(run["artifact_path"], "historical_envelope.parquet").exists()
