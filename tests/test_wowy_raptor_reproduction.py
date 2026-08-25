import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.rapm_lab.run_wowy_raptor_reproduction import (
    darko_reproduction_metrics,
    reproduced_darko_panel,
    raptor_table_reproduction,
)


def test_darko_game_means_reproduce_published_season(tmp_path: Path, monkeypatch) -> None:
    history_root = tmp_path / "game_history"
    history_root.mkdir()
    payload = {
        "rows": [
            {
                "nba_id": 7,
                "season": 2026,
                "wowy_orapm": 1.0,
                "wowy_drapm": 0.5,
                "wowy_rapm": 1.5,
            },
            {
                "nba_id": 7,
                "season": 2026,
                "wowy_orapm": 3.0,
                "wowy_drapm": -0.5,
                "wowy_rapm": 2.5,
            },
        ],
        "truncated": False,
        "maxRows": None,
    }
    (history_root / "7.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        "research.rapm_lab.run_wowy_raptor_reproduction.DARKO_HISTORY",
        history_root,
    )
    reproduced = reproduced_darko_panel([7])
    assert reproduced.iloc[0]["reproduced_offense"] == 2.0
    assert reproduced.iloc[0]["reproduced_defense"] == 0.0
    assert reproduced.iloc[0]["reproduced_net"] == 2.0
    assert reproduced.iloc[0]["reproduced_games"] == 2

    published = pd.DataFrame(
        {
            "PLAYER_ID": [7],
            "season": [2026],
            "reference_offense": [2.0],
            "reference_defense": [0.0],
            "reference_net": [2.0],
        }
    )
    metrics, _ = darko_reproduction_metrics(published, reproduced)
    assert metrics["maximum_absolute_error"].max() == 0.0


def test_raptor_semantic_identity_ignores_csv_serialization(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "player_id": ["playea01", "playeb01"],
            "season": [2022, 2022],
            "raptor_onoff_offense": [1.25, -0.5],
            "raptor_onoff_defense": [0.25, 0.75],
            "raptor_onoff_total": [1.5, 0.25],
        }
    )
    local = tmp_path / "local.csv"
    official = tmp_path / "official.csv"
    frame.to_csv(local, index=False, lineterminator="\n")
    frame.to_csv(official, index=False, lineterminator="\r\n")
    metrics, matches = raptor_table_reproduction(local, official)
    assert len(matches) == 2
    assert np.allclose(metrics["pearson"], 1.0)
    assert metrics["maximum_absolute_error"].max() == 0.0


def test_raptor_semantic_identity_rejects_different_row_sets(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "player_id": ["playea01", "playeb01"],
            "season": [2022, 2022],
            "raptor_onoff_offense": [1.25, -0.5],
            "raptor_onoff_defense": [0.25, 0.75],
            "raptor_onoff_total": [1.5, 0.25],
        }
    )
    local = tmp_path / "local.csv"
    official = tmp_path / "official.csv"
    frame.to_csv(local, index=False)
    frame.iloc[:1].to_csv(official, index=False)
    with pytest.raises(ValueError, match="row sets differ"):
        raptor_table_reproduction(local, official)
