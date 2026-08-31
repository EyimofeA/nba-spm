from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models/compact_spm_correlated_v1.json"


def test_compact_contract_removes_every_audited_high_correlation_pair() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads((ROOT / config["source_feature_contract_path"]).read_text())
    drops = {
        side: set(config["dropped_features"][side])
        for side in ("offense", "defense")
    }
    selected = {}

    for side in ("offense", "defense"):
        source_features = source["feature_contract"][side]
        assert drops[side] <= set(source_features)
        selected[side] = set(source_features) - drops[side]
        assert len(selected[side]) == config["selected_feature_count"][side]

    with (ROOT / config["correlation_audit_path"]).open(newline="") as handle:
        pairs = list(csv.DictReader(handle))

    assert pairs
    assert all(
        row["feature_a"] not in selected[row["side"]]
        or row["feature_b"] not in selected[row["side"]]
        for row in pairs
    )
    assert "crafted_spacing_stable_v1" in selected["offense"]
    assert "FG3A_p100" not in selected["offense"]
    assert "FG3M_p100" not in selected["offense"]
