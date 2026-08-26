from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.five_year_spm_feature_audit import (
    feature_encoding,
    redundancy_audit,
)


def test_feature_encoding_separates_reencoded_signals() -> None:
    assert feature_encoding("fg3_pct_eb") == "empirical_bayes"
    assert feature_encoding("PTS_p100_relative") == "era_relative"
    assert feature_encoding("behavioral_passer_score_v1") == "composite"
    assert feature_encoding("AST_p100") == "direct"


def test_redundancy_audit_finds_near_duplicate_pair() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=300)
    features = pd.DataFrame(
        {
            "direct": base,
            "direct_eb": base + rng.normal(scale=1e-4, size=300),
            "other": rng.normal(size=300),
        }
    )
    registry, pairs, summary = redundancy_audit(
        features,
        {"offense": ("direct", "direct_eb", "other"), "defense": ("other",)},
    )
    assert len(registry) == 4
    assert summary["pairs_at_or_above_0_995"] == 1
    assert set(pairs.iloc[0][["feature_left", "feature_right"]]) == {
        "direct",
        "direct_eb",
    }
