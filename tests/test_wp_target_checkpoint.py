import pandas as pd
import pytest

from nba_impact.models.win_probability_rapm import build_conserved_wp_target
from research.rapm_lab import run_wp_spm_aio


def test_cached_wp_target_rejects_stale_credit_without_overwriting(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "possession_id": ["g:1:1", "g:1:2", "g:2:1", "g:2:2"],
        "gameid": ["g"] * 4, "period": [1, 1, 2, 2], "num": [1, 2, 1, 2],
        "home_poss": [1, 0, 1, 0], "home_win": [1] * 4,
        "probability_context": [0.5, 0.6, 0.7, 0.8],
    })
    target, _ = build_conserved_wp_target(frame)
    target["pts"] = target["offense_wp_change"]
    checkpoint = tmp_path / "target.parquet"
    monkeypatch.setattr(run_wp_spm_aio, "CHECKPOINT", checkpoint)
    target.to_parquet(checkpoint, index=False)
    pd.testing.assert_frame_equal(run_wp_spm_aio._build_target({}), target)

    # Even correctly ordered rows must not retain a stale response.
    target.loc[0, "pts"] += 1
    target.to_parquet(checkpoint, index=False)
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="stale possession credit"):
        run_wp_spm_aio._build_target({})
    assert checkpoint.read_bytes() == original

    target["pts"] = target["offense_wp_change"]
    target.sort_values(["gameid", "num"]).to_parquet(checkpoint, index=False)
    with pytest.raises(ValueError, match="stale possession credit"):
        run_wp_spm_aio._build_target({})
