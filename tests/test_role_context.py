import pandas as pd
import pytest

from nba_impact.data.role_context import compute_role_context_features


def _source(*, jump: bool = False) -> pd.DataFrame:
    rows = []
    for bucket, attempts, makes in [
        ("0", 10, 5), ("1", 5, 2), ("2", 5, 3), ("3_6", 4, 2), ("7+", 6, 3)
    ]:
        row = {
            "PLAYER_ID": 1.0,
            "PLAYER": "Example Player",
            "dribbles": bucket,
            "year": 2024.0,
            "FGM": makes,
            "FGA": attempts,
            "AGE": 99,
            "GP": 82,
        }
        if jump:
            row["TEAM"] = pd.NA
            row["FREQ%"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def test_role_context_is_count_derived_and_excludes_display_fields():
    result = compute_role_context_features(_source(), _source(jump=True), seasons=(2024,))
    assert result.columns[:2].tolist() == ["PLAYER_ID", "Season"]
    assert "AGE" not in result and "GP" not in result and "TEAM" not in result
    row = result.iloc[0]
    assert row["all_shot_fga"] == 30
    assert row["all_shot_zero_dribble_share"] == pytest.approx(10 / 30)
    assert row["all_shot_one_two_dribble_share"] == pytest.approx(10 / 30)
    assert row["all_shot_three_plus_dribble_share"] == pytest.approx(10 / 30)
    assert row["jump_shot_zero_dribble_fg_pct"] == pytest.approx(0.5)


def test_role_context_rejects_unknown_or_duplicate_buckets():
    unknown = _source()
    unknown.loc[0, "dribbles"] = "4"
    with pytest.raises(ValueError, match="unknown dribble buckets"):
        compute_role_context_features(unknown, _source(jump=True), seasons=(2024,))
    duplicate = pd.concat([_source(), _source().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player-season-dribble keys"):
        compute_role_context_features(duplicate, _source(jump=True), seasons=(2024,))
