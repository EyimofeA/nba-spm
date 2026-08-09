from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from nba_impact.data.manifest import sha256_file
from nba_impact.models.statistical_priors import (
    build_cross_fitted_statistical_priors,
)


def _write_inputs(root: Path, *, shift_2019_target: float = 0.0) -> tuple[Path, Path, Path]:
    feature_rows = []
    target_rows = []
    for window_end in range(2016, 2021):
        for player_id in range(12):
            signal = player_id / 10 + window_end - 2016
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "OffPoss": 1000.0,
                    "DefPoss": 1000.0,
                    "signal": signal,
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "Off": signal + (shift_2019_target if window_end == 2019 else 0.0),
                    "Def": -0.5 * signal,
                    "Poss_Off": 1000.0,
                    "Poss_Def": 1000.0,
                }
            )
    feature_rows.append(
        {
            "PLAYER_ID": 99,
            "Window_End": 2019,
            "OffPoss": 50.0,
            "DefPoss": 50.0,
            "signal": 1.5,
        }
    )
    features = root / f"features_{shift_2019_target:g}.parquet"
    targets = root / f"targets_{shift_2019_target:g}.csv"
    pd.DataFrame(feature_rows).to_parquet(features, index=False)
    pd.DataFrame(target_rows).to_csv(targets, index=False)
    reference = root / f"reference_{shift_2019_target:g}"
    reference.mkdir()
    (reference / "run.json").write_text(
        json.dumps(
            {
                "run_id": "reference",
                "config": {
                    "offense_model": {"family": "ridge"},
                    "defense_model": {"family": "ridge"},
                    "source_hashes": {
                        "features": sha256_file(features),
                        "targets": sha256_file(targets),
                    },
                },
                "selected_features": {
                    "offense": ["signal"],
                    "defense": ["signal"],
                },
            }
        )
    )
    return features, targets, reference


def _ridge() -> Pipeline:
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("model", Ridge(alpha=1.0))]
    )


def test_cross_fitted_priors_exclude_same_window_targets_and_cover_unlabeled_players(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "nba_impact.models.statistical_priors._frozen_model", lambda side: _ridge()
    )
    features, targets, reference = _write_inputs(tmp_path)
    first = build_cross_fitted_statistical_priors(
        features,
        targets,
        reference,
        artifact_root=tmp_path / "first",
        prediction_window_ends=(2019,),
    )
    shifted_features, shifted_targets, shifted_reference = _write_inputs(
        tmp_path, shift_2019_target=1000.0
    )
    shifted = build_cross_fitted_statistical_priors(
        shifted_features,
        shifted_targets,
        shifted_reference,
        artifact_root=tmp_path / "shifted",
        prediction_window_ends=(2019,),
    )
    first_priors = pd.read_parquet(first["priors_path"]).sort_values("PLAYER_ID")
    shifted_priors = pd.read_parquet(shifted["priors_path"]).sort_values("PLAYER_ID")
    np.testing.assert_allclose(
        first_priors["prior_offense_per_100"],
        shifted_priors["prior_offense_per_100"],
    )
    assert 99 in first_priors["PLAYER_ID"].to_numpy()
    assert len(first_priors) == 13
    assert first_priors["train_max_window_end"].eq(2016).all()
    assert not {"target_offense", "target_defense", "target_net"}.intersection(
        first_priors.columns
    )
    assert first["quality"]["duplicate_keys"] == 0
    assert first["quality"]["all_eligible_feature_rows_scored"] is True


def test_cross_fitted_priors_reject_overlapping_early_window(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "nba_impact.models.statistical_priors._frozen_model", lambda side: _ridge()
    )
    features, targets, reference = _write_inputs(tmp_path)
    try:
        build_cross_fitted_statistical_priors(
            features,
            targets,
            reference,
            artifact_root=tmp_path,
            prediction_window_ends=(2018,),
        )
    except ValueError as exc:
        assert "non-overlapping older target" in str(exc)
    else:
        raise AssertionError("Expected the early overlapping window to be rejected.")
