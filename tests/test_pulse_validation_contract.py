import json

import pandas as pd
import pytest

from nba_impact.api.web_snapshot import _pulse_evidence
from nba_impact.models.pulse_validation import load_pulse_validation


@pytest.fixture
def pulse_run(tmp_path):
    path = tmp_path / "artifacts/models/pulse/test_pulse"
    path.mkdir(parents=True)
    manifest = {
        "run_id": path.name,
        "config": {"interpretation": {"validation_rows": "past_only"}},
        "validation": {"rating_seasons": [2024, 2024]},
        "artifacts": {name: f"{name}.parquet" for name in (
            "validation_games", "validation_folds", "validation_priors",
        )},
    }
    (path / "run.json").write_text(json.dumps(manifest))
    games = pd.DataFrame([
        {"candidate": candidate, "rating_season": 2024, "outcome_season": 2025,
         "game_id": str(game), "actual_margin": game, "predicted_margin": game - 1}
        for candidate in ("prior", "pulse", "rapm") for game in (1, 2)
    ])
    folds = games[["candidate", "rating_season", "outcome_season"]].drop_duplicates().assign(
        training_start=2005, training_end=2023, games=2, mse=1.0,
        correlation=1.0, calibration_slope=1.0,
    )
    priors = pd.DataFrame({"PLAYER_ID": [1], "rating_season": [2024], "Window_End": [2024]})
    for name, frame in (("games", games), ("folds", folds), ("priors", priors)):
        frame.to_parquet(path / f"validation_{name}.parquet", index=False)
    return path


def test_export_uses_checked_fold_predictions_not_saved_or_legacy_summary(pulse_run):
    games, folds = load_pulse_validation(pulse_run)
    assert len(games) == 6 and len(folds) == 3
    pd.DataFrame({"candidate": ["pulse"], "equal_season_rmse": [999]}).to_parquet(
        pulse_run / "validation_summary.parquet"
    )
    evidence = _pulse_evidence(pulse_run.parents[3], json.loads((pulse_run / "run.json").read_text()))
    assert {row["equal_season_rmse"] for row in evidence["comparison"]} == {1.0}
    (pulse_run / "validation_games.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        _pulse_evidence(pulse_run.parents[3], json.loads((pulse_run / "run.json").read_text()))


@pytest.mark.parametrize("file,column,value", [
    ("folds", "training_end", 2024),
    ("folds", "training_end", float("nan")),
    ("games", "outcome_season", 2024),
    ("games", "outcome_season", 2027),
    ("games", "candidate", "box15_9y_normal_aio"),
    ("games", "predicted_margin", float("inf")),
    ("games", "actual_margin", 99),
    ("folds", "mse", 99),
    ("folds", "correlation", 0.5),
    ("folds", "calibration_slope", 3),
    ("priors", "Window_End", 2025),
    ("priors", "PLAYER_ID", float("nan")),
])
def test_rejects_leaky_misaligned_or_renamed_inputs(pulse_run, file, column, value):
    path = pulse_run / f"validation_{file}.parquet"
    frame = pd.read_parquet(path)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        load_pulse_validation(pulse_run)


def test_rejects_duplicate_games_and_incomplete_candidates(pulse_run):
    path = pulse_run / "validation_games.parquet"
    games = pd.read_parquet(path)
    pd.concat([games, games.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_pulse_validation(pulse_run)
    games.iloc[1:].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="identical games"):
        load_pulse_validation(pulse_run)


def test_rejects_descriptive_artifact_substitution_and_missing_declared_fold(pulse_run):
    path = pulse_run / "run.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["validation_games"] = "ratings.parquet"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="descriptive"):
        load_pulse_validation(pulse_run)
    manifest["artifacts"]["validation_games"] = "validation_games.parquet"
    manifest["validation"]["rating_seasons"] = [2023, 2024]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing declared"):
        load_pulse_validation(pulse_run)
