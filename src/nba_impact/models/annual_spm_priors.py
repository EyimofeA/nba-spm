"""Forward-chained annual SPM priors for honest next-season RAPM tests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


PRIOR_OUTPUT_COLUMNS = (
    "prior_offense_per_100",
    "prior_defense_per_100",
    "prior_net_per_100",
)


def build_leave_one_season_out_annual_spm_priors(
    spm_run_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Convert held-out annual SPM predictions into leakage-safe RAPM centers."""
    source = Path(spm_run_path)
    source_manifest_path = source / "run.json"
    source_predictions_path = source / "oof_predictions.parquet"
    source_manifest = json.loads(source_manifest_path.read_text())
    predictions = pd.read_parquet(source_predictions_path)
    required = {
        "PLAYER_ID",
        "Season",
        "spm_offense",
        "spm_defense",
        "spm_net",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Annual SPM OOF predictions are missing {missing}.")
    if predictions.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual SPM OOF prediction keys must be unique.")

    training_seasons = tuple(
        int(value) for value in source_manifest.get("config", {}).get("training_seasons", [])
    )
    output_seasons = tuple(sorted(int(value) for value in predictions["Season"].unique()))
    if not training_seasons or not output_seasons:
        raise ValueError("Annual SPM run must declare training and output seasons.")
    if not set(output_seasons).issubset(training_seasons):
        raise ValueError("Annual SPM OOF output seasons must be training seasons.")

    priors = predictions[
        ["PLAYER_ID", "Season", "spm_offense", "spm_defense", "spm_net"]
    ].rename(
        columns={
            "spm_offense": "prior_offense_per_100",
            "spm_defense": "prior_defense_per_100",
            "spm_net": "prior_net_per_100",
        }
    )
    priors["Window_End"] = priors["Season"].astype(int)
    priors["spm_training_rule"] = "leave_one_season_out"
    priors["spm_training_season_count"] = len(training_seasons) - 1
    if not np.isfinite(priors[list(PRIOR_OUTPUT_COLUMNS)].to_numpy()).all():
        raise ValueError("Annual SPM OOF priors must be finite.")

    identity = uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps(
            {
                "source_run_id": source_manifest.get("run_id"),
                "source_predictions_sha256": sha256_file(source_predictions_path),
                "training_rule": "leave_one_season_out",
            },
            sort_keys=True,
        ),
    ).hex[:10]
    run_id = f"annual_spm_oof_priors_v1_{identity}"
    output = Path(artifact_root) / "models" / "annual_spm_priors" / run_id
    output.mkdir(parents=True, exist_ok=False)
    priors.to_parquet(output / "priors.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "leave_one_season_out_annual_statistical_plus_minus",
        "estimand": source_manifest.get("estimand"),
        "status": "research_priors_for_retrospective_aio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "source_spm_run_id": source_manifest.get("run_id"),
            "training_rule": "all labeled seasons except the rated season",
            "training_seasons": list(training_seasons),
            "output_seasons": list(output_seasons),
            "source_hashes": {
                "source_run": sha256_file(source_manifest_path),
                "oof_predictions": sha256_file(source_predictions_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "quality": {
            "rows": len(priors),
            "players": int(priors["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "nonfinite_values": 0,
        },
        "metrics": {},
        "priors_path": str((output / "priors.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "The SPM mapping excludes the rated season's RAPM labels.",
            "Later seasons can train earlier retrospective ratings, so this is not a forecast.",
            "The features summarize the complete rated season.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run


def _load_contract(path: str | Path) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Annual SPM contract must be frozen before prior generation.")
    if contract.get("validation_rules", {}).get("rapm_scale_search_allowed") is not False:
        raise ValueError("Annual SPM contract must forbid RAPM prior-scale search.")
    return contract


def build_forward_chained_annual_spm_priors(
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    contract_path: str | Path,
    *,
    artifact_root: str | Path,
    output_seasons: tuple[int, ...] = tuple(range(2017, 2024)),
    minimum_training_seasons: int = 3,
    train_window_seasons: int | None = None,
) -> dict:
    """Train only before season T, then create an SPM prior for season T.

    ``train_window_seasons=None`` uses every earlier season.  A positive value
    uses only the most recent complete seasons before T.  This makes expanding,
    one-year, three-year, and five-year histories directly comparable on the
    same forecast-season rows.
    """
    if not output_seasons or tuple(sorted(set(output_seasons))) != output_seasons:
        raise ValueError("output_seasons must be unique and increasing.")
    if minimum_training_seasons < 1:
        raise ValueError("minimum_training_seasons must be positive.")
    if train_window_seasons is not None and train_window_seasons < 1:
        raise ValueError("train_window_seasons must be positive when set.")

    contract = _load_contract(contract_path)
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    if features.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual SPM feature keys must be unique.")
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual SPM target keys must be unique.")

    selected = _selected_single_season_features(reference_run_path)
    for side in ("offense", "defense"):
        additions = tuple(contract["components"][side]["additional_features"])
        selected[side] = tuple(dict.fromkeys((*selected[side], *additions)))
    required = {feature for side in selected.values() for feature in side}
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Forward annual SPM features are missing {missing}.")

    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    fitted_models: dict[int, dict[str, object]] = {}
    for season in output_seasons:
        train = panel.loc[panel["Season"].lt(season)].copy()
        if train_window_seasons is not None:
            available = sorted(int(value) for value in train["Season"].unique())
            keep = set(available[-train_window_seasons:])
            train = train.loc[train["Season"].isin(keep)].copy()
        prediction_frame = features.loc[features["Season"].eq(season)].copy()
        train_seasons = tuple(sorted(int(value) for value in train["Season"].unique()))
        if len(train_seasons) < minimum_training_seasons:
            raise ValueError(
                f"Season {season} has {len(train_seasons)} prior training seasons; "
                f"at least {minimum_training_seasons} are required."
            )
        if prediction_frame.empty:
            raise ValueError(f"Season {season} has no SPM feature rows.")

        predictions = prediction_frame[["PLAYER_ID", "Season"]].copy()
        fitted_models[season] = {}
        for side in ("offense", "defense"):
            model = _fit_model(
                _frozen_model(side), train, selected[side], f"target_{side}"
            )
            predictions[f"prior_{side}_per_100"] = model.predict(
                prediction_frame.loc[:, selected[side]]
            )
            fitted_models[season][side] = model
        predictions["prior_net_per_100"] = (
            predictions["prior_offense_per_100"]
            + predictions["prior_defense_per_100"]
        )
        predictions["Window_End"] = season
        predictions["spm_training_start"] = train_seasons[0]
        predictions["spm_training_end"] = train_seasons[-1]
        prediction_rows.append(predictions)

        evaluation = predictions.merge(
            targets,
            on=["PLAYER_ID", "Season"],
            how="inner",
            validate="one_to_one",
        )
        evaluation["sample_weight"] = np.sqrt(
            np.minimum(evaluation["Poss_Off"], evaluation["Poss_Def"]).clip(lower=1)
        )
        for side in ("offense", "defense", "net"):
            metric_rows.append(
                {
                    "rated_season": season,
                    "component": side,
                    "training_start": train_seasons[0],
                    "training_end": train_seasons[-1],
                    "training_seasons": len(train_seasons),
                    "training_rows": len(train),
                    "prediction_rows": len(predictions),
                    "evaluation_rows": len(evaluation),
                    **_metrics(
                        evaluation[f"target_{side}"].to_numpy(),
                        evaluation[f"prior_{side}_per_100"].to_numpy(),
                        evaluation["sample_weight"].to_numpy(),
                    ),
                }
            )

    priors = pd.concat(prediction_rows, ignore_index=True)
    if priors.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Forward annual SPM prior keys must be unique.")
    if not np.isfinite(priors[list(PRIOR_OUTPUT_COLUMNS)].to_numpy()).all():
        raise ValueError("Forward annual SPM priors must be finite.")
    metrics = pd.DataFrame(metric_rows)

    run_id = f"annual_spm_priors_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "annual_spm_priors" / run_id
    output.mkdir(parents=True, exist_ok=False)
    priors.to_parquet(output / "priors.parquet", index=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    for season, models in fitted_models.items():
        for side, model in models.items():
            joblib.dump(model, output / f"model_{side}_through_{season - 1}.joblib")

    run = {
        "run_id": run_id,
        "model_family": "forward_chained_annual_statistical_plus_minus",
        "estimand": contract["estimand"],
        "status": "research_priors_for_next_season_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "contract_version": contract["contract_version"],
            "output_seasons": list(output_seasons),
            "minimum_training_seasons": minimum_training_seasons,
            "training_rule": "strictly earlier seasons only",
            "train_window_seasons": train_window_seasons,
            "features": {side: list(values) for side, values in selected.items()},
            "learners": {
                side: contract["components"][side]["learner"]
                for side in ("offense", "defense")
            },
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "reference_run": sha256_file(Path(reference_run_path) / "run.json"),
                "contract": sha256_file(contract_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "quality": {
            "rows": len(priors),
            "players": int(priors["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "nonfinite_values": 0,
            "minimum_prediction_rows": int(
                priors.groupby("Window_End").size().min()
            ),
        },
        "metrics": {
            "folds": metrics.to_dict(orient="records"),
        },
        "priors_path": str((output / "priors.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "The SPM mapping uses only seasons before the rated season.",
            "A finite training window retains only the most recent earlier seasons.",
            "The features summarize the complete rated season, so the rating is descriptive rather than preseason.",
            "One-season zero-prior RAPM labels are noisy and are not ground truth.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
