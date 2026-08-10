"""Score the frozen annual SPM once on a new current-data season."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    load_current_possessions,
    ratings_table,
)
from nba_impact.models.statistical_impact import _metrics


def run_current_spm_confirmation(
    features_path: str | Path,
    frozen_spm_run: str | Path,
    possessions_path: str | Path,
    segments_path: str | Path,
    names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
    season: int = 2025,
) -> dict:
    """Evaluate saved 2014-24 SPM models on an untouched season target."""
    frozen_dir = Path(frozen_spm_run)
    frozen = json.loads((frozen_dir / "run.json").read_text())
    features = pd.read_parquet(features_path)
    season_features = features.loc[features["Window_End"].eq(season)].copy()
    if season_features.empty:
        raise ValueError(f"No statistical features exist for season {season}.")

    possessions = load_current_possessions(
        possessions_path,
        segments_path,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    possessions = possessions.loc[possessions["season"].eq(season)].copy()
    if possessions.empty:
        raise ValueError(f"No current possessions exist for season {season}.")
    design = build_design(possessions)
    config = RapmConfig(
        seasons=(season,),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        game_types=("regular",),
        data_scope="current_single_season_terminal_lineup_target",
    )
    beta, intercept = fit_coefficients(design, config)
    names = load_current_player_names(names_path, player_games_path)
    targets = ratings_table(design, beta, names=names).rename(
        columns={
            "player_id": "PLAYER_ID",
            "offense_per_100": "target_offense",
            "defense_per_100": "target_defense",
            "net_per_100": "target_net",
            "off_possessions": "Poss_Off",
            "def_possessions": "Poss_Def",
            "player_name": "PLAYER_NAME",
        }
    )
    scored = targets.merge(
        season_features,
        on="PLAYER_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_target", "_feature"),
    )
    metric_rows = []
    weights = np.sqrt(np.minimum(scored["Poss_Off"], scored["Poss_Def"]).clip(lower=1))
    for component in ("offense", "defense"):
        model_info = frozen["models"][component]
        feature_names = model_info["features"]
        missing = sorted(set(feature_names) - set(scored.columns))
        if missing:
            raise ValueError(f"Frozen {component} model features are missing: {missing}")
        model = joblib.load(frozen_dir / f"model_{component}.joblib")
        scored[f"spm_{component}"] = model.predict(scored[feature_names])
        metric_rows.append(
            {
                "component": component,
                **_metrics(
                    scored[f"target_{component}"].to_numpy(),
                    scored[f"spm_{component}"].to_numpy(),
                    weights.to_numpy(),
                ),
            }
        )
    scored["spm_net"] = scored["spm_offense"] + scored["spm_defense"]
    metric_rows.append(
        {
            "component": "net",
            **_metrics(
                scored["target_net"].to_numpy(),
                scored["spm_net"].to_numpy(),
                weights.to_numpy(),
            ),
        }
    )
    metrics = pd.DataFrame(metric_rows)
    historical_metrics_path = frozen_dir / "fold_metrics.parquet"
    historical_metrics = pd.read_parquet(historical_metrics_path)
    historical_envelope = (
        historical_metrics.groupby("component", as_index=False)
        .agg(
            historical_folds=("test_season", "nunique"),
            historical_worst_rmse=("weighted_rmse", "max"),
            historical_worst_correlation=("correlation", "min"),
        )
        .merge(metrics, on="component", validate="one_to_one")
    )
    historical_envelope["rmse_outside_historical_range"] = (
        historical_envelope["weighted_rmse"]
        > historical_envelope["historical_worst_rmse"]
    )
    historical_envelope["correlation_outside_historical_range"] = (
        historical_envelope["correlation"]
        < historical_envelope["historical_worst_correlation"]
    )
    historical_envelope["outside_historical_range"] = historical_envelope[
        ["rmse_outside_historical_range", "correlation_outside_historical_range"]
    ].any(axis=1)
    source_hashes = {
        "features": sha256_file(features_path),
        "frozen_run": sha256_file(frozen_dir / "run.json"),
        "frozen_offense_model": sha256_file(frozen_dir / "model_offense.joblib"),
        "frozen_defense_model": sha256_file(frozen_dir / "model_defense.joblib"),
        "frozen_fold_metrics": sha256_file(historical_metrics_path),
        "possessions": sha256_file(possessions_path),
        "segments": sha256_file(segments_path),
        "names": sha256_file(names_path),
        "player_games": sha256_file(player_games_path),
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps({"season": season, "sources": source_hashes}, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"current_spm_confirmation_v1_{identity}"
    output = Path(artifact_root) / "models" / "current_spm_confirmation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    keep = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "Poss_Off",
        "Poss_Def",
        "target_offense",
        "target_defense",
        "target_net",
        "spm_offense",
        "spm_defense",
        "spm_net",
    ]
    scored[keep].to_parquet(output / "player_scores.parquet", index=False)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    historical_envelope.to_parquet(output / "historical_envelope.parquet", index=False)
    outside_components = historical_envelope.loc[
        historical_envelope["outside_historical_range"], "component"
    ].tolist()
    run = {
        "run_id": run_id,
        "model_family": "frozen_annual_spm_new_season_confirmation",
        "estimand": "single_regular_season_normal_rapm_offense_defense_and_net",
        "status": "untouched_season_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "season": season,
            "frozen_spm_run_id": frozen["run_id"],
            "target_lineup_policy": "terminal",
            "target_penalties": [3000.0, 3000.0, 300.0],
            "source_hashes": source_hashes,
            "builder_sha256": source_hashes["builder"],
        },
        "quality": {
            "feature_rows": len(season_features),
            "target_rows": len(targets),
            "matched_rows": len(scored),
            "match_rate": float(len(scored) / len(targets)),
            "missing_names": int(targets["PLAYER_NAME"].isna().sum()),
            "nonfinite_predictions": int(
                (~np.isfinite(scored[["spm_offense", "spm_defense", "spm_net"]])).sum().sum()
            ),
            "games": int(possessions["gameid"].nunique()),
            "possessions": len(possessions),
        },
        "metrics": metric_rows,
        "historical_envelope": historical_envelope.to_dict(orient="records"),
        "decision": {
            "promotion": "do_not_promote" if outside_components else "eligible_for_review",
            "components_outside_historical_range": outside_components,
            "gate_was_predeclared": False,
            "basis": (
                "Transparent post-confirmation comparison with the worst held-out 2017-24 fold; "
                "no model or feature tuning used the 2025 result."
            ),
        },
        "target_intercept_points_per_possession": intercept,
        "artifact_path": str(output.resolve()),
        "scores_path": str((output / "player_scores.parquet").resolve()),
        "caveats": [
            "The target is noisy one-season RAPM, not ground truth.",
            "This is the first untouched seasonal check of the frozen 2014-24 SPM mapping.",
        ],
    }
    if run["quality"]["nonfinite_predictions"]:
        raise ValueError("Current SPM confirmation produced non-finite predictions.")
    write_json_atomic(run, output / "run.json")
    return run
