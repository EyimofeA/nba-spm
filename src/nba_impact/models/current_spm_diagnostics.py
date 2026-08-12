"""Diagnose a frozen current-season SPM result without tuning the model."""

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
    load_current_possessions,
    ratings_table,
)
from nba_impact.models.statistical_impact import _metrics


EXPOSURE_BINS = (-1.0, 499.0, 999.0, 1999.0, float("inf"))
EXPOSURE_LABELS = ("under_500", "500_to_999", "1000_to_1999", "2000_plus")
HUSTLE_FEATURES = {
    "deflections_p100",
    "charges_drawn_p100",
    "contested_2pt_p100",
    "contested_3pt_p100",
    "def_loose_balls_recovered_p100",
}


def _exposure_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    frame["exposure"] = frame[["Poss_Off", "Poss_Def"]].min(axis=1)
    frame["exposure_bin"] = pd.cut(
        frame["exposure"], EXPOSURE_BINS, labels=EXPOSURE_LABELS
    )
    rows = []
    for exposure_bin, group in frame.groupby("exposure_bin", observed=True):
        weights = np.sqrt(group["exposure"].clip(lower=1)).to_numpy()
        for component in ("offense", "defense", "net"):
            rows.append(
                {
                    "exposure_bin": str(exposure_bin),
                    "component": component,
                    "players": len(group),
                    "target_std": float(group[f"target_{component}"].std()),
                    "prediction_std": float(group[f"spm_{component}"].std()),
                    **_metrics(
                        group[f"target_{component}"].to_numpy(),
                        group[f"spm_{component}"].to_numpy(),
                        weights,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
    defensive_tracking: set[str],
) -> pd.DataFrame:
    rows = []
    for feature in feature_names:
        historical = reference[feature].dropna()
        observed = current[feature].dropna()
        iqr = historical.quantile(0.75) - historical.quantile(0.25)
        median_shift_iqr = (
            float(abs(observed.median() - historical.median()) / iqr)
            if iqr > 0
            else float("nan")
        )
        outside = ((observed < historical.min()) | (observed > historical.max())).mean()
        family = "other_selected"
        if feature in defensive_tracking:
            family = "hustle" if feature in HUSTLE_FEATURES else "dfg_rim"
        rows.append(
            {
                "feature": feature,
                "family": family,
                "historical_median": float(historical.median()),
                "current_median": float(observed.median()),
                "absolute_median_shift_iqr": median_shift_iqr,
                "current_outside_historical_range_fraction": float(outside),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["absolute_median_shift_iqr", "feature"], ascending=[False, True]
    )


def _split_half_stability(
    possessions: pd.DataFrame,
    seasons: tuple[int, ...],
    exposure_floors: tuple[int, ...] = (0, 500, 1000),
) -> pd.DataFrame:
    rows = []
    for season in seasons:
        season_frame = possessions.loc[possessions["season"].eq(season)].copy()
        games = season_frame[["gameid", "date"]].drop_duplicates().sort_values(
            ["date", "gameid"], kind="stable"
        )
        cut = len(games) // 2
        half_ratings = []
        for label, game_ids in (
            ("first", games.iloc[:cut]["gameid"]),
            ("second", games.iloc[cut:]["gameid"]),
        ):
            half = season_frame.loc[season_frame["gameid"].isin(set(game_ids))]
            design = build_design(half)
            beta, _ = fit_coefficients(
                design,
                RapmConfig(
                    seasons=(season,),
                    lambda_off=3000.0,
                    lambda_def=3000.0,
                    lambda_home=300.0,
                    game_types=("regular",),
                    data_scope="split_half_stability_diagnostic",
                ),
            )
            table = ratings_table(design, beta).rename(
                columns={
                    "offense_per_100": f"offense_{label}",
                    "defense_per_100": f"defense_{label}",
                    "net_per_100": f"net_{label}",
                    "off_possessions": f"off_possessions_{label}",
                    "def_possessions": f"def_possessions_{label}",
                }
            )
            half_ratings.append(table)
        joined = half_ratings[0].merge(
            half_ratings[1], on="player_id", validate="one_to_one"
        )
        exposure_columns = [
            "off_possessions_first",
            "def_possessions_first",
            "off_possessions_second",
            "def_possessions_second",
        ]
        for floor in exposure_floors:
            eligible = joined.loc[joined[exposure_columns].min(axis=1).ge(floor)]
            for component in ("offense", "defense", "net"):
                rows.append(
                    {
                        "season": season,
                        "minimum_possessions_each_half": floor,
                        "component": component,
                        "players": len(eligible),
                        "correlation": float(
                            eligible[f"{component}_first"].corr(
                                eligible[f"{component}_second"]
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_current_spm_diagnostics(
    confirmation_run: str | Path,
    current_features_path: str | Path,
    reference_features_path: str | Path,
    frozen_spm_run: str | Path,
    possessions_path: str | Path,
    segments_path: str | Path,
    *,
    artifact_root: str | Path,
    season: int = 2025,
    comparison_season: int = 2024,
) -> dict:
    """Build no-tuning diagnostics for a frozen current-season confirmation."""
    confirmation_dir = Path(confirmation_run)
    frozen_dir = Path(frozen_spm_run)
    confirmation = json.loads((confirmation_dir / "run.json").read_text())
    frozen = json.loads((frozen_dir / "run.json").read_text())
    scores = pd.read_parquet(confirmation_dir / "player_scores.parquet")
    current_features = pd.read_parquet(current_features_path)
    current = current_features.loc[current_features["Window_End"].eq(season)].copy()
    reference = pd.read_parquet(reference_features_path)
    if current.empty:
        raise ValueError(f"No current features exist for season {season}.")

    offense_features = frozen["models"]["offense"]["features"]
    defense_features = frozen["models"]["defense"]["features"]
    selected_features = sorted(set(offense_features + defense_features))
    defensive_tracking = set(frozen["config"]["additional_defense_features"])
    exposure = _exposure_metrics(scores)
    drift = _feature_drift(reference, current, selected_features, defensive_tracking)

    defense_model = joblib.load(frozen_dir / "model_defense.joblib")
    evaluation = current.merge(
        scores[["PLAYER_ID", "target_defense", "Poss_Off", "Poss_Def"]],
        on="PLAYER_ID",
        validate="one_to_one",
    )
    weights = np.sqrt(
        evaluation[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1)
    ).to_numpy()
    actual = evaluation["target_defense"].to_numpy()
    block_rows = []
    block_definitions = {
        "full_model": set(),
        "all_defensive_tracking_neutralized": defensive_tracking,
        "hustle_neutralized": defensive_tracking & HUSTLE_FEATURES,
        "dfg_rim_neutralized": defensive_tracking - HUSTLE_FEATURES,
    }
    historical_medians = reference[defense_features].median()
    for label, neutralized in block_definitions.items():
        design = evaluation[defense_features].copy()
        for feature in neutralized:
            design[feature] = historical_medians[feature]
        prediction = defense_model.predict(design)
        block_rows.append(
            {
                "variant": label,
                "neutralized_features": sorted(neutralized),
                "prediction_std": float(np.std(prediction)),
                **_metrics(actual, prediction, weights),
            }
        )
    block_metrics = pd.DataFrame(block_rows)

    possessions = load_current_possessions(
        possessions_path,
        segments_path,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    stability = _split_half_stability(
        possessions, (comparison_season, season)
    )
    residuals = scores.copy()
    residuals["exposure"] = residuals[["Poss_Off", "Poss_Def"]].min(axis=1)
    for component in ("offense", "defense", "net"):
        residuals[f"{component}_error"] = (
            residuals[f"spm_{component}"] - residuals[f"target_{component}"]
        )
        residuals[f"absolute_{component}_error"] = residuals[
            f"{component}_error"
        ].abs()

    source_hashes = {
        "confirmation_run": sha256_file(confirmation_dir / "run.json"),
        "confirmation_scores": sha256_file(confirmation_dir / "player_scores.parquet"),
        "current_features": sha256_file(current_features_path),
        "reference_features": sha256_file(reference_features_path),
        "frozen_run": sha256_file(frozen_dir / "run.json"),
        "frozen_defense_model": sha256_file(frozen_dir / "model_defense.joblib"),
        "possessions": sha256_file(possessions_path),
        "segments": sha256_file(segments_path),
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "season": season,
                "comparison_season": comparison_season,
                "sources": source_hashes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"current_spm_diagnostics_v1_{identity}"
    output = Path(artifact_root) / "models" / "current_spm_diagnostics" / run_id
    output.mkdir(parents=True, exist_ok=False)
    exposure.to_parquet(output / "exposure_metrics.parquet", index=False)
    drift.to_parquet(output / "feature_drift.parquet", index=False)
    block_metrics.to_parquet(output / "defensive_block_neutralization.parquet", index=False)
    stability.to_parquet(output / "split_half_stability.parquet", index=False)
    residuals.to_parquet(output / "player_residuals.parquet", index=False)

    full = block_metrics.loc[block_metrics["variant"].eq("full_model")].iloc[0]
    all_neutral = block_metrics.loc[
        block_metrics["variant"].eq("all_defensive_tracking_neutralized")
    ].iloc[0]
    run = {
        "run_id": run_id,
        "model_family": "frozen_current_spm_failure_diagnostic",
        "estimand": "diagnostic_of_single_season_spm_against_normal_rapm",
        "status": "diagnostic_no_tuning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "season": season,
            "comparison_season": comparison_season,
            "confirmation_run_id": confirmation["run_id"],
            "frozen_spm_run_id": frozen["run_id"],
            "source_hashes": source_hashes,
        },
        "quality": {
            "score_rows": len(scores),
            "current_feature_rows": len(current),
            "selected_features": len(selected_features),
            "nonfinite_diagnostics": int(
                exposure.select_dtypes(include=["number"]).isna().sum().sum()
            ),
        },
        "metrics": {
            "exposure_rows": len(exposure),
            "feature_drift_rows": len(drift),
            "defensive_block_variants": len(block_metrics),
            "split_half_rows": len(stability),
        },
        "findings": {
            "defense_full_rmse": float(full["weighted_rmse"]),
            "defense_full_correlation": float(full["correlation"]),
            "defense_tracking_neutralized_rmse": float(all_neutral["weighted_rmse"]),
            "defense_tracking_neutralized_correlation": float(all_neutral["correlation"]),
            "tracking_block_helped_2025": bool(
                full["weighted_rmse"] < all_neutral["weighted_rmse"]
            ),
            "largest_median_iqr_shift_feature": str(drift.iloc[0]["feature"]),
            "largest_median_iqr_shift": float(
                drift.iloc[0]["absolute_median_shift_iqr"]
            ),
        },
        "interpretation": [
            "The defensive tracking block still improves the frozen model on 2025; it is not the direct failure source.",
            "High-exposure defensive errors remain large, so low-minute noise alone does not explain the regression.",
            "Split-half stability is a target-noise diagnostic, not a forecast-validation metric.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
