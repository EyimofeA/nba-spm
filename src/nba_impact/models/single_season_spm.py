"""Modern single-season statistical plus-minus and external disagreement audit."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.external_impact_benchmark import (
    normalize_player_name,
    parse_bpm_html,
    parse_xrapm_html,
)
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_legacy_possessions,
    ratings_table,
)
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


TEMPORAL_SUFFIXES = ("_latest", "_trend", "_volatility")


def build_single_season_rapm_targets(
    cache_dir: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2025)),
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> dict:
    """Fit one zero-prior normal RAPM per season as noisy SPM training labels."""
    rows = []
    source_hashes = {}
    season_quality = []
    for season in seasons:
        frame = load_legacy_possessions(cache_dir, (season,), game_types=("regular",))
        design = build_design(frame)
        config = RapmConfig(
            seasons=(season,),
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
            game_types=("regular",),
            data_scope="legacy_single_season_normal_rapm_target",
        )
        beta, _ = fit_coefficients(design, config)
        ratings = ratings_table(design, beta).rename(
            columns={
                "player_id": "PLAYER_ID",
                "offense_per_100": "target_offense",
                "defense_per_100": "target_defense",
                "net_per_100": "target_net",
                "off_possessions": "Poss_Off",
                "def_possessions": "Poss_Def",
            }
        )
        ratings["Season"] = season
        rows.append(
            ratings[
                [
                    "PLAYER_ID",
                    "Season",
                    "target_offense",
                    "target_defense",
                    "target_net",
                    "Poss_Off",
                    "Poss_Def",
                ]
            ]
        )
        season_quality.append(
            {
                "season": season,
                "possession_rows": len(frame),
                "games": int(frame["gameid"].nunique()),
                "players": len(design.players),
            }
        )
        for source_path in frame.attrs.get("source_paths", []):
            source_hashes[source_path] = sha256_file(source_path)
    targets = pd.concat(rows, ignore_index=True)
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Single-season RAPM targets have duplicate player-season keys.")
    if not np.isfinite(
        targets[["target_offense", "target_defense", "target_net"]].to_numpy()
    ).all():
        raise ValueError("Single-season RAPM targets contain non-finite ratings.")
    run_id = f"single_season_rapm_targets_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "single_season_rapm_targets" / run_id
    output.mkdir(parents=True, exist_ok=False)
    targets.to_parquet(output / "targets.parquet", index=False)
    pd.DataFrame(season_quality).to_parquet(output / "season_quality.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "single_season_zero_prior_normal_rapm_targets",
        "estimand": "single_regular_season_offense_defense_and_net_points_per_100",
        "status": "research_training_labels",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seasons": list(seasons),
            "builder_sha256": sha256_file(Path(__file__)),
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_home": lambda_home,
            "lineup_policy": "legacy possession terminal lineup",
            "prior": "zero",
            "source_hashes": source_hashes,
        },
        "quality": {
            "rows": len(targets),
            "players": int(targets["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "minimum_games_per_season": int(
                min(item["games"] for item in season_quality)
            ),
        },
        "metrics": {
            "season_quality": season_quality,
        },
        "targets_path": str((output / "targets.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "One-season RAPM is noisy and is used as a training label, not ground truth.",
            "Legacy possessions are stale after 2024.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run


def _selected_single_season_features(reference_run_path: str | Path) -> dict[str, tuple[str, ...]]:
    reference = json.loads((Path(reference_run_path) / "run.json").read_text())
    selected = reference.get("selected_features", {})
    output = {}
    for side in ("offense", "defense"):
        values = tuple(
            feature
            for feature in selected.get(side, ())
            if not feature.endswith(TEMPORAL_SUFFIXES)
        )
        if not values:
            raise ValueError(f"Single-season SPM has no selected {side} features.")
        output[side] = values
    return output


def _external_annual(raw_root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        bpm = parse_bpm_html(
            (raw_root / "basketball_reference_bpm" / f"season={season}" / "page.html").read_text(),
            season,
        )
        xrapm = parse_xrapm_html(
            (raw_root / "xrapm" / f"season={season}" / "page.html").read_text(),
            season,
        )
        rows.append(
            bpm.merge(
                xrapm,
                on=["season", "normalized_name"],
                how="outer",
                validate="one_to_one",
            )
        )
    return pd.concat(rows, ignore_index=True)


def _external_source_hashes(raw_root: Path, seasons: tuple[int, ...]) -> dict[str, str]:
    hashes = {}
    for season in seasons:
        for source in ("basketball_reference_bpm", "xrapm"):
            path = raw_root / source / f"season={season}" / "page.html"
            hashes[str(path.resolve())] = sha256_file(path)
    return hashes


def _external_metrics(frame: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    for external in ("bpm", "xrapm"):
        for component in ("offense", "defense", "net"):
            left = f"spm_{component}"
            right = f"{external}_{component}"
            valid = frame[[left, right]].dropna()
            if len(valid) < 3:
                continue
            pearson = (
                float(valid[left].corr(valid[right], method="pearson"))
                if valid[left].nunique() > 1 and valid[right].nunique() > 1
                else float("nan")
            )
            spearman = (
                float(valid[left].corr(valid[right], method="spearman"))
                if valid[left].nunique() > 1 and valid[right].nunique() > 1
                else float("nan")
            )
            rows.append(
                {
                    "scope": scope,
                    "external_metric": external,
                    "component": component,
                    "rows": len(valid),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return rows


def fit_single_season_spm(
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    names_path: str | Path,
    external_raw_root: str | Path,
    *,
    artifact_root: str | Path,
    output_seasons: tuple[int, ...] = tuple(range(2017, 2025)),
    minimum_possessions_per_side: float = 1000.0,
    additional_offense_features: tuple[str, ...] = (),
    additional_defense_features: tuple[str, ...] = (),
) -> dict:
    """Fit season-held-out SPM predictions, a final full-panel model, and disagreements."""
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    selected = _selected_single_season_features(reference_run_path)
    selected["offense"] = tuple(dict.fromkeys((*selected["offense"], *additional_offense_features)))
    selected["defense"] = tuple(dict.fromkeys((*selected["defense"], *additional_defense_features)))
    required = {feature for values in selected.values() for feature in values}
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Single-season SPM features are missing {missing}.")
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    reliability = np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    panel["sample_weight"] = np.sqrt(reliability)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Single-season SPM panel keys must be unique.")

    prediction_rows = []
    fold_rows = []
    for test_season in output_seasons:
        train = panel.loc[panel["Season"].ne(test_season)]
        test = panel.loc[panel["Season"].eq(test_season)].copy()
        if min(len(train), len(test)) == 0:
            raise ValueError(f"Single-season SPM fold {test_season} has an empty partition.")
        fold = test[
            [
                "PLAYER_ID",
                "Season",
                "target_offense",
                "target_defense",
                "target_net",
                "Poss_Off",
                "Poss_Def",
                "sample_weight",
            ]
        ].copy()
        for side in ("offense", "defense"):
            model = _fit_model(
                _frozen_model(side), train, selected[side], f"target_{side}"
            )
            fold[f"spm_{side}"] = model.predict(test.loc[:, selected[side]])
            fold_rows.append(
                {
                    "test_season": test_season,
                    "component": side,
                    "train_seasons": int(train["Season"].nunique()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **_metrics(
                        test[f"target_{side}"].to_numpy(),
                        fold[f"spm_{side}"].to_numpy(),
                        test["sample_weight"].to_numpy(),
                    ),
                }
            )
        fold["spm_net"] = fold["spm_offense"] + fold["spm_defense"]
        fold_rows.append(
            {
                "test_season": test_season,
                "component": "net",
                "train_seasons": int(train["Season"].nunique()),
                "train_rows": len(train),
                "test_rows": len(test),
                **_metrics(
                    test["target_net"].to_numpy(),
                    fold["spm_net"].to_numpy(),
                    test["sample_weight"].to_numpy(),
                ),
            }
        )
        prediction_rows.append(fold)
    oof = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)

    output_feature_rows = features.loc[features["Season"].isin(output_seasons)].copy()
    final = output_feature_rows[["PLAYER_ID", "Season", "OffPoss", "DefPoss"]].copy()
    final_models = {}
    fitted = {}
    for side in ("offense", "defense"):
        model = _fit_model(
            _frozen_model(side), panel, selected[side], f"target_{side}"
        )
        final[f"spm_{side}"] = model.predict(output_feature_rows.loc[:, selected[side]])
        fitted[side] = model
    final["spm_net"] = final["spm_offense"] + final["spm_defense"]

    names = pd.read_csv(names_path)[["PLAYER_ID", "PLAYER_NAME"]]
    names["normalized_name"] = names["PLAYER_NAME"].map(normalize_player_name)
    duplicate_names = names["normalized_name"].duplicated(keep=False)
    names.loc[duplicate_names, "normalized_name"] = pd.NA
    oof = oof.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    final = final.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    external = _external_annual(Path(external_raw_root), output_seasons)
    benchmark = oof.merge(
        external,
        left_on=["Season", "normalized_name"],
        right_on=["season", "normalized_name"],
        how="left",
        validate="many_to_one",
    )
    benchmark["high_exposure"] = benchmark["Poss_Off"].ge(
        minimum_possessions_per_side
    ) & benchmark["Poss_Def"].ge(minimum_possessions_per_side)
    external_metric_rows = _external_metrics(benchmark, "all_matched")
    external_metric_rows += _external_metrics(
        benchmark.loc[benchmark["high_exposure"]], "high_exposure"
    )
    for season in output_seasons:
        external_metric_rows += _external_metrics(
            benchmark.loc[
                benchmark["high_exposure"] & benchmark["Season"].eq(season)
            ],
            f"season_{season}_high_exposure",
        )
    external_metrics = pd.DataFrame(external_metric_rows)

    disagreement = benchmark.loc[
        benchmark["high_exposure"]
        & benchmark["spm_defense"].notna()
        & benchmark["xrapm_defense"].notna()
    ].copy()
    disagreement["spm_defense_percentile"] = disagreement.groupby("Season")[
        "spm_defense"
    ].rank(pct=True)
    disagreement["xrapm_defense_percentile"] = disagreement.groupby("Season")[
        "xrapm_defense"
    ].rank(pct=True)
    disagreement["defense_percentile_gap"] = (
        disagreement["spm_defense_percentile"]
        - disagreement["xrapm_defense_percentile"]
    )
    disagreement["disagreement_direction"] = np.select(
        [
            disagreement["defense_percentile_gap"].ge(0.25),
            disagreement["defense_percentile_gap"].le(-0.25),
        ],
        ["spm_higher", "xrapm_higher"],
        default="aligned",
    )
    disagreement = disagreement.sort_values(
        "defense_percentile_gap", key=lambda values: values.abs(), ascending=False
    )
    player_disagreement = (
        disagreement.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
        .agg(
            seasons=("Season", "nunique"),
            mean_defense_percentile_gap=("defense_percentile_gap", "mean"),
            mean_absolute_defense_percentile_gap=(
                "defense_percentile_gap",
                lambda values: float(values.abs().mean()),
            ),
        )
        .sort_values("mean_absolute_defense_percentile_gap", ascending=False)
    )

    summary = (
        fold_metrics.groupby("component", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_season", "nunique"),
        )
    )
    run_id = f"single_season_spm_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "single_season_spm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for side, model in fitted.items():
        path = output / f"model_{side}.joblib"
        joblib.dump(model, path)
        final_models[side] = {
            "path": str(path.resolve()),
            "features": list(selected[side]),
            "feature_count": len(selected[side]),
        }
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    final.to_parquet(output / "leaderboard.parquet", index=False)
    benchmark.to_parquet(output / "external_benchmark.parquet", index=False)
    disagreement.to_parquet(output / "defensive_disagreements.parquet", index=False)
    player_disagreement.to_parquet(
        output / "defensive_player_summary.parquet", index=False
    )
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    external_metrics.to_parquet(output / "external_metrics.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "single_season_statistical_plus_minus",
        "estimand": "single_regular_season_normal_rapm_offense_defense_and_net",
        "status": "research_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "output_seasons": list(output_seasons),
            "training_seasons": sorted(int(value) for value in panel["Season"].unique()),
            "builder_sha256": sha256_file(Path(__file__)),
            "evaluation": "leave-one-season-out across the full modern panel",
            "final_fit": "all available labeled seasons",
            "temporal_features_excluded": True,
            "minimum_possessions_per_side": minimum_possessions_per_side,
            "additional_offense_features": list(additional_offense_features),
            "additional_defense_features": list(additional_defense_features),
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "reference_run": sha256_file(Path(reference_run_path) / "run.json"),
                "names": sha256_file(names_path),
                "external": _external_source_hashes(
                    Path(external_raw_root), output_seasons
                ),
            },
        },
        "quality": {
            "panel_rows": len(panel),
            "oof_rows": len(oof),
            "leaderboard_rows": len(final),
            "duplicate_oof_keys": int(oof.duplicated(["PLAYER_ID", "Season"]).sum()),
            "external_benchmark_rows": len(benchmark),
            "xrapm_matched_rows": int(benchmark["xrapm_net"].notna().sum()),
            "bpm_matched_rows": int(benchmark["bpm_net"].notna().sum()),
            "high_exposure_rows": int(benchmark["high_exposure"].sum()),
            "high_exposure_xrapm_matched_rows": int(
                (benchmark["high_exposure"] & benchmark["xrapm_net"].notna()).sum()
            ),
            "nonfinite_prediction_values": int(
                (~np.isfinite(oof[["spm_offense", "spm_defense", "spm_net"]])).sum().sum()
            ),
        },
        "metrics": {
            "rapm_target_summary": summary.to_dict(orient="records"),
            "external": external_metrics.to_dict(orient="records"),
        },
        "models": final_models,
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Annual zero-prior RAPM labels are noisy and are not ground truth.",
            "Leave-one-season-out evaluation uses both earlier and later seasons to learn the descriptive SPM mapping; it is not a forecast.",
            "xRAPM uses a box prior and multiple information windows, so its annual table is an external comparator rather than a pure one-year target.",
            "The final leaderboard refits on every available labeled season and is separate from the out-of-fold evaluation table.",
            "Defensive disagreement summaries require at least the configured possession threshold on both sides.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
