"""Five-year-target SPM and its single-season RAPM posterior update.

The challenger matches rolling five-year player features to rolling five-year
zero-prior RAPM labels.  For rating season ``s``, every training label ends
before ``s``.  The predicted SPM for the five-year window ending in ``s`` then
centers a RAPM fit using possessions from season ``s`` only.  Both SPM and AIO
are evaluated on season ``s + 1``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.data.statistical_features_v2 import _engineer_window
from nba_impact.models.predictive_spm import _predictive_metrics
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    _game_margin_metrics,
    build_design,
    fit_coefficient_center_path,
    load_current_player_names,
    load_unified_terminal_possessions,
)
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model


EXPERIMENT_ID = "five_year_target_spm_v1"
ARMS = (
    "annual_target_spm",
    "five_year_target_spm",
    "zero_prior_rapm",
    "annual_target_aio",
    "five_year_target_aio",
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_contract(path: str | Path) -> dict:
    contract = yaml.safe_load(Path(path).read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    model = contract["model_contract"]
    rating_seasons = tuple(int(value) for value in model["rating_seasons"])
    if rating_seasons != (2021, 2022, 2023, 2024, 2025, 2026):
        raise ValueError("Rating seasons must remain 2021 through 2026.")
    evaluation = contract["evaluation"]
    test_seasons = tuple(
        int(value)
        for value in (
            *evaluation["development_test_seasons"],
            *evaluation["reused_diagnostic_test_seasons"],
        )
    )
    if test_seasons != (2022, 2023, 2024, 2025, 2026):
        raise ValueError("Evaluation seasons must remain 2022 through 2026.")
    if 2027 not in {int(v) for v in evaluation["untouched_confirmation_seasons"]}:
        raise ValueError("Season 2027 must remain an untouched confirmation.")
    aio = contract["aio_contract"]
    if int(aio["likelihood_seasons"]) != 1:
        raise ValueError("AIO must use one season of possession evidence.")
    if float(aio["center_scale"]) != 1.0:
        raise ValueError("This target-only comparison fixes the center scale at 1.")
    return json.loads(json.dumps(contract, default=str))


def _feature_lists(
    reference_manifest_path: str | Path,
    contract: dict,
) -> dict[str, tuple[str, ...]]:
    manifest = json.loads(Path(reference_manifest_path).read_text())
    raw = manifest.get("features")
    if not isinstance(raw, dict):
        raise ValueError("Feature reference manifest has no feature lists.")
    model_contract = contract["model_contract"]
    selected = {}
    for side in ("offense", "defense"):
        common = tuple(str(value) for value in raw.get(side, ()))
        additions = tuple(
            str(value)
            for value in model_contract.get(f"additional_{side}_features", ())
        )
        selected[side] = tuple(dict.fromkeys((*common, *additions)))
    if any(not values for values in selected.values()):
        raise ValueError("Feature reference must define both offense and defense.")
    forbidden = {
        "OnOffRtg",
        "OnDefRtg",
        "Age",
        "height",
        "position",
        "minutes",
        "games",
        "BPM",
        "xRAPM",
    }
    if overlap := sorted(forbidden & set((*selected["offense"], *selected["defense"]))):
        raise ValueError(f"Frozen SPM features contain forbidden fields: {overlap}.")
    return selected


def _target_panels(
    annual_features: pd.DataFrame,
    annual_targets: pd.DataFrame,
    five_year_features: pd.DataFrame,
    five_year_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if annual_features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual feature keys must be unique.")
    if five_year_features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year feature keys must be unique.")
    annual = annual_targets.rename(columns={"Season": "Window_End"}).copy()
    five = five_year_targets.rename(
        columns={
            "window_end": "Window_End",
            "offense": "target_offense",
            "defense": "target_defense",
            "net": "target_net",
        }
    ).copy()
    target_columns = [
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    ]
    if annual.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual target keys must be unique.")
    if five.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year target keys must be unique.")

    def merge(feature: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        panel = feature.merge(
            target[target_columns],
            on=["PLAYER_ID", "Window_End"],
            how="inner",
            validate="one_to_one",
        )
        panel["sample_weight"] = np.sqrt(
            np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
        )
        finite = np.isfinite(
            panel[["target_offense", "target_defense", "sample_weight"]].to_numpy(
                dtype=float
            )
        ).all(axis=1)
        return panel.loc[finite].copy()

    return merge(annual_features, annual), merge(five_year_features, five)


def _matched_five_year_inputs(
    *,
    reference_features_path: str | Path,
    reference_targets_path: str | Path,
    rolling_targets_path: str | Path,
    player_sheet_dir: str | Path,
    annual_features: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Extend the exact matched-window 2014-23 panel through 2026."""
    reference_features = pd.read_parquet(reference_features_path)
    reference_features = reference_features.loc[
        reference_features["Window_End"].between(2018, 2023)
    ].copy()
    reference_targets = pd.read_parquet(reference_targets_path)
    reference_targets = reference_targets.loc[
        reference_targets["horizon"].eq("5y")
        & reference_targets["Window_End"].ge(2018)
        & reference_targets["Window_End"].le(2023)
    ].drop(columns="horizon")
    rolling = pd.read_parquet(rolling_targets_path).rename(
        columns={
            "window_end": "Window_End",
            "offense": "target_offense",
            "defense": "target_defense",
            "net": "target_net",
        }
    )
    overlap = reference_targets.merge(
        rolling,
        on=["PLAYER_ID", "Window_End"],
        suffixes=("_reference", "_rolling"),
        validate="one_to_one",
    )
    maximum_overlap_error = float(
        max(
            (
                overlap[f"target_{side}_reference"]
                - overlap[f"target_{side}_rolling"]
            ).abs().max()
            for side in ("offense", "defense", "net")
        )
    )
    if maximum_overlap_error > 1e-6:
        raise ValueError(
            "Reference and rolling five-year RAPM labels differ beyond solver tolerance."
        )

    sheet_root = Path(player_sheet_dir)
    loaded = {
        season: _load_source(sheet_root / f"{season}.csv", season)[0]
        for season in range(2020, 2027)
    }
    current_features = []
    for end in (2024, 2025, 2026):
        frames = [loaded[season] for season in range(end - 4, end + 1)]
        temporal = [
            _aggregate_window([loaded[season]], season)
            for season in range(end - 2, end + 1)
        ]
        current_features.append(
            _engineer_window(_aggregate_window(frames, end), frames, temporal)
        )
    five_year_features = pd.concat(
        [reference_features, *current_features], ignore_index=True
    )
    five_year_targets = rolling.copy()

    # The original horizon bake-off used the feature families common to every
    # window. Add the frozen annual model's zTS and defense fields by pooling
    # their annual values over the identical five seasons. Possessions are the
    # aggregation weights; they remain excluded from the fitted feature matrix.
    missing_by_side = {
        side: tuple(value for value in values if value not in five_year_features)
        for side, values in selected.items()
    }
    pooled_rows = []
    for end in range(2018, 2027):
        window = annual_features.loc[
            annual_features["Window_End"].between(end - 4, end)
        ]
        player_ids = window["PLAYER_ID"].drop_duplicates().astype(int)
        pooled = pd.DataFrame({"PLAYER_ID": player_ids, "Window_End": end})
        for side, fields in missing_by_side.items():
            weight_field = "OffPoss" if side == "offense" else "DefPoss"
            weights = pd.to_numeric(window[weight_field], errors="coerce").clip(lower=0)
            for field in fields:
                values = pd.to_numeric(window[field], errors="coerce")
                valid = values.notna() & weights.notna() & weights.gt(0)
                numerator = (values.where(valid, 0.0) * weights.where(valid, 0.0)).groupby(
                    window["PLAYER_ID"]
                ).sum()
                denominator = weights.where(valid, 0.0).groupby(window["PLAYER_ID"]).sum()
                pooled[field] = pooled["PLAYER_ID"].map(
                    (numerator / denominator.replace(0.0, np.nan)).to_dict()
                )
        pooled_rows.append(pooled)
    pooled_features = pd.concat(pooled_rows, ignore_index=True)
    added_fields = tuple(
        dict.fromkeys(value for values in missing_by_side.values() for value in values)
    )
    five_year_features = five_year_features.merge(
        pooled_features[["PLAYER_ID", "Window_End", *added_fields]],
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    if five_year_features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Extended five-year feature keys are not unique.")
    if five_year_targets.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Extended five-year target keys are not unique.")
    return five_year_features, five_year_targets, maximum_overlap_error


def _fit_spm(
    panel: pd.DataFrame,
    inference: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    *,
    rating_season: int,
    target_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
    ends = tuple(sorted(int(value) for value in train["Window_End"].unique()))
    if target_name == "five_year" and len(ends) < 3:
        raise ValueError(f"Season {rating_season} has fewer than three five-year labels.")
    scored = inference.loc[inference["Window_End"].eq(rating_season)].copy()
    if train.empty or scored.empty:
        raise ValueError(f"SPM fold {rating_season} has an empty train or score set.")
    output = scored[["PLAYER_ID", "Window_End"]].copy()
    models = {}
    for side in ("offense", "defense"):
        model = _fit_model(
            _frozen_model(side), train, selected[side], f"target_{side}"
        )
        output[f"prior_{side}_per_100"] = model.predict(
            scored.loc[:, selected[side]]
        )
        models[side] = model
    output["prior_net_per_100"] = (
        output["prior_offense_per_100"] + output["prior_defense_per_100"]
    )
    output["target_kind"] = target_name
    output["training_target_start"] = ends[0]
    output["training_target_end"] = ends[-1]
    output["training_target_seasons"] = len(ends)
    return output, models


def _rating_table(
    design,
    beta: np.ndarray,
    train_mask: np.ndarray,
    names: pd.DataFrame,
) -> pd.DataFrame:
    n = len(design.players)
    train = design.X[train_mask]
    off = np.asarray(train[:, :n].sum(axis=0)).ravel()
    deff = np.asarray(train[:, n : 2 * n].sum(axis=0)).ravel()
    output = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": off,
            "Poss_Def": deff,
        }
    )
    output["net"] = output["offense"] + output["defense"]
    return output.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")


def _player_metrics(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    test_season: int,
    candidate: str,
    common_player_ids: set[int],
) -> list[dict]:
    target = targets.loc[targets["Season"].eq(test_season), [
        "PLAYER_ID", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"
    ]].rename(
        columns={"Poss_Off": "target_Poss_Off", "Poss_Def": "target_Poss_Def"}
    )
    predictions = predictions.loc[
        predictions["PLAYER_ID"].astype(int).isin(common_player_ids)
    ]
    target = target.loc[target["PLAYER_ID"].astype(int).isin(common_player_ids)]
    merged = predictions.merge(target, on="PLAYER_ID", how="inner", validate="one_to_one")
    if len(merged) != len(common_player_ids):
        raise ValueError("Player comparators must use exactly the same rows.")
    merged["sample_weight"] = np.sqrt(
        np.minimum(merged["target_Poss_Off"], merged["target_Poss_Def"]).clip(lower=1)
    )
    rows = []
    for side in ("offense", "defense", "net"):
        predicted = (
            merged[f"prior_{side}_per_100"]
            if f"prior_{side}_per_100" in merged
            else merged[side]
        )
        rows.append(
            {
                "evaluation": "next_season_annual_rapm",
                "candidate": candidate,
                "component": side,
                "test_season": test_season,
                "rows": len(merged),
                **_predictive_metrics(
                    merged[f"target_{side}"].to_numpy(dtype=float),
                    predicted.to_numpy(dtype=float),
                    merged["sample_weight"].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def _paired_bootstrap(
    games: pd.DataFrame,
    *,
    challenger: str,
    baseline: str,
    seasons: tuple[int, ...],
    draws: int,
    seed: int,
) -> dict:
    scope = games.loc[
        games["test_season"].isin(seasons)
        & games["candidate"].isin([challenger, baseline])
    ]
    wide = scope.pivot(
        index=["test_season", "game_id"], columns="candidate", values="squared_error"
    )
    if wide[[challenger, baseline]].isna().any().any():
        raise ValueError("Bootstrap candidates do not share identical games.")
    groups = [
        group[challenger].to_numpy(dtype=float)
        - group[baseline].to_numpy(dtype=float)
        for _, group in wide.groupby(level="test_season", sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        samples[index] = np.mean(
            [rng.choice(values, size=len(values), replace=True).mean() for values in groups]
        )
    observed = float(np.mean([values.mean() for values in groups]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "challenger": challenger,
        "baseline": baseline,
        "seasons": list(seasons),
        "matched_games": int(len(wide)),
        "observed_equal_season_mse_delta": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "probability_challenger_better": float(np.mean(samples < 0)),
        "draws": draws,
        "seed": seed,
    }


def build_five_year_target_spm(
    *,
    annual_features_path: str | Path,
    annual_targets_path: str | Path,
    five_year_reference_features_path: str | Path,
    five_year_reference_targets_path: str | Path,
    five_year_rolling_targets_path: str | Path,
    player_sheet_dir: str | Path,
    reference_manifest_path: str | Path,
    legacy_cache_dir: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
    player_games_path: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    contract = _load_contract(contract_path)
    input_files = {
        "annual_features": Path(annual_features_path),
        "annual_targets": Path(annual_targets_path),
        "five_year_reference_features": Path(five_year_reference_features_path),
        "five_year_reference_targets": Path(five_year_reference_targets_path),
        "five_year_rolling_targets": Path(five_year_rolling_targets_path),
        "feature_reference": Path(reference_manifest_path),
        "current_possessions": Path(current_possessions_path),
        "current_segments": Path(current_segments_path),
        "player_games": Path(player_games_path),
        "contract": Path(contract_path),
    }
    source_hashes = {name: sha256_file(path) for name, path in input_files.items()}
    for season in range(2020, 2027):
        path = Path(player_sheet_dir) / f"{season}.csv"
        source_hashes[f"player_sheet_{season}"] = sha256_file(path)
    for season in (2021, 2022, 2023):
        path = Path(legacy_cache_dir) / f"matchups_{season}.parquet"
        source_hashes[f"legacy_possessions_{season}"] = sha256_file(path)
    identity = hashlib.sha256(
        json.dumps(
            {
                "source_hashes": source_hashes,
                "source_code": sha256_file(Path(__file__)),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"five_year_target_spm_v1_{identity}"
    output = Path(artifact_root) / "models" / "five_year_target_spm" / run_id
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    selected = _feature_lists(reference_manifest_path, contract)
    features = pd.read_parquet(annual_features_path)
    annual_targets = pd.read_parquet(annual_targets_path)
    five_features, five_targets, maximum_overlap_error = _matched_five_year_inputs(
        reference_features_path=five_year_reference_features_path,
        reference_targets_path=five_year_reference_targets_path,
        rolling_targets_path=five_year_rolling_targets_path,
        player_sheet_dir=player_sheet_dir,
        annual_features=features,
        selected=selected,
    )
    if max(
        features["Window_End"].max(),
        annual_targets["Season"].max(),
        five_targets["Window_End"].max(),
    ) > 2026:
        raise ValueError("Season 2027 must not enter the SPM inputs.")
    selected_fields = {value for values in selected.values() for value in values}
    for label, source_features in (
        ("Annual", features),
        ("Five-year", five_features),
    ):
        if missing := sorted(selected_fields - set(source_features)):
            raise ValueError(f"{label} features are missing frozen fields: {missing}.")
    annual_panel, five_panel = _target_panels(
        features,
        annual_targets,
        five_features,
        five_targets,
    )
    rating_seasons = tuple(int(v) for v in contract["model_contract"]["rating_seasons"])

    frame = load_unified_terminal_possessions(
        legacy_cache_dir,
        current_possessions_path,
        current_segments_path,
        tuple(range(rating_seasons[0], rating_seasons[-1] + 1)),
        transition_season=2024,
        game_types=("regular",),
    )
    if int(frame["season"].max()) > 2026:
        raise ValueError("Season 2027 entered the possession design.")
    design = build_design(frame, include_home=True)
    names = load_current_player_names(
        Path(legacy_cache_dir).parent / "all_names.csv", player_games_path
    )
    rolling_names = five_targets[["PLAYER_ID", "PLAYER_NAME"]].dropna().drop_duplicates(
        "PLAYER_ID", keep="last"
    )
    names = pd.concat(
        [names.loc[~names["PLAYER_ID"].isin(rolling_names["PLAYER_ID"])], rolling_names],
        ignore_index=True,
    ).drop_duplicates("PLAYER_ID", keep="last")

    predictions = []
    ratings = []
    games = []
    metrics = []
    coverage = []
    latest_models: dict[str, dict[str, object]] = {}
    aio = contract["aio_contract"]
    config = RapmConfig(
        seasons=rating_seasons,
        lambda_off=float(aio["lambda_off"]),
        lambda_def=float(aio["lambda_def"]),
        lambda_home=float(aio["lambda_home"]),
        include_home=True,
        game_types=("regular",),
        data_scope="five_year_target_spm_single_season_aio",
    )
    for rating_season in rating_seasons:
        annual_spm, annual_models = _fit_spm(
            annual_panel,
            features,
            selected,
            rating_season=rating_season,
            target_name="annual",
        )
        five_spm, five_models = _fit_spm(
            five_panel,
            five_features,
            selected,
            rating_season=rating_season,
            target_name="five_year",
        )
        predictions.extend([annual_spm, five_spm])
        latest_models = {"annual": annual_models, "five_year": five_models}

        train_mask = design.seasons == rating_season
        if not train_mask.any():
            raise ValueError(f"No single-season possessions for {rating_season}.")
        test_season = rating_season + 1
        test_mask = design.seasons == test_season if test_season <= 2026 else train_mask
        centers = {}
        for target_name, prior in (("annual", annual_spm), ("five_year", five_spm)):
            center, report = build_prior_center(
                design,
                prior,
                prior_window_end=rating_season,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            centers[target_name] = center
            report.update({"rating_season": rating_season, "target_kind": target_name})
            coverage.append(report)

        fit_map = {
            "zero_prior_rapm": fit_coefficient_center_path(
                design,
                config,
                np.zeros(design.X.shape[1]),
                center_scales=(0.0,),
                row_mask=train_mask,
            )[0.0],
            "annual_target_aio": fit_coefficient_center_path(
                design,
                config,
                centers["annual"],
                center_scales=(1.0,),
                row_mask=train_mask,
            )[1.0],
            "five_year_target_aio": fit_coefficient_center_path(
                design,
                config,
                centers["five_year"],
                center_scales=(1.0,),
                row_mask=train_mask,
            )[1.0],
        }
        fold_ratings = {}
        for candidate, (beta, intercept) in fit_map.items():
            table = _rating_table(design, beta, train_mask, names)
            table["rating_season"] = rating_season
            table["candidate"] = candidate
            ratings.append(table)
            fold_ratings[candidate] = table
            if test_season <= 2026:
                game_metric = _game_margin_metrics(
                    design, beta, intercept, test_mask, train_mask
                )
                metrics.append(
                    {
                        "evaluation": "next_season_game_margin",
                        "candidate": candidate,
                        "component": "net",
                        "rating_season": rating_season,
                        "test_season": test_season,
                        "rows": game_metric["games"],
                        "weighted_rmse": game_metric["margin_rmse"],
                        "weighted_correlation": game_metric["margin_correlation"],
                        "correlation": game_metric["margin_correlation"],
                        "dispersion_ratio": game_metric["predicted_margin_sd"]
                        / game_metric["actual_margin_sd"],
                        "calibration_slope": game_metric["calibration_slope"],
                        "calibration_intercept": game_metric["calibration_intercept"],
                    }
                )
                game = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
                game["candidate"] = candidate
                game["rating_season"] = rating_season
                game["test_season"] = test_season
                game["squared_error"] = (game["actual_margin"] - game["predicted_margin"]) ** 2
                games.append(game)

        if test_season <= 2026:
            target_ids = set(
                annual_targets.loc[
                    annual_targets["Season"].eq(test_season), "PLAYER_ID"
                ].astype(int)
            )
            common_player_ids = (
                set(annual_spm["PLAYER_ID"].astype(int))
                & set(five_spm["PLAYER_ID"].astype(int))
                & target_ids
            )
            metrics.extend(
                _player_metrics(
                    annual_spm,
                    annual_targets,
                    test_season=test_season,
                    candidate="annual_target_spm",
                    common_player_ids=common_player_ids,
                )
            )
            metrics.extend(
                _player_metrics(
                    five_spm,
                    annual_targets,
                    test_season=test_season,
                    candidate="five_year_target_spm",
                    common_player_ids=common_player_ids,
                )
            )
            for candidate, table in fold_ratings.items():
                metrics.extend(
                    _player_metrics(
                        table,
                        annual_targets,
                        test_season=test_season,
                        candidate=candidate,
                        common_player_ids=common_player_ids,
                    )
                )
            print(
                f"five-year-target SPM fold {rating_season}->{test_season}: complete",
                flush=True,
            )
        else:
            print(f"five-year-target SPM final {rating_season}: complete", flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    rating_frame = pd.concat(ratings, ignore_index=True)
    game_frame = pd.concat(games, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    summary = (
        metric_frame.groupby(["evaluation", "candidate", "component"], as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_rmse=("weighted_rmse", "mean"),
            mean_correlation=("weighted_correlation", "mean"),
        )
        .sort_values(["evaluation", "component", "mean_rmse", "candidate"], kind="stable")
    )
    development = tuple(int(v) for v in contract["evaluation"]["development_test_seasons"])
    diagnostics = tuple(int(v) for v in contract["evaluation"]["reused_diagnostic_test_seasons"])
    bootstrap = {
        "development": _paired_bootstrap(
            game_frame,
            challenger="five_year_target_aio",
            baseline="annual_target_aio",
            seasons=development,
            draws=2000,
            seed=20260826,
        ),
        "reused_diagnostics": _paired_bootstrap(
            game_frame,
            challenger="five_year_target_aio",
            baseline="annual_target_aio",
            seasons=diagnostics,
            draws=2000,
            seed=20260827,
        ),
    }
    game_rows = metric_frame.loc[metric_frame["evaluation"].eq("next_season_game_margin")]
    wide = game_rows.pivot(index="test_season", columns="candidate", values="weighted_rmse")
    dev_win = float(wide.loc[list(development), "five_year_target_aio"].mean()) < float(
        wide.loc[list(development), "annual_target_aio"].mean()
    )
    diagnostic_wins = {
        str(season): bool(
            wide.loc[season, "five_year_target_aio"]
            < wide.loc[season, "annual_target_aio"]
        )
        for season in diagnostics
    }
    research_replacement = dev_win and all(diagnostic_wins.values())

    _atomic_parquet(prediction_frame, output / "spm_predictions.parquet")
    _atomic_parquet(rating_frame, output / "aio_ratings.parquet")
    _atomic_parquet(game_frame, output / "game_predictions.parquet")
    _atomic_parquet(metric_frame, output / "fold_metrics.parquet")
    _atomic_parquet(summary, output / "summary.parquet")
    _atomic_parquet(pd.DataFrame(coverage), output / "prior_coverage.parquet")
    _atomic_parquet(five_targets, output / "five_year_targets.parquet")
    model_dir = output / "models"
    model_dir.mkdir()
    for target_name, models in latest_models.items():
        for side, model in models.items():
            joblib.dump(model, model_dir / f"{target_name}_{side}.joblib")

    manifest = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "estimand_id": contract["estimand_id"],
        "status": "research_replacement" if research_replacement else "research_null",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "source_hashes": source_hashes,
        "features": {side: list(values) for side, values in selected.items()},
        "quality": {
            "annual_panel_rows": len(annual_panel),
            "five_year_panel_rows": len(five_panel),
            "spm_prediction_rows": len(prediction_frame),
            "aio_rating_rows": len(rating_frame),
            "game_prediction_rows": len(game_frame),
            "identical_game_rows": True,
            "maximum_loaded_season": int(design.seasons.max()),
            "season_2027_loaded": False,
            "five_year_reference_overlap_max_error": maximum_overlap_error,
            "component_identity_max_error": float(
                max(
                    (
                        prediction_frame["prior_offense_per_100"]
                        + prediction_frame["prior_defense_per_100"]
                        - prediction_frame["prior_net_per_100"]
                    )
                    .abs()
                    .max(),
                    (
                        rating_frame["offense"]
                        + rating_frame["defense"]
                        - rating_frame["net"]
                    )
                    .abs()
                    .max(),
                )
            ),
        },
        "decision": {
            "development_mean_game_rmse_win": dev_win,
            "reused_diagnostic_wins": diagnostic_wins,
            "research_replacement": research_replacement,
            "production_replacement": False,
            "reason": (
                "The matched five-year challenger passed the frozen research gate; "
                "Season 2027 remains required for production promotion."
                if research_replacement
                else "The matched five-year challenger failed the frozen research gate."
            ),
        },
        "bootstrap": bootstrap,
        "summary": summary.to_dict("records"),
        "paths": {
            "spm_predictions": "spm_predictions.parquet",
            "aio_ratings": "aio_ratings.parquet",
            "game_predictions": "game_predictions.parquet",
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "prior_coverage": "prior_coverage.parquet",
            "five_year_targets": "five_year_targets.parquet",
            "models": "models",
        },
        "forbidden_interpretation": (
            "Production SPM or AIO promotion before untouched Season 2027."
        ),
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest
