"""Walk-forward aging selection and current player/team projections."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic


METHODS = (
    "ar1",
    "linear_age",
    "quadratic_age",
    "spline_age",
    "spline_age_minutes",
    "spline_age_impact",
)
KNOTS = (22.0, 25.0, 28.0, 31.0, 34.0)


def _metadata(player_sheets_dir: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = player_sheets_dir / f"{season}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda column: column
            in {"PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "MIN"},
        )
        frame["Season"] = season
        frame = frame.dropna(subset=["PLAYER_ID"]).drop_duplicates("PLAYER_ID", keep="last")
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frames.append(frame)
    if not frames:
        raise ValueError("No player-season metadata was found.")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Player-season metadata keys are not unique.")
    return result


def _design(frame: pd.DataFrame, method: str, side: str) -> np.ndarray:
    if method not in METHODS or method == "ar1":
        raise ValueError(f"Unsupported adjustment method: {method}.")
    age = frame["AGE"].to_numpy(dtype=float) - 27.0
    columns = [age]
    if method in {"quadratic_age", "spline_age", "spline_age_minutes", "spline_age_impact"}:
        columns.append(age**2 / 25.0)
    if method in {"spline_age", "spline_age_minutes", "spline_age_impact"}:
        columns.extend(np.maximum(0.0, frame["AGE"].to_numpy(dtype=float) - knot) for knot in KNOTS)
    if method == "spline_age_minutes":
        log_minutes = np.log1p(frame["MIN"].clip(lower=0).to_numpy(dtype=float))
        columns.extend((log_minutes, age * log_minutes))
    if method == "spline_age_impact":
        impact = frame[f"filtered_{side}"].to_numpy(dtype=float)
        columns.extend((impact, age * impact))
    return np.column_stack(columns)


def _weighted_metrics(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict:
    mse = float(np.average((actual - predicted) ** 2, weights=weight))
    mean = float(np.average(actual, weights=weight))
    denominator = float(np.sum(weight * (actual - mean) ** 2))
    numerator = float(np.sum(weight * (actual - predicted) ** 2))
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0
        else np.nan
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "correlation": correlation,
        "r2": float(1.0 - numerator / denominator) if denominator > 0 else np.nan,
    }


def _transitions(
    trajectories: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    minimum_side_possessions: float,
) -> pd.DataFrame:
    current = trajectories[
        [
            "PLAYER_ID", "Season", "filtered_offense", "filtered_defense", "filtered_net",
            "Poss_Off", "Poss_Def", "phi",
        ]
    ].merge(metadata, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one")
    next_target = targets[
        ["PLAYER_ID", "Season", "target_offense", "target_defense", "Poss_Off", "Poss_Def"]
    ].copy()
    next_target["Season"] -= 1
    next_target = next_target.rename(
        columns={
            "target_offense": "next_offense",
            "target_defense": "next_defense",
            "Poss_Off": "next_Poss_Off",
            "Poss_Def": "next_Poss_Def",
        }
    )
    frame = current.merge(next_target, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one")
    frame = frame.dropna(subset=["AGE", "MIN"]).copy()
    exposure = frame[["Poss_Off", "Poss_Def", "next_Poss_Off", "next_Poss_Def"]].min(axis=1)
    frame = frame.loc[exposure.ge(minimum_side_possessions)].copy()
    frame["evaluation_weight"] = np.sqrt(exposure.loc[frame.index])
    frame["origin_season"] = frame["Season"].astype(int)
    frame["target_season"] = frame["origin_season"] + 1
    return frame.reset_index(drop=True)


def _predict_method(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    side: str,
    *,
    alpha: float,
) -> np.ndarray:
    base_train = train["phi"].to_numpy(dtype=float) * train[f"filtered_{side}"].to_numpy(dtype=float)
    base_test = test["phi"].to_numpy(dtype=float) * test[f"filtered_{side}"].to_numpy(dtype=float)
    if method == "ar1":
        return base_test
    residual = train[f"next_{side}"].to_numpy(dtype=float) - base_train
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        _design(train, method, side),
        residual,
        ridge__sample_weight=train["evaluation_weight"].to_numpy(dtype=float),
    )
    return base_test + model.predict(_design(test, method, side))


def _score_folds(
    transitions: pd.DataFrame,
    origins: tuple[int, ...],
    *,
    alpha: float,
    minimum_training_origins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for origin in origins:
        train = transitions.loc[transitions["origin_season"].lt(origin)]
        test = transitions.loc[transitions["origin_season"].eq(origin)].copy()
        if train["origin_season"].nunique() < minimum_training_origins or test.empty:
            continue
        for method in METHODS:
            scored = test[
                ["PLAYER_ID", "PLAYER_NAME", "origin_season", "target_season", "AGE", "MIN", "TEAM_ABBREVIATION", "filtered_net", "evaluation_weight"]
            ].copy()
            for side in ("offense", "defense"):
                scored[f"actual_{side}"] = test[f"next_{side}"].to_numpy(dtype=float)
                scored[f"projected_{side}"] = _predict_method(train, test, method, side, alpha=alpha)
            scored["actual_net"] = scored["actual_offense"] + scored["actual_defense"]
            scored["projected_net"] = scored["projected_offense"] + scored["projected_defense"]
            scored["method"] = method
            prediction_rows.append(scored)
            for component in ("offense", "defense", "net"):
                metrics = _weighted_metrics(
                    scored[f"actual_{component}"].to_numpy(dtype=float),
                    scored[f"projected_{component}"].to_numpy(dtype=float),
                    scored["evaluation_weight"].to_numpy(dtype=float),
                )
                metric_rows.append(
                    {
                        "origin_season": origin,
                        "target_season": origin + 1,
                        "method": method,
                        "component": component,
                        "rows": len(scored),
                        "training_origins": int(train["origin_season"].nunique()),
                        **metrics,
                    }
                )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def _subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["age_group"] = pd.cut(values["AGE"], [-np.inf, 23, 27, 31, np.inf], labels=["23_or_younger", "24_27", "28_31", "32_plus"])
    values["minutes_group"] = pd.cut(values["MIN"], [-np.inf, 999, 1999, np.inf], labels=["under_1000", "1000_1999", "2000_plus"])
    values["impact_group"] = pd.cut(values["filtered_net"], [-np.inf, -1, 1, 3, np.inf], labels=["below_minus_1", "minus_1_to_1", "1_to_3", "above_3"])
    rows: list[dict] = []
    for dimension in ("age_group", "minutes_group", "impact_group"):
        for group, selected in values.groupby(dimension, observed=True):
            if len(selected) < 20:
                continue
            metrics = _weighted_metrics(
                selected["actual_net"].to_numpy(dtype=float),
                selected["projected_net"].to_numpy(dtype=float),
                selected["evaluation_weight"].to_numpy(dtype=float),
            )
            rows.append({"dimension": dimension, "group": str(group), "rows": len(selected), **metrics})
    return pd.DataFrame(rows)


def _historical_player_projections(
    predictions: pd.DataFrame,
    *,
    selected_method: str,
    selection_origins: tuple[int, ...],
    diagnostic_origins: tuple[int, ...],
) -> pd.DataFrame:
    """Keep causal player backtests for the selected projection method.

    Each row is fit only on transitions with an origin before its origin season.
    The selection rows remain reused evidence because their losses selected the
    displayed method; diagnostic rows are later, still-reused diagnostics.
    """
    result = predictions.loc[predictions["method"].eq(selected_method)].copy()
    if result.empty:
        raise ValueError("Selected aging method has no walk-forward predictions.")
    known_origins = set(selection_origins) | set(diagnostic_origins)
    actual_origins = set(result["origin_season"].astype(int))
    if actual_origins != known_origins:
        raise ValueError("Historical projections do not cover every configured origin.")
    if (result["target_season"] != result["origin_season"] + 1).any():
        raise ValueError("Historical projection targets must follow their origin by one season.")
    if result.duplicated(["PLAYER_ID", "origin_season"]).any():
        raise ValueError("Historical player projections must have unique player-origin keys.")
    result["projection_kind"] = "walk_forward_backtest"
    result["evidence_status"] = np.where(
        result["origin_season"].isin(selection_origins),
        "selection_reused",
        "diagnostic_reused",
    )
    result["method"] = selected_method
    ordered = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "origin_season", "target_season",
        "AGE", "MIN", "filtered_net", "evaluation_weight", "method",
        "projection_kind", "evidence_status",
        "actual_offense", "actual_defense", "actual_net",
        "projected_offense", "projected_defense", "projected_net",
    ]
    return result[ordered].sort_values(["target_season", "projected_net"], ascending=[True, False])


def build_aging_projection(
    trajectories_path: str | Path,
    targets_path: str | Path,
    player_sheets_dir: str | Path,
    *,
    artifact_root: str | Path,
    selection_origins: tuple[int, ...] = (2018, 2019, 2020, 2021),
    diagnostic_origins: tuple[int, ...] = (2022, 2023),
    projection_origin: int = 2026,
    minimum_side_possessions: float = 1000.0,
    minimum_training_origins: int = 3,
    alpha: float = 25.0,
) -> dict:
    trajectories = pd.read_parquet(trajectories_path)
    targets = pd.read_parquet(targets_path)
    seasons = tuple(sorted(int(value) for value in trajectories["Season"].unique()))
    metadata = _metadata(Path(player_sheets_dir), seasons)
    transitions = _transitions(trajectories, targets, metadata, minimum_side_possessions)
    origins = tuple(sorted(set(selection_origins + diagnostic_origins)))
    fold_metrics, predictions = _score_folds(
        transitions, origins, alpha=alpha, minimum_training_origins=minimum_training_origins
    )
    selection = fold_metrics.loc[
        fold_metrics["origin_season"].isin(selection_origins) & fold_metrics["component"].eq("net")
    ].groupby("method", as_index=False).agg(mean_rmse=("rmse", "mean"), mean_correlation=("correlation", "mean"), mean_r2=("r2", "mean"), folds=("origin_season", "nunique"))
    if selection.empty:
        raise ValueError("No aging method has complete selection metrics.")
    selected_method = str(selection.sort_values(["mean_rmse", "method"]).iloc[0]["method"])
    diagnostic = fold_metrics.loc[
        fold_metrics["origin_season"].isin(diagnostic_origins) & fold_metrics["component"].eq("net")
    ].groupby("method", as_index=False).agg(mean_rmse=("rmse", "mean"), mean_correlation=("correlation", "mean"), mean_r2=("r2", "mean"), folds=("origin_season", "nunique"))
    selected_diagnostic_predictions = predictions.loc[
        predictions["method"].eq(selected_method) & predictions["origin_season"].isin(diagnostic_origins)
    ]
    subgroups = _subgroup_metrics(selected_diagnostic_predictions)
    historical_player_projections = _historical_player_projections(
        predictions,
        selected_method=selected_method,
        selection_origins=selection_origins,
        diagnostic_origins=diagnostic_origins,
    )

    final_train = transitions.loc[transitions["origin_season"].lt(projection_origin)].copy()
    current = trajectories.loc[trajectories["Season"].eq(projection_origin)].merge(
        metadata.loc[metadata["Season"].eq(projection_origin)].drop(columns="PLAYER_NAME"),
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    ).dropna(subset=["AGE", "MIN", "TEAM_ABBREVIATION"]).copy()
    current["evaluation_weight"] = np.sqrt(current[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1))
    for side in ("offense", "defense"):
        current[f"projected_{side}"] = _predict_method(final_train, current, selected_method, side, alpha=alpha)
    current["projected_net"] = current["projected_offense"] + current["projected_defense"]
    current["projection_season"] = projection_origin + 1
    player_columns = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season", "projection_season", "AGE", "MIN",
        "Poss_Off", "Poss_Def", "filtered_offense", "filtered_defense", "filtered_net",
        "projected_offense", "projected_defense", "projected_net",
    ]
    player_projections = current[player_columns].sort_values("projected_net", ascending=False)
    player_projections["projection_kind"] = "forecast"
    player_projections["evidence_status"] = "unscored_future"
    team_rows = []
    for team, group in player_projections.groupby("TEAM_ABBREVIATION", sort=True):
        total_minutes = float(group["MIN"].sum())
        projected_net = float(5.0 * np.average(group["projected_net"], weights=group["MIN"]))
        team_rows.append(
            {
                "TEAM_ABBREVIATION": team,
                "projection_season": projection_origin + 1,
                "players": len(group),
                "minutes": total_minutes,
                "projected_net_rating": projected_net,
                "projected_win_pace": float(np.clip(41.0 + 2.7 * projected_net, 0.0, 82.0)),
            }
        )
    team_projections = pd.DataFrame(team_rows).sort_values("projected_win_pace", ascending=False)

    run_id = f"aging_projection_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "aging_projection" / run_id
    output.mkdir(parents=True, exist_ok=False)
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    selection.to_parquet(output / "selection_summary.parquet", index=False)
    diagnostic.to_parquet(output / "diagnostic_summary.parquet", index=False)
    subgroups.to_parquet(output / "subgroup_metrics.parquet", index=False)
    historical_player_projections.to_parquet(
        output / "historical_player_projections.parquet", index=False
    )
    player_projections.to_parquet(output / "player_projections.parquet", index=False)
    team_projections.to_parquet(output / "team_projections.parquet", index=False)
    selected_selection = selection.set_index("method").loc[selected_method].to_dict()
    selected_diagnostic = diagnostic.set_index("method").loc[selected_method].to_dict()
    run = {
        "run_id": run_id,
        "model_family": "walk_forward_aging_adjusted_state_space_projection",
        "estimand": "next_season_player_strength_proxy_and_returning_minutes_team_baseline",
        "status": "research_projection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "methods": list(METHODS), "selected_method": selected_method, "alpha": alpha,
            "selection_origins": list(selection_origins), "diagnostic_origins": list(diagnostic_origins),
            "projection_origin": projection_origin, "projection_season": projection_origin + 1,
            "minimum_side_possessions": minimum_side_possessions,
            "team_translation": "five_times_minutes_weighted_player_net; wins=clip(41+2.7*net,0,82)",
            "historical_player_projection_targets": sorted(
                int(value) for value in historical_player_projections["target_season"].unique()
            ),
            "source_hashes": {
                "trajectories": sha256_file(trajectories_path), "targets": sha256_file(targets_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {"selection": selected_selection, "diagnostic": selected_diagnostic},
        "quality": {
            "transition_rows": len(transitions), "player_projection_rows": len(player_projections),
            "historical_player_projection_rows": len(historical_player_projections),
            "team_projection_rows": len(team_projections),
            "maximum_component_identity_error": float(np.abs(player_projections["projected_net"] - player_projections["projected_offense"] - player_projections["projected_defense"]).max()),
        },
        "artifact_path": str(output.resolve()),
        "caveats": [
            "The method is selected only on 2018-21; 2022-23 are reused diagnostics.",
            "Season 2027 outcomes are not used. The 2027 rows are predictions, not confirmation evidence.",
            "Team rows hold the 2026 team and minutes distribution fixed; they are a returning-minutes baseline, not a roster or schedule simulation.",
            "Historical player rows are causal walk-forward backtests conditional on a fixed candidate method. Rows used to select the method are marked selection_reused, and later rows are marked diagnostic_reused.",
            "Historical team rows are intentionally omitted. A team projection requires a contemporaneous returning-roster and minutes assumption; do not backfill it from actual next-season rosters or minutes.",
            "Canonical position history is unavailable, so position is not a model input or subgroup.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
