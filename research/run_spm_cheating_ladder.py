#!/usr/bin/env python3
"""Test deliberately impure additions to the five-year SPM.

The public statistical model excludes lineup outcomes, demographics, position,
and team strength.  This research runner adds those sources one at a time so we
can measure the price and benefit of relaxing that boundary.  Every feature for
rating season ``Y`` is computed with seasons no later than ``Y``.  Ratings are
scored against season ``Y+1`` team wins using the same player-minute rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.data.statistical_features_v2 import _engineer_window
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit as _fit_box_model,
    _select_alpha,
)
from nba_impact.models.public_aio_benchmark import build_team_win_benchmark
from nba_impact.models.statistical_impact import _metrics


ROOT = Path(__file__).resolve().parents[1]
RATING_SEASONS = (2022, 2023, 2024)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
RESIDUAL_ALPHA = 100.0


def _schedule_frame(schedule_root: Path, seasons: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return team efficiency context and team-game win rows."""
    contexts: list[pd.DataFrame] = []
    game_rows: list[pd.DataFrame] = []
    for season in seasons:
        path = schedule_root / f"leaguegamelog_{season}.json.gz"
        with gzip.open(path, "rt") as handle:
            result = json.load(handle)["resultSets"][0]
        games = pd.DataFrame(result["rowSet"], columns=result["headers"])
        if games.groupby("GAME_ID").size().ne(2).any():
            raise ValueError(f"Schedule season {season} has a non-paired game.")
        games["poss_estimate"] = (
            games["FGA"] + 0.44 * games["FTA"] - games["OREB"] + games["TOV"]
        )
        opponent = games[["GAME_ID", "TEAM_ID", "PTS", "poss_estimate"]].rename(
            columns={
                "TEAM_ID": "opponent_team_id",
                "PTS": "opponent_points",
                "poss_estimate": "opponent_possessions",
            }
        )
        paired = games.merge(opponent, on="GAME_ID", how="inner")
        paired = paired.loc[paired["TEAM_ID"].ne(paired["opponent_team_id"])].copy()
        paired["game_possessions"] = (
            paired["poss_estimate"] + paired["opponent_possessions"]
        ) / 2.0
        context = paired.groupby("TEAM_ID", as_index=False).agg(
            points=("PTS", "sum"),
            opponent_points=("opponent_points", "sum"),
            team_possessions=("game_possessions", "sum"),
        )
        context["team_net_rating"] = 100.0 * (
            context["points"] - context["opponent_points"]
        ) / context["team_possessions"]
        context["Season"] = int(season)
        contexts.append(context)
        game_rows.append(
            pd.DataFrame(
                {
                    "Season": int(season),
                    "team_id": games["TEAM_ID"].astype(int),
                    "won": games["WL"].eq("W"),
                }
            )
        )
    return pd.concat(contexts, ignore_index=True), pd.concat(game_rows, ignore_index=True)


def _first_text(values: pd.Series) -> str | None:
    values = values.dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    return None if values.empty else str(values.iloc[0])


def _legacy_fields(path: Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        usecols=["PLAYER_ID", "Season", "POSITION", "AuPM"],
    )
    source["PLAYER_ID"] = pd.to_numeric(source["PLAYER_ID"], errors="coerce")
    source = source.dropna(subset=["PLAYER_ID", "Season"]).copy()
    source["PLAYER_ID"] = source["PLAYER_ID"].astype(int)
    source["Season"] = source["Season"].astype(int)
    return source.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
        position=("POSITION", _first_text),
        aupm=("AuPM", "median"),
    )


def _annual_player_context(
    player_sheet_dir: Path,
    legacy_fields: pd.DataFrame,
    team_context: pd.DataFrame,
    seasons: range,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annual player context and derive a raw on/off net rating."""
    rows: list[pd.DataFrame] = []
    minute_rows: list[pd.DataFrame] = []
    for season in seasons:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            source, _ = _load_source(player_sheet_dir / f"{season}.csv", season)
        required = {
            "PLAYER_ID",
            "TEAM_ID",
            "MIN",
            "AGE",
            "OffPoss",
            "DefPoss",
            "OnOffRtg",
            "OnDefRtg",
        }
        if missing := sorted(required - set(source.columns)):
            raise ValueError(f"Player sheet {season} is missing {missing}.")
        frame = source.loc[:, sorted(required)].copy()
        frame["Season"] = int(season)
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frame["TEAM_ID"] = frame["TEAM_ID"].astype(int)
        frame["on_possessions"] = frame[["OffPoss", "DefPoss"]].min(axis=1)
        frame["on_net_rating"] = frame["OnOffRtg"] - frame["OnDefRtg"]
        frame = frame.merge(
            team_context.loc[team_context["Season"].eq(season), [
                "Season", "TEAM_ID", "team_possessions", "team_net_rating"
            ]],
            on=["Season", "TEAM_ID"],
            how="left",
            validate="many_to_one",
        )
        frame["off_possessions"] = frame["team_possessions"] - frame["on_possessions"]
        valid = (
            frame["off_possessions"].gt(0)
            & frame["on_possessions"].gt(0)
            & frame["team_possessions"].notna()
        )
        off_net = (
            frame["team_net_rating"] * frame["team_possessions"]
            - frame["on_net_rating"] * frame["on_possessions"]
        ) / frame["off_possessions"]
        frame["raw_onoff_net"] = (frame["on_net_rating"] - off_net).where(valid)
        frame = frame.merge(
            legacy_fields.loc[legacy_fields["Season"].eq(season)],
            on=["PLAYER_ID", "Season"],
            how="left",
            validate="one_to_one",
        )
        rows.append(frame)
        minute_rows.append(
            frame[["PLAYER_ID", "Season", "TEAM_ID", "MIN"]]
            .rename(columns={"TEAM_ID": "team_id", "MIN": "minutes"})
            .drop_duplicates()
        )
    annual = pd.concat(rows, ignore_index=True)
    if annual.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual player context has duplicate player-season keys.")
    return annual, pd.concat(minute_rows, ignore_index=True)


def _weighted_average(group: pd.DataFrame, value: str, weight: str) -> float:
    valid = group[value].notna() & group[weight].notna() & group[weight].gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(group.loc[valid, value], weights=group.loc[valid, weight]))


def _position_flags(position: str | None) -> tuple[float, float, float]:
    value = "" if position is None or pd.isna(position) else str(position).upper()
    return float("G" in value), float("F" in value), float("C" in value)


def _rolling_context(annual: pd.DataFrame, window_ends: range) -> pd.DataFrame:
    rows: list[dict] = []
    for window_end in window_ends:
        window = annual.loc[annual["Season"].between(window_end - 4, window_end)]
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            latest = group.sort_values("Season", kind="stable").iloc[-1]
            guard, forward, center = _position_flags(latest.get("position"))
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "Window_End": int(window_end),
                    "age_end": float(latest["AGE"]) if pd.notna(latest["AGE"]) else np.nan,
                    "log_minutes_5y": float(np.log1p(group["MIN"].clip(lower=0).sum())),
                    "position_guard": guard,
                    "position_forward": forward,
                    "position_center": center,
                    "raw_onoff_net_5y": _weighted_average(
                        group, "raw_onoff_net", "on_possessions"
                    ),
                    "aupm_5y": _weighted_average(group, "aupm", "on_possessions"),
                }
            )
    output = pd.DataFrame(rows)
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Rolling context has duplicate player-window keys.")
    return output


def _raptor_context(path: Path, window_ends: range) -> pd.DataFrame:
    source = pd.read_parquet(path)
    required = {
        "PLAYER_ID",
        "season",
        "official_possessions",
        "target_offense",
        "target_defense",
        "target_net",
    }
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"RAPTOR on/off source is missing {missing}.")
    rows: list[dict] = []
    for window_end in window_ends:
        window = source.loc[source["season"].between(window_end - 4, window_end)]
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            available = int(group["season"].nunique())
            rows.append(
                {
                    "PLAYER_ID": int(player_id),
                    "Window_End": int(window_end),
                    "raptor_onoff_offense_5y": _weighted_average(
                        group, "target_offense", "official_possessions"
                    ),
                    "raptor_onoff_defense_5y": _weighted_average(
                        group, "target_defense", "official_possessions"
                    ),
                    "raptor_onoff_net_5y": _weighted_average(
                        group, "target_net", "official_possessions"
                    ),
                    "raptor_season_coverage": available / 5.0,
                }
            )
    return pd.DataFrame(rows)


def _extend_features(
    reference_path: Path, player_sheet_dir: Path, maximum_window_end: int
) -> pd.DataFrame:
    reference = pd.read_parquet(reference_path)
    missing_ends = [
        end
        for end in range(int(reference["Window_End"].min()), maximum_window_end + 1)
        if end not in set(reference["Window_End"].astype(int))
    ]
    additions: list[pd.DataFrame] = []
    for end in missing_ends:
        loaded = {}
        for season in range(end - 4, end + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                loaded[season] = _load_source(player_sheet_dir / f"{season}.csv", season)[0]
        frames = [loaded[season] for season in range(end - 4, end + 1)]
        temporal = [
            _aggregate_window([loaded[season]], season)
            for season in range(end - 2, end + 1)
        ]
        additions.append(_engineer_window(_aggregate_window(frames, end), frames, temporal))
    output = pd.concat([reference, *additions], ignore_index=True) if additions else reference
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Extended feature panel has duplicate player-window keys.")
    return output


def _ridge_residual_model(alpha: float = RESIDUAL_ALPHA) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _base_ratings(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    source = source.loc[source["variant"].eq("selected_combined")].copy()
    output = source.rename(
        columns={
            "Window_End": "Season",
            "prior_offense_per_100": "offense",
            "prior_defense_per_100": "defense",
            "prior_net_per_100": "net",
        }
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Selected five-year SPM has duplicate keys.")
    return output


def _residual_variant(
    base: pd.DataFrame,
    targets: pd.DataFrame,
    context: pd.DataFrame,
    feature_map: dict[str, tuple[str, ...]],
    *,
    metric: str,
    metric_label: str,
) -> pd.DataFrame:
    panel = base.rename(columns={"Season": "Window_End"}).merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).merge(
        context,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    outputs: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 1 or test.empty:
            raise ValueError(f"Residual fold {metric} {season} is empty.")
        fold = test[["PLAYER_ID", "Window_End", "offense", "defense"]].copy()
        for side in ("offense", "defense"):
            features = feature_map[side]
            train[f"residual_{side}"] = train[f"target_{side}"] - train[side]
            model = _ridge_residual_model()
            model.fit(
                train.loc[:, features],
                train[f"residual_{side}"],
                ridge__sample_weight=train["sample_weight"],
            )
            fold[side] = fold[side] + model.predict(test.loc[:, features])
        fold["net"] = fold["offense"] + fold["defense"]
        fold = fold.rename(columns={"Window_End": "Season"})
        outputs.append(fold)
    output = pd.concat(outputs, ignore_index=True)
    output["metric"] = metric
    output["metric_label"] = metric_label
    output["category"] = "impure five-year SPM challenger"
    return output


def _box_ratings(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={"Window_End": "Season"})
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    outputs: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Season"].lt(season)].copy()
        test = panel.loc[panel["Season"].eq(season)].copy()
        fold = test[["PLAYER_ID", "Season"]].copy()
        for side in ("offense", "defense"):
            target = f"target_{side}"
            alpha = _select_alpha(train, BOX_PIPM_STYLE_FEATURES, target, ALPHA_GRID)
            model = _fit_box_model(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
            fold[side] = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
        fold["net"] = fold["offense"] + fold["defense"]
        outputs.append(fold)
    output = pd.concat(outputs, ignore_index=True)
    output["metric"] = "box_pipm_5y"
    output["metric_label"] = "Five-year BoxPIPM-style"
    output["category"] = "box-only five-year SPM"
    return output


def apply_team_reconciliation(
    ratings: pd.DataFrame,
    player_minutes: pd.DataFrame,
    team_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a BPM-style constant so each team's player sum equals team net."""
    allocation = ratings.merge(
        player_minutes,
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).merge(
        team_context[["Season", "TEAM_ID", "team_net_rating"]].rename(
            columns={"TEAM_ID": "team_id"}
        ),
        on=["Season", "team_id"],
        how="left",
        validate="many_to_one",
    )
    valid = allocation.dropna(subset=["minutes", "team_id", "team_net_rating"])
    predicted = valid.groupby(["Season", "team_id"], as_index=False).apply(
        lambda group: pd.Series(
            {"raw_team_rating": 5.0 * np.average(group["net"], weights=group["minutes"])}
        ),
        include_groups=False,
    ).reset_index(drop=True)
    adjustments = predicted.merge(
        team_context[["Season", "TEAM_ID", "team_net_rating"]].rename(
            columns={"TEAM_ID": "team_id"}
        ),
        on=["Season", "team_id"],
        validate="one_to_one",
    )
    adjustments["player_constant"] = (
        adjustments["team_net_rating"] - adjustments["raw_team_rating"]
    ) / 5.0
    output = allocation.merge(
        adjustments[["Season", "team_id", "player_constant"]],
        on=["Season", "team_id"],
        how="left",
        validate="many_to_one",
    )
    output["player_constant"] = output["player_constant"].fillna(0.0)
    output["defense"] = output["defense"] + output["player_constant"]
    output["net"] = output["offense"] + output["defense"]
    output = output[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    output["metric"] = "bpm_team_reconciled_spm"
    output["metric_label"] = "SPM + team reconciliation"
    output["category"] = "team-informed five-year SPM"
    return output, adjustments


def _metric_frame(frame: pd.DataFrame, metric: str, label: str, category: str) -> pd.DataFrame:
    output = frame.copy()
    output["metric"] = metric
    output["metric_label"] = label
    output["category"] = category
    return output


def _player_metrics(ratings: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season in RATING_SEASONS:
        season_target = targets.loc[targets["Window_End"].eq(season)].copy()
        metric_frames = {
            metric: group.loc[group["Season"].eq(season)].copy()
            for metric, group in ratings.groupby("metric", sort=False)
        }
        common = set(season_target["PLAYER_ID"].astype(int))
        for frame in metric_frames.values():
            common &= set(frame["PLAYER_ID"].astype(int))
        target = season_target.loc[season_target["PLAYER_ID"].isin(common)].copy()
        target["sample_weight"] = np.sqrt(
            np.minimum(target["Poss_Off"], target["Poss_Def"]).clip(lower=1)
        )
        for metric, frame in metric_frames.items():
            matched = target.merge(
                frame.loc[frame["PLAYER_ID"].isin(common), [
                    "PLAYER_ID", "offense", "defense", "net", "metric_label"
                ]],
                on="PLAYER_ID",
                validate="one_to_one",
            )
            for side in ("offense", "defense", "net"):
                rows.append(
                    {
                        "rating_season": season,
                        "metric": metric,
                        "metric_label": matched["metric_label"].iloc[0],
                        "component": side,
                        "rows": len(matched),
                        **_metrics(
                            matched[f"target_{side}"].to_numpy(dtype=float),
                            matched[side].to_numpy(dtype=float),
                            matched["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _default_team_predictions(
    ratings: pd.DataFrame,
    player_minutes: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    replacement_value: float = -2.0,
    minimum_metric_minutes: float = 250.0,
) -> pd.DataFrame:
    """Return the team rows behind the default next-season benchmark."""
    minutes = player_minutes.groupby(
        ["PLAYER_ID", "Season", "team_id"], as_index=False
    )["minutes"].sum()
    metric_minutes = minutes.groupby(["PLAYER_ID", "Season"], as_index=False)[
        "minutes"
    ].sum().rename(columns={"minutes": "metric_year_minutes"})
    outcomes = team_games.groupby(["Season", "team_id"], as_index=False).agg(
        games=("won", "size"), wins=("won", "sum")
    )
    outcomes["win_pct"] = outcomes["wins"] / outcomes["games"]
    rows: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        next_minutes = minutes.loc[minutes["Season"].eq(season + 1)].copy()
        next_outcomes = outcomes.loc[outcomes["Season"].eq(season + 1)].copy()
        for metric, frame in ratings.loc[ratings["Season"].eq(season)].groupby(
            "metric", sort=False
        ):
            allocation = next_minutes.merge(
                frame[["PLAYER_ID", "net", "metric_label"]],
                on="PLAYER_ID",
                how="left",
                validate="many_to_one",
            ).merge(
                metric_minutes.loc[metric_minutes["Season"].eq(season), [
                    "PLAYER_ID", "metric_year_minutes"
                ]],
                on="PLAYER_ID",
                how="left",
                validate="many_to_one",
            )
            allocation["qualified"] = allocation["net"].notna() & allocation[
                "metric_year_minutes"
            ].ge(minimum_metric_minutes)
            allocation["adjusted_rating"] = allocation["net"].where(
                allocation["qualified"], replacement_value
            )
            teams = allocation.groupby("team_id", as_index=False).apply(
                lambda group: pd.Series(
                    {
                        "team_rating": 5.0
                        * np.average(group["adjusted_rating"], weights=group["minutes"])
                    }
                ),
                include_groups=False,
            ).reset_index(drop=True)
            teams = teams.merge(
                next_outcomes[["team_id", "win_pct"]],
                on="team_id",
                how="inner",
                validate="one_to_one",
            )
            teams["rating_season"] = int(season)
            teams["outcome_season"] = int(season + 1)
            teams["metric"] = metric
            teams["metric_label"] = frame["metric_label"].iloc[0]
            rows.append(teams)
    return pd.concat(rows, ignore_index=True)


def _paired_team_bootstrap(
    team_predictions: pd.DataFrame,
    *,
    baseline_metric: str = "five_year_spm",
    draws: int = 10_000,
    seed: int = 20260826,
) -> pd.DataFrame:
    """Paired team bootstrap for equal-season mean R-squared differences."""
    rng = np.random.default_rng(seed)
    wide = team_predictions.pivot(
        index=["rating_season", "team_id", "win_pct"],
        columns="metric",
        values="team_rating",
    ).reset_index()
    metrics = [column for column in wide.columns if column not in {
        "rating_season", "team_id", "win_pct"
    }]
    if baseline_metric not in metrics:
        raise ValueError("Bootstrap baseline is absent from team predictions.")
    metric_labels = team_predictions[["metric", "metric_label"]].drop_duplicates()
    rows: list[dict] = []
    for candidate in metrics:
        if candidate == baseline_metric:
            continue
        draws_delta = np.empty(draws, dtype=float)
        observed_deltas = []
        season_frames = []
        for season in RATING_SEASONS:
            frame = wide.loc[wide["rating_season"].eq(season), [
                "win_pct", baseline_metric, candidate
            ]].dropna()
            if len(frame) < 20:
                raise ValueError(f"Bootstrap {candidate} season {season} has too few teams.")
            season_frames.append(frame)
            observed_deltas.append(
                frame[candidate].corr(frame["win_pct"]) ** 2
                - frame[baseline_metric].corr(frame["win_pct"]) ** 2
            )
        for draw in range(draws):
            fold_deltas = []
            for frame in season_frames:
                sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
                fold_deltas.append(
                    sampled[candidate].corr(sampled["win_pct"]) ** 2
                    - sampled[baseline_metric].corr(sampled["win_pct"]) ** 2
                )
            draws_delta[draw] = float(np.nanmean(fold_deltas))
        label = metric_labels.loc[metric_labels["metric"].eq(candidate), "metric_label"].iloc[0]
        rows.append(
            {
                "metric": candidate,
                "metric_label": label,
                "baseline_metric": baseline_metric,
                "fold_wins": int(sum(delta > 0 for delta in observed_deltas)),
                "observed_mean_r_squared_delta": float(np.mean(observed_deltas)),
                "bootstrap_95_low": float(np.nanquantile(draws_delta, 0.025)),
                "bootstrap_95_high": float(np.nanquantile(draws_delta, 0.975)),
                "probability_delta_above_zero": float(np.nanmean(draws_delta > 0)),
                "bootstrap_draws": int(draws),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "observed_mean_r_squared_delta", ascending=False
    )


def _onoff_validation(annual: pd.DataFrame, raptor_path: Path) -> dict:
    reference = pd.read_parquet(raptor_path)[
        ["PLAYER_ID", "season", "raw_onoff_net", "possessions_offense", "possessions_defense"]
    ]
    matched = annual.rename(columns={"Season": "season"}).merge(
        reference, on=["PLAYER_ID", "season"], how="inner", validate="one_to_one"
    )
    matched["reference_possessions"] = matched[
        ["possessions_offense", "possessions_defense"]
    ].min(axis=1)
    matched = matched.loc[matched["reference_possessions"].ge(1000)].dropna(
        subset=["raw_onoff_net_x", "raw_onoff_net_y"]
    )
    return {
        "rows_at_1000_possessions": int(len(matched)),
        "pearson": float(matched["raw_onoff_net_x"].corr(matched["raw_onoff_net_y"])),
        "spearman": float(
            matched["raw_onoff_net_x"].corr(matched["raw_onoff_net_y"], method="spearman")
        ),
        "mean_absolute_difference": float(
            (matched["raw_onoff_net_x"] - matched["raw_onoff_net_y"]).abs().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player-sheet-dir",
        type=Path,
        default=Path("/Users/eadebayo/Documents/Projects/Sports Analytics/Basketball/New SPM/data/raw/playersheets/year_totals"),
    )
    parser.add_argument(
        "--legacy-features",
        type=Path,
        default=Path("/Users/eadebayo/Documents/Projects/Sports Analytics/Basketball/New SPM/data/processed/merged_per100_with_rTS_AuPM.csv"),
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=ROOT / "artifacts/models/five_year_spm_feature_research/five_year_spm_feature_research_v1_93c148510e/spm_predictions.parquet",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/research/spm_target_horizon_full/spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet",
    )
    parser.add_argument(
        "--raptor-onoff",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/raptor_onoff_proxy/raptor_onoff_proxy_v1_bb23b07cc8/matches.parquet",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=ROOT / "data/lake/bronze/official_game_schedule_1997_2026",
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    source_paths = {
        "base_predictions": args.base_predictions,
        "features": args.features,
        "legacy_features": args.legacy_features,
        "raptor_onoff": args.raptor_onoff,
        "targets": args.targets,
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    team_context, team_games = _schedule_frame(args.schedule_root, range(2014, 2026))
    legacy = _legacy_fields(args.legacy_features)
    annual, player_minutes = _annual_player_context(
        args.player_sheet_dir, legacy, team_context, range(2014, 2026)
    )
    context = _rolling_context(annual, range(2018, 2025)).merge(
        _raptor_context(args.raptor_onoff, range(2018, 2025)),
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    features = _extend_features(args.features, args.player_sheet_dir, 2024)
    targets = pd.read_parquet(args.targets).loc[
        lambda frame: frame["Window_End"].between(2018, 2024)
    ].copy()
    base = _base_ratings(args.base_predictions).loc[
        lambda frame: frame["Season"].between(2021, 2024)
    ].copy()

    base_ratings = _metric_frame(
        base.loc[base["Season"].isin(RATING_SEASONS)],
        "five_year_spm",
        "Five-year SPM",
        "stat-only five-year SPM",
    )
    feature_sets = {
        "demographics": {
            "label": "SPM + age/minutes/position",
            "offense": ("age_end", "log_minutes_5y", "position_guard", "position_forward", "position_center"),
            "defense": ("age_end", "log_minutes_5y", "position_guard", "position_forward", "position_center"),
        },
        "raw_onoff": {
            "label": "SPM + raw on/off",
            "offense": ("raw_onoff_net_5y",),
            "defense": ("raw_onoff_net_5y",),
        },
        "aupm": {
            "label": "SPM + legacy AuPM",
            "offense": ("aupm_5y",),
            "defense": ("aupm_5y",),
        },
        "raptor_onoff": {
            "label": "SPM + RAPTOR on/off",
            "offense": ("raptor_onoff_offense_5y", "raptor_onoff_net_5y", "raptor_season_coverage"),
            "defense": ("raptor_onoff_defense_5y", "raptor_onoff_net_5y", "raptor_season_coverage"),
        },
        "all_full_coverage": {
            "label": "SPM + all full-coverage cheats",
            "offense": (
                "age_end", "log_minutes_5y", "position_guard", "position_forward", "position_center",
                "raw_onoff_net_5y", "aupm_5y",
            ),
            "defense": (
                "age_end", "log_minutes_5y", "position_guard", "position_forward", "position_center",
                "raw_onoff_net_5y", "aupm_5y",
            ),
        },
    }
    ratings = [base_ratings, _box_ratings(features, targets)]
    for metric, spec in feature_sets.items():
        ratings.append(
            _residual_variant(
                base,
                targets,
                context,
                {"offense": spec["offense"], "defense": spec["defense"]},
                metric=f"five_year_spm_{metric}",
                metric_label=spec["label"],
            )
        )
    reconciled, team_adjustments = apply_team_reconciliation(
        base.loc[base["Season"].isin(RATING_SEASONS)],
        player_minutes.loc[player_minutes["Season"].isin(RATING_SEASONS)],
        team_context.loc[team_context["Season"].isin(RATING_SEASONS)],
    )
    ratings.append(reconciled)
    rating_panel = pd.concat(ratings, ignore_index=True)
    if rating_panel.duplicated(["PLAYER_ID", "Season", "metric"]).any():
        raise ValueError("Rating panel has duplicate player-season-metric keys.")
    identity_error = (rating_panel["offense"] + rating_panel["defense"] - rating_panel["net"]).abs().max()
    if identity_error > 1e-10:
        raise ValueError("Offense plus defense must equal net.")

    player_metrics = _player_metrics(rating_panel, targets)
    win_folds, win_summary, coverage = build_team_win_benchmark(
        rating_panel,
        player_minutes,
        team_games,
        rating_seasons=RATING_SEASONS,
        minimum_metric_minutes=250.0,
        replacement_values=(-3.0, -2.5, -2.0, -1.5),
    )
    team_predictions = _default_team_predictions(
        rating_panel, player_minutes, team_games
    )
    team_bootstrap = _paired_team_bootstrap(team_predictions)
    player_summary = player_metrics.groupby(
        ["metric", "metric_label", "component"], as_index=False
    ).agg(
        folds=("rating_season", "nunique"),
        mean_weighted_rmse=("weighted_rmse", "mean"),
        mean_correlation=("correlation", "mean"),
        minimum_rows=("rows", "min"),
    )
    onoff_validation = _onoff_validation(annual, args.raptor_onoff)

    exact_public = pd.read_parquet(
        ROOT / "artifacts/research/public_aio_benchmark/public_aio_benchmark_v1_e411f910ea/team_win_folds.parquet"
    )
    exact_public = exact_public.loc[
        exact_public["metric"].eq("annual_spm")
        & exact_public["rating_season"].isin(RATING_SEASONS)
        & exact_public["replacement_value"].eq(-2.0)
    ]
    approximate_check_ratings = []
    for season in RATING_SEASONS:
        source = pd.read_json(ROOT / f"web/public/data/leaderboard-{season}.json")
        approximate_check_ratings.append(
            source.rename(
                columns={"spm_offense": "offense", "spm_defense": "defense", "spm_net": "net"}
            )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
        )
    approximate_check = pd.concat(approximate_check_ratings, ignore_index=True)
    approximate_check = _metric_frame(approximate_check, "annual_spm", "Annual SPM", "qa")
    approx_folds, _, _ = build_team_win_benchmark(
        approximate_check,
        player_minutes,
        team_games,
        rating_seasons=RATING_SEASONS,
        minimum_metric_minutes=250.0,
        replacement_values=(-2.0,),
    )
    benchmark_drift = float(approx_folds["r_squared"].mean() - exact_public["r_squared"].mean())

    config = {
        "rating_seasons": list(RATING_SEASONS),
        "outcome_seasons": [season + 1 for season in RATING_SEASONS],
        "primary_metric": "equal-season mean next-season team win R-squared",
        "secondary_metric": "five-year RAPM weighted RMSE and correlation",
        "residual_alpha": RESIDUAL_ALPHA,
        "box_alpha_grid": list(ALPHA_GRID),
        "minimum_rating_season_minutes": 250.0,
        "replacement_values": [-3.0, -2.5, -2.0, -1.5],
        "team_minutes_policy": "primary-team year totals; identical for every arm",
        "feature_timing": "five trailing seasons ending in rating season; demographics use rating-season endpoint",
        "source_hashes": {
            **{name: sha256_file(path) for name, path in source_paths.items()},
            "player_sheets": {
                str(season): sha256_file(args.player_sheet_dir / f"{season}.csv")
                for season in range(2014, 2026)
            },
            "schedules": {
                str(season): sha256_file(
                    args.schedule_root / f"leaguegamelog_{season}.json.gz"
                )
                for season in range(2014, 2026)
            },
            "exact_stint_benchmark": sha256_file(
                ROOT / "artifacts/research/public_aio_benchmark/public_aio_benchmark_v1_e411f910ea/team_win_folds.parquet"
            ),
            "website_annual_spm": {
                str(season): sha256_file(
                    ROOT / f"web/public/data/leaderboard-{season}.json"
                )
                for season in RATING_SEASONS
            },
        },
        "runner_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research"
        / "spm_cheating_ladder"
        / f"spm_cheating_ladder_v1_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    rating_panel.to_parquet(output / "ratings.parquet", index=False)
    player_metrics.to_parquet(output / "player_metrics.parquet", index=False)
    player_summary.to_parquet(output / "player_summary.parquet", index=False)
    win_folds.to_parquet(output / "team_win_folds.parquet", index=False)
    win_summary.to_parquet(output / "team_win_summary.parquet", index=False)
    coverage.to_parquet(output / "coverage.parquet", index=False)
    team_adjustments.to_parquet(output / "team_adjustments.parquet", index=False)
    team_predictions.to_parquet(output / "team_predictions.parquet", index=False)
    team_bootstrap.to_parquet(output / "team_bootstrap.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": "spm_cheating_ladder_v1",
        "status": "research_pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rating_rows": int(len(rating_panel)),
            "duplicate_rating_keys": 0,
            "component_identity_max_error": float(identity_error),
            "derived_onoff_validation": onoff_validation,
            "primary_team_minutes_r2_drift_vs_exact_stints": benchmark_drift,
            "raptor_window_coverage": {
                str(int(season)): float(
                    context.loc[context["Window_End"].eq(season), "raptor_season_coverage"].mean()
                )
                for season in RATING_SEASONS
            },
            "season_2027_rows": int(rating_panel["Season"].eq(2027).sum()),
        },
        "paths": {
            "ratings": "ratings.parquet",
            "player_metrics": "player_metrics.parquet",
            "player_summary": "player_summary.parquet",
            "team_win_folds": "team_win_folds.parquet",
            "team_win_summary": "team_win_summary.parquet",
            "coverage": "coverage.parquet",
            "team_adjustments": "team_adjustments.parquet",
            "team_predictions": "team_predictions.parquet",
            "team_bootstrap": "team_bootstrap.parquet",
        },
        "caveats": [
            "This deliberately breaks the stat-only SPM boundary and cannot replace the public SPM without a separate promotion test.",
            "Observed next-season minutes make the team-win test an oracle-minutes retrodiction, not a preseason forecast.",
            "The retained year totals assign a traded player to one primary team; the run reports the resulting R-squared drift against the exact-stint benchmark.",
            "RAPTOR on/off ends in 2022, so later rolling windows have declining source-season coverage.",
            "The team reconciliation matches observed rating-season team net efficiency and therefore imports team context by construction.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    headline = win_summary.loc[win_summary["replacement_value"].eq(-2.0)].sort_values(
        "mean_r_squared", ascending=False
    )[["metric_label", "mean_r_squared", "mean_pearson", "mean_spearman"]]
    print(json.dumps({"run_id": run["run_id"], "output": str(output), "quality": run["quality"]}, indent=2))
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
