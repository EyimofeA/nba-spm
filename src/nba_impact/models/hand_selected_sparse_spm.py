"""Hand-selected twelve-feature five-year SPM research challenger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.defensive_tracking_features import (
    _load_box as _load_defensive_box,
    compute_defensive_tracking_features,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.playtype_features import _load_playtypes, compute_playtype_features
from nba_impact.data.statistical_features import _load_source
from nba_impact.models.sparse_function_spm import (
    PLAYER_TEST_SEASONS,
    RATING_SEASONS,
    RIDGE_ALPHA,
    TEAM_WIN_RATING_SEASONS,
    build_five_year_features,
    evaluate_next_season_players,
    evaluate_team_wins,
    feature_registry,
    fit_historical_predictions,
    selected_features,
    standardize_within_window,
    summarize_player_metrics,
)


EXPERIMENT_ID = "hand_selected_sparse_spm_v1"
CANDIDATE_NAME = "hand_selected_sparse_spm"

FEATURE_SPECS = {
    "offense": (
        ("scoring", "PTS_p100", "Points per 100 offensive possessions."),
        (
            "efficiency",
            "zts_pct_points",
            "True shooting percentage minus expected true shooting from playtype mix.",
        ),
        (
            "turnovers",
            "turnover_to_load_2017_eb",
            "Empirical-Bayes turnovers divided by offensive load.",
        ),
        (
            "passing",
            "box_creation_2017_eb_p100",
            "Box Creation estimate of shots created for teammates per 100.",
        ),
        (
            "offensive_rebounding",
            "OREB_p100",
            "Offensive rebounds per 100 offensive possessions.",
        ),
        (
            "spacing",
            "crafted_spacing_stable_v1",
            "Stabilized three-point volume and accuracy relative to league shot value.",
        ),
        (
            "offensive_load",
            "offensive_load_2017_eb_p100",
            "Empirical-Bayes estimate of offensive actions directly used per 100.",
        ),
        (
            "rim_pressure",
            "at_rim_fga_p100",
            "At-rim field-goal attempts per 100 offensive possessions.",
        ),
    ),
    "defense": (
        (
            "event_stops",
            "event_stops_p100",
            "Steals, recovered blocks, charges and offensive fouls drawn per 100 defensive possessions.",
        ),
        (
            "rim_protection",
            "rim_points_saved_p100",
            "Empirical-Bayes rim points saved versus location expectation per 100 defensive possessions.",
        ),
        (
            "contested_rebounding",
            "dreb_contests_p100",
            "Defensive rebound contests per 100 defensive possessions.",
        ),
        (
            "foul_discipline",
            "shooting_fouls_committed_p100",
            "Shooting fouls committed per 100 defensive possessions.",
        ),
    ),
}


def hand_selected_features() -> dict[str, tuple[str, ...]]:
    return selected_features(FEATURE_SPECS)


def _annual_box(
    player_sheet_dir: str | Path, seasons: tuple[int, ...]
) -> pd.DataFrame:
    rows = []
    for season in seasons:
        frame = _load_source(Path(player_sheet_dir) / f"{season}.parquet", season)[0]
        minutes = "Minutes" if "Minutes" in frame else "MIN"
        required = {"PLAYER_ID", "PTS", "FGA", "FTA", minutes}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Player sheet {season} is missing {missing}.")
        keep = frame[["PLAYER_ID", "PTS", "FGA", "FTA", minutes]].copy()
        keep = keep.rename(columns={minutes: "Minutes"})
        keep["Season"] = season
        for column in ("PLAYER_ID", "PTS", "FGA", "FTA", "Minutes"):
            keep[column] = pd.to_numeric(keep[column], errors="coerce")
        keep = keep.dropna(subset=["PLAYER_ID", "PTS", "FGA", "FTA"])
        keep["PLAYER_ID"] = keep["PLAYER_ID"].astype(int)
        rows.append(keep)
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual box rows are not unique.")
    return output


def build_annual_auxiliary_features(
    player_sheet_dir: str | Path,
    *,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2027)),
) -> tuple[pd.DataFrame, dict]:
    """Build annual zTS and rim-points-saved features from pinned raw sources."""
    box = _annual_box(player_sheet_dir, seasons)
    playtypes = _load_playtypes(playtype_path, seasons)
    playtype = compute_playtype_features(box, playtypes)

    overrides = {
        season: Path(player_sheet_dir) / f"{season}.parquet" for season in seasons
    }
    defensive_box, _ = _load_defensive_box(
        player_sheet_dir, seasons, source_overrides=overrides
    )
    dfg = pd.read_csv(dfg_path, low_memory=False)
    rim = pd.read_csv(rim_dfg_path, low_memory=False)
    hustle = pd.read_csv(hustle_path, low_memory=False)
    for frame in (dfg, rim, hustle):
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    defensive, defensive_quality = compute_defensive_tracking_features(
        defensive_box,
        dfg.loc[dfg["year"].isin(seasons)],
        rim.loc[rim["year"].isin(seasons)],
        hustle.loc[hustle["year"].isin(seasons)],
    )
    actual_rim_seasons = set(
        pd.to_numeric(rim["year"], errors="coerce").dropna().astype(int)
    )
    defensive["rim_source_observed"] = defensive["Season"].isin(actual_rim_seasons)
    output = box[["PLAYER_ID", "Season"]].merge(
        playtype[["PLAYER_ID", "Season", "zts_pct_points", "synergy_possessions"]],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).merge(
        defensive[
            [
                "PLAYER_ID",
                "Season",
                "rim_points_saved_p100",
                "rim_source_observed",
            ]
        ],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).merge(
        defensive_box[["PLAYER_ID", "Season", "DefPoss"]],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    )
    quality = {
        "playtype_rows": int(len(playtype)),
        "playtype_seasons": sorted(playtype["Season"].unique().astype(int).tolist()),
        "rim_source_seasons": sorted(actual_rim_seasons & set(seasons)),
        "rim_2026_observed": 2026 in actual_rim_seasons,
        "defensive_source_join_quality": defensive_quality["source_join_quality"],
    }
    return output, quality


def _pool_metric(
    annual: pd.DataFrame,
    *,
    metric: str,
    weight: str,
    window_ends: tuple[int, ...],
    observed: str | None = None,
) -> pd.DataFrame:
    outputs = []
    for window_end in window_ends:
        frame = annual.loc[annual["Season"].between(window_end - 4, window_end)].copy()
        valid = frame[metric].notna() & frame[weight].gt(0)
        if observed is not None:
            valid &= frame[observed].fillna(False)
        frame["_numerator"] = frame[metric].where(valid) * frame[weight].where(valid)
        frame["_denominator"] = frame[weight].where(valid)
        pooled = frame.groupby("PLAYER_ID", as_index=False).agg(
            _numerator=("_numerator", "sum"),
            _denominator=("_denominator", "sum"),
            source_seasons=("Season", lambda values: int(valid.loc[values.index].sum())),
        )
        pooled[metric] = pooled["_numerator"] / pooled["_denominator"].where(
            pooled["_denominator"].gt(0)
        )
        pooled["Window_End"] = window_end
        outputs.append(pooled[["PLAYER_ID", "Window_End", metric, "source_seasons"]])
    return pd.concat(outputs, ignore_index=True)


def build_hand_selected_five_year_features(
    player_sheet_dir: str | Path,
    *,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    window_ends: tuple[int, ...] = tuple(range(2018, 2027)),
) -> tuple[pd.DataFrame, dict]:
    base = build_five_year_features(player_sheet_dir, window_ends=window_ends)
    annual, auxiliary_quality = build_annual_auxiliary_features(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
    )
    zts = _pool_metric(
        annual,
        metric="zts_pct_points",
        weight="synergy_possessions",
        window_ends=window_ends,
    ).rename(columns={"source_seasons": "zts_source_seasons"})
    rim = _pool_metric(
        annual,
        metric="rim_points_saved_p100",
        weight="DefPoss",
        window_ends=window_ends,
        observed="rim_source_observed",
    ).rename(columns={"source_seasons": "rim_source_seasons"})
    features = base.merge(
        zts,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    ).merge(
        rim,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    required = set((*hand_selected_features()["offense"], *hand_selected_features()["defense"]))
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Hand-selected five-year features are missing {missing}.")
    quality = {
        **auxiliary_quality,
        "feature_rows": int(len(features)),
        "missing_fraction": {
            feature: float(features[feature].isna().mean()) for feature in sorted(required)
        },
        "minimum_zts_source_seasons": int(features["zts_source_seasons"].min()),
        "minimum_rim_source_seasons": int(features["rim_source_seasons"].min()),
    }
    return features, quality


def run_hand_selected_sparse_spm(
    *,
    player_sheet_dir: str | Path,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    five_year_targets_path: str | Path,
    full_predictions_path: str | Path,
    annual_ratings_path: str | Path,
    html_root: str | Path,
    identity_root: str | Path,
    schedule_root: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """Fit, evaluate and persist the frozen hand-selected challenger."""
    source_paths = {
        "playtype": Path(playtype_path),
        "dfg": Path(dfg_path),
        "rim_dfg": Path(rim_dfg_path),
        "hustle": Path(hustle_path),
        "five_year_targets": Path(five_year_targets_path),
        "full_predictions": Path(full_predictions_path),
        "annual_ratings": Path(annual_ratings_path),
        "contract": Path(contract_path),
        "source_code": Path(__file__),
    }
    for season in range(2014, 2027):
        source_paths[f"player_sheet_{season}"] = Path(player_sheet_dir) / f"{season}.parquet"
    for season in range(min(TEAM_WIN_RATING_SEASONS), max(TEAM_WIN_RATING_SEASONS) + 2):
        source_paths[f"bbref_totals_{season}"] = Path(html_root) / f"nba_{season}_totals.html"
        source_paths[f"team_identity_{season}"] = Path(identity_root) / f"{season}.parquet"
        source_paths[f"team_schedule_{season}"] = Path(schedule_root) / f"leaguegamelog_{season}.json.gz"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    names = hand_selected_features()
    config = {
        "alpha": alpha,
        "features": {side: list(values) for side, values in names.items()},
        "rating_seasons": list(RATING_SEASONS),
        "player_test_seasons": list(PLAYER_TEST_SEASONS),
        "team_win_rating_seasons": list(TEAM_WIN_RATING_SEASONS),
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "research" / "hand_selected_sparse_spm" / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())

    raw_features, feature_quality = build_hand_selected_five_year_features(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
    )
    features = standardize_within_window(raw_features, feature_names=names)
    predictions, coefficients, models = fit_historical_predictions(
        features,
        pd.read_parquet(five_year_targets_path),
        alpha=alpha,
        feature_names=names,
    )
    full_predictions = pd.read_parquet(full_predictions_path)
    annual_ratings = pd.read_parquet(annual_ratings_path)
    player_metrics, matched_players = evaluate_next_season_players(
        predictions,
        full_predictions,
        annual_ratings,
        candidate_name=CANDIDATE_NAME,
    )
    player_summary = summarize_player_metrics(player_metrics)
    team_folds, team_summary, team_coverage, team_source_coverage = evaluate_team_wins(
        predictions,
        full_predictions,
        html_root=html_root,
        identity_root=identity_root,
        schedule_root=schedule_root,
        candidate_name=CANDIDATE_NAME,
        candidate_label="Hand-selected sparse SPM",
    )

    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    predictions.to_parquet(output / "predictions.parquet", index=False)
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    feature_registry(FEATURE_SPECS).to_parquet(output / "feature_registry.parquet", index=False)
    player_metrics.to_parquet(output / "player_fold_metrics.parquet", index=False)
    player_summary.to_parquet(output / "player_summary.parquet", index=False)
    matched_players.to_parquet(output / "matched_player_predictions.parquet", index=False)
    team_folds.to_parquet(output / "team_win_folds.parquet", index=False)
    team_summary.to_parquet(output / "team_win_summary.parquet", index=False)
    team_coverage.to_parquet(output / "team_win_coverage.parquet", index=False)
    team_source_coverage.to_parquet(output / "team_win_source_coverage.parquet", index=False)
    for side, model in models.items():
        joblib.dump(model, output / "models" / f"{side}.joblib")

    candidate_net = player_summary.loc[
        player_summary["candidate"].eq(CANDIDATE_NAME)
        & player_summary["side"].eq("net")
    ].iloc[0]
    full_net = player_summary.loc[
        player_summary["candidate"].eq("full_five_year_spm")
        & player_summary["side"].eq("net")
    ].iloc[0]
    candidate_team = team_summary.loc[team_summary["metric"].eq(CANDIDATE_NAME)].iloc[0]
    full_team = team_summary.loc[team_summary["metric"].eq("full_five_year_spm")].iloc[0]
    status = (
        "research_challenger_retained"
        if float(candidate_team["mean_r_squared"]) >= float(full_team["mean_r_squared"])
        else "research_null"
    )
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": json.loads(Path(contract_path).read_text()),
        "config": config,
        "quality": {
            **feature_quality,
            "prediction_rows": int(len(predictions)),
            "player_test_folds": int(player_metrics["test_season"].nunique()),
            "team_win_folds": int(team_folds["rating_season"].nunique()),
            "component_identity_max_error": float(
                (
                    predictions["prediction_offense"]
                    + predictions["prediction_defense"]
                    - predictions["prediction_net"]
                ).abs().max()
            ),
            "season_2027_loaded": False,
        },
        "result": {
            "candidate_next_season_net_pearson": float(candidate_net["weighted_pearson"]),
            "full_next_season_net_pearson": float(full_net["weighted_pearson"]),
            "candidate_next_season_net_rmse": float(candidate_net["weighted_rmse"]),
            "full_next_season_net_rmse": float(full_net["weighted_rmse"]),
            "candidate_team_win_mean_r_squared": float(candidate_team["mean_r_squared"]),
            "full_team_win_mean_r_squared": float(full_team["mean_r_squared"]),
        },
        "decision": (
            "Retain as a research challenger; it matched or improved the two-fold team-win diagnostic."
            if status == "research_challenger_retained"
            else "Keep the full five-year SPM; the hand-selected model did not improve the primary team-win diagnostic."
        ),
        "caveats": [
            "The team-win benchmark uses observed next-season minutes and is not a preseason forecast.",
            "Only two team-win folds are available from the local source bundle.",
            "Rim defended-shot data end in 2025; the 2026 five-year window pools 2022-25 observed rim data.",
            "All evaluated seasons are reused historical diagnostics; Season 2027 remains untouched.",
            "The fixed ridge alpha was not tuned on these outcomes.",
        ],
        "paths": {
            "predictions": "predictions.parquet",
            "coefficients": "coefficients.parquet",
            "feature_registry": "feature_registry.parquet",
            "player_fold_metrics": "player_fold_metrics.parquet",
            "player_summary": "player_summary.parquet",
            "matched_player_predictions": "matched_player_predictions.parquet",
            "team_win_folds": "team_win_folds.parquet",
            "team_win_summary": "team_win_summary.parquet",
            "team_win_coverage": "team_win_coverage.parquet",
            "team_win_source_coverage": "team_win_source_coverage.parquet",
            "models": "models",
        },
        "forbidden_interpretation": "Public SPM promotion, causal skill value, or untouched confirmation.",
    }
    write_json_atomic(run, output / "run.json")
    return run
