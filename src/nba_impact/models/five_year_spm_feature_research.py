"""Leak-safe feature-family research for the five-year SPM.

Every candidate starts as a player-season statistic stabilized only against
that same season.  The five-year model then possession-weights those frozen
annual rows over its explicit five-season window.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.five_year_target_spm import (
    _feature_lists,
    _fit_spm,
    _load_contract,
    _matched_five_year_inputs,
    _rating_table,
    _target_panels,
)
from nba_impact.models.predictive_spm import _predictive_metrics
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_metrics,
    build_design,
    fit_coefficient_center_path,
    load_current_player_names,
    load_unified_terminal_possessions,
)


EXPERIMENT_ID = "five_year_spm_feature_research_v1"
HISTORICAL_HUSTLE_FIELDS = (
    "deflections_p100",
    "charges_drawn_p100",
    "contested_2pt_p100",
    "contested_3pt_p100",
    "contested_3pt_share",
    "def_loose_balls_recovered_p100",
)
CANDIDATE_GROUPS: dict[str, dict[str, object]] = {
    "bbi_shooting": {
        "side": "offense",
        "features": (
            "shot_difficulty_expected_points_per_attempt_relative",
            "shot_making_points_above_expected_p100_eb",
            "tight_shot_attempt_share_eb",
        ),
        "source": "Basketball Index inspired shot context",
    },
    "bbi_passing": {
        "side": "offense",
        "features": (
            "pass_creation_points_per_potential_assist_eb",
            "high_value_assist_share_eb",
            "bad_pass_turnovers_per_100_passes_eb",
        ),
        "source": "Basketball Index inspired passing context",
    },
    "bbi_screening": {
        "side": "offense",
        "features": ("screen_assist_points_p100_eb",),
        "source": "Basketball Index inspired screening value",
    },
    "raptor_playtype": {
        "side": "offense",
        "features": (
            "playtype_difficulty_pct_points",
            "playtype_poe_per_75",
            "transition_share",
            "transition_poe_per_75",
        ),
        "source": "RAPTOR style contextual offense",
    },
    "bbi_defense_hustle": {
        "side": "defense",
        "features": (
            "deflections_p100_eb",
            "charges_drawn_p100_eb",
            "defensive_boxouts_p100_eb",
            "loose_balls_recovered_p100_eb",
        ),
        "source": "Basketball Index inspired defensive activity",
    },
    "raptor_shot_defense": {
        "side": "defense",
        "features": (
            "dfg_two_point_equivalent_saved_p100",
            "rim_matchup_attempt_share",
            "contested_3pt_share",
        ),
        "source": "RAPTOR style defended 2P value and 3PA contest volume",
    },
    "raptor_matchup_volume": {
        "side": "defense",
        "features": ("matchup_3pa_share",),
        "source": "RAPTOR style matchup shot-volume context",
    },
    "noisy_opponent_shooting": {
        "side": "defense",
        "features": ("matchup_efg_pct_allowed_eb",),
        "source": "Opponent shooting outcome falsification test",
    },
}


def enforce_same_season_stabilization(features: pd.DataFrame) -> pd.DataFrame:
    """Remove known cross-season neutral fills from the historical artifact."""
    output = features.copy()
    if int(output["Window_End"].max()) > 2026:
        raise ValueError("Season 2027 must not enter feature research.")
    for field in HISTORICAL_HUSTLE_FIELDS:
        if field in output:
            # Hustle tracking begins in 2018. Older constant values came from a
            # pooled median fallback and are not legitimate season estimates.
            output.loc[output["Window_End"].lt(2018), field] = 0.0
    return output


def pool_annual_candidates(
    annual: pd.DataFrame,
    *,
    window_ends: tuple[int, ...] = tuple(range(2018, 2027)),
) -> pd.DataFrame:
    """Possession-weight already-stabilized annual values over five seasons."""
    fields = tuple(
        dict.fromkeys(
            feature
            for group in CANDIDATE_GROUPS.values()
            for feature in group["features"]
        )
    )
    if missing := sorted(set(fields) - set(annual.columns)):
        raise ValueError(f"Annual feature artifact is missing candidates {missing}.")
    side_by_feature = {
        feature: str(group["side"])
        for group in CANDIDATE_GROUPS.values()
        for feature in group["features"]
    }
    outputs = []
    for end in window_ends:
        window = annual.loc[annual["Window_End"].between(end - 4, end)]
        output = pd.DataFrame(
            {"PLAYER_ID": sorted(window["PLAYER_ID"].astype(int).unique()), "Window_End": end}
        )
        for feature in fields:
            weight_field = "OffPoss" if side_by_feature[feature] == "offense" else "DefPoss"
            values = pd.to_numeric(window[feature], errors="coerce")
            weights = pd.to_numeric(window[weight_field], errors="coerce").clip(lower=0)
            valid = values.notna() & weights.gt(0)
            numerator = (values.where(valid, 0.0) * weights.where(valid, 0.0)).groupby(
                window["PLAYER_ID"]
            ).sum()
            denominator = weights.where(valid, 0.0).groupby(window["PLAYER_ID"]).sum()
            pooled = numerator / denominator.replace(0.0, np.nan)
            output[feature] = output["PLAYER_ID"].map(pooled.to_dict())
        outputs.append(output)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Pooled candidate keys are not unique.")
    return result


def _primary_team_map(paths: tuple[str | Path, ...]) -> dict[tuple[int, int], str]:
    frames = [pd.read_parquet(path) for path in paths]
    games = pd.concat(frames, ignore_index=True)
    games = games.loc[
        games["season_type"].eq("regular") & games["played"].fillna(False)
    ].copy()
    games["minutes_seconds"] = pd.to_numeric(
        games["minutes_seconds"], errors="coerce"
    ).fillna(0.0)
    primary = (
        games.groupby(["season_end", "player_id", "team_tricode"], as_index=False)[
            "minutes_seconds"
        ]
        .sum()
        .sort_values(
            ["season_end", "player_id", "minutes_seconds"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["season_end", "player_id"])
    )
    return {
        (int(row.season_end), int(row.player_id)): str(row.team_tricode)
        for row in primary.itertuples()
    }


def _metric_rows(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    *,
    variant: str,
    rating_season: int,
    switchers: set[int],
) -> list[dict]:
    merged = prediction.merge(target, on="PLAYER_ID", validate="one_to_one")
    rows = []
    for scope, frame in (
        ("all", merged),
        ("team_changers", merged.loc[merged["PLAYER_ID"].isin(switchers)]),
    ):
        weights = np.sqrt(np.minimum(frame["Poss_Off"], frame["Poss_Def"]).clip(lower=1))
        for side in ("offense", "defense", "net"):
            rows.append(
                {
                    "variant": variant,
                    "rating_season": rating_season,
                    "test_season": rating_season + 1,
                    "scope": scope,
                    "component": side,
                    "rows": len(frame),
                    **_predictive_metrics(
                        frame[f"target_{side}"].to_numpy(dtype=float),
                        frame[f"prior_{side}_per_100"].to_numpy(dtype=float),
                        weights.to_numpy(dtype=float),
                    ),
                }
            )
    return rows


def _candidate_decisions(metrics: pd.DataFrame) -> pd.DataFrame:
    development = metrics.loc[metrics["test_season"].isin((2022, 2023, 2024))]
    records = []
    for group, spec in CANDIDATE_GROUPS.items():
        side = str(spec["side"])
        candidate = development.loc[
            development["variant"].eq(group) & development["component"].eq(side)
        ]
        baseline = development.loc[
            development["variant"].eq("baseline") & development["component"].eq(side)
        ]
        wide = candidate.merge(
            baseline,
            on=["rating_season", "test_season", "scope", "component"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        wide["rmse_delta"] = wide["weighted_rmse_candidate"] - wide["weighted_rmse_baseline"]
        wide["correlation_delta"] = (
            wide["weighted_correlation_candidate"] - wide["weighted_correlation_baseline"]
        )
        all_rows = wide.loc[wide["scope"].eq("all")]
        changer_rows = wide.loc[wide["scope"].eq("team_changers")]
        mean_all = float(all_rows["rmse_delta"].mean())
        mean_changer = float(changer_rows["rmse_delta"].mean())
        corr_changer = float(changer_rows["correlation_delta"].mean())
        fold_wins = int(all_rows["rmse_delta"].lt(0).sum())
        selected = (
            mean_all < 0
            and fold_wins >= 2
            and mean_changer <= 0.01
            and corr_changer >= -0.01
            and group != "noisy_opponent_shooting"
        )
        records.append(
            {
                "group": group,
                "side": side,
                "feature_count": len(spec["features"]),
                "features": json.dumps(list(spec["features"])),
                "source": str(spec["source"]),
                "development_mean_rmse_delta": mean_all,
                "development_fold_wins": fold_wins,
                "team_changer_mean_rmse_delta": mean_changer,
                "team_changer_mean_correlation_delta": corr_changer,
                "selected": selected,
                "decision": "add" if selected else "reject",
            }
        )
    return pd.DataFrame(records)


def build_five_year_spm_feature_research(
    *,
    annual_features_path: str | Path,
    annual_targets_path: str | Path,
    five_year_reference_features_path: str | Path,
    five_year_reference_targets_path: str | Path,
    five_year_rolling_targets_path: str | Path,
    player_sheet_dir: str | Path,
    reference_manifest_path: str | Path,
    baseline_contract_path: str | Path,
    legacy_cache_dir: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
    current_player_games_path: str | Path,
    historical_player_games_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    inputs = {
        "annual_features": Path(annual_features_path),
        "annual_targets": Path(annual_targets_path),
        "five_year_reference_features": Path(five_year_reference_features_path),
        "five_year_reference_targets": Path(five_year_reference_targets_path),
        "five_year_rolling_targets": Path(five_year_rolling_targets_path),
        "reference_manifest": Path(reference_manifest_path),
        "baseline_contract": Path(baseline_contract_path),
        "current_possessions": Path(current_possessions_path),
        "current_segments": Path(current_segments_path),
        "current_player_games": Path(current_player_games_path),
        "historical_player_games": Path(historical_player_games_path),
    }
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    identity = hashlib.sha256(
        json.dumps(
            {"inputs": hashes, "code": sha256_file(Path(__file__))}, sort_keys=True
        ).encode()
    ).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{identity}"
    output = Path(artifact_root) / "models" / "five_year_spm_feature_research" / run_id
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    contract = _load_contract(baseline_contract_path)
    baseline_features = _feature_lists(reference_manifest_path, contract)
    annual = enforce_same_season_stabilization(pd.read_parquet(annual_features_path))
    annual_targets = pd.read_parquet(annual_targets_path)
    five, five_targets, overlap_error = _matched_five_year_inputs(
        reference_features_path=five_year_reference_features_path,
        reference_targets_path=five_year_reference_targets_path,
        rolling_targets_path=five_year_rolling_targets_path,
        player_sheet_dir=player_sheet_dir,
        annual_features=annual,
        selected=baseline_features,
    )
    pooled = pool_annual_candidates(annual)
    five = five.merge(pooled, on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one")
    _, panel = _target_panels(annual, annual_targets, five, five_targets)
    team_map = _primary_team_map(
        (historical_player_games_path, current_player_games_path)
    )

    variants = {"baseline": baseline_features}
    for group, spec in CANDIDATE_GROUPS.items():
        side = str(spec["side"])
        feature_map = {key: tuple(value) for key, value in baseline_features.items()}
        feature_map[side] = tuple(dict.fromkeys((*feature_map[side], *spec["features"])))
        variants[group] = feature_map

    metrics = []
    predictions = []
    for rating_season in (2021, 2022, 2023):
        target = annual_targets.loc[
            annual_targets["Season"].eq(rating_season + 1),
            ["PLAYER_ID", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"],
        ]
        switchers = {
            int(player_id)
            for player_id in target["PLAYER_ID"]
            if team_map.get((rating_season, int(player_id)))
            and team_map.get((rating_season + 1, int(player_id)))
            and team_map[(rating_season, int(player_id))]
            != team_map[(rating_season + 1, int(player_id))]
        }
        for variant, fields in variants.items():
            prediction, _ = _fit_spm(
                panel, five, fields, rating_season=rating_season, target_name="five_year"
            )
            prediction["variant"] = variant
            predictions.append(prediction)
            metrics.extend(
                _metric_rows(
                    prediction,
                    target,
                    variant=variant,
                    rating_season=rating_season,
                    switchers=switchers,
                )
            )
    metric_frame = pd.DataFrame(metrics)
    decisions = _candidate_decisions(metric_frame)
    selected_features = {key: tuple(value) for key, value in baseline_features.items()}
    for row in decisions.loc[decisions["selected"]].itertuples():
        spec = CANDIDATE_GROUPS[row.group]
        side = str(spec["side"])
        selected_features[side] = tuple(
            dict.fromkeys((*selected_features[side], *spec["features"]))
        )

    # Refit the combined selected specification forward through 2026.
    baseline_full_predictions = []
    selected_predictions = []
    final_models = None
    for rating_season in range(2021, 2027):
        baseline_prediction, _ = _fit_spm(
            panel,
            five,
            baseline_features,
            rating_season=rating_season,
            target_name="five_year",
        )
        baseline_prediction["variant"] = "baseline"
        baseline_full_predictions.append(baseline_prediction)
        prediction, models = _fit_spm(
            panel,
            five,
            selected_features,
            rating_season=rating_season,
            target_name="five_year",
        )
        prediction["variant"] = "selected_combined"
        selected_predictions.append(prediction)
        final_models = models

    frame = load_unified_terminal_possessions(
        legacy_cache_dir,
        current_possessions_path,
        current_segments_path,
        tuple(range(2021, 2027)),
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    names = load_current_player_names(
        Path(legacy_cache_dir).parent / "all_names.csv", current_player_games_path
    )
    config = RapmConfig(
        seasons=tuple(range(2021, 2027)),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        include_home=True,
        game_types=("regular",),
        data_scope="five_year_spm_selected_feature_aio",
    )
    aio_metrics = []
    aio_ratings = []
    selected_prediction_frame = pd.concat(selected_predictions, ignore_index=True)
    baseline_prediction_frame = pd.concat(baseline_full_predictions, ignore_index=True)
    for rating_season in range(2021, 2027):
        train_mask = design.seasons == rating_season
        test_mask = design.seasons == rating_season + 1 if rating_season < 2026 else train_mask
        for variant, source in (
            ("baseline", baseline_prediction_frame),
            ("selected_combined", selected_prediction_frame),
        ):
            prior = source.loc[source["Window_End"].eq(rating_season)]
            center, _ = build_prior_center(
                design,
                prior,
                prior_window_end=rating_season,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            beta, intercept = fit_coefficient_center_path(
                design, config, center, center_scales=(1.0,), row_mask=train_mask
            )[1.0]
            table = _rating_table(design, beta, train_mask, names)
            table["rating_season"] = rating_season
            table["variant"] = variant
            aio_ratings.append(table)
            if rating_season < 2026:
                game = _game_margin_metrics(design, beta, intercept, test_mask, train_mask)
                aio_metrics.append(
                    {
                        "variant": variant,
                        "rating_season": rating_season,
                        "test_season": rating_season + 1,
                        **game,
                    }
                )

    model_dir = output / "models"
    model_dir.mkdir()
    assert final_models is not None
    for side, model in final_models.items():
        joblib.dump(model, model_dir / f"{side}.joblib")
    candidate_predictions = [
        prediction
        for prediction in predictions
        if not prediction["variant"].eq("baseline").all()
    ]
    pd.concat(
        [*candidate_predictions, *baseline_full_predictions, *selected_predictions],
        ignore_index=True,
    ).to_parquet(
        output / "spm_predictions.parquet", index=False
    )
    metric_frame.to_parquet(output / "feature_group_metrics.parquet", index=False)
    decisions.to_parquet(output / "feature_group_decisions.parquet", index=False)
    pd.concat(aio_ratings, ignore_index=True).to_parquet(output / "aio_ratings.parquet", index=False)
    pd.DataFrame(aio_metrics).to_parquet(output / "aio_metrics.parquet", index=False)

    run = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_challenger",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stabilization_contract": {
            "center": "same season league distribution only",
            "opportunities": "same player-season only",
            "five_year_aggregation": "possession weighted frozen annual estimates",
            "prior_season_stabilization": False,
            "future_season_stabilization": False,
            "unavailable_source_season": "explicit neutral zero",
        },
        "selection_gate": {
            "development_test_seasons": [2022, 2023, 2024],
            "overall_mean_rmse_delta": "less_than_zero",
            "minimum_fold_wins": 2,
            "team_changer_rmse_regression_limit": 0.01,
            "team_changer_correlation_regression_limit": -0.01,
            "opponent_shooting_outcome_falsification_can_promote": False,
            "reused_diagnostic_test_seasons": [2025, 2026],
            "untouched_confirmation_season": 2027,
        },
        "candidate_groups": CANDIDATE_GROUPS,
        "decisions": decisions.to_dict(orient="records"),
        "selected_features": {key: list(value) for key, value in selected_features.items()},
        "quality": {
            "maximum_reference_target_overlap_error": overlap_error,
            "feature_keys_unique": True,
            "season_2027_rows": 0,
        },
        "source_hashes": hashes,
        "paths": {
            "spm_predictions": "spm_predictions.parquet",
            "feature_group_metrics": "feature_group_metrics.parquet",
            "feature_group_decisions": "feature_group_decisions.parquet",
            "aio_ratings": "aio_ratings.parquet",
            "aio_metrics": "aio_metrics.parquet",
            "models": "models",
        },
        "forbidden_interpretation": "Feature-family selection is predictive evidence, not causal attribution or untouched confirmation.",
    }
    write_json_atomic(run, output / "run.json")
    return run
