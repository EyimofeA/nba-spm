#!/usr/bin/env python3
"""Compare raw and same-season-stabilized five-year SPM/AIO arms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from nba_impact.data.full_spm_features import _pool_annual_field
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features_v2 import _weighted_zscore
from nba_impact.models.rapm import load_legacy_possessions
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model
from run_aio_prior_bakeoff import _game_metrics, _prior_frame
from run_aio_prior_canonical_followup import (
    _annual_from_frame,
    _center,
    _recover_annual,
    _remap_annual,
    _solve,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_stabilization_ablation_v1"
RATING_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
ARMS = ("raw_spm", "stabilized_spm")
MODEL_ORDER = (
    "raw_spm",
    "stabilized_spm",
    "zero_prior_rapm",
    "raw_spm_aio",
    "stabilized_spm_aio",
)
PRIMARY_PAIRS = {
    frozenset(("raw_spm", "stabilized_spm")),
    frozenset(("raw_spm_aio", "stabilized_spm_aio")),
}


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must equal {value!r}.")
    cutoff = contract["information_cutoff"]
    if tuple(cutoff["rating_seasons"]) != RATING_SEASONS:
        raise ValueError("The rating-season contract changed.")
    if tuple(cutoff["evaluated_rating_seasons"]) != EVALUATED_RATING_SEASONS:
        raise ValueError("The evaluated rating-season contract changed.")
    if tuple(cutoff["test_seasons"]) != tuple(
        season + 1 for season in EVALUATED_RATING_SEASONS
    ):
        raise ValueError("The test-season contract changed.")
    if int(cutoff["final_unscored_rating_season"]) != 2026:
        raise ValueError("The final current rating must remain 2026 and unscored.")
    if cutoff["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return json.loads(json.dumps(contract, default=str))


def _ratio_recovery(stabilized: pd.Series, exposure: pd.Series, prior: float) -> pd.Series:
    reliability = exposure.clip(lower=0) / (exposure.clip(lower=0) + prior)
    return stabilized / reliability.where(reliability.gt(0))


def _add_raw_defense_features(
    annual: pd.DataFrame,
    matchup: pd.DataFrame,
) -> pd.DataFrame:
    output = annual.copy()
    dfg_attempts = output["dfg_attempts_p100"] * output["DefPoss"] / 100.0
    rim_attempts = output["rim_dfga_p100"] * output["DefPoss"] / 100.0
    output["dfg_diff_pct_raw"] = _ratio_recovery(
        output["dfg_diff_pct_eb"], dfg_attempts, 200.0
    )
    output["rim_diff_pct_raw"] = _ratio_recovery(
        output["rim_diff_pct_eb"], rim_attempts, 100.0
    )

    matchup = matchup.rename(columns={"Season": "Window_End"}).copy()
    required = {
        "PLAYER_ID",
        "Window_End",
        "matchup_possessions",
        "matchup_fga_p100",
        "matchup_opponent_adjusted_points_saved_p100",
        "matchup_fga_suppressed_vs_scorer_p100_eb",
        "matchup_shotmaking_points_saved_vs_scorer_p100_eb",
        "matchup_three_pa_suppressed_vs_scorer_p100_eb",
        "matchup_turnovers_forced_vs_scorer_p100_eb",
        "matchup_assists_suppressed_vs_scorer_p100_eb",
        "matchup_shooting_fouls_prevented_vs_scorer_p100_eb",
    }
    if missing := sorted(required - set(matchup.columns)):
        raise ValueError(f"Matchup panel lacks raw-recovery fields: {missing}")
    if matchup.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Matchup feature keys are not unique.")

    raw_matchup = matchup[["PLAYER_ID", "Window_End", "matchup_possessions"]].copy()
    raw_matchup["matchup_opponent_adjusted_points_saved_p100"] = matchup[
        "matchup_opponent_adjusted_points_saved_p100"
    ]
    possession_pairs = {
        "matchup_fga_suppressed_vs_scorer_p100_eb": "matchup_fga_suppressed_vs_scorer_p100_raw",
        "matchup_three_pa_suppressed_vs_scorer_p100_eb": "matchup_three_pa_suppressed_vs_scorer_p100_raw",
        "matchup_turnovers_forced_vs_scorer_p100_eb": "matchup_turnovers_forced_vs_scorer_p100_raw",
        "matchup_assists_suppressed_vs_scorer_p100_eb": "matchup_assists_suppressed_vs_scorer_p100_raw",
        "matchup_shooting_fouls_prevented_vs_scorer_p100_eb": "matchup_shooting_fouls_prevented_vs_scorer_p100_raw",
    }
    for stable, raw in possession_pairs.items():
        raw_matchup[raw] = _ratio_recovery(
            matchup[stable], matchup["matchup_possessions"], 500.0
        )
    fga = matchup["matchup_fga_p100"] * matchup["matchup_possessions"] / 100.0
    raw_matchup["matchup_shotmaking_points_saved_vs_scorer_p100_raw"] = (
        _ratio_recovery(
            matchup["matchup_shotmaking_points_saved_vs_scorer_p100_eb"],
            fga,
            200.0,
        )
    )
    raw_columns = [
        column
        for column in raw_matchup.columns
        if column not in {"PLAYER_ID", "Window_End", "matchup_possessions"}
    ]
    output = output.drop(
        columns=[column for column in raw_columns if column in output],
        errors="ignore",
    ).merge(
        raw_matchup[["PLAYER_ID", "Window_End", *raw_columns]],
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    return output


def _offense_pair_map(selected: tuple[str, ...], contract: dict) -> dict[str, str]:
    selected_set = set(selected)
    pairs = {
        stable: stable.removesuffix("_eb")
        for stable in selected
        if stable.endswith("_eb") and stable.removesuffix("_eb") in selected_set
    }
    pairs.update(contract["feature_contract"]["offense_composite_pairs"])
    if len(pairs) != 37:
        raise ValueError(f"Expected 37 frozen offense pairs, found {len(pairs)}.")
    if missing := sorted(set(pairs) - selected_set):
        raise ValueError(f"Stable offense fields left the selected contract: {missing}")
    return pairs


def _arm_features(
    selected: dict[str, tuple[str, ...]],
    contract: dict,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], pd.DataFrame]:
    offense_pairs = _offense_pair_map(selected["offense"], contract)
    defense_pairs = dict(contract["feature_contract"]["defense_pairs"])
    if len(defense_pairs) != 10:
        raise ValueError("The frozen defense map must contain 10 pairs.")

    raw_offense_duplicates = set(offense_pairs.values()) & set(selected["offense"])
    stable_offense = tuple(
        feature
        for feature in selected["offense"]
        if feature not in raw_offense_duplicates
    )
    raw_offense = tuple(offense_pairs.get(feature, feature) for feature in stable_offense)
    stable_defense = selected["defense"]
    raw_defense = tuple(defense_pairs.get(feature, feature) for feature in stable_defense)
    arms = {
        "stabilized_spm": {"offense": stable_offense, "defense": stable_defense},
        "raw_spm": {"offense": raw_offense, "defense": raw_defense},
    }
    for side in ("offense", "defense"):
        lengths = {len(arms[arm][side]) for arm in ARMS}
        if len(lengths) != 1:
            raise ValueError(f"{side} arm feature counts differ.")
        for arm in ARMS:
            if len(set(arms[arm][side])) != len(arms[arm][side]):
                raise ValueError(f"{arm} {side} contains duplicate concepts.")

    rows = []
    for side, pairs in (("offense", offense_pairs), ("defense", defense_pairs)):
        for stable, raw in pairs.items():
            rows.append({"component": side, "stabilized_feature": stable, "raw_feature": raw})
    return arms, pd.DataFrame(rows)


def _raw_passer_score(frame: pd.DataFrame) -> pd.Series:
    required = (
        "offensive_load_2017_p100",
        "assist_to_load_2017",
        "turnover_to_load_2017",
        "creation_to_load_2017",
        "OffPoss",
    )
    if missing := sorted(set(required) - set(frame.columns)):
        raise ValueError(f"Raw passer score lacks fields: {missing}")
    output = pd.Series(index=frame.index, dtype=float)
    for _, indexes in frame.groupby("Window_End", sort=True).groups.items():
        block = frame.loc[indexes]
        weights = block["OffPoss"]
        output.loc[indexes] = (
            _weighted_zscore(block["offensive_load_2017_p100"], weights).clip(-4.0, 4.0)
            + 3.0 * _weighted_zscore(block["assist_to_load_2017"], weights).clip(-4.0, 4.0)
            - 2.0 * _weighted_zscore(block["turnover_to_load_2017"], weights).clip(-4.0, 4.0)
            + 0.5 * _weighted_zscore(block["creation_to_load_2017"], weights).clip(-4.0, 4.0)
        ).to_numpy()
    return output


def _build_panel(
    annual_path: Path,
    matchup_path: Path,
    five_year_features_path: Path,
    selected_manifest_path: Path,
    targets_path: Path,
    contract: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    annual = pd.read_parquet(annual_path)
    matchup = pd.read_parquet(matchup_path)
    manifest = json.loads(selected_manifest_path.read_text())
    selected = {
        side: tuple(manifest["feature_contract"][side])
        for side in ("offense", "defense")
    }
    arms, pair_map = _arm_features(selected, contract)
    annual = _add_raw_defense_features(annual, matchup)
    annual["behavioral_passer_score_raw_v1"] = _raw_passer_score(annual)

    features = pd.read_parquet(five_year_features_path)
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year feature keys are not unique.")
    paired_fields = tuple(
        dict.fromkeys(
            (*pair_map["stabilized_feature"], *pair_map["raw_feature"])
        )
    )
    if missing := sorted(set(paired_fields) - set(annual.columns)):
        raise ValueError(f"Annual panel lacks paired fields: {missing}")
    for end, indexes in features.groupby("Window_End", sort=True).groups.items():
        window = annual.loc[annual["Window_End"].between(end - 4, end)]
        players = features.loc[indexes, "PLAYER_ID"]
        for row in pair_map.itertuples(index=False):
            weight = "OffPoss" if row.component == "offense" else "DefPoss"
            features.loc[indexes, row.stabilized_feature] = _pool_annual_field(
                window, players, row.stabilized_feature, weight
            ).to_numpy()
            features.loc[indexes, row.raw_feature] = _pool_annual_field(
                window, players, row.raw_feature, weight
            ).to_numpy()
    required_features = {
        feature
        for arm in arms.values()
        for fields in arm.values()
        for feature in fields
    }
    if missing := sorted(required_features - set(features.columns)):
        raise ValueError(f"Paired feature panel lacks fields: {missing}")
    if features["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered the feature panel.")

    targets = pd.read_parquet(targets_path)
    target_columns = (
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    )
    panel = features.merge(
        targets[list(target_columns)],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if panel.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("The stabilization panel has duplicate keys.")

    coverage_rows = []
    for row in pair_map.itertuples(index=False):
        for season, block in features.groupby("Window_End", sort=True):
            stable_missing = block[row.stabilized_feature].isna()
            raw_missing = block[row.raw_feature].isna()
            jointly_observed = ~(stable_missing | raw_missing)
            correlation = (
                block.loc[jointly_observed, [row.stabilized_feature, row.raw_feature]]
                .corr()
                .iloc[0, 1]
                if jointly_observed.sum() >= 2
                else np.nan
            )
            coverage_rows.append(
                {
                    "component": row.component,
                    "stabilized_feature": row.stabilized_feature,
                    "raw_feature": row.raw_feature,
                    "window_end": int(season),
                    "rows": len(block),
                    "stabilized_missing_fraction": float(stable_missing.mean()),
                    "raw_missing_fraction": float(raw_missing.mean()),
                    "jointly_observed_rows": int(jointly_observed.sum()),
                    "pearson_correlation": correlation,
                }
            )
    return panel, arms, pair_map, pd.DataFrame(coverage_rows)


def _fit_priors(
    panel: pd.DataFrame,
    arms: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prior_rows = []
    metric_rows = []
    selection_rows = []
    models = {}
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Rating season {season} lacks chronological history.")
        for arm in ARMS:
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                fields = arms[arm][side]
                target = f"target_{side}"
                model = _fit_model(_frozen_model(side), train, fields, target)
                prediction = model.predict(test.loc[:, fields])
                prior[side] = prediction
                models[(season, arm, side)] = model
                selection_rows.append(
                    {
                        "rating_season": season,
                        "candidate": arm,
                        "component": side,
                        "feature_count": len(fields),
                        "family": "histogram_gbm" if side == "offense" else "ridge",
                    }
                )
                metric_rows.append(
                    {
                        "rating_season": season,
                        "candidate": arm,
                        "component": side,
                        "players": len(test),
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prediction,
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
            prior["net"] = prior["offense"] + prior["defense"]
            metric_rows.append(
                {
                    "rating_season": season,
                    "candidate": arm,
                    "component": "net",
                    "players": len(test),
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["net"].to_numpy(dtype=float),
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prior_rows.append(_prior_frame(prior, arm))
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        models,
    )


def _annual_bundles(cache_root: Path, matrix_root: Path) -> tuple[dict, pd.DataFrame]:
    annual = {}
    for season in range(2020, 2024):
        frame = load_legacy_possessions(cache_root, (season,), game_types=("regular",))
        annual[season] = _annual_from_frame(frame, season)
    reconstruction = []
    for season in (2024, 2025, 2026):
        bundle, quality = _recover_annual(
            matrix_root / f"5y_end_{season}", season, annual
        )
        annual[season] = bundle
        reconstruction.append(quality)
    return annual, pd.DataFrame(reconstruction)


def _score_models(
    priors: pd.DataFrame,
    annual: dict,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rating_rows = []
    game_rows = []
    coverage_rows = []
    for season in RATING_SEASONS:
        matrix_dir = matrix_root / f"5y_end_{season}"
        matrix_players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[season], matrix_players)
        zero_center = np.zeros(2 * len(bundle.players) + 1)
        zero_beta, zero_intercept = _solve(bundle, zero_center, scale=0.0)
        season_models = [("zero_prior_rapm", zero_beta, zero_intercept)]
        for arm in ARMS:
            prior = priors.loc[
                priors["candidate"].eq(arm) & priors["Window_End"].eq(season)
            ]
            center, coverage = _center(prior, bundle)
            standalone = center.copy()
            standalone[-1] = zero_beta[-1]
            posterior, posterior_intercept = _solve(bundle, center, scale=1.0)
            season_models.extend(
                [(arm, standalone, zero_intercept), (f"{arm}_aio", posterior, posterior_intercept)]
            )
            coverage_rows.append({"candidate": arm, "rating_season": season, **coverage})
        for candidate, beta, intercept in season_models:
            if season in EVALUATED_RATING_SEASONS:
                game = stored_evaluation_predictions(matrix_dir, beta, intercept)
                game["candidate"] = candidate
                game["rating_season"] = season
                game["test_season"] = season + 1
                game["squared_error"] = (game["actual_margin"] - game["predicted_margin"]) ** 2
                game_rows.append(game)
            n = len(bundle.players)
            rating = pd.DataFrame(
                {
                    "PLAYER_ID": bundle.players,
                    "offense": 100.0 * beta[:n],
                    "defense": -100.0 * beta[n : 2 * n],
                    "Poss_Off": bundle.off_possessions,
                    "Poss_Def": bundle.def_possessions,
                }
            )
            rating["net"] = rating["offense"] + rating["defense"]
            rating["candidate"] = candidate
            rating["rating_season"] = season
            rating_rows.append(rating)
    games = pd.concat(game_rows, ignore_index=True)
    for season, frame in games.groupby("test_season"):
        counts = frame.groupby("candidate")["game_id"].nunique()
        if set(counts.index) != set(MODEL_ORDER) or counts.nunique() != 1:
            raise ValueError(f"Candidates do not score identical {season} games.")
        hashes = frame.groupby("candidate").apply(
            lambda group: hashlib.sha256(
                "|".join(
                    sorted(group["game_id"].astype(str) + ":" + group["actual_margin"].astype(str))
                ).encode()
            ).hexdigest(),
            include_groups=False,
        )
        if hashes.nunique() != 1:
            raise ValueError(f"Candidates do not share {season} outcomes.")
    return pd.concat(rating_rows, ignore_index=True), games, pd.DataFrame(coverage_rows)


def _game_metric_frames(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (season, candidate), frame in games.groupby(["test_season", "candidate"]):
        rows.append(
            {
                "test_season": int(season),
                "rating_season": int(season) - 1,
                "candidate": candidate,
                **_game_metrics(frame),
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    return folds, summary


def paired_game_bootstrap(
    games: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_errors = []
    for _, frame in games.groupby("test_season", sort=True):
        wide = frame.pivot(
            index="game_id", columns="candidate", values=["actual_margin", "predicted_margin"]
        )
        actual = wide["actual_margin"]
        if actual.isna().any().any() or not actual.nunique(axis=1).eq(1).all():
            raise ValueError("Candidates must score identical complete outcomes.")
        predictions = wide["predicted_margin"].reindex(columns=MODEL_ORDER)
        if predictions.isna().any().any():
            raise ValueError("A candidate missed a scored game.")
        actual_values = actual.iloc[:, 0].to_numpy(dtype=float)
        season_errors.append((actual_values[:, None] - predictions.to_numpy(dtype=float)) ** 2)
    point_mse = np.mean([values.mean(axis=0) for values in season_errors], axis=0)
    season_mse = np.asarray([values.mean(axis=0) for values in season_errors])
    rng = np.random.default_rng(seed)
    draw_mse = np.empty((draws, len(MODEL_ORDER)), dtype=float)
    for draw in range(draws):
        draw_mse[draw] = np.mean(
            [
                values[rng.integers(0, len(values), len(values))].mean(axis=0)
                for values in season_errors
            ],
            axis=0,
        )
    ranks = np.argsort(np.argsort(draw_mse, axis=1), axis=1) + 1
    model_rows = []
    for index, candidate in enumerate(MODEL_ORDER):
        low, high = np.quantile(draw_mse[:, index], [0.025, 0.975])
        model_rows.append(
            {
                "candidate": candidate,
                "folds": len(season_errors),
                "equal_season_mse": float(point_mse[index]),
                "equal_season_rmse": float(np.sqrt(point_mse[index])),
                "bootstrap_rmse_95_low": float(np.sqrt(low)),
                "bootstrap_rmse_95_high": float(np.sqrt(high)),
                "probability_best": float(np.mean(ranks[:, index] == 1)),
                "mean_bootstrap_rank": float(ranks[:, index].mean()),
                "bootstrap_draws": draws,
            }
        )
    pair_rows = []
    for left, right in itertools.combinations(MODEL_ORDER, 2):
        left_index = MODEL_ORDER.index(left)
        right_index = MODEL_ORDER.index(right)
        delta = draw_mse[:, left_index] - draw_mse[:, right_index]
        low, high = np.quantile(delta, [0.025, 0.975])
        pair_rows.append(
            {
                "candidate": left,
                "reference": right,
                "primary_comparison": frozenset((left, right)) in PRIMARY_PAIRS,
                "mean_mse_delta": float(point_mse[left_index] - point_mse[right_index]),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(delta < 0)),
                "candidate_fold_wins": int(np.sum(season_mse[:, left_index] < season_mse[:, right_index])),
                "reference_fold_wins": int(np.sum(season_mse[:, right_index] < season_mse[:, left_index])),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/spm_stabilization_ablation_v1.yml",
    )
    parser.add_argument(
        "--annual-features",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/features/statistical_impact/statistical_features_v2_cb03edaf32/features.parquet",
    )
    parser.add_argument(
        "--matchup-features",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/features/matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet",
    )
    parser.add_argument(
        "--five-year-features",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/panels/full_spm_features_2014_2026_v1_4c77ae6acc/five_year_features.parquet",
    )
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/panels/full_spm_features_2014_2026_v1_4c77ae6acc/run.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument(
        "--possession-cache",
        type=Path,
        default=ROOT / "rapm/data/possession_cache",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    contract = _load_contract(args.contract)
    source_paths = {
        "contract": args.contract,
        "annual_features": args.annual_features,
        "matchup_features": args.matchup_features,
        "five_year_features": args.five_year_features,
        "selected_manifest": args.selected_manifest,
        "targets": args.targets,
        "runner": Path(__file__),
        **{
            f"matrix_{season}": args.matrix_root / f"5y_end_{season}/manifest.json"
            for season in RATING_SEASONS
        },
        **{
            f"possessions_{season}": args.possession_cache / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    if missing := [str(path) for path in source_paths.values() if not path.exists()]:
        raise FileNotFoundError(f"Missing stabilization inputs: {missing}")

    panel, arms, pair_map, feature_coverage = _build_panel(
        args.annual_features,
        args.matchup_features,
        args.five_year_features,
        args.selected_manifest,
        args.targets,
        contract,
    )
    priors, target_metrics, selections, models = _fit_priors(panel, arms)
    annual, reconstruction = _annual_bundles(args.possession_cache, args.matrix_root)
    ratings, games, prior_coverage = _score_models(priors, annual, args.matrix_root)
    fold_metrics, summary = _game_metric_frames(games)
    bootstrap_models, bootstrap_pairs = paired_game_bootstrap(
        games, draws=args.draws, seed=args.seed
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(RATING_SEASONS),
        "evaluated_rating_seasons": list(EVALUATED_RATING_SEASONS),
        "test_seasons": [season + 1 for season in EVALUATED_RATING_SEASONS],
        "final_unscored_rating_season": 2026,
        "feature_counts": {
            arm: {side: len(fields) for side, fields in sides.items()}
            for arm, sides in arms.items()
        },
        "paired_concepts": {
            side: int(pair_map["component"].eq(side).sum())
            for side in ("offense", "defense")
        },
        "rapm_penalties": {"offense": 3000.0, "defense": 3000.0, "home": 300.0},
        "center_scale": 1.0,
        "bootstrap": {
            "draws": args.draws,
            "seed": args.seed,
            "unit": "whole game within test season",
            "aggregation": "equal-season mean MSE",
        },
        "sources": {
            name: {"path": _relative(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research"
        / "spm_stabilization_ablation"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "feature_pair_map.parquet": pair_map,
        "feature_pair_coverage.parquet": feature_coverage,
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "model_selection.parquet": selections,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": fold_metrics,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_coverage.parquet": prior_coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    model_root = output / "models"
    model_root.mkdir()
    for (season, arm, side), model in models.items():
        joblib.dump(model, model_root / f"{season}_{arm}_{side}.joblib")
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season game margin from prior-season player ratings",
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "identical_games_within_fold": True,
            "equal_feature_counts_within_side": True,
            "raw_and_stabilized_use_one_value_per_paired_concept": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "season_2027_loaded": False,
            "final_rating_season": int(ratings["rating_season"].max()),
        },
        "files": {},
        "forbidden_interpretation": (
            "All five test seasons are reused evidence. This run cannot promote a public model."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    run["files"]["models"] = {"path": "models", "files": len(models)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(
        bootstrap_pairs.loc[bootstrap_pairs["primary_comparison"]]
        .sort_values("mean_mse_delta")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
