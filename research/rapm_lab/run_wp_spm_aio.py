"""Test Box15 and rich statistical priors inside conserved WP-RAPM."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit as fit_box
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    fit_coefficients_with_center,
    load_current_player_names,
    load_unified_terminal_possessions,
    ratings_table,
)
from nba_impact.models.win_probability_rapm import build_conserved_wp_target
try:
    from research.rapm_lab.run_rolling_5y_wp_rapm import _states
    from research.run_annual_spm_learner_screen import ModelSpec, _fit as fit_rich, _prune_features, _weighted_metrics
except ModuleNotFoundError:
    from run_rolling_5y_wp_rapm import _states
    from run_annual_spm_learner_screen import ModelSpec, _fit as fit_rich, _prune_features, _weighted_metrics


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/wp_spm_aio_v1.yml"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/wp_spm_aio"
CHECKPOINT = OUTPUT_ROOT / "checkpoints/wp_target_2014_2026.parquet"
WP_TARGET_RUN = ROOT / "research/rapm_lab/outputs/rolling_5y_wp_rapm/rolling_5y_wp_rapm_v1_2704c10da7"
BOX = ROOT / "artifacts/research/historical_box15_extension/historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet"
RICH_RUN = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_fdee01ec4e"
RICH = RICH_RUN / "annual_features.parquet"


def _build_target(contract: dict) -> pd.DataFrame:
    if CHECKPOINT.exists():
        cached = pd.read_parquet(CHECKPOINT)
        rebuilt, _ = build_conserved_wp_target(cached)
        columns = ["home_wp_after", "home_wp_change", "offense_wp_change", "pts"]
        rebuilt["pts"] = rebuilt["offense_wp_change"]
        if not cached["possession_id"].equals(rebuilt["possession_id"]) or not np.allclose(
            cached[columns], rebuilt[columns], rtol=0, atol=1e-12,
        ):
            raise ValueError(
                "WP checkpoint has stale possession credit. Preserve it and create "
                "a new versioned target and dependent priors before rerunning."
            )
        return cached
    start, end = map(int, contract["seasons"]["possession_source"])
    source = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        tuple(range(int(contract["surface"]["source_start"]), end + 1)),
        transition_season=2024,
        game_types=("regular",),
    )
    states, features = _states(source)
    probabilities = np.full(len(states), np.nan)
    stride = int(contract["surface"]["training_stride"])
    for season in range(start, end + 1):
        train = states["season"].lt(season).to_numpy()
        sampled = train & states["possession_index_before"].mod(stride).eq(0).to_numpy()
        test = states["season"].eq(season).to_numpy()
        model = LogisticRegression(
            C=float(contract["surface"]["logistic_c"]), solver="lbfgs", max_iter=1000
        )
        model.fit(features.loc[sampled], states.loc[sampled, "home_win"])
        probabilities[test] = model.predict_proba(features.loc[test])[:, 1]
    keep = states["season"].between(start, end).to_numpy()
    frame = states.loc[keep].copy().reset_index(drop=True)
    frame["probability_context"] = probabilities[keep]
    if frame["probability_context"].isna().any():
        raise AssertionError("Every rating possession needs a past-only WP state.")
    target, conservation = build_conserved_wp_target(frame)
    if float(conservation["conservation_error"].abs().max()) > 1e-10:
        raise AssertionError("WP changes must conserve the game result.")
    target["pts"] = target["offense_wp_change"]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    target.to_parquet(CHECKPOINT, index=False)
    return target


def _statistical_panel() -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    target = pd.read_parquet(WP_TARGET_RUN / "ratings.parquet").rename(
        columns={
            "player_id": "PLAYER_ID",
            "window_end": "Season",
            "offense_per_100": "target_offense",
            "defense_per_100": "target_defense",
            "net_per_100": "target_net",
            "off_possessions": "Poss_Off",
            "def_possessions": "Poss_Def",
        }
    )
    target = target[["PLAYER_ID", "Season", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]]
    target["sample_weight"] = np.sqrt(np.minimum(target["Poss_Off"], target["Poss_Def"]).clip(lower=1))
    box = pd.read_parquet(BOX).rename(columns={"Window_End": "Season"}).merge(
        target, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    rich = pd.read_parquet(RICH).rename(columns={"Window_End": "Season"}).merge(
        target, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    manifest = json.loads((RICH_RUN / "run.json").read_text())
    return box, rich, tuple(manifest["feature_contract"]["offense"]), tuple(manifest["feature_contract"]["defense"])


def _fit_priors(contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    box, rich, rich_offense, rich_defense = _statistical_panel()
    rating_seasons = tuple(map(int, contract["seasons"]["rating"]))
    predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    selections: list[dict] = []
    for season in rating_seasons:
        for candidate, panel in (("box15", box), ("rich", rich)):
            train = panel.loc[panel["Season"].lt(season)].copy()
            test = panel.loc[panel["Season"].eq(season)].copy()
            if train["Season"].nunique() < 3 or test.empty:
                continue
            output = test[["PLAYER_ID", "Season", "target_offense", "target_defense", "target_net", "sample_weight"]].copy()
            for side in ("offense", "defense"):
                target = f"target_{side}"
                if candidate == "box15":
                    alpha = float(contract["spm"]["box15"][f"{side}_alpha"])
                    model = fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
                    features = BOX_PIPM_STYLE_FEATURES
                else:
                    candidates = rich_offense if side == "offense" else rich_defense
                    features = _prune_features(
                        train,
                        candidates,
                        threshold=float(contract["spm"]["rich"]["correlation_prune_threshold"]),
                    )
                    spec = (
                        ModelSpec("elastic_net", {
                            "alpha": float(contract["spm"]["rich"]["offense_alpha"]),
                            "l1_ratio": float(contract["spm"]["rich"]["offense_l1_ratio"]),
                        })
                        if side == "offense"
                        else ModelSpec("ridge", {"alpha": float(contract["spm"]["rich"]["defense_alpha"])})
                    )
                    model = fit_rich(spec, train, features, target)
                output[f"prior_{side}"] = model.predict(test.loc[:, features])
                values = _weighted_metrics(
                    test[target].to_numpy(), output[f"prior_{side}"].to_numpy(), test["sample_weight"].to_numpy()
                )
                metrics.append({"candidate": candidate, "rating_season": season, "component": side, **values})
                selections.append({"candidate": candidate, "rating_season": season, "component": side, "features": len(features)})
            output["prior_net"] = output["prior_offense"] + output["prior_defense"]
            values = _weighted_metrics(
                test["target_net"].to_numpy(), output["prior_net"].to_numpy(), test["sample_weight"].to_numpy()
            )
            metrics.append({"candidate": candidate, "rating_season": season, "component": "net", **values})
            output["candidate"] = candidate
            predictions.append(output)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(selections)


def _center(design, prior: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    n = len(design.players)
    center = np.zeros(design.X.shape[1])
    indexed = prior.set_index("PLAYER_ID")
    shared, design_index, prior_index = np.intersect1d(
        design.players, indexed.index.to_numpy(dtype=np.int64), return_indices=True
    )
    if not len(shared):
        return center
    center[design_index] = indexed.iloc[prior_index]["prior_offense"].to_numpy() / 100.0
    center[n + design_index] = -indexed.iloc[prior_index]["prior_defense"].to_numpy() / 100.0
    off_weight = np.asarray(design.X[train_mask, :n].sum(axis=0)).ravel()
    def_weight = np.asarray(design.X[train_mask, n : 2 * n].sum(axis=0)).ravel()
    center[:n] -= np.average(center[:n], weights=off_weight)
    center[n : 2 * n] -= np.average(center[n : 2 * n], weights=def_weight)
    return center


def _game_predictions(design, beta: np.ndarray, intercept: float, mask: np.ndarray, candidate: str, outcome_season: int) -> pd.DataFrame:
    predicted = intercept + np.asarray(design.X[mask] @ beta).ravel()
    sign = np.where(design.home_offense[mask], 1.0, -1.0)
    return pd.DataFrame({
        "outcome_season": outcome_season,
        "candidate": candidate,
        "game_id": design.game_ids[mask],
        "actual": design.y[mask] * sign,
        "predicted": predicted * sign,
    }).groupby(["outcome_season", "candidate", "game_id"], as_index=False).sum()


def _fold_metrics(games: pd.DataFrame) -> dict:
    error = games["actual"].to_numpy() - games["predicted"].to_numpy()
    predicted = games["predicted"].to_numpy()
    actual = games["actual"].to_numpy()
    variance = float(np.var(predicted))
    return {
        "games": len(games),
        "mse": float(np.mean(error**2)),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "calibration_slope": float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance > 0 else math.nan,
    }


def _bootstrap(games: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = games.pivot(index=["outcome_season", "game_id"], columns="candidate", values=["actual", "predicted"])
    candidates = [value for value in ("box15_aio", "rich_aio") if ("predicted", value) in wide.columns]
    rows = []
    for candidate in candidates:
        deltas = []
        for _ in range(draws):
            season_values = []
            for season in sorted(wide.index.get_level_values(0).unique()):
                frame = wide.xs(season, level=0)
                take = rng.integers(0, len(frame), len(frame))
                actual = frame[("actual", candidate)].to_numpy()[take]
                candidate_error = actual - frame[("predicted", candidate)].to_numpy()[take]
                zero_error = actual - frame[("predicted", "zero_wp_rapm")].to_numpy()[take]
                season_values.append(np.mean(candidate_error**2 - zero_error**2))
            deltas.append(np.mean(season_values))
        rows.append({
            "candidate": candidate,
            "reference": "zero_wp_rapm",
            "draws": draws,
            "mean_mse_delta": float(np.mean(deltas)),
            "lower_95": float(np.quantile(deltas, 0.025)),
            "upper_95": float(np.quantile(deltas, 0.975)),
            "probability_better": float(np.mean(np.asarray(deltas) < 0)),
        })
    return pd.DataFrame(rows)


def run() -> dict:
    started = time.perf_counter()
    contract = yaml.safe_load(CONTRACT.read_text())
    prior_predictions, prior_metrics, selections = _fit_priors(contract)
    target = _build_target(contract)
    design = build_design(target, include_home=True)
    config = RapmConfig(
        seasons=tuple(map(int, contract["seasons"]["rating"])),
        lambda_off=float(contract["wp_rapm"]["lambda_offense"]),
        lambda_def=float(contract["wp_rapm"]["lambda_defense"]),
        lambda_home=float(contract["wp_rapm"]["lambda_home"]),
        data_scope="annual_conserved_progress_wp_aio",
    )
    games = []
    fold_rows = []
    latest_tables = []
    names = load_current_player_names(ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet")
    for rating_season in map(int, contract["seasons"]["rating"]):
        train = design.seasons == rating_season
        if not train.any():
            continue
        zero_beta, zero_intercept = fit_coefficients(design, config, row_mask=train)
        candidates = {"zero_wp_rapm": (zero_beta, zero_intercept)}
        for candidate in ("box15", "rich"):
            prior = prior_predictions.loc[
                prior_predictions["rating_season"].eq(rating_season)
                if "rating_season" in prior_predictions.columns
                else prior_predictions["Season"].eq(rating_season)
            ]
            prior = prior.loc[prior["candidate"].eq(candidate)]
            if prior.empty:
                continue
            center = _center(design, prior, train)
            candidates[f"{candidate}_aio"] = fit_coefficients_with_center(
                design, config, center, center_scale=float(contract["wp_rapm"]["prior_trust"]), row_mask=train
            )
        if rating_season == max(map(int, contract["seasons"]["rating"])):
            for candidate, (beta, _) in candidates.items():
                table = ratings_table(design, beta, names=names)
                table = table.loc[table[["off_possessions", "def_possessions"]].min(axis=1).gt(0)].copy()
                table["candidate"] = candidate
                table["Season"] = rating_season
                latest_tables.append(table)
        outcome = rating_season + 1
        test = design.seasons == outcome
        if not test.any() or outcome not in set(map(int, contract["seasons"]["outcome"])):
            continue
        for candidate, (beta, intercept) in candidates.items():
            game = _game_predictions(design, beta, intercept, test, candidate, outcome)
            games.append(game)
            fold_rows.append({"rating_season": rating_season, "outcome_season": outcome, "candidate": candidate, **_fold_metrics(game)})
    games_frame = pd.concat(games, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"), mean_mse=("mse", "mean"), mean_rmse=("rmse", "mean"), mean_correlation=("correlation", "mean"), mean_calibration_slope=("calibration_slope", "mean")
    )
    bootstrap = _bootstrap(games_frame, int(contract["evaluation"]["bootstrap_draws"]), int(contract["evaluation"]["bootstrap_seed"]))
    identity = hashlib.sha256(json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__)), "wp_target": sha256_file(WP_TARGET_RUN / "ratings.parquet"), "box": sha256_file(BOX), "rich": sha256_file(RICH)}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"wp_spm_aio_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    prior_predictions.to_parquet(output / "prior_predictions.parquet", index=False)
    prior_metrics.to_parquet(output / "prior_metrics.parquet", index=False)
    selections.to_parquet(output / "feature_counts.parquet", index=False)
    folds.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    bootstrap.to_parquet(output / "paired_bootstrap.parquet", index=False)
    games_frame.to_parquet(output / "game_predictions.parquet", index=False)
    if latest_tables:
        pd.concat(latest_tables, ignore_index=True).to_parquet(output / "leaderboard_2026.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "reused_historical_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "quality": {"possessions": len(target), "games": int(target["gameid"].nunique()), "folds": int(folds["outcome_season"].nunique())},
        "summary": summary.to_dict("records"),
        "paired_bootstrap": bootstrap.to_dict("records"),
        "decision_rule": "A statistical prior must improve equal-season next-season conserved-WP game MSE on identical games.",
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
