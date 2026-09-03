"""Correct WP chronology, choose a lower penalty on earlier folds, and gate it."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, ratings_table, load_unified_terminal_possessions
from nba_impact.models.win_probability_rapm import build_log_odds_wp_target, build_conserved_wp_target
from research.rapm_lab.run_log_odds_wp_rapm_lambda import _game_predictions, _metrics, _past_only_predictions
from research.rapm_lab.run_rolling_5y_wp_rapm import _states

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/wp_chronology_release_v2.json"
SOURCE = ROOT / "research/rapm_lab/outputs/wp_spm_aio/checkpoints/wp_target_2014_2026.parquet"
PULSE = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a"
OUTPUT = ROOT / "research/rapm_lab/outputs/wp_chronology_release"
SCORES = ROOT / "data/lake/bronze/official_game_scores"


def official_scores(start=1997, end=2026):
    frames = []
    for season in range(start, end + 1):
        path = SCORES / f"project_season={season}/regular.parquet"
        manifest = json.loads(path.with_suffix(".parquet.manifest.json").read_text())
        if not manifest["passed"] or sha256_file(path) != manifest["output_sha256"]:
            raise ValueError("Official-score source hash or quality gate failed.")
        part = pd.read_parquet(path)
        if not part.project_season.eq(season).all() or not part.season_type.eq("regular").all():
            raise ValueError("Wrong official-score partition.")
        frames.append(part)
    scores = pd.concat(frames, ignore_index=True).rename(columns={"project_season": "season", "game_id": "gameid"})
    # Schedule entries with no score are not played games. A possession join to
    # one still fails below rather than fabricating a winner.
    scores = scores.loc[~(scores.home_score.eq(0) & scores.away_score.eq(0))].copy()
    if scores.duplicated(["season", "gameid"]).any() or scores.home_score.eq(scores.away_score).any():
        raise ValueError("Duplicate or tied official game.")
    if not np.isfinite(scores[["home_score", "away_score"]]).all().all():
        raise ValueError("Nonfinite official score.")
    scores["official_home_win"] = scores.home_score.gt(scores.away_score).astype(int)
    scores["actual_margin"] = scores.home_score - scores.away_score
    return scores


def label_states(states, scores):
    labeled = states.merge(scores[["season", "gameid", "official_home_win"]],
                           on=["season", "gameid"], how="left", validate="many_to_one", sort=False)
    if labeled.official_home_win.isna().any():
        raise ValueError("WP source game has no official winner; no cached-label fallback.")
    pd.testing.assert_frame_equal(labeled[states.columns], states.reset_index(drop=True))
    changed = labeled.loc[labeled.home_win.ne(labeled.official_home_win), ["season", "gameid"]].drop_duplicates()
    labeled["home_win"] = labeled.pop("official_home_win").astype(int)
    return labeled, changed.groupby("season").size().to_dict()


def corrected_target(contract: dict, *, pilot: bool = False):
    end = 2014 if pilot else 2026
    source = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache", ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        tuple(range(1997, end + 1)), transition_season=2024, game_types=("regular",))
    print(f"audited source through {end}: {len(source):,} possessions", flush=True)
    states, features = _states(source)
    if list(features) != ["home_score_diff", "possession_progress", "is_overtime", "score_pressure", "score_late_interaction", "home_possession"]:
        raise ValueError("Frozen WP feature contract changed.")
    states, changed_labels = label_states(states, official_scores(1997, end))
    if not np.isfinite(features).all().all():
        raise ValueError("Nonfinite progress-WP features.")
    probabilities = np.full(len(states), np.nan)
    surface = []
    for season in range(2014, end + 1):
        sampled = states.season.lt(season) & states.possession_index_before.mod(contract["surface"]["training_stride"]).eq(0)
        test = states.season.eq(season)
        model = LogisticRegression(C=contract["surface"]["logistic_c"], solver="lbfgs", max_iter=contract["surface"]["max_iter"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features.loc[sampled], states.loc[sampled, "home_win"])
        probability = model.predict_proba(features.loc[test])[:, 1]
        probabilities[test] = probability
        surface.append({"season": season, "train_end": season - 1, "train_rows": int(sampled.sum()),
                        "brier": float(brier_score_loss(states.loc[test, "home_win"], probability)),
                        "log_loss": float(log_loss(states.loc[test, "home_win"], probability)),
                        "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])})
        print(f"rebuilt past-only WP surface for {season}", flush=True)
    keep = states.season.ge(2014)
    source = states.loc[keep].copy().reset_index(drop=True)
    source["probability_context"] = probabilities[keep]
    if not pilot:
        old = pd.read_parquet(SOURCE, columns=["possession_id", "home_poss", "home_score_diff_before",
                                             "possession_index_before", *AWAY_PLAYER_COLUMNS, *HOME_PLAYER_COLUMNS])
        cols = list(old.columns)
        pd.testing.assert_frame_equal(source[cols].sort_values("possession_id").reset_index(drop=True),
                                      old.sort_values("possession_id").reset_index(drop=True), check_dtype=False)
    target, conservation = build_log_odds_wp_target(source, epsilon=contract["epsilon"])
    _, raw_conservation = build_conserved_wp_target(source)
    expected = target.groupby("gameid", sort=False).cumcount().to_numpy()
    if not np.array_equal(target.possession_index_before.to_numpy(), expected):
        raise ValueError("Each game's chronology must start at zero and be contiguous.")
    if max(conservation.conservation_error.abs().max(), raw_conservation.conservation_error.abs().max()) > 1e-10:
        raise ValueError("Log-odds conservation failed.")
    quality = {"possessions": len(target), "games": int(target.gameid.nunique()),
               "maximum_conservation_error": float(conservation.conservation_error.abs().max()),
               "official_winner_corrections_by_season": changed_labels, "surface": surface,
               "surface_metric_weighting": "possession-state weighted; diagnostic only, not selected on",
               "state_features": contract["surface_features"]}
    target["pts"] = target.offense_log_odds_change
    return target.reset_index(drop=True), quality


def paired_rmse(frame: pd.DataFrame, candidates: list[str], contract: dict):
    """Common whole-game draws within each fixed season; equal season weights."""
    if frame.groupby(["outcome_season", "game_id"]).actual_margin.nunique().ne(1).any():
        raise ValueError("Paired candidates have inconsistent actual margins.")
    errors = frame.assign(squared_error=(frame.actual_margin - frame.predicted_margin) ** 2)
    wide = errors.pivot(index=["outcome_season", "game_id"], columns="candidate", values="squared_error")[candidates]
    if wide.empty or not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("Paired candidates must have identical finite game rows.")
    rng = np.random.default_rng(contract["bootstrap_seed"])
    draws = contract["bootstrap_draws"]
    boot = np.zeros((draws, len(candidates)))
    means = []
    groups = list(wide.groupby(level="outcome_season", sort=True))
    for _, part in groups:
        values = part.to_numpy(dtype=float)
        means.append(values.mean(axis=0))
        for start in range(0, draws, 100):
            stop = min(start + 100, draws)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            boot[start:stop] += values[indices].mean(axis=1) / len(groups)
    return np.sqrt(np.mean(means, axis=0)), np.sqrt(boot)


def evaluation_period(predictions: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    frame = predictions.loc[predictions.outcome_season.isin(seasons)]
    for _, candidate in frame.groupby("candidate"):
        if set(candidate.outcome_season) != set(seasons):
            raise ValueError("Candidate is missing a declared evaluation season.")
    if frame.empty:
        raise ValueError("Evaluation period is empty.")
    return frame


def select_lambda(predictions: pd.DataFrame, contract: dict) -> dict:
    development = evaluation_period(predictions, contract["development_outcomes"])
    lambdas = contract["shared_lambdas"]
    candidates = [f"logit_{value}" for value in lambdas]
    point, boot = paired_rmse(development, candidates, contract)
    best = int(np.argmin(point))
    centered = boot - point
    # Maximum over all ordered pair errors equals the range across arms.
    critical = float(np.quantile(np.ptp(centered, axis=1), contract["bootstrap_quantile"]))
    excess = point - point[best]
    upper = excess + critical
    upper[best] = 0.0  # An arm's difference from itself is exactly zero.
    eligible = (excess <= contract["rmse_tolerance"]) & (upper <= contract["rmse_tolerance"])
    selected = min(value for value, ok in zip(lambdas, eligible, strict=True) if ok)
    return {"selected_lambda": selected, "empirical_best_lambda": lambdas[best],
            "simultaneous_upper_critical": critical,
            "selection_outcomes": contract["development_outcomes"],
            "candidates": [{"lambda": value, "rmse": float(point[i]), "excess_rmse": float(excess[i]),
                            "simultaneous_upper_excess": float(upper[i]), "eligible": bool(eligible[i])}
                           for i, value in enumerate(lambdas)]}


def publication_gate(predictions: pd.DataFrame, selected: int, contract: dict) -> dict:
    later = evaluation_period(predictions, contract["diagnostic_outcomes"])
    reference = contract["reference_lambda"]
    if selected == reference:
        paired_rmse(later, [f"logit_{reference}"], contract)
        return {"passed": True, "published_lambda": reference, "upper_rmse_delta": 0.0,
                "rmse_delta": 0.0, "maximum_season_rmse_delta": 0.0}
    names = [f"logit_{selected}", f"logit_{reference}"]
    point, boot = paired_rmse(later, names, contract)
    upper = float(np.quantile(boot[:, 0] - boot[:, 1], contract["bootstrap_quantile"]))
    folds = _metrics(later.loc[later.candidate.isin(names)]).pivot(index="outcome_season", columns="candidate", values="rmse")
    worst = float((folds[names[0]] - folds[names[1]]).max())
    passed = upper <= contract["rmse_tolerance"] and worst <= contract["max_season_rmse_deterioration"]
    return {"passed": bool(passed), "published_lambda": selected if passed else reference,
            "rmse_delta": float(point[0] - point[1]), "upper_rmse_delta": upper,
            "maximum_season_rmse_delta": worst}


def fit(design, start: int, end: int, off: int, defense: int, home: int):
    mask = (design.seasons >= start) & (design.seasons <= end)
    beta, intercept = fit_coefficients(design, RapmConfig(
        seasons=tuple(range(start, end + 1)), lambda_off=off, lambda_def=defense,
        lambda_home=home, data_scope="corrected_progress_wp"), row_mask=mask)
    if not np.isfinite(beta).all() or not np.isfinite(intercept):
        raise ValueError("Nonfinite fit.")
    return beta, intercept


def public_table(design, names, start, end, off, defense, home):
    beta, _ = fit(design, start, end, off, defense, home)
    table = ratings_table(design, beta, names=names)
    mask = (design.seasons >= start) & (design.seasons <= end)
    n = len(design.players)
    counts = pd.DataFrame({"player_id": design.players,
                           "off_possessions": np.asarray(design.X[mask, :n].sum(axis=0)).ravel(),
                           "def_possessions": np.asarray(design.X[mask, n:2*n].sum(axis=0)).ravel()})
    table = table.drop(columns=["off_possessions", "def_possessions"]).merge(counts, on="player_id", validate="one_to_one")
    table = table.loc[table[["off_possessions", "def_possessions"]].min(axis=1).gt(0)].copy()
    if table.player_name.isna().any():
        raise ValueError("Public WP ratings contain unnamed players.")
    table["Season"], table["window_start"], table["window_end"] = end, start, end
    return table


def run(*, pilot: bool = False):
    started = time.perf_counter()
    contract = json.loads(CONTRACT.read_text())
    if pilot:
        target, quality = corrected_target(contract, pilot=True)
        design = build_design(target, include_home=True)
        fit_started = time.perf_counter()
        fit(design, 2014, 2014, 100, 100, 300)
        print(json.dumps({"pilot_fit_seconds": time.perf_counter() - fit_started,
                          "total_seconds": time.perf_counter() - started, "games": quality["games"]}))
        return
    paths = [CONTRACT, SOURCE, PULSE / "validation_games.parquet", PULSE / "ratings.parquet",
             PULSE / "run.json", Path(__file__),
             ROOT / "research/rapm_lab/run_log_odds_wp_rapm_lambda.py",
             ROOT / "research/rapm_lab/run_rolling_5y_wp_rapm.py",
             ROOT / "src/nba_impact/models/win_probability_rapm.py", ROOT / "src/nba_impact/models/rapm.py"]
    paths += [ROOT / f"rapm/data/possession_cache/matchups_{season}.parquet" for season in range(1997, 2024)]
    paths += [ROOT / "data/lake/silver/possessions.parquet", ROOT / "data/lake/silver/possession_lineup_segments.parquet"]
    for season in range(1997, 2027):
        path = SCORES / f"project_season={season}/regular.parquet"
        paths += [path, path.with_suffix(".parquet.manifest.json")]
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"wp_chronology_release_v2_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic({"config": contract, "hashes": hashes, "status": "running"}, output / "inputs.json")
    target, quality = corrected_target(contract)
    design = build_design(target, include_home=True)
    # Preserve only the corrected target in this new immutable run, never the old checkpoint.
    target.to_parquet(output / "corrected_target.parquet", index=False)
    pulse = pd.read_parquet(PULSE / "validation_games.parquet")
    actual = pulse.loc[pulse.candidate.eq("pulse"), ["outcome_season", "game_id"]].copy()
    scores = official_scores(2014, 2026).rename(columns={"season": "outcome_season", "gameid": "game_id"})
    actual = actual.merge(scores[["outcome_season", "game_id", "actual_margin"]],
                          on=["outcome_season", "game_id"], how="left", validate="one_to_one")
    if not np.isfinite(actual.actual_margin).all():
        raise ValueError("Validation games are missing official scores.")
    actual["game_id"] = actual.game_id.astype(str)
    if actual.duplicated(["outcome_season", "game_id"]).any():
        raise ValueError("PULSE actual game keys must be unique.")
    keys = target[["season", "gameid"]].drop_duplicates().rename(columns={"season": "outcome_season", "gameid": "game_id"})
    keys["game_id"] = keys.game_id.astype(str)
    coverage = actual.merge(keys, on=["outcome_season", "game_id"], how="outer", indicator=True, validate="one_to_one")
    coverage = coverage.loc[coverage.outcome_season.between(2015, 2026)]
    excluded = coverage.loc[coverage._merge.ne("both")].copy()
    excluded["reason"] = excluded._merge.astype(str)
    excluded.drop(columns="_merge").to_parquet(output / "excluded_games.parquet", index=False)
    common = coverage.loc[coverage._merge.eq("both"), actual.columns].sort_values(["outcome_season", "game_id"])
    raw_design = replace(design, y=target.offense_wp_change.to_numpy())
    frames = []
    specifications = [(f"logit_{value}", design, value, value) for value in contract["shared_lambdas"]]
    specifications.append(("raw_wp", raw_design, contract["raw_wp"]["lambda_offense"], contract["raw_wp"]["lambda_defense"]))
    for candidate, candidate_design, off, defense in specifications:
        for season in range(2014, 2026):
            beta, intercept = fit(candidate_design, season, season, off, defense, contract["lambda_home"])
            game = _game_predictions(candidate_design, beta, intercept, candidate_design.seasons == season + 1)
            required = common.loc[common.outcome_season.eq(season + 1)]
            game = required.merge(game, on="game_id", how="left", validate="one_to_one")
            if not np.isfinite(game[["actual_margin", "raw_prediction"]]).all().all():
                raise ValueError("WP candidate lost common games.")
            game["candidate"] = candidate
            frames.append(game)
        print(f"fitted {candidate}", flush=True)
    for candidate in ("pulse", "rapm"):
        part = pulse.loc[pulse.candidate.eq(candidate), ["outcome_season", "game_id", "predicted_margin"]].rename(columns={"predicted_margin": "raw_prediction"})
        part["game_id"] = part.game_id.astype(str)
        part = common.merge(part, on=["outcome_season", "game_id"], how="left", validate="one_to_one")
        part["candidate"] = candidate
        if not np.isfinite(part.raw_prediction).all():
            raise ValueError("Reference candidate lost common games.")
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)
    scored = contract["development_outcomes"] + contract["diagnostic_outcomes"]
    predictions = _past_only_predictions(raw, scored)
    selection = select_lambda(predictions, contract)
    write_json_atomic(selection, output / "frozen_selection.json")
    print(f"frozen development lambda: {selection['selected_lambda']}", flush=True)
    gate = publication_gate(predictions, selection["selected_lambda"], contract)
    published = gate["published_lambda"]
    names = pd.read_parquet(PULSE / "ratings.parquet", columns=["PLAYER_ID", "PLAYER_NAME", "Season"])
    names = names.sort_values("Season").drop_duplicates("PLAYER_ID", keep="last")[["PLAYER_ID", "PLAYER_NAME"]]
    logit_tables, raw_tables = [], []
    for season in contract["publication_seasons"]:
        logit_tables.append(public_table(design, names, season, season, published, published, contract["lambda_home"]))
        settings = contract["raw_wp"]
        raw_tables.append(public_table(raw_design, names, season - settings["window"] + 1, season,
                                       settings["lambda_offense"], settings["lambda_defense"], contract["lambda_home"]))
    pd.concat(logit_tables).to_parquet(output / "public_logit_ratings.parquet", index=False)
    pd.concat(raw_tables).to_parquet(output / "public_raw_rolling_ratings.parquet", index=False)
    raw.to_parquet(output / "raw_game_predictions.parquet", index=False)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    folds = _metrics(predictions)
    folds.to_parquet(output / "fold_metrics.parquet", index=False)
    benchmark = folds.loc[folds.outcome_season.isin(contract["diagnostic_outcomes"]) & folds.candidate.isin([f"logit_{published}", "raw_wp", "pulse", "rapm"])]
    summary = benchmark.groupby("candidate").agg(mean_mse=("mse", "mean"), mean_correlation=("correlation", "mean"))
    summary["equal_season_rmse"] = np.sqrt(summary.mean_mse)
    summary.reset_index().to_parquet(output / "benchmark_summary.parquet", index=False)
    artifacts = {path.name: sha256_file(path) for path in output.glob("*.parquet")}
    record = {"run_id": output.name, "status": contract["status"], "created_at": datetime.now(timezone.utc).isoformat(),
              "config": contract, "input_hashes": hashes, "artifact_hashes": artifacts, "quality": quality,
              "selection": selection, "publication_gate": gate,
              "games_per_outcome": common.groupby("outcome_season").size().astype(int).to_dict(),
              "common_game_keys_sha256": hashlib.sha256(common[["outcome_season", "game_id"]].to_csv(index=False).encode()).hexdigest(),
              "excluded_games": len(excluded), "benchmark_summary": summary.reset_index().to_dict("records"),
              "solver": "Canonical CG rtol 1e-8, maxiter 10000, direct fallback; finite coefficient and prediction checks",
              "elapsed_seconds": time.perf_counter() - started,
              "supersedes": ["log_odds_wp_rapm_lambda_v1_91e7dccec4", "rolling_5y_wp_rapm_v1_39800d31b3", "wp_rapm_vs_pulse_v1_3d2995995c"]}
    if any(sha256_file(ROOT / path) != digest for path, digest in hashes.items()):
        raise ValueError("An input changed during the run; publication is blocked.")
    write_json_atomic(record, output / "run.json")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    run(pilot=parser.parse_args().pilot)
