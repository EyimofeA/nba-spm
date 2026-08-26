"""Four-arm shooting-luck RAPM experiment.

The experiment separates player-neutral shot opportunity from repeatable
shooter conversion.  Shot expectations never see the game they adjust.  The
chosen shot-model and empirical-Bayes settings use information through 2024;
2025 and 2026 remain reused diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.expected_shot_quality import _feature_frame
from nba_impact.models.possession_outcome_rapm import (
    assign_events_to_possessions,
    canonical_terminal_frame,
)
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficients,
    load_current_player_names,
    ratings_table,
)


ARMS = (
    "normal_realized_points",
    "opponent_luck_adjusted",
    "teammate_and_opponent_luck_adjusted",
    "full_expected_outcome",
)
CATEGORY_COLUMNS = {
    "ft": ("FTM", "FTA"),
    "rim": ("AtRimFGM", "AtRimFGA"),
    "short_mid": ("ShortMidRangeFGM", "ShortMidRangeFGA"),
    "long_mid": ("LongMidRangeFGM", "LongMidRangeFGA"),
    "corner_3": ("Corner3FGM", "Corner3FGA"),
    "above_break_3": ("Arc3FGM", "Arc3FGA"),
}


def load_contract(path: str | Path) -> dict:
    """Validate the frozen experiment before any data path is opened."""
    contract = yaml.safe_load(Path(path).read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": "luck_adjusted_rapm_spm_v1",
        "status": "preregistered_reused_diagnostic",
        "estimand_id": "luck_adjusted_player_impact_v1",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError(f"arms must be exactly {ARMS}.")
    policy = contract["season_policy"]
    if tuple(policy["diagnostic_seasons"]) != (2025, 2026):
        raise ValueError("Diagnostic seasons must remain 2025 and 2026.")
    if int(policy["forbidden_season"]) != 2027:
        raise ValueError("Season 2027 must remain forbidden.")
    referenced = [
        int(policy["expectation_development_end"]),
        *map(int, policy["diagnostic_seasons"]),
        *map(int, policy["final_training_seasons"]),
        int(policy["output_season"]),
    ]
    if max(referenced) > 2026:
        raise ValueError("Season 2027 must be rejected before reading any input.")
    if tuple(contract["data"]["shot_event_seasons"]) != (2024, 2025, 2026):
        raise ValueError("Shot-event seasons must remain 2024 through 2026.")
    return contract


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def _stable_game_fold(game_id: object, folds: int) -> int:
    value = hashlib.sha256(str(game_id).encode()).digest()
    return int.from_bytes(value[:8], "little") % folds


def load_annual_shooting(
    source_dir: str | Path,
    *,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    """Load only the numerator, denominator, and exposure fields we need."""
    columns = {
        "PLAYER_ID",
        "OffPoss",
        *(column for pair in CATEGORY_COLUMNS.values() for column in pair),
    }
    outputs: list[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        path = Path(source_dir) / f"{season}.csv"
        frame = pd.read_csv(path, usecols=sorted(columns), low_memory=False)
        frame = frame.drop_duplicates().copy()
        if frame["PLAYER_ID"].duplicated().any():
            raise ValueError(f"Annual shooting source has duplicate players in {season}.")
        frame["season"] = season
        for column in columns - {"PLAYER_ID"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def _half_life_weight(season: np.ndarray, test_season: int, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(season), dtype=float)
    return np.power(2.0, (season.astype(float) - (test_season - 1.0)) / half_life)


def history_rate_table(
    annual: pd.DataFrame,
    *,
    category: str,
    target_season: int,
    prior_attempts: float,
    half_life: float | None,
) -> tuple[pd.DataFrame, float]:
    """Return pre-season player EB rates; no target-season row is used."""
    makes_column, attempts_column = CATEGORY_COLUMNS[category]
    history = annual.loc[annual["season"].lt(target_season)].copy()
    if history.empty:
        raise ValueError(f"No shooting history precedes {target_season}.")
    weight = _half_life_weight(
        history["season"].to_numpy(dtype=float), target_season, half_life
    )
    history["weighted_makes"] = history[makes_column].to_numpy(dtype=float) * weight
    history["weighted_attempts"] = history[attempts_column].to_numpy(dtype=float) * weight
    league_attempts = float(history["weighted_attempts"].sum())
    if league_attempts <= 0:
        raise ValueError(f"No {category} attempts precede {target_season}.")
    league_rate = float(history["weighted_makes"].sum() / league_attempts)
    players = history.groupby("PLAYER_ID", as_index=False).agg(
        weighted_makes=("weighted_makes", "sum"),
        weighted_attempts=("weighted_attempts", "sum"),
    )
    players["expected_make"] = (
        players["weighted_makes"] + prior_attempts * league_rate
    ) / (players["weighted_attempts"] + prior_attempts)
    return players, league_rate


def _grouped_binary_scores(
    makes: np.ndarray,
    attempts: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float]:
    valid = attempts > 0
    makes = makes[valid].astype(float)
    attempts = attempts[valid].astype(float)
    prediction = np.clip(prediction[valid].astype(float), 1e-6, 1.0 - 1e-6)
    misses = attempts - makes
    total = float(attempts.sum())
    negative_log_likelihood = -float(
        (makes * np.log(prediction) + misses * np.log1p(-prediction)).sum() / total
    )
    brier = float(
        (makes * (1.0 - prediction) ** 2 + misses * prediction**2).sum() / total
    )
    return negative_log_likelihood, brier


def tune_shooting_history(
    annual: pd.DataFrame,
    *,
    categories: tuple[str, ...],
    selection_seasons: tuple[int, ...],
    prior_grid: tuple[float, ...],
    half_life_grid: tuple[float | None, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune EB settings on next-season binomial likelihood through 2024."""
    rows: list[dict] = []
    for category in categories:
        makes_column, attempts_column = CATEGORY_COLUMNS[category]
        for prior_attempts in prior_grid:
            for half_life in half_life_grid:
                for test_season in selection_seasons:
                    rates, league_rate = history_rate_table(
                        annual,
                        category=category,
                        target_season=test_season,
                        prior_attempts=prior_attempts,
                        half_life=half_life,
                    )
                    target = annual.loc[
                        annual["season"].eq(test_season),
                        ["PLAYER_ID", makes_column, attempts_column],
                    ].merge(
                        rates[["PLAYER_ID", "expected_make"]],
                        on="PLAYER_ID",
                        how="left",
                        validate="one_to_one",
                    )
                    target["expected_make"] = target["expected_make"].fillna(league_rate)
                    score, brier = _grouped_binary_scores(
                        target[makes_column].to_numpy(dtype=float),
                        target[attempts_column].to_numpy(dtype=float),
                        target["expected_make"].to_numpy(dtype=float),
                    )
                    rows.append(
                        {
                            "category": category,
                            "prior_attempts": float(prior_attempts),
                            "half_life_years": "none" if half_life is None else f"{half_life:g}",
                            "test_season": int(test_season),
                            "log_loss": score,
                            "brier": brier,
                            "attempts": float(target[attempts_column].sum()),
                        }
                    )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["category", "prior_attempts", "half_life_years"], as_index=False)
        .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"), folds=("test_season", "nunique"))
        .sort_values(
            ["category", "mean_log_loss", "mean_brier", "prior_attempts", "half_life_years"],
            kind="stable",
        )
    )
    selected = summary.groupby("category", as_index=False, sort=False).head(1).copy()
    selected["selected"] = True
    return folds, selected


def _fit_player_neutral_shot_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    c: float,
) -> np.ndarray:
    combined = pd.concat([train, score], ignore_index=True)
    features, _ = _feature_frame(combined)
    train_features = features[: len(train)]
    score_features = features[len(train) :]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    score_features = scaler.transform(score_features)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=300)
    model.fit(train_features, train["shot_made"].to_numpy(dtype=int))
    return model.predict_proba(score_features)[:, 1]


def select_shot_model_c(
    panel_2024: pd.DataFrame,
    candidates: tuple[float, ...],
    *,
    train_game_share: float,
) -> pd.DataFrame:
    """Use a chronological 2024 split; later seasons cannot select C."""
    if not 0.0 < train_game_share < 1.0:
        raise ValueError("train_game_share must be strictly between zero and one.")
    games = (
        panel_2024[["game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"], kind="stable")
    )
    split = max(
        1,
        min(len(games) - 1, int(np.floor(train_game_share * len(games)))),
    )
    train_ids = set(games.iloc[:split]["game_id"].astype(str))
    train = panel_2024.loc[panel_2024["game_id"].astype(str).isin(train_ids)].copy()
    validation = panel_2024.loc[~panel_2024["game_id"].astype(str).isin(train_ids)].copy()
    rows = []
    for c in candidates:
        prediction = _fit_player_neutral_shot_model(train, validation, c=c)
        actual = validation["shot_made"].to_numpy(dtype=int)
        rows.append(
            {
                "c": float(c),
                "train_games": int(train["game_id"].nunique()),
                "validation_games": int(validation["game_id"].nunique()),
                "validation_log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
                "validation_brier": float(np.mean((actual - prediction) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier", "c"], kind="stable"
    ).reset_index(drop=True)


def build_player_neutral_shot_predictions(
    panel: pd.DataFrame,
    *,
    selected_c: float,
    current_game_out_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit 2024 by game, then forecast 2025 and 2026 from prior seasons."""
    outputs: list[pd.DataFrame] = []
    metrics: list[dict] = []
    for season in (2024, 2025, 2026):
        score = panel.loc[panel["season_end"].eq(season)].copy()
        if season == 2024:
            score["fold"] = score["game_id"].map(
                lambda value: _stable_game_fold(value, current_game_out_folds)
            )
            predictions = np.full(len(score), np.nan, dtype=float)
            for fold in range(current_game_out_folds):
                test_mask = score["fold"].eq(fold).to_numpy()
                train = score.loc[~test_mask]
                predictions[test_mask] = _fit_player_neutral_shot_model(
                    train, score.loc[test_mask], c=selected_c
                )
            evidence = "whole_game_cross_fit_within_2024"
        else:
            train = panel.loc[panel["season_end"].lt(season)].copy()
            predictions = _fit_player_neutral_shot_model(train, score, c=selected_c)
            evidence = f"trained_through_{season - 1}"
        if not np.isfinite(predictions).all():
            raise ValueError(f"Expected-shot predictions are incomplete for {season}.")
        actual = score["shot_made"].to_numpy(dtype=int)
        scored = score[
            ["shot_id", "game_id", "actionNumber", "season_end", "shooter_id", "shot_zone", "shot_value", "shot_made"]
        ].copy()
        scored["neutral_expected_make"] = predictions
        scored["expectation_evidence"] = evidence
        outputs.append(scored)
        metrics.append(
            {
                "season": season,
                "shots": int(len(score)),
                "games": int(score["game_id"].nunique()),
                "log_loss": float(log_loss(actual, predictions, labels=[0, 1])),
                "brier": float(np.mean((actual - predictions) ** 2)),
                "actual_make_rate": float(actual.mean()),
                "predicted_make_rate": float(predictions.mean()),
                "evidence": evidence,
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(metrics)


def add_shooter_skill_expectations(
    shots: pd.DataFrame,
    annual: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add only repeatable pre-season shooter conversion above neutral context."""
    output = shots.copy()
    output["category"] = output["shot_zone"].astype(str)
    output["skill_expected_make"] = np.nan
    lineage: list[dict] = []
    settings = selected.set_index("category")
    for season in (2024, 2025, 2026):
        for category in tuple(value for value in CATEGORY_COLUMNS if value != "ft"):
            setting = settings.loc[category]
            half_life_label = str(setting["half_life_years"])
            half_life = None if half_life_label == "none" else float(half_life_label)
            rates, league = history_rate_table(
                annual,
                category=category,
                target_season=season,
                prior_attempts=float(setting["prior_attempts"]),
                half_life=half_life,
            )
            rate_map = rates.set_index("PLAYER_ID")["expected_make"]
            mask = output["season_end"].eq(season) & output["category"].eq(category)
            player_rate = output.loc[mask, "shooter_id"].map(rate_map).fillna(league)
            neutral = output.loc[mask, "neutral_expected_make"].to_numpy(dtype=float)
            output.loc[mask, "skill_expected_make"] = np.clip(
                neutral + player_rate.to_numpy(dtype=float) - league,
                0.01,
                0.99,
            )
            lineage.append(
                {
                    "season": season,
                    "category": category,
                    "league_rate": league,
                    "players_with_history": int(len(rates)),
                    "prior_attempts": float(setting["prior_attempts"]),
                    "half_life_years": half_life_label,
                }
            )
    if output["skill_expected_make"].isna().any():
        missing = output.loc[output["skill_expected_make"].isna(), "category"].value_counts()
        raise ValueError(f"Unsupported shot zones in skill expectation: {missing.to_dict()}.")
    return output, pd.DataFrame(lineage)


def _free_throw_ledger(
    base: pd.DataFrame,
    events: pd.DataFrame,
    annual: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    free_throws = events.loc[
        events["season_type"].eq("regular")
        & events["season_end"].isin((2024, 2025, 2026))
        & events["actionType"].eq("Free Throw")
    ].copy()
    mapped = assign_events_to_possessions(base, free_throws).merge(
        base[["possession_id", "offense_team_id"]],
        on="possession_id",
        validate="many_to_one",
    )
    mapped = mapped.loc[
        pd.to_numeric(mapped["teamId"], errors="coerce").eq(mapped["offense_team_id"])
        & pd.to_numeric(mapped["personId"], errors="coerce").notna()
    ].copy()
    points = pd.to_numeric(mapped["points_added"], errors="coerce")
    inferred = ~mapped["description"].astype("string").str.contains(
        "MISS", case=False, na=False
    )
    mapped["actual_points"] = points.where(points.isin([0.0, 1.0]), inferred.astype(float))
    mapped["shooter_id"] = pd.to_numeric(mapped["personId"], errors="raise").astype(int)
    settings = selected.set_index("category").loc["ft"]
    half_life_label = str(settings["half_life_years"])
    half_life = None if half_life_label == "none" else float(half_life_label)
    lineage = []
    pieces = []
    for season in (2024, 2025, 2026):
        rates, league = history_rate_table(
            annual,
            category="ft",
            target_season=season,
            prior_attempts=float(settings["prior_attempts"]),
            half_life=half_life,
        )
        rate_map = rates.set_index("PLAYER_ID")["expected_make"]
        block = mapped.loc[mapped["season_end"].eq(season)].copy()
        block["neutral_expected_points"] = league
        block["skill_expected_points"] = block["shooter_id"].map(rate_map).fillna(league)
        pieces.append(block)
        lineage.append(
            {
                "season": season,
                "category": "ft",
                "league_rate": league,
                "players_with_history": int(len(rates)),
                "prior_attempts": float(settings["prior_attempts"]),
                "half_life_years": half_life_label,
            }
        )
    ledger = pd.concat(pieces, ignore_index=True)
    ledger["category"] = "ft"
    return ledger[
        ["possession_id", "game_id", "season_end", "shooter_id", "category", "actual_points", "neutral_expected_points", "skill_expected_points"]
    ], pd.DataFrame(lineage)


def build_conversion_ledger(
    base: pd.DataFrame,
    expected_shots: pd.DataFrame,
    events: pd.DataFrame,
    annual: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shots, shot_lineage = add_shooter_skill_expectations(expected_shots, annual, selected)
    mapped = assign_events_to_possessions(base, shots)
    mapped["actual_points"] = mapped["shot_value"].to_numpy(dtype=float) * mapped["shot_made"].to_numpy(dtype=float)
    mapped["neutral_expected_points"] = mapped["shot_value"].to_numpy(dtype=float) * mapped["neutral_expected_make"].to_numpy(dtype=float)
    mapped["skill_expected_points"] = mapped["shot_value"].to_numpy(dtype=float) * mapped["skill_expected_make"].to_numpy(dtype=float)
    field_goals = mapped[
        ["possession_id", "game_id", "season_end", "shooter_id", "category", "actual_points", "neutral_expected_points", "skill_expected_points"]
    ].copy()
    free_throws, ft_lineage = _free_throw_ledger(base, events, annual, selected)
    ledger = pd.concat([field_goals, free_throws], ignore_index=True)
    if ledger.duplicated(["game_id", "possession_id", "shooter_id", "category", "actual_points", "neutral_expected_points"]).any():
        # Multiple identical free throws or shots in one possession are valid;
        # only exact event identity was removed upstream, so do not deduplicate.
        pass
    return ledger, pd.concat([shot_lineage, ft_lineage], ignore_index=True)


def build_expected_outcome_frame(base: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    conversion = ledger.groupby("possession_id", as_index=False).agg(
        actual_conversion_points=("actual_points", "sum"),
        neutral_expected_conversion_points=("neutral_expected_points", "sum"),
        skill_expected_conversion_points=("skill_expected_points", "sum"),
    )
    output = base.merge(conversion, on="possession_id", how="left", validate="one_to_one")
    columns = [
        "actual_conversion_points",
        "neutral_expected_conversion_points",
        "skill_expected_conversion_points",
    ]
    output[columns] = output[columns].fillna(0.0)
    output["expected_pts"] = (
        output["pts"]
        - output["actual_conversion_points"]
        + output["neutral_expected_conversion_points"]
    )
    if not np.isfinite(output["expected_pts"]).all():
        raise ValueError("Expected-outcome possession target contains nonfinite values.")
    return output


def shooter_skill_bonus(
    design,
    ledger: pd.DataFrame,
    *,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Return offense coefficient add-backs in points per possession."""
    train_games = set(design.game_ids[train_mask])
    source = ledger.loc[ledger["game_id"].astype(str).isin(train_games)].copy()
    source["skill_delta"] = source["skill_expected_points"] - source["neutral_expected_points"]
    points = source.groupby("shooter_id")["skill_delta"].sum()
    exposures = np.asarray(design.X[train_mask, : len(design.players)].sum(axis=0)).ravel()
    bonus = np.divide(
        pd.Series(design.players).map(points).fillna(0.0).to_numpy(dtype=float),
        exposures,
        out=np.zeros(len(design.players), dtype=float),
        where=exposures > 0,
    )
    mean = float(np.average(bonus, weights=exposures))
    return bonus - mean


def compose_arm_beta(
    normal_beta: np.ndarray,
    expected_beta: np.ndarray,
    *,
    n_players: int,
    arm: str,
    shooting_bonus: np.ndarray,
) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(f"Unknown luck arm {arm}.")
    if arm == "normal_realized_points":
        return normal_beta.copy()
    if arm == "opponent_luck_adjusted":
        return np.concatenate(
            [normal_beta[:n_players], expected_beta[n_players : 2 * n_players], normal_beta[-1:]]
        )
    offense = expected_beta[:n_players].copy()
    if arm == "teammate_and_opponent_luck_adjusted":
        offense += shooting_bonus
    return np.concatenate([offense, expected_beta[n_players : 2 * n_players], expected_beta[-1:]])


def _game_metrics(games: pd.DataFrame) -> dict:
    actual = games["actual_margin"].to_numpy(dtype=float)
    predicted = games["predicted_margin"].to_numpy(dtype=float)
    error = predicted - actual
    variance = float(np.var(predicted))
    slope = float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance > 0 else np.nan
    return {
        "games": int(len(games)),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "calibration_slope": slope,
        "calibration_intercept": float(actual.mean() - slope * predicted.mean()),
        "predicted_margin_sd": float(predicted.std(ddof=0)),
    }


def _rating_components(design, beta: np.ndarray, names: pd.DataFrame, arm: str, training_end: int) -> pd.DataFrame:
    table = ratings_table(design, beta, names=names).rename(
        columns={
            "offense_per_100": "offense",
            "defense_per_100": "defense",
            "net_per_100": "net",
        }
    )
    table["arm"] = arm
    table["training_end"] = training_end
    return table


def run_luck_adjusted_rapm(
    contract_path: str | Path,
    *,
    project_root: str | Path,
) -> dict:
    """Run the frozen target experiment and save a resumable research artifact."""
    started = time.perf_counter()
    root = Path(project_root)
    contract_path = Path(contract_path)
    contract = load_contract(contract_path)
    paths = {
        "annual": root / "data/raw/playersheets/year_totals",
        "shots": root / "data/lake/silver/shot_defense_events.parquet",
        "events": root / "data/lake/silver/event_states.parquet",
        "possessions": root / "data/lake/silver/possessions.parquet",
        "segments": root / "data/lake/silver/possession_lineup_segments.parquet",
    }
    identity_config = {
        "contract_sha256": sha256_file(contract_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "shot_sha256": sha256_file(paths["shots"]),
        "event_sha256": sha256_file(paths["events"]),
        "possession_sha256": sha256_file(paths["possessions"]),
        "segment_sha256": sha256_file(paths["segments"]),
        "annual_source_hashes": {
            str(season): sha256_file(paths["annual"] / f"{season}.csv")
            for season in range(2014, 2026)
        },
    }
    identity = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode()).hexdigest()[:10]
    output = root / "artifacts/models/luck_adjusted_rapm" / f"luck_adjusted_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    annual = load_annual_shooting(paths["annual"], start_season=2014, end_season=2025)
    tuning_path = checkpoints / "shooting_history_folds.parquet"
    selected_path = checkpoints / "shooting_history_selected.parquet"
    if tuning_path.exists() and selected_path.exists():
        tuning = pd.read_parquet(tuning_path)
        selected = pd.read_parquet(selected_path)
    else:
        history = contract["shooting_skill"]
        tuning, selected = tune_shooting_history(
            annual,
            categories=tuple(history["categories"]),
            selection_seasons=tuple(map(int, history["selection_seasons"])),
            prior_grid=tuple(map(float, history["prior_attempt_grid"])),
            half_life_grid=tuple(
                None if value is None else float(value) for value in history["half_life_grid"]
            ),
        )
        _atomic_parquet(tuning, tuning_path)
        _atomic_parquet(selected, selected_path)

    shot_panel = pd.read_parquet(paths["shots"])
    c_metrics_path = checkpoints / "shot_model_c_selection.parquet"
    if c_metrics_path.exists():
        c_metrics = pd.read_parquet(c_metrics_path)
    else:
        c_metrics = select_shot_model_c(
            shot_panel.loc[shot_panel["season_end"].eq(2024)].copy(),
            tuple(map(float, contract["shot_expectation"]["logistic_c_grid"])),
            train_game_share=float(
                contract["shot_expectation"]["tuning_train_game_share"]
            ),
        )
        _atomic_parquet(c_metrics, c_metrics_path)
    selected_c = float(c_metrics.iloc[0]["c"])
    shot_predictions_path = checkpoints / "expected_field_goals.parquet"
    shot_metrics_path = checkpoints / "expected_field_goal_metrics.parquet"
    if shot_predictions_path.exists() and shot_metrics_path.exists():
        expected_shots = pd.read_parquet(shot_predictions_path)
        shot_metrics = pd.read_parquet(shot_metrics_path)
    else:
        expected_shots, shot_metrics = build_player_neutral_shot_predictions(
            shot_panel,
            selected_c=selected_c,
            current_game_out_folds=int(contract["shot_expectation"]["current_game_out_folds"]),
        )
        _atomic_parquet(expected_shots, shot_predictions_path)
        _atomic_parquet(shot_metrics, shot_metrics_path)

    possessions = pd.read_parquet(paths["possessions"])
    segments = pd.read_parquet(paths["segments"])
    events = pd.read_parquet(paths["events"])
    base = canonical_terminal_frame(possessions, segments, seasons=(2024, 2025, 2026))
    ledger_path = checkpoints / "conversion_ledger.parquet"
    lineage_path = checkpoints / "expectation_lineage.parquet"
    if ledger_path.exists() and lineage_path.exists():
        ledger = pd.read_parquet(ledger_path)
        expectation_lineage = pd.read_parquet(lineage_path)
    else:
        ledger, expectation_lineage = build_conversion_ledger(
            base, expected_shots, events, annual, selected
        )
        _atomic_parquet(ledger, ledger_path)
        _atomic_parquet(expectation_lineage, lineage_path)
    adjusted = build_expected_outcome_frame(base, ledger)
    normal_design = build_design(base, include_home=True)
    expected_design = build_design(adjusted.assign(pts=adjusted["expected_pts"]), include_home=True)
    if not np.array_equal(normal_design.players, expected_design.players):
        raise ValueError("Normal and expected-outcome designs must use identical players.")

    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=float(contract["rapm"]["lambda_off"]),
        lambda_def=float(contract["rapm"]["lambda_def"]),
        lambda_home=float(contract["rapm"]["lambda_home"]),
        data_scope="luck_adjusted_rapm_spm_v1",
    )
    names = load_current_player_names(
        root / "rapm/data/all_names.csv", root / "data/lake/silver/player_games.parquet"
    )
    metric_rows: list[dict] = []
    game_rows: list[pd.DataFrame] = []
    rating_rows: list[pd.DataFrame] = []
    for test_season in (2025, 2026):
        train_mask = normal_design.seasons < test_season
        test_mask = normal_design.seasons == test_season
        normal_beta, normal_intercept = fit_coefficients(normal_design, config, row_mask=train_mask)
        expected_beta, expected_intercept = fit_coefficients(expected_design, config, row_mask=train_mask)
        bonus = shooter_skill_bonus(normal_design, ledger, train_mask=train_mask)
        for arm in ARMS:
            beta = compose_arm_beta(
                normal_beta,
                expected_beta,
                n_players=len(normal_design.players),
                arm=arm,
                shooting_bonus=bonus,
            )
            intercept = normal_intercept if arm in {"normal_realized_points", "opponent_luck_adjusted"} else expected_intercept
            games = _game_margin_frame(normal_design, beta, intercept, test_mask, train_mask)
            games["arm"] = arm
            games["test_season"] = test_season
            game_rows.append(games)
            metric_rows.append({"arm": arm, "test_season": test_season, **_game_metrics(games)})
            rating_rows.append(_rating_components(normal_design, beta, names, arm, test_season - 1))

    # Descriptive 2026 ratings use exactly 2024-25 evidence.
    final_train = normal_design.seasons <= 2025
    normal_beta, _ = fit_coefficients(normal_design, config, row_mask=final_train)
    expected_beta, _ = fit_coefficients(expected_design, config, row_mask=final_train)
    bonus = shooter_skill_bonus(normal_design, ledger, train_mask=final_train)
    final_ratings = []
    for arm in ARMS:
        beta = compose_arm_beta(
            normal_beta,
            expected_beta,
            n_players=len(normal_design.players),
            arm=arm,
            shooting_bonus=bonus,
        )
        final_ratings.append(_rating_components(normal_design, beta, names, arm, 2025))

    metrics = pd.DataFrame(metric_rows)
    games = pd.concat(game_rows, ignore_index=True)
    ratings = pd.concat(final_ratings, ignore_index=True)
    folds_ratings = pd.concat(rating_rows, ignore_index=True)
    _atomic_parquet(metrics, output / "future_game_metrics.parquet")
    _atomic_parquet(games, output / "future_game_predictions.parquet")
    _atomic_parquet(ratings, output / "ratings_2026.parquet")
    _atomic_parquet(folds_ratings, output / "fold_ratings.parquet")
    _atomic_parquet(tuning, output / "shooting_history_folds.parquet")
    _atomic_parquet(selected, output / "shooting_history_selected.parquet")
    _atomic_parquet(c_metrics, output / "shot_model_c_selection.parquet")
    _atomic_parquet(shot_metrics, output / "expected_field_goal_metrics.parquet")
    _atomic_parquet(expectation_lineage, output / "expectation_lineage.parquet")

    normal = ratings.loc[ratings["arm"].eq("normal_realized_points"), ["player_id", "offense", "defense", "net"]]
    movement_rows = []
    for arm in ARMS[1:]:
        challenger = ratings.loc[ratings["arm"].eq(arm)].merge(
            normal, on="player_id", suffixes=("", "_normal"), validate="one_to_one"
        )
        for component in ("offense", "defense", "net"):
            delta = challenger[component] - challenger[f"{component}_normal"]
            movement_rows.append(
                {
                    "arm": arm,
                    "component": component,
                    "correlation_with_normal": float(challenger[[component, f"{component}_normal"]].corr().iloc[0, 1]),
                    "mean_absolute_movement": float(delta.abs().mean()),
                    "maximum_absolute_movement": float(delta.abs().max()),
                }
            )
    movement = pd.DataFrame(movement_rows)
    _atomic_parquet(movement, output / "rating_movement.parquet")

    run = {
        "run_id": output.name,
        "status": "research_target_experiment_complete",
        "evidence_status": "development_through_2024_reused_2025_2026_diagnostics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": contract["estimand_id"],
        "identity": identity_config,
        "selected_shot_model_c": selected_c,
        "quality": {
            "possessions": int(len(base)),
            "games": int(base["game_id"].nunique()),
            "field_goal_attempts": int(expected_shots.shape[0]),
            "conversion_events_mapped": int(len(ledger)),
            "conversion_actual_points_share": float(
                ledger["actual_points"].sum() / base["pts"].sum()
            ),
        },
        "diagnostic_metrics": metrics.to_dict(orient="records"),
        "paths": {
            "future_game_metrics": "future_game_metrics.parquet",
            "future_game_predictions": "future_game_predictions.parquet",
            "ratings_2026": "ratings_2026.parquet",
            "rating_movement": "rating_movement.parquet",
            "shooting_history_selected": "shooting_history_selected.parquet",
            "shot_model_c_selection": "shot_model_c_selection.parquet",
        },
        "spm_status": (
            "not_fit: only 2024 and 2025 possess shot-level expected-outcome labels before the 2026 output; "
            "that is insufficient for chronological SPM selection"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "forbidden_interpretation": (
            "Production RAPM, confirmed luck adjustment, causal shooting or defense, "
            "or a validated luck-adjusted SPM."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run
