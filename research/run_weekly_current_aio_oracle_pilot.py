"""Package one frozen 2024-26 weekly oracle evaluation of current AIO."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.predictive_current_aio import (
    _game_metrics,
    build_weekly_cutoff_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "weekly_current_aio_oracle_pilot_v1"
TEST_SEASONS = (2024, 2025, 2026)
SELECTED_ARM = "selected_decay_spm_prior_aio"
CONTRACT = ROOT / "research/experiments/weekly_current_aio_oracle_pilot_v1.yml"
VALIDATION_CONTRACT = ROOT / "research/experiments/impact_validation_v2.yml"
SOURCE_RUN = (
    ROOT
    / "artifacts/models/predictive_current_aio"
    / "predictive_current_aio_2026_v1_c18e2472ec"
)
PRIORS = (
    ROOT
    / "artifacts/models/predictive_spm_trajectory_ablation"
    / "predictive_spm_trajectory_ablation_v1_8d310a2ad6"
    / "selected_predictions.parquet"
)
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
OUTPUT_ROOT = ROOT / "artifacts/research/weekly_current_aio_oracle"


def load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    validation = yaml.safe_load(VALIDATION_CONTRACT.read_text())
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_mechanical_reused_diagnostic",
        "validation_contract": "impact_validation_v2",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    source = contract["source_run"]
    evaluation = contract["evaluation"]
    if tuple(source["test_seasons"]) != TEST_SEASONS:
        raise ValueError("Pilot seasons must remain 2024 through 2026.")
    if source["arm"] != SELECTED_ARM or int(source["half_life_years"]) != 2:
        raise ValueError("Pilot must reuse the frozen selected AIO arm.")
    if source["same_season_update"] != "none_in_mechanical_pilot":
        raise ValueError("This pilot cannot introduce a same-season update rule.")
    if int(evaluation["horizon_days"]) != 14:
        raise ValueError("The frozen oracle horizon is 14 days.")
    current = validation["current_strength"]
    if (
        current["cutoffs"] != "weekly Mondays from November 1 through April 1"
        or int(str(current["horizon"]).split()[1]) != 14
        or current["primary_test"]["name"] != "oracle-lineup player-quality test"
    ):
        raise ValueError("impact_validation_v2 no longer matches the pilot.")
    return contract


def game_dates_from_possessions(possessions: pd.DataFrame) -> pd.DataFrame:
    required = {"season_end", "season_type", "game_date", "game_id"}
    if missing := sorted(required - set(possessions.columns)):
        raise ValueError(f"Possessions are missing {missing}.")
    regular = possessions.loc[
        possessions["season_type"].eq("regular")
        & possessions["season_end"].isin(TEST_SEASONS),
        ["season_end", "game_date", "game_id"],
    ].copy()
    regular["game_date"] = pd.to_datetime(
        regular["game_date"], errors="raise"
    ).dt.normalize()
    if (
        regular.groupby(["season_end", "game_id"])["game_date"].nunique() > 1
    ).any():
        raise ValueError("Each source game must have one date.")
    return regular.groupby(
        ["season_end", "game_id"], as_index=False, sort=False
    ).agg(game_date=("game_date", "first"))


def expand_weekly_oracle_predictions(
    predictions: pd.DataFrame,
    game_dates: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.merge(
        game_dates,
        left_on=["test_season", "game_id"],
        right_on=["season_end", "game_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["game_date"].isna().any():
        raise ValueError("Every scored game must resolve to one source date.")
    frames: list[pd.DataFrame] = []
    for cutoff in ledger.itertuples(index=False):
        window = merged.loc[
            merged["test_season"].eq(cutoff.target_season)
            & merged["game_date"].ge(cutoff.cutoff_date)
            & merged["game_date"].lt(cutoff.horizon_end_exclusive)
        ].copy()
        rowset_hash = hashlib.sha256(
            "\n".join(sorted(window["game_id"].astype(str))).encode()
        ).hexdigest()
        if len(window) != cutoff.oracle_games or rowset_hash != cutoff.oracle_game_rowset_hash:
            raise ValueError("Saved predictions and cutoff ledger use different games.")
        window["cutoff_date"] = cutoff.cutoff_date
        window["horizon_end_exclusive"] = cutoff.horizon_end_exclusive
        window["snapshot_id"] = (
            f"{cutoff.target_season}_{cutoff.cutoff_date:%Y%m%d}"
        )
        window["rating_information_end_season"] = cutoff.target_season - 1
        frames.append(window)
    output = pd.concat(frames, ignore_index=True).drop(columns="season_end")
    if output.duplicated(["test_season", "cutoff_date", "game_id"]).any():
        raise ValueError("Weekly oracle prediction keys must be unique.")
    return output


def evidence_backed_ratings(
    ratings: pd.DataFrame,
    player_games: pd.DataFrame,
    priors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = player_games.loc[
        player_games["season_end"].isin(TEST_SEASONS)
        & player_games["season_type"].eq("regular")
        & player_games["played"].fillna(False),
        ["season_end", "player_id"],
    ].drop_duplicates()
    played = played.rename(
        columns={"season_end": "target_season", "player_id": "PLAYER_ID"}
    )
    selected = ratings.loc[
        ratings["target_season"].isin(TEST_SEASONS)
        & ratings["arm"].eq(SELECTED_ARM)
    ].copy()
    population = played.merge(
        selected,
        on=["target_season", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
    )
    if population["arm"].isna().any():
        raise ValueError("Every active player must resolve to the saved AIO design.")
    prior_ids = priors.loc[
        priors["Target_Season"].isin(TEST_SEASONS), ["Target_Season", "PLAYER_ID"]
    ].drop_duplicates()
    prior_ids["has_predictive_spm_prior"] = True
    population = population.merge(
        prior_ids,
        left_on=["target_season", "PLAYER_ID"],
        right_on=["Target_Season", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
    ).drop(columns="Target_Season")
    population["has_predictive_spm_prior"] = population[
        "has_predictive_spm_prior"
    ].fillna(False)
    population["has_history"] = population["Poss_Off"].gt(0) & population[
        "Poss_Def"
    ].gt(0)
    population["research_display_eligible"] = population["has_history"] | population[
        "has_predictive_spm_prior"
    ]
    population["rating_status"] = "preseason_frozen_reused_oracle_research"
    audit = (
        population.groupby("target_season", as_index=False)
        .agg(
            active_players=("PLAYER_ID", "size"),
            history_players=("has_history", "sum"),
            prior_players=("has_predictive_spm_prior", "sum"),
            eligible_players=("research_display_eligible", "sum"),
            names_present=("PLAYER_NAME", "count"),
        )
        .sort_values("target_season", kind="stable")
    )
    audit["excluded_no_history_or_prior"] = (
        audit["active_players"] - audit["eligible_players"]
    )
    eligible = population.loc[population["research_display_eligible"]].copy()
    if eligible["PLAYER_NAME"].isna().any():
        raise ValueError("Eligible ratings require player names.")
    return eligible.sort_values(
        ["target_season", "net"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True), audit.reset_index(drop=True)


def metric_frames(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (season, cutoff), games in predictions.groupby(
        ["test_season", "cutoff_date"], sort=True
    ):
        rows.append(
            {
                "test_season": int(season),
                "cutoff_date": cutoff,
                "horizon_end_exclusive": games["horizon_end_exclusive"].iloc[0],
                **_game_metrics(games),
            }
        )
    cutoffs = pd.DataFrame(rows)
    seasons = cutoffs.groupby("test_season", as_index=False).agg(
        cutoffs=("cutoff_date", "nunique"),
        mean_cutoff_margin_rmse=("margin_rmse", "mean"),
        mean_cutoff_margin_correlation=("margin_correlation", "mean"),
        mean_cutoff_calibration_slope=("calibration_slope", "mean"),
    )
    instances = predictions.groupby("test_season").size()
    unique_games = predictions.groupby("test_season")["game_id"].nunique()
    seasons["snapshot_game_instances"] = seasons["test_season"].map(instances)
    seasons["unique_games"] = seasons["test_season"].map(unique_games)
    return cutoffs, seasons


def main() -> None:
    started = time.perf_counter()
    contract = load_contract()
    source_run = json.loads((SOURCE_RUN / "run.json").read_text())
    if (
        source_run["run_id"] != contract["source_run"]["run_id"]
        or source_run["selected_arm"] != SELECTED_ARM
        or source_run["selected_half_life_years"] != "2"
    ):
        raise ValueError("Saved current-AIO run does not match the pilot contract.")

    possessions = pd.read_parquet(
        POSSESSIONS,
        columns=["season_end", "season_type", "game_date", "game_id"],
    )
    if int(possessions["season_end"].max()) > 2026:
        raise ValueError("Season 2027 is forbidden in this pilot.")
    game_dates = game_dates_from_possessions(possessions)
    ledger_input = possessions.rename(
        columns={"season_end": "season", "game_date": "date", "game_id": "gameid"}
    )
    ledger_input = ledger_input.loc[ledger_input["season_type"].eq("regular")]
    ledger = pd.concat(
        [
            build_weekly_cutoff_ledger(ledger_input, target_season=season)
            for season in TEST_SEASONS
        ],
        ignore_index=True,
    )

    predictions = pd.read_parquet(SOURCE_RUN / "game_predictions.parquet")
    predictions = predictions.loc[
        predictions["test_season"].isin(TEST_SEASONS)
        & predictions["arm"].eq(SELECTED_ARM)
    ].copy()
    weekly = expand_weekly_oracle_predictions(predictions, game_dates, ledger)
    cutoff_metrics, season_summary = metric_frames(weekly)

    priors = pd.read_parquet(PRIORS)
    relevant_priors = priors.loc[priors["Target_Season"].isin(TEST_SEASONS)]
    if set(relevant_priors["method"]) != {"raw"} or not (
        relevant_priors["training_target_end"] < relevant_priors["Target_Season"]
    ).all():
        raise ValueError("Predictive SPM priors violate the frozen cutoff.")
    fold_metrics = pd.read_parquet(SOURCE_RUN / "fold_metrics.parquet")
    folds = fold_metrics.loc[
        fold_metrics["test_season"].isin(TEST_SEASONS)
        & fold_metrics["arm"].eq(SELECTED_ARM)
    ].copy()
    if not (
        folds["training_end"].eq(folds["test_season"] - 1)
        & folds["training_start"].eq(folds["test_season"] - 5)
    ).all():
        raise ValueError("Saved ratings do not use five completed seasons.")

    ratings = pd.read_parquet(SOURCE_RUN / "ratings.parquet")
    player_games = pd.read_parquet(PLAYER_GAMES)
    eligible_ratings, population_audit = evidence_backed_ratings(
        ratings, player_games, priors
    )
    excluded_2026 = int(
        population_audit.loc[
            population_audit["target_season"].eq(2026),
            "excluded_no_history_or_prior",
        ].iloc[0]
    )
    expected_excluded = int(
        contract["rating_population"]["expected_2026_excluded_no_evidence"]
    )
    if excluded_2026 != expected_excluded:
        raise ValueError(
            f"Expected {expected_excluded} excluded 2026 rows; found {excluded_2026}."
        )

    component_error = float(
        (eligible_ratings["net"] - eligible_ratings["offense"] - eligible_ratings["defense"])
        .abs()
        .max()
    )
    unknown_instances = int(weekly["unknown_player_slots"].gt(0).sum())
    unknown_slots = int(weekly["unknown_player_slots"].sum())
    checks = pd.DataFrame(
        [
            {
                "check": "season_2027_absent",
                "status": "pass",
                "detail": "maximum source season is 2026",
            },
            {
                "check": "model_information_cutoff",
                "status": "pass",
                "detail": "every rating uses five seasons ending at T-1",
            },
            {
                "check": "predictive_prior_cutoff",
                "status": "pass",
                "detail": "every prior training target ends before T",
            },
            {
                "check": "weekly_window_dates",
                "status": "pass",
                "detail": "all scored dates are in [cutoff, cutoff + 14 days)",
            },
            {
                "check": "oracle_game_rowsets",
                "status": "pass",
                "detail": "every scored rowset matches its ledger hash",
            },
            {
                "check": "rating_population",
                "status": "pass",
                "detail": f"excluded {excluded_2026} no-evidence 2026 rows",
            },
            {
                "check": "oracle_no_history_exposure",
                "status": "caveat",
                "detail": (
                    f"{unknown_instances} snapshot-game rows contain "
                    f"{unknown_slots} no-history lineup slots"
                ),
            },
            {
                "check": "operational_forecast",
                "status": "blocked",
                "detail": (
                    "historical projected minutes, roster status, and injuries "
                    "are not inputs"
                ),
            },
            {
                "check": "public_release",
                "status": "forbidden",
                "detail": "reused oracle diagnostic with no 2027 confirmation",
            },
        ]
    )

    sources = {
        "pilot_contract": CONTRACT,
        "validation_contract": VALIDATION_CONTRACT,
        "source_run_manifest": SOURCE_RUN / "run.json",
        "source_game_predictions": SOURCE_RUN / "game_predictions.parquet",
        "source_ratings": SOURCE_RUN / "ratings.parquet",
        "source_fold_metrics": SOURCE_RUN / "fold_metrics.parquet",
        "predictive_priors": PRIORS,
        "possessions": POSSESSIONS,
        "player_games": PLAYER_GAMES,
        "runner": Path(__file__),
        "cutoff_code": ROOT / "src/nba_impact/models/predictive_current_aio.py",
    }
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{identity}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "cutoff_ledger.parquet": ledger,
        "snapshot_game_predictions.parquet": weekly,
        "cutoff_metrics.parquet": cutoff_metrics,
        "season_summary.parquet": season_summary,
        "eligible_ratings.parquet": eligible_ratings,
        "rating_population_audit.parquet": population_audit,
        "validation_checks.parquet": checks,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_historical_mechanical_pilot_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_run_id": source_run["run_id"],
        "source_arm": SELECTED_ARM,
        "test_seasons": list(TEST_SEASONS),
        "snapshot_policy": contract["source_run"]["snapshot_policy"],
        "source_hashes": source_hashes,
        "quality": {
            "cutoffs": int(len(ledger)),
            "snapshot_game_instances": int(len(weekly)),
            "unique_games": int(weekly["game_id"].nunique()),
            "maximum_component_identity_error": component_error,
            "eligible_rating_rows": int(len(eligible_ratings)),
            "excluded_2026_no_evidence_rows": excluded_2026,
            "snapshot_games_with_no_history_slots": unknown_instances,
            "no_history_lineup_slots": unknown_slots,
            "required_checks_passed": bool(checks.loc[:5, "status"].eq("pass").all()),
        },
        "readiness": {
            "weekly_oracle_evaluation": "ready_reused_mechanical_pilot_with_no_history_caveat",
            "eligible_ratings": "ready_local_research_only",
            "operational_forecast": "blocked_missing_historical_pregame_inputs",
            "public_release": "forbidden",
        },
        "files": {},
        "forbidden_interpretation": (
            "Preseason ratings are frozen within season and observed future lineups "
            "supply exposure. Overlapping windows are not independent. This is not an "
            "operational forecast, model comparison, promotion test, or public rating."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "sha256": sha256_file(output / name),
            "rows": int(len(frame)),
        }
    write_json_atomic(run, output / "run.json")
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
