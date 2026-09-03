"""Compare one-season points RAPM with player-neutral expected-conversion targets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.expected_shot_quality import _feature_frame
from nba_impact.models.luck_adjusted_rapm import (
    CATEGORY_COLUMNS,
    _game_metrics,
    _stable_game_fold,
    build_conversion_ledger,
    build_expected_outcome_frame,
)
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, _game_margin_frame, build_design, fit_coefficients
from nba_impact.models.shot_model_suite import enrich_shots


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/single_season_expected_outcome_rapm_v1.yml"
REFERENCE = ROOT / "artifacts/models/luck_adjusted_rapm/luck_adjusted_rapm_v1_8580bb30e9"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/single_season_expected_outcome_rapm"


def _load_annual_shooting() -> pd.DataFrame:
    source = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
    columns = {
        "PLAYER_ID",
        "OffPoss",
        *(column for pair in CATEGORY_COLUMNS.values() for column in pair),
    }
    outputs = []
    for season in range(2014, 2026):
        frame = pd.read_parquet(source / f"{season}.parquet", columns=sorted(columns))
        frame = frame.drop_duplicates().copy()
        if frame["PLAYER_ID"].duplicated().any():
            raise ValueError(f"Annual shooting source has duplicate players in {season}.")
        frame["season"] = season
        for column in columns - {"PLAYER_ID"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def _cross_fit_shots(
    panel: pd.DataFrame,
    *,
    seasons: tuple[int, ...],
    folds: int,
    c: float,
    include_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    metrics: list[dict] = []
    for season in seasons:
        block = panel.loc[panel["season_end"].eq(season)].reset_index(drop=True).copy()
        block["fold"] = block["game_id"].map(lambda value: _stable_game_fold(value, folds))
        predictions = np.full(len(block), np.nan)
        for fold in range(folds):
            test = block["fold"].eq(fold).to_numpy()
            train = ~test
            train_x, _ = _feature_frame(
                block.loc[train], include_possession_context=include_context
            )
            test_x, _ = _feature_frame(
                block.loc[test], include_possession_context=include_context
            )
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=300)
            model.fit(train_x, block.loc[train, "shot_made"].to_numpy(dtype=int))
            predictions[test] = model.predict_proba(test_x)[:, 1]
        if not np.isfinite(predictions).all():
            raise ValueError(f"Incomplete shot expectations for {season}.")
        actual = block["shot_made"].to_numpy(dtype=float)
        output = block[
            ["shot_id", "game_id", "actionNumber", "season_end", "shooter_id", "shot_zone", "shot_value", "shot_made"]
        ].copy()
        output["neutral_expected_make"] = predictions
        outputs.append(output)
        metrics.append(
            {
                "season": season,
                "model": "context" if include_context else "location",
                "shots": len(block),
                "games": block["game_id"].nunique(),
                "brier": float(np.mean((actual - predictions) ** 2)),
                "mean_actual": float(actual.mean()),
                "mean_expected": float(predictions.mean()),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(metrics)


def _paired_bootstrap(
    games: pd.DataFrame, *, challenger: str, season: int, draws: int, seed: int
) -> dict:
    block = games.loc[games["test_season"].eq(season)]
    normal = block.loc[block["arm"].eq("normal_realized_points")]
    candidate = block.loc[block["arm"].eq(challenger)]
    paired = normal[["game_id", "actual_margin", "predicted_margin"]].merge(
        candidate[["game_id", "actual_margin", "predicted_margin"]],
        on="game_id",
        suffixes=("_normal", "_candidate"),
        validate="one_to_one",
    )
    if not np.array_equal(paired["actual_margin_normal"], paired["actual_margin_candidate"]):
        raise ValueError("Expected-outcome arms do not share identical game outcomes.")
    actual = paired["actual_margin_normal"].to_numpy(dtype=float)
    normal_sq = (actual - paired["predicted_margin_normal"].to_numpy(dtype=float)) ** 2
    candidate_sq = (actual - paired["predicted_margin_candidate"].to_numpy(dtype=float)) ** 2
    rng = np.random.default_rng(seed + season)
    samples = np.empty(draws)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        index = rng.integers(0, len(paired), size=(stop - start, len(paired)))
        samples[start:stop] = np.sqrt(candidate_sq[index].mean(axis=1)) - np.sqrt(
            normal_sq[index].mean(axis=1)
        )
    delta = float(np.sqrt(candidate_sq.mean()) - np.sqrt(normal_sq.mean()))
    return {
        "test_season": season,
        "challenger": challenger,
        "games": len(paired),
        "rmse_delta": delta,
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "probability_better": float(np.mean(samples < 0)),
    }


def run() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["season_policy"]["forbidden_season"] != 2027:
        raise ValueError("Season 2027 must remain forbidden.")
    rating_seasons = tuple(contract["season_policy"]["rating_seasons"])
    test_seasons = tuple(contract["season_policy"]["test_seasons"])
    if rating_seasons != (2024, 2025) or test_seasons != (2025, 2026):
        raise ValueError("The frozen one-season folds changed.")
    contract = json.loads(json.dumps(contract, default=str))

    paths = {
        "shots": ROOT / "data/lake/silver/shot_defense_events.parquet",
        "possessions": ROOT / "data/lake/silver/possessions.parquet",
        "segments": ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        "events": ROOT / "data/lake/silver/event_states.parquet",
    }
    raw_paths = tuple(
        sorted((ROOT / "data/lake/bronze/nba_data_archive/cdnnba").glob("season=*/regular.parquet"))
    )
    if len(raw_paths) != 3:
        raise FileNotFoundError("The pinned 2024-26 event context is incomplete.")

    possessions = pd.read_parquet(paths["possessions"])
    segments = pd.read_parquet(paths["segments"])
    events = pd.read_parquet(paths["events"])
    base = canonical_terminal_frame(possessions, segments, seasons=(2024, 2025, 2026))
    rating_base = base.loc[base["season"].isin(rating_seasons)].copy()
    enriched = enrich_shots(pd.read_parquet(paths["shots"]), possessions, raw_paths)
    shot_outputs: dict[str, pd.DataFrame] = {}
    shot_metrics: list[pd.DataFrame] = []
    for arm, include_context in (
        ("location_expected_conversion", False),
        ("context_expected_conversion", True),
    ):
        predictions, metrics = _cross_fit_shots(
            enriched,
            seasons=rating_seasons,
            folds=int(contract["shot_expectation"]["current_game_out_folds"]),
            c=float(contract["shot_expectation"]["logistic_c"]),
            include_context=include_context,
        )
        shot_outputs[arm] = predictions
        shot_metrics.append(metrics.assign(arm=arm))

    annual_dir = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
    annual_paths = tuple(annual_dir / f"{season}.parquet" for season in range(2014, 2026))
    annual = _load_annual_shooting()
    selected_path = REFERENCE / "shooting_history_selected.parquet"
    selected = pd.read_parquet(selected_path)
    target_frames = {"normal_realized_points": base}
    coverage = []
    for arm, shots in shot_outputs.items():
        ledger, _ = build_conversion_ledger(rating_base, shots, events, annual, selected)
        adjusted = build_expected_outcome_frame(rating_base, ledger)[
            ["possession_id", "expected_pts"]
        ]
        target = base.merge(adjusted, on="possession_id", how="left", validate="one_to_one")
        target["expected_pts"] = target["expected_pts"].fillna(target["pts"])
        target_frames[arm] = target.assign(pts=target["expected_pts"])
        coverage.append(
            {
                "arm": arm,
                "conversion_events": len(ledger),
                "actual_conversion_share": float(
                    ledger["actual_points"].sum() / rating_base["pts"].sum()
                ),
            }
        )

    designs = {arm: build_design(frame, include_home=True) for arm, frame in target_frames.items()}
    players = designs["normal_realized_points"].players
    if any(not np.array_equal(design.players, players) for design in designs.values()):
        raise ValueError("Every arm must have an identical player design.")
    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=float(contract["rapm"]["lambda_off"]),
        lambda_def=float(contract["rapm"]["lambda_def"]),
        lambda_home=float(contract["rapm"]["lambda_home"]),
        data_scope="single_season_expected_outcome_rapm_v1",
    )
    game_parts = []
    metric_rows = []
    normal_design = designs["normal_realized_points"]
    for test_season in test_seasons:
        train = normal_design.seasons == test_season - 1
        test = normal_design.seasons == test_season
        for arm, design in designs.items():
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            games = _game_margin_frame(normal_design, beta, intercept, test, train)
            games["arm"] = arm
            games["test_season"] = test_season
            game_parts.append(games)
            metric_rows.append({"arm": arm, "test_season": test_season, **_game_metrics(games)})
    games = pd.concat(game_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap(
                games,
                challenger=arm,
                season=season,
                draws=int(contract["evaluation"]["bootstrap_draws"]),
                seed=20260903,
            )
            for season in test_seasons
            for arm in ("location_expected_conversion", "context_expected_conversion")
        ]
    )

    identity_sources = [
        CONTRACT,
        Path(__file__),
        selected_path,
        *paths.values(),
        *raw_paths,
        *annual_paths,
    ]
    identity = hashlib.sha256(
        json.dumps({str(path.relative_to(ROOT)): sha256_file(path) for path in identity_sources}, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"single_season_expected_outcome_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / "future_game_metrics.parquet", index=False)
    bootstrap.to_parquet(output / "paired_game_bootstrap.parquet", index=False)
    pd.concat(shot_metrics, ignore_index=True).to_parquet(output / "shot_metrics.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "research_null_complete",
        "evidence_status": "reused_2025_2026_diagnostics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": contract["estimand_id"],
        "config": contract,
        "quality": {
            "possessions": len(base),
            "games": base["gameid"].nunique(),
            "rating_seasons": list(rating_seasons),
            "test_seasons": list(test_seasons),
            "season_2027_loaded": False,
            "coverage": coverage,
        },
        "diagnostic_metrics": metrics.to_dict(orient="records"),
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "artifact_path": str(output.resolve()),
        "decision": "retain_normal_realized_points_unless_an_expected_conversion_arm_wins_both_reused_folds",
        "forbidden_interpretation": "Production RAPM, literal qSQ, causal shot quality, or untouched confirmation.",
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
