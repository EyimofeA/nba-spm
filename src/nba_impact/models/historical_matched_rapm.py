"""Research-only comparison of historical V3 and legacy terminal-lineup RAPM.

The comparison is deliberately narrow.  It uses the same regular-season game
IDs in both sources, fits the frozen zero-prior 3000/3000/300 model separately
to each source, and never writes a canonical or public artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficients,
    load_legacy_possessions,
    ratings_table,
)


@dataclass(frozen=True)
class MatchedRapmConfig:
    """Frozen settings for the comparison experiment."""

    seasons: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023)
    lambda_off: float = 3000.0
    lambda_def: float = 3000.0
    lambda_home: float = 300.0
    holdout_fraction: float = 0.20

    def rapm(self, season: int) -> RapmConfig:
        return RapmConfig(
            seasons=(int(season),),
            lambda_off=self.lambda_off,
            lambda_def=self.lambda_def,
            lambda_home=self.lambda_home,
            include_home=True,
            game_types=("regular",),
            data_scope="matched_historical_research",
        )


def adapt_v3_terminal_lineups(
    possessions: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """Adapt one V3 candidate partition to the RAPM design contract.

    The final ordinal segment is used for every possession.  This is the
    explicit terminal-lineup sensitivity, not a claim that the lineup was
    constant during the possession.
    """
    required_p = {
        "possession_id", "game_id", "season_end", "game_date", "period",
        "possession_number", "offense_is_home", "points", "season_type",
    }
    missing = sorted(required_p - set(possessions.columns))
    if missing:
        raise ValueError(f"V3 possessions missing columns: {missing}")
    required_s = {
        "possession_id", "segment_number", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5",
    }
    missing = sorted(required_s - set(segments.columns))
    if missing:
        raise ValueError(f"V3 segments missing columns: {missing}")
    possessions = possessions.loc[possessions["season_type"].eq("regular")].copy()
    segments = segments.loc[segments["possession_id"].isin(possessions["possession_id"])].copy()
    segments = segments.sort_values(["possession_id", "segment_number"], kind="stable")
    terminal = segments.groupby("possession_id", sort=False, as_index=False).tail(1)
    terminal = terminal[
        ["possession_id", *[f"{side}_player_{i}" for side in ("home", "away") for i in range(1, 6)]]
    ]
    frame = possessions.merge(terminal, on="possession_id", how="inner", validate="one_to_one")
    player_cols = [f"{side}_player_{i}" for side in ("home", "away") for i in range(1, 6)]
    if frame.empty:
        raise ValueError("V3 terminal-lineup adaptation produced no possessions.")
    if frame[player_cols].isna().any().any():
        raise ValueError("V3 terminal-lineup adaptation contains missing players.")
    for row in frame[player_cols].to_numpy(dtype=np.int64):
        if len(set(row[:5])) != 5 or len(set(row[5:])) != 5 or set(row[:5]) & set(row[5:]):
            raise ValueError("V3 terminal lineup must contain ten unique players.")
    return pd.DataFrame(
        {
            "home_poss": frame["offense_is_home"].astype(int),
            "pts": frame["points"].astype(float),
            **{
                f"a{i}": frame[f"away_player_{i}"].astype("int64") for i in range(1, 6)
            },
            **{
                f"h{i}": frame[f"home_player_{i}"].astype("int64") for i in range(1, 6)
            },
            "season": frame["season_end"].astype(int),
            "date": frame["game_date"],
            "period": frame["period"].astype(int),
            "num": frame["possession_number"].astype(int),
            "gameid": frame["game_id"].astype(str),
        }
    )


def _regular_game_ids(frame: pd.DataFrame) -> set[str]:
    ids = frame["gameid"].astype(str)
    # 002 is the NBA regular-season game prefix.  Explicit filtering avoids
    # accidentally comparing the candidate to legacy playoff rows.
    return set(ids.loc[ids.str.startswith("002")])


def _stable_holdout_games(frame: pd.DataFrame, fraction: float) -> set[str]:
    game_dates = frame[["gameid", "date"]].drop_duplicates("gameid").copy()
    game_dates["date"] = pd.to_datetime(game_dates["date"], errors="coerce")
    game_dates = game_dates.sort_values(["date", "gameid"], na_position="last", kind="stable")
    n_test = max(1, int(np.ceil(len(game_dates) * fraction)))
    return set(game_dates.tail(n_test)["gameid"].astype(str))


def _rating_comparison(
    candidate: pd.DataFrame, legacy: pd.DataFrame, *, minimum_possessions: int = 0
) -> dict:
    merged = candidate.merge(legacy, on="player_id", suffixes=("_candidate", "_legacy"))
    exposure_columns = [
        f"{side}_possessions_{source}"
        for source in ("candidate", "legacy")
        for side in ("off", "def")
    ]
    minimum_exposure = merged[exposure_columns].min(axis=1)
    merged = merged.loc[minimum_exposure.ge(minimum_possessions)].copy()
    result: dict[str, float | int] = {
        "minimum_possessions_per_source_side": int(minimum_possessions),
        "matched_players": int(len(merged)),
    }
    for component in ("offense_per_100", "defense_per_100", "net_per_100"):
        x = merged[f"{component}_candidate"].to_numpy(float)
        y = merged[f"{component}_legacy"].to_numpy(float)
        if len(merged) < 2:
            result[f"{component}_pearson"] = float("nan")
            result[f"{component}_spearman"] = float("nan")
        else:
            result[f"{component}_pearson"] = float(pearsonr(x, y).statistic)
            result[f"{component}_spearman"] = float(spearmanr(x, y).statistic)
        result[f"{component}_rmse"] = float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else float("nan")
    return result


def _official_margin_metrics(
    design,
    beta: np.ndarray,
    intercept: float,
    test_mask: np.ndarray,
    train_mask: np.ndarray,
    official_margins: Mapping[str, float],
) -> dict:
    """Score one source against the common official final-margin target."""
    games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
    reconstructed = games["actual_margin"].copy()
    official = games["game_id"].astype(str).map(official_margins)
    if official.isna().any():
        missing = sorted(games.loc[official.isna(), "game_id"].astype(str).unique())
        raise ValueError(f"Official margins missing games: {missing[:10]}")
    games["actual_margin"] = official.astype(float)
    error = games["actual_margin"] - games["predicted_margin"]
    reconstruction_error = reconstructed - games["actual_margin"]
    correlation = float(games[["actual_margin", "predicted_margin"]].corr().iloc[0, 1])
    predicted_variance = float(np.var(games["predicted_margin"], ddof=0))
    calibration_slope = (
        float(
            np.cov(games["actual_margin"], games["predicted_margin"], ddof=0)[0, 1]
            / predicted_variance
        )
        if predicted_variance > 0
        else float("nan")
    )
    calibration_intercept = float(
        games["actual_margin"].mean()
        - calibration_slope * games["predicted_margin"].mean()
    )
    return {
        "games": int(len(games)),
        "margin_target": "official_final_score",
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": correlation,
        "actual_margin_sd": float(games["actual_margin"].std(ddof=0)),
        "predicted_margin_sd": float(games["predicted_margin"].std(ddof=0)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "known_player_rate": float(games.attrs["known_player_rate"]),
        "games_with_unknown_players": int((games["unknown_player_slots"] > 0).sum()),
        "reconstructed_margin_rmse": float(np.sqrt(np.mean(reconstruction_error**2))),
        "reconstructed_margin_mae": float(np.mean(np.abs(reconstruction_error))),
        "games_with_reconstructed_margin_mismatch": int((reconstruction_error != 0).sum()),
        "max_abs_reconstructed_margin_error": float(np.abs(reconstruction_error).max()),
    }


def compare_season(
    candidate: pd.DataFrame,
    legacy: pd.DataFrame,
    season: int,
    config: MatchedRapmConfig,
    official_margins: Mapping[str, float],
) -> tuple[dict, dict, list[dict], list[dict], list[dict]]:
    """Return coverage, rating, and held-out margin rows for one season."""
    candidate = candidate.loc[candidate["season"].eq(season)].copy()
    legacy = legacy.loc[legacy["season"].eq(season)].copy()
    candidate_ids = _regular_game_ids(candidate)
    legacy_ids = _regular_game_ids(legacy)
    matched_ids = candidate_ids & legacy_ids
    if not matched_ids:
        raise ValueError(f"No matched regular-season games for {season}.")
    candidate = candidate.loc[candidate["gameid"].isin(matched_ids)].reset_index(drop=True)
    legacy = legacy.loc[legacy["gameid"].isin(matched_ids)].reset_index(drop=True)
    if set(candidate["gameid"]) != matched_ids or set(legacy["gameid"]) != matched_ids:
        raise ValueError(f"Matched game restriction failed for {season}.")
    candidate_dates = candidate[["gameid", "date"]].drop_duplicates("gameid")
    legacy_dates = legacy[["gameid", "date"]].drop_duplicates("gameid")
    dates = candidate_dates.merge(
        legacy_dates, on="gameid", suffixes=("_candidate", "_legacy"), validate="one_to_one"
    )
    if not pd.to_datetime(dates["date_candidate"]).eq(pd.to_datetime(dates["date_legacy"])).all():
        raise ValueError(f"Candidate and legacy game dates differ for {season}.")
    coverage = {
        "season": season,
        "candidate_games": len(candidate_ids),
        "legacy_games": len(legacy_ids),
        "matched_games": len(matched_ids),
        "candidate_only_games": len(candidate_ids - legacy_ids),
        "legacy_only_games": len(legacy_ids - candidate_ids),
        "candidate_possessions_matched": len(candidate),
        "legacy_possessions_matched": len(legacy),
        "candidate_points_matched": float(candidate["pts"].sum()),
        "legacy_points_matched": float(legacy["pts"].sum()),
        "points_delta": float(candidate["pts"].sum() - legacy["pts"].sum()),
    }
    rows: dict[str, dict] = {}
    margin_rows: list[dict] = []
    test_games = _stable_holdout_games(candidate, config.holdout_fraction)
    if not test_games.issubset(matched_ids):
        raise ValueError(f"Held-out game restriction failed for {season}.")
    split_rows = [
        {
            "season": season,
            "game_id": game_id,
            "split": "test" if game_id in test_games else "train",
        }
        for game_id in sorted(matched_ids)
    ]
    for name, frame in (("candidate_v3_terminal", candidate), ("legacy_terminal", legacy)):
        design = build_design(frame, include_home=True)
        beta, intercept = fit_coefficients(design, config.rapm(season))
        ratings = ratings_table(design, beta).drop(columns=["uncertainty_status"])
        rows[name] = ratings
        test_mask = np.isin(design.game_ids, list(test_games))
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            metrics = {"status": "not_available", "reason": "insufficient games"}
        else:
            train_beta, train_intercept = fit_coefficients(design, config.rapm(season), train_mask)
            metrics = _official_margin_metrics(
                design,
                train_beta,
                train_intercept,
                test_mask,
                train_mask,
                official_margins,
            )
            metrics["status"] = "complete"
        margin_rows.append({"season": season, "source": name, **metrics})
    rating_row = {"season": season, **_rating_comparison(rows["candidate_v3_terminal"], rows["legacy_terminal"])}
    exposure_rows = [
        {
            "season": season,
            **_rating_comparison(
                rows["candidate_v3_terminal"],
                rows["legacy_terminal"],
                minimum_possessions=threshold,
            ),
        }
        for threshold in (0, 500, 1000, 2000)
    ]
    return coverage, rating_row, margin_rows, split_rows, exposure_rows


def run_matched_comparison(
    candidate_roots: Mapping[int, str | Path],
    legacy_cache: str | Path,
    official_scores_path: str | Path,
    artifact_root: str | Path,
    *,
    config: MatchedRapmConfig = MatchedRapmConfig(),
) -> dict:
    """Run and persist the versioned research comparison."""
    missing = sorted(set(config.seasons) - set(candidate_roots))
    if missing:
        raise ValueError(f"Candidate roots missing seasons: {missing}")
    coverage_rows: list[dict] = []
    rating_rows: list[dict] = []
    margin_rows: list[dict] = []
    split_rows: list[dict] = []
    exposure_rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    official_scores_path = Path(official_scores_path)
    official_scores = pd.read_parquet(official_scores_path)
    official_scores = official_scores.loc[official_scores["season_type"].eq("regular")].copy()
    if official_scores["game_id"].astype(str).duplicated().any():
        raise ValueError("Official final scores must be unique by regular-season game ID.")
    official_margins = dict(
        zip(
            official_scores["game_id"].astype(str),
            official_scores["home_score"].astype(float)
            - official_scores["away_score"].astype(float),
            strict=True,
        )
    )
    source_hashes["official_game_scores"] = sha256_file(official_scores_path)
    source_hashes["comparison_code"] = sha256_file(Path(__file__))
    source_hashes["rapm_code"] = sha256_file(Path(__file__).with_name("rapm.py"))
    for season in config.seasons:
        root = Path(candidate_roots[season])
        possessions_path = root / "possessions.parquet"
        segments_path = root / "possession_lineup_segments.parquet"
        lineup_quality_path = root / "lineup_quality.parquet"
        attachment_quality_path = root / "attachment_quality.parquet"
        candidate = adapt_v3_terminal_lineups(
            pd.read_parquet(possessions_path), pd.read_parquet(segments_path)
        )
        candidate_ids = set(candidate["gameid"].astype(str))
        for label, quality_path in (
            ("lineup", lineup_quality_path),
            ("attachment", attachment_quality_path),
        ):
            quality = pd.read_parquet(quality_path)
            quality["game_id"] = quality["game_id"].astype(str)
            accepted = quality.loc[quality["game_id"].isin(candidate_ids)]
            if set(accepted["game_id"]) != candidate_ids or not accepted["passed"].all():
                raise ValueError(f"Candidate {label} quality gate failed for {season}.")
        legacy = load_legacy_possessions(legacy_cache, (season,), game_types=("regular",))
        coverage, rating, margins, splits, exposure = compare_season(
            candidate, legacy, season, config, official_margins
        )
        coverage_rows.append(coverage)
        rating_rows.append(rating)
        margin_rows.extend(margins)
        split_rows.extend(splits)
        exposure_rows.extend(exposure)
        source_hashes[f"candidate_{season}_possessions"] = sha256_file(possessions_path)
        source_hashes[f"candidate_{season}_segments"] = sha256_file(segments_path)
        source_hashes[f"candidate_{season}_lineup_quality"] = sha256_file(lineup_quality_path)
        source_hashes[f"candidate_{season}_attachment_quality"] = sha256_file(attachment_quality_path)
        source_hashes[f"legacy_{season}"] = sha256_file(Path(legacy_cache) / f"matchups_{season}.parquet")
    payload = {"config": asdict(config), "source_hashes": source_hashes, "rows": coverage_rows}
    digest = __import__("hashlib").sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(artifact_root) / "research" / "historical_matched_rapm" / f"historical_matched_rapm_v1_{digest}"
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(coverage_rows).to_parquet(output / "season_coverage.parquet", index=False)
    pd.DataFrame(rating_rows).to_parquet(output / "rating_comparison.parquet", index=False)
    pd.DataFrame(margin_rows).to_parquet(output / "game_margin_comparison.parquet", index=False)
    pd.DataFrame(split_rows).to_parquet(output / "game_splits.parquet", index=False)
    pd.DataFrame(exposure_rows).to_parquet(
        output / "rating_comparison_by_exposure.parquet", index=False
    )
    run = {
        "run_id": f"historical_matched_rapm_v1_{digest}",
        "model_family": "research_matched_v3_vs_legacy_terminal_rapm",
        "estimand": "retrospective_lineup_adjusted_points_per_100",
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "source_hashes": source_hashes,
        "artifacts": [
            "season_coverage.parquet",
            "rating_comparison.parquet",
            "game_margin_comparison.parquet",
            "game_splits.parquet",
            "rating_comparison_by_exposure.parquet",
        ],
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Sources are fit separately after restriction to identical regular-season game IDs.",
            "V3 uses the terminal ordinal lineup and cannot represent within-possession substitutions.",
            "Held-out margin metrics are chronological within-season retrodictions using observed test lineups, not forecasts.",
            "Both sources use identical persisted train/test game IDs and the same official final-score margin target.",
            "No public or canonical artifact is modified by this research run.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
