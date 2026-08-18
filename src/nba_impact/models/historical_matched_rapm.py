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
    _game_margin_metrics,
    build_design,
    fit_coefficients,
    load_legacy_possessions,
    ratings_table,
)


@dataclass(frozen=True)
class MatchedRapmConfig:
    """Frozen settings for the comparison experiment."""

    seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023)
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


def _rating_comparison(candidate: pd.DataFrame, legacy: pd.DataFrame) -> dict:
    merged = candidate.merge(legacy, on="player_id", suffixes=("_candidate", "_legacy"))
    result: dict[str, float | int] = {"matched_players": int(len(merged))}
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


def compare_season(
    candidate: pd.DataFrame, legacy: pd.DataFrame, season: int, config: MatchedRapmConfig
) -> tuple[dict, dict, dict]:
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
    for name, frame in (("candidate_v3_terminal", candidate), ("legacy_terminal", legacy)):
        design = build_design(frame, include_home=True)
        beta, intercept = fit_coefficients(design, config.rapm(season))
        ratings = ratings_table(design, beta).drop(columns=["uncertainty_status"])
        rows[name] = ratings
        test_games = _stable_holdout_games(frame, config.holdout_fraction)
        test_mask = np.isin(design.game_ids, list(test_games))
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            metrics = {"status": "not_available", "reason": "insufficient games"}
        else:
            train_beta, train_intercept = fit_coefficients(design, config.rapm(season), train_mask)
            metrics = _game_margin_metrics(design, train_beta, train_intercept, test_mask, train_mask)
            metrics["status"] = "complete"
        margin_rows.append({"season": season, "source": name, **metrics})
    rating_row = {"season": season, **_rating_comparison(rows["candidate_v3_terminal"], rows["legacy_terminal"])}
    return coverage, rating_row, pd.DataFrame(margin_rows).to_dict(orient="records")


def run_matched_comparison(
    candidate_roots: Mapping[int, str | Path],
    legacy_cache: str | Path,
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
    source_hashes: dict[str, str] = {}
    for season in config.seasons:
        root = Path(candidate_roots[season])
        possessions_path = root / "possessions.parquet"
        segments_path = root / "possession_lineup_segments.parquet"
        candidate = adapt_v3_terminal_lineups(
            pd.read_parquet(possessions_path), pd.read_parquet(segments_path)
        )
        legacy = load_legacy_possessions(legacy_cache, (season,), game_types=("regular",))
        coverage, rating, margins = compare_season(candidate, legacy, season, config)
        coverage_rows.append(coverage)
        rating_rows.append(rating)
        margin_rows.extend(margins)
        source_hashes[f"candidate_{season}_possessions"] = sha256_file(possessions_path)
        source_hashes[f"candidate_{season}_segments"] = sha256_file(segments_path)
        source_hashes[f"legacy_{season}"] = sha256_file(Path(legacy_cache) / f"matchups_{season}.parquet")
    payload = {"config": asdict(config), "source_hashes": source_hashes, "rows": coverage_rows}
    digest = __import__("hashlib").sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(artifact_root) / "research" / "historical_matched_rapm" / f"historical_matched_rapm_v1_{digest}"
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(coverage_rows).to_parquet(output / "season_coverage.parquet", index=False)
    pd.DataFrame(rating_rows).to_parquet(output / "rating_comparison.parquet", index=False)
    pd.DataFrame(margin_rows).to_parquet(output / "game_margin_comparison.parquet", index=False)
    run = {
        "run_id": f"historical_matched_rapm_v1_{digest}",
        "model_family": "research_matched_v3_vs_legacy_terminal_rapm",
        "estimand": "retrospective_lineup_adjusted_points_per_100",
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "source_hashes": source_hashes,
        "artifacts": ["season_coverage.parquet", "rating_comparison.parquet", "game_margin_comparison.parquet"],
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Sources are fit separately after restriction to identical regular-season game IDs.",
            "V3 uses the terminal ordinal lineup and cannot represent within-possession substitutions.",
            "Held-out margin metrics are chronological within-season retrodictions using observed test lineups, not forecasts.",
            "No public or canonical artifact is modified by this research run.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run

