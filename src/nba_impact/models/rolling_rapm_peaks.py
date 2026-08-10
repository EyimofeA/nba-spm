"""Independent rolling normal-RAPM ratings and all-time peak tables."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_legacy_possessions,
    ratings_table,
)


COMPONENTS = ("offense", "defense", "net")


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(f"{path.suffix}.partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def load_peak_player_names(
    names_path: str | Path,
    player_sheets_dir: str | Path,
    seasons: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Prefer the RAPM crosswalk and fill its old-ID gaps from annual sheets."""
    names = pd.read_csv(names_path)[["PLAYER_ID", "PLAYER_NAME"]]
    if names["PLAYER_ID"].duplicated().any():
        raise ValueError("Rolling RAPM player-name IDs must be unique.")
    names["PLAYER_ID"] = pd.to_numeric(names["PLAYER_ID"], errors="raise").astype(int)
    names["name_source"] = "rapm_all_names"
    hashes = {str(Path(names_path).resolve()): sha256_file(names_path)}

    fallback_rows = []
    for season in seasons:
        path = Path(player_sheets_dir) / f"{season}.csv"
        if not path.exists():
            raise ValueError(f"Missing annual player sheet {path}.")
        sheet = pd.read_csv(path, usecols=["PLAYER_ID", "PLAYER_NAME"])
        sheet["PLAYER_ID"] = pd.to_numeric(
            sheet["PLAYER_ID"], errors="coerce"
        ).astype("Int64")
        sheet = sheet.dropna(subset=["PLAYER_ID", "PLAYER_NAME"])
        sheet["Season"] = season
        fallback_rows.append(sheet)
        hashes[str(path.resolve())] = sha256_file(path)
    fallback = (
        pd.concat(fallback_rows, ignore_index=True)
        .sort_values("Season", kind="stable")
        .drop_duplicates("PLAYER_ID", keep="last")
    )
    fallback["PLAYER_ID"] = fallback["PLAYER_ID"].astype(int)
    fallback["name_source"] = "annual_player_sheet_fallback"
    missing = fallback.loc[~fallback["PLAYER_ID"].isin(names["PLAYER_ID"])]
    combined = pd.concat(
        [names, missing[["PLAYER_ID", "PLAYER_NAME", "name_source"]]],
        ignore_index=True,
    )
    if combined["PLAYER_ID"].duplicated().any():
        raise ValueError("Resolved peak player-name IDs must be unique.")
    return combined, hashes


def fit_rolling_rapm_window(
    frame: pd.DataFrame,
    config: RapmConfig,
    *,
    window_start: int,
    window_end: int,
    minimum_possessions_per_window_season: int,
) -> tuple[pd.DataFrame, dict]:
    """Fit one fixed zero-prior rolling RAPM window."""
    window_seasons = window_end - window_start + 1
    if tuple(config.seasons) != tuple(range(window_start, window_end + 1)):
        raise ValueError("RAPM config seasons must equal the complete rolling window.")
    season_means = frame.groupby("season")["pts"].mean().sort_index()
    if len(season_means) != window_seasons:
        raise ValueError("Every rolling-window season must contain possessions.")
    overall_mean = float(frame["pts"].mean())
    adjusted = frame.copy()
    adjusted["pts"] = (
        adjusted["pts"]
        - adjusted["season"].map(season_means)
        + overall_mean
    )
    design = build_design(adjusted, include_home=config.include_home)
    beta, intercept = fit_coefficients(design, config)
    ratings = ratings_table(design, beta).rename(
        columns={
            "player_id": "PLAYER_ID",
            "offense_per_100": "offense",
            "defense_per_100": "defense",
            "net_per_100": "net",
            "off_possessions": "Poss_Off",
            "def_possessions": "Poss_Def",
        }
    )
    ratings = ratings.drop(columns=["uncertainty_status"])
    ratings["window_start"] = window_start
    ratings["window_end"] = window_end
    ratings["window_seasons"] = window_seasons
    ratings["minimum_side_possessions"] = ratings[["Poss_Off", "Poss_Def"]].min(axis=1)
    threshold = minimum_possessions_per_window_season * window_seasons
    ratings["peak_eligible"] = ratings["minimum_side_possessions"].ge(threshold)
    for component in COMPONENTS:
        rank = ratings.loc[ratings["peak_eligible"], component].rank(
            method="min", ascending=False
        )
        ratings[f"{component}_rank"] = pd.Series(pd.NA, index=ratings.index, dtype="Int64")
        ratings.loc[rank.index, f"{component}_rank"] = rank.astype("Int64")
    quality = {
        "window_start": window_start,
        "window_end": window_end,
        "window_seasons": window_seasons,
        "games": int(frame["gameid"].nunique()),
        "possession_rows": len(frame),
        "players": len(ratings),
        "peak_eligible_players": int(ratings["peak_eligible"].sum()),
        "minimum_peak_possessions_per_side": threshold,
        "intercept_per_possession": float(intercept),
        "season_scoring_environment_per_100": {
            str(int(key)): float((value - overall_mean) * 100.0)
            for key, value in season_means.items()
        },
        "max_component_identity_error": float(
            np.abs(ratings["net"] - ratings["offense"] - ratings["defense"]).max()
        ),
    }
    return ratings, quality


def extract_player_peaks(ratings: pd.DataFrame) -> pd.DataFrame:
    """Select one eligible peak window per player, length, and component."""
    required = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "window_start",
        "window_end",
        "window_seasons",
        "Poss_Off",
        "Poss_Def",
        "minimum_side_possessions",
        "peak_eligible",
        *COMPONENTS,
    }
    if missing := sorted(required - set(ratings.columns)):
        raise ValueError(f"Peak extraction is missing columns {missing}.")
    eligible = ratings.loc[ratings["peak_eligible"]].copy()
    if eligible.empty:
        raise ValueError("Peak extraction requires at least one eligible rating.")
    rows = []
    for component in COMPONENTS:
        ordered = eligible.sort_values(
            [
                "PLAYER_ID",
                "window_seasons",
                component,
                "minimum_side_possessions",
                "window_end",
            ],
            ascending=[True, True, False, False, True],
            kind="stable",
        )
        selected = ordered.drop_duplicates(["PLAYER_ID", "window_seasons"])
        output = selected[
            [
                "PLAYER_ID",
                "PLAYER_NAME",
                "window_seasons",
                "window_start",
                "window_end",
                "Poss_Off",
                "Poss_Def",
                "offense",
                "defense",
                "net",
            ]
        ].copy()
        output["peak_component"] = component
        output["peak_value"] = selected[component].to_numpy()
        rows.append(output)
    peaks = pd.concat(rows, ignore_index=True)
    peaks["all_time_rank"] = peaks.groupby(
        ["window_seasons", "peak_component"]
    )["peak_value"].rank(method="min", ascending=False).astype(int)
    if peaks.duplicated(["PLAYER_ID", "window_seasons", "peak_component"]).any():
        raise ValueError("Peak keys must be unique.")
    return peaks.sort_values(
        ["window_seasons", "peak_component", "all_time_rank"], kind="stable"
    )


def build_rolling_rapm_peaks(
    cache_dir: str | Path,
    names_path: str | Path,
    player_sheets_dir: str | Path,
    contract_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Fit all configured rolling windows and save ratings plus player peaks."""
    contract = json.loads(Path(contract_path).read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Rolling RAPM peak contract must be frozen.")
    season_start, season_end = (int(value) for value in contract["season_range"])
    window_lengths = tuple(int(value) for value in contract["window_lengths"])
    if not window_lengths or any(length < 1 for length in window_lengths):
        raise ValueError("Rolling RAPM window lengths must be positive.")
    model = contract["model"]
    if model.get("prior") != "zero":
        raise ValueError("Rolling normal RAPM peaks require a zero prior.")
    minimum_per_season = int(
        contract["peak_eligibility"]["minimum_offensive_possessions_per_window_season"]
    )
    if minimum_per_season != int(
        contract["peak_eligibility"]["minimum_defensive_possessions_per_window_season"]
    ):
        raise ValueError("Offensive and defensive peak thresholds must match in v1.")

    all_seasons = tuple(range(season_start, season_end + 1))
    names, name_source_hashes = load_peak_player_names(
        names_path, player_sheets_dir, all_seasons
    )
    possession_paths = [
        Path(cache_dir) / f"matchups_{season}.parquet" for season in all_seasons
    ]
    source_hashes = {str(path.resolve()): sha256_file(path) for path in possession_paths}
    identity_payload = {
        "contract": sha256_file(contract_path),
        "source_code": sha256_file(Path(__file__)),
        "names": name_source_hashes,
        "possessions": source_hashes,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"rolling_rapm_peaks_v1_{identity}"
    output = Path(artifact_root) / "models" / "rolling_rapm_peaks" / run_id
    completed_run = output / "run.json"
    if completed_run.exists():
        return json.loads(completed_run.read_text())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "window_checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    all_possessions = load_legacy_possessions(
        cache_dir, all_seasons, game_types=("regular",)
    )

    rating_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    for window_seasons in window_lengths:
        first_end = season_start + window_seasons - 1
        for window_end in range(first_end, season_end + 1):
            window_start = window_end - window_seasons + 1
            checkpoint = checkpoint_root / f"{window_seasons}y_end_{window_end}.parquet"
            quality_checkpoint = checkpoint_root / f"{window_seasons}y_end_{window_end}.json"
            if checkpoint.exists() and quality_checkpoint.exists():
                ratings = pd.read_parquet(checkpoint)
                quality = json.loads(quality_checkpoint.read_text())
            else:
                frame = all_possessions.loc[
                    all_possessions["season"].between(window_start, window_end)
                ]
                config = RapmConfig(
                    seasons=tuple(range(window_start, window_end + 1)),
                    lambda_off=float(model["lambda_off"]),
                    lambda_def=float(model["lambda_def"]),
                    lambda_home=float(model["lambda_home"]),
                    game_types=("regular",),
                    data_scope="legacy_rolling_normal_rapm_peaks",
                )
                ratings, quality = fit_rolling_rapm_window(
                    frame,
                    config,
                    window_start=window_start,
                    window_end=window_end,
                    minimum_possessions_per_window_season=minimum_per_season,
                )
                ratings = ratings.merge(
                    names, on="PLAYER_ID", how="left", validate="one_to_one"
                )
                _write_parquet_atomic(ratings, checkpoint)
                write_json_atomic(quality, quality_checkpoint)
            rating_frames.append(ratings)
            quality_rows.append(quality)

    rolling = pd.concat(rating_frames, ignore_index=True)
    if rolling.duplicated(["PLAYER_ID", "window_seasons", "window_end"]).any():
        raise ValueError("Rolling RAPM rating keys must be unique.")
    if not np.isfinite(rolling[[*COMPONENTS, "Poss_Off", "Poss_Def"]].to_numpy()).all():
        raise ValueError("Rolling RAPM ratings must be finite.")
    peaks = extract_player_peaks(rolling)
    quality = pd.DataFrame(quality_rows)

    _write_parquet_atomic(rolling, output / "rolling_ratings.parquet")
    _write_parquet_atomic(peaks, output / "player_peaks.parquet")
    _write_parquet_atomic(quality, output / "window_quality.parquet")
    top_peaks = peaks.loc[peaks["all_time_rank"].le(10)].to_dict(orient="records")
    run = {
        "run_id": run_id,
        "model_family": "rolling_zero_prior_normal_rapm_peaks",
        "estimand": contract["estimand"],
        "status": "research_leaderboard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "contract_version": contract["contract_version"],
            "season_range": [season_start, season_end],
            "window_lengths": list(window_lengths),
            "minimum_possessions_per_window_season": minimum_per_season,
            "lambda_off": float(model["lambda_off"]),
            "lambda_def": float(model["lambda_def"]),
            "lambda_home": float(model["lambda_home"]),
            "source_hashes": {
                "contract": identity_payload["contract"],
                "names": name_source_hashes,
                "source_code": identity_payload["source_code"],
                "possessions": source_hashes,
            },
        },
        "quality": {
            "windows": len(quality),
            "rolling_rating_rows": len(rolling),
            "peak_rows": len(peaks),
            "players": int(rolling["PLAYER_ID"].nunique()),
            "duplicate_rating_keys": 0,
            "duplicate_peak_keys": 0,
            "missing_rating_names": int(rolling["PLAYER_NAME"].isna().sum()),
            "missing_peak_names": int(peaks["PLAYER_NAME"].isna().sum()),
            "unresolved_player_ids": sorted(
                int(value)
                for value in rolling.loc[rolling["PLAYER_NAME"].isna(), "PLAYER_ID"].unique()
            ),
            "maximum_component_identity_error": float(
                quality["max_component_identity_error"].max()
            ),
        },
        "metrics": {"top_ten_peaks": top_peaks},
        "rolling_ratings_path": str((output / "rolling_ratings.parquet").resolve()),
        "player_peaks_path": str((output / "player_peaks.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": contract["caveats"],
    }
    write_json_atomic(run, output / "run.json")
    return run
