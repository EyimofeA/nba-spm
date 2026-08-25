"""Build sequential rolling RAPM windows across the unified 2014-26 timeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import RapmConfig, load_unified_terminal_possessions
from nba_impact.models.rapm_sufficient_statistics import store_lambda_research_matrices
from nba_impact.models.rolling_rapm_peaks import (
    _write_parquet_atomic,
    extract_player_peaks,
    fit_rolling_rapm_window,
    load_peak_player_names,
)


LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "research" / "experiments" / "rolling_5y_rapm_2014_2026.json"
DEFAULT_CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
DEFAULT_PLAYER_GAMES = REPO_ROOT / "data" / "lake" / "silver" / "player_games.parquet"
DEFAULT_NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"
DEFAULT_SHEETS = REPO_ROOT / "data" / "raw" / "playersheets" / "year_totals"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "models" / "rolling_rapm_peaks"
DEFAULT_OUTPUT_ROOT = LAB_ROOT / "outputs" / "rolling_5y_2014_2026"
COMPONENTS = ("offense", "defense", "net")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _source_hashes(
    *,
    seasons: tuple[int, ...],
    transition_season: int,
    cache_dir: Path,
    possessions_path: Path,
    segments_path: Path,
    names_path: Path,
    player_sheets_dir: Path,
    player_games_path: Path,
) -> dict[str, str]:
    paths = [
        *(cache_dir / f"matchups_{season}.parquet" for season in seasons if season < transition_season),
        possessions_path,
        segments_path,
        player_games_path,
        names_path,
        *(player_sheets_dir / f"{season}.csv" for season in seasons),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Rolling RAPM inputs are missing: {missing}")
    return {_relative(path): sha256_file(path) for path in paths}


def compare_to_reference(
    rolling: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    window_seasons: int,
    transition_season: int,
) -> pd.DataFrame:
    """Compare common windows to the pinned legacy-only rolling build."""
    rows: list[dict] = []
    current = rolling.loc[rolling["window_seasons"].eq(window_seasons)]
    reference = reference.loc[reference["window_seasons"].eq(window_seasons)]
    for window_end in sorted(set(current["window_end"]) & set(reference["window_end"])):
        left = current.loc[current["window_end"].eq(window_end), ["PLAYER_ID", *COMPONENTS]]
        right = reference.loc[
            reference["window_end"].eq(window_end), ["PLAYER_ID", *COMPONENTS]
        ]
        matched = left.merge(right, on="PLAYER_ID", suffixes=("_current", "_reference"))
        row: dict[str, float | int | bool] = {
            "window_end": int(window_end),
            "matched_players": int(len(matched)),
            "same_input_contract": bool(window_end < transition_season),
        }
        for component in COMPONENTS:
            delta = matched[f"{component}_current"] - matched[f"{component}_reference"]
            row[f"{component}_max_abs_error"] = float(delta.abs().max())
            row[f"{component}_rmse"] = float(np.sqrt(np.mean(delta**2)))
            row[f"{component}_correlation"] = float(
                matched[[f"{component}_current", f"{component}_reference"]].corr().iloc[0, 1]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_rolling_rapm(
    contract_path: Path,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    possessions_path: Path = DEFAULT_POSSESSIONS,
    segments_path: Path = DEFAULT_SEGMENTS,
    names_path: Path = DEFAULT_NAMES,
    player_sheets_dir: Path = DEFAULT_SHEETS,
    player_games_path: Path = DEFAULT_PLAYER_GAMES,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    requested_window_ends: tuple[int, ...] | None = None,
    store_lambda_matrices: bool = True,
    resume_run_dir: Path | None = None,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "active_research":
        raise ValueError("The rolling RAPM experiment contract must be active research.")
    season_start, season_end = (int(value) for value in contract["season_range"])
    window_seasons = int(contract["window_seasons"])
    transition_season = int(contract["transition_season"])
    contract_ends = tuple(int(value) for value in contract["window_ends"])
    window_ends = requested_window_ends or contract_ends
    invalid = sorted(set(window_ends) - set(contract_ends))
    if invalid:
        raise ValueError(f"Requested window ends are outside the contract: {invalid}")
    all_seasons = tuple(range(season_start, season_end + 1))
    hashes = _source_hashes(
        seasons=all_seasons,
        transition_season=transition_season,
        cache_dir=cache_dir,
        possessions_path=possessions_path,
        segments_path=segments_path,
        names_path=names_path,
        player_sheets_dir=player_sheets_dir,
        player_games_path=player_games_path,
    )
    identity_payload = {
        "contract": sha256_file(contract_path),
        "source_code": sha256_file(Path(__file__)),
        "sufficient_statistics_code": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "rapm_sufficient_statistics.py"
        ),
        "sources": hashes,
        "window_ends": list(window_ends),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"rolling_5y_rapm_2014_2026_{identity}"
    output = output_root / run_id if resume_run_dir is None else resume_run_dir
    run_id = output.name
    completed = output / "run.json"
    if completed.exists():
        return json.loads(completed.read_text())
    checkpoint_root = output / "window_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    names, name_hashes = load_peak_player_names(names_path, player_sheets_dir, all_seasons)
    player_games = pd.read_parquet(
        player_games_path,
        columns=["player_id", "player_name", "game_date"],
    ).dropna(subset=["player_id", "player_name"])
    player_games["PLAYER_ID"] = pd.to_numeric(
        player_games["player_id"], errors="coerce"
    ).astype("Int64")
    current_names = (
        player_games.dropna(subset=["PLAYER_ID"])
        .sort_values(["game_date", "PLAYER_ID"], kind="stable")
        .drop_duplicates("PLAYER_ID", keep="last")
        .rename(columns={"player_name": "PLAYER_NAME"})
        [["PLAYER_ID", "PLAYER_NAME"]]
    )
    current_names["PLAYER_ID"] = current_names["PLAYER_ID"].astype(int)
    current_names["name_source"] = "canonical_player_games_fallback"
    names = pd.concat(
        [names, current_names.loc[~current_names["PLAYER_ID"].isin(names["PLAYER_ID"])]],
        ignore_index=True,
    )
    name_hashes[str(player_games_path.resolve())] = sha256_file(player_games_path)
    model = contract["model"]
    threshold = int(contract["peak_eligibility"]["minimum_possessions_per_side_per_season"])
    rating_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    matrix_rows: list[dict] = []

    for window_end in window_ends:
        window_start = window_end - window_seasons + 1
        seasons = tuple(range(window_start, window_end + 1))
        checkpoint = checkpoint_root / f"{window_seasons}y_end_{window_end}.parquet"
        quality_checkpoint = checkpoint_root / f"{window_seasons}y_end_{window_end}.json"
        matrix_root = output / "lambda_matrices" / f"{window_seasons}y_end_{window_end}"
        matrix_manifest_path = matrix_root / "manifest.json"
        frame: pd.DataFrame | None = None
        if checkpoint.exists() and quality_checkpoint.exists():
            ratings = pd.read_parquet(checkpoint)
            quality = json.loads(quality_checkpoint.read_text())
            if ratings["PLAYER_NAME"].isna().any():
                lookup = names.set_index("PLAYER_ID")
                missing = ratings["PLAYER_NAME"].isna()
                ratings.loc[missing, "PLAYER_NAME"] = ratings.loc[
                    missing, "PLAYER_ID"
                ].map(lookup["PLAYER_NAME"])
                ratings.loc[missing, "name_source"] = ratings.loc[
                    missing, "PLAYER_ID"
                ].map(lookup["name_source"])
                _write_parquet_atomic(ratings, checkpoint)
            print(f"{window_start}-{window_end}: checkpointed", flush=True)
        else:
            frame = load_unified_terminal_possessions(
                cache_dir,
                possessions_path,
                segments_path,
                seasons,
                transition_season=transition_season,
                game_types=("regular",),
            )
            source_by_season = dict(frame.attrs["source_by_season"])
            config = RapmConfig(
                seasons=seasons,
                lambda_off=float(model["lambda_off"]),
                lambda_def=float(model["lambda_def"]),
                lambda_home=float(model["lambda_home"]),
                game_types=("regular",),
                data_scope="unified_rolling_5y_terminal_research",
            )
            ratings, quality = fit_rolling_rapm_window(
                frame,
                config,
                window_start=window_start,
                window_end=window_end,
                minimum_possessions_per_window_season=threshold,
            )
            ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
            quality["source_by_season"] = source_by_season
            _write_parquet_atomic(ratings, checkpoint)
            write_json_atomic(quality, quality_checkpoint)
            print(
                f"{window_start}-{window_end}: {quality['games']:,} games, "
                f"{quality['possession_rows']:,} possessions, {len(ratings):,} players",
                flush=True,
            )
        if store_lambda_matrices:
            evaluation = None
            if matrix_manifest_path.exists():
                matrix_manifest = json.loads(matrix_manifest_path.read_text())
            else:
                if frame is None:
                    frame = load_unified_terminal_possessions(
                        cache_dir,
                        possessions_path,
                        segments_path,
                        seasons,
                        transition_season=transition_season,
                        game_types=("regular",),
                    )
                evaluation_season = window_end + 1
                evaluation_sources: dict[str, str] = {}
                if evaluation_season <= season_end:
                    evaluation = load_unified_terminal_possessions(
                        cache_dir,
                        possessions_path,
                        segments_path,
                        (evaluation_season,),
                        transition_season=transition_season,
                        game_types=("regular",),
                    )
                    evaluation_sources = dict(evaluation.attrs["source_by_season"])
                matrix_manifest = store_lambda_research_matrices(
                    frame,
                    matrix_root,
                    evaluation_frame=evaluation,
                    metadata={
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_seasons": window_seasons,
                        "train_source_by_season": dict(frame.attrs["source_by_season"]),
                        "evaluation_source_by_season": evaluation_sources,
                        "source_hash_digest": hashlib.sha256(
                            json.dumps(hashes, sort_keys=True).encode()
                        ).hexdigest(),
                        "season_2027_loaded": False,
                    },
                )
            matrix_rows.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "train_rows": matrix_manifest["train"]["possession_rows"],
                    "players": matrix_manifest["train"]["players_in_train_or_evaluation"],
                    "matrix_columns": matrix_manifest["train"]["matrix_shape"][0],
                    "matrix_nonzero": matrix_manifest["train"]["matrix_nonzero"],
                    "evaluation_status": matrix_manifest["evaluation"]["status"],
                    "evaluation_season": matrix_manifest["evaluation"].get("season"),
                    "evaluation_games": matrix_manifest["evaluation"].get("games", 0),
                    "manifest": _relative(matrix_manifest_path),
                }
            )
            print(
                f"  lambda matrices: {matrix_manifest['train']['matrix_shape']} "
                f"nnz={matrix_manifest['train']['matrix_nonzero']:,} "
                f"eval={matrix_manifest['evaluation']['status']}",
                flush=True,
            )
            del evaluation
        if frame is not None:
            del frame
        gc.collect()
        rating_frames.append(ratings)
        quality_rows.append(quality)

    rolling = pd.concat(rating_frames, ignore_index=True)
    quality_frame = pd.DataFrame(quality_rows).sort_values("window_end")
    matrix_catalog = pd.DataFrame(matrix_rows)
    if not matrix_catalog.empty:
        matrix_catalog = matrix_catalog.sort_values("window_end")
    if rolling.duplicated(["PLAYER_ID", "window_seasons", "window_end"]).any():
        raise ValueError("Rolling RAPM rating keys must be unique.")
    if not np.isfinite(rolling[[*COMPONENTS, "Poss_Off", "Poss_Def"]].to_numpy()).all():
        raise ValueError("Rolling RAPM ratings must be finite.")
    if rolling["PLAYER_NAME"].isna().any():
        unresolved = sorted(rolling.loc[rolling["PLAYER_NAME"].isna(), "PLAYER_ID"].unique())
        raise ValueError(f"Rolling RAPM has unresolved player names: {unresolved}")
    peaks = extract_player_peaks(rolling)

    reference_run_id = contract["reference_run_id"]
    reference_path = artifact_root / reference_run_id / "rolling_ratings.parquet"
    reference = pd.read_parquet(reference_path)
    comparison = compare_to_reference(
        rolling,
        reference,
        window_seasons=window_seasons,
        transition_season=transition_season,
    )
    acceptance = contract["acceptance"]
    unchanged = comparison.loc[
        comparison["window_end"].isin(acceptance["unchanged_source_window_ends"])
    ]
    component_error_columns = [f"{component}_max_abs_error" for component in COMPONENTS]
    max_reference_error = float(unchanged[component_error_columns].to_numpy().max())
    max_identity_error = float(quality_frame["max_component_identity_error"].max())
    is_full_run = tuple(window_ends) == contract_ends
    gates = {
        "full_window_count": (not is_full_run) or len(quality_frame) == int(acceptance["expected_windows"]),
        "unchanged_windows_reproduce_reference": max_reference_error
        <= float(acceptance["maximum_reference_component_error"]),
        "component_identity": max_identity_error
        <= float(acceptance["maximum_component_identity_error"]),
        "finite_ratings": bool(np.isfinite(rolling[[*COMPONENTS, "Poss_Off", "Poss_Def"]]).all().all()),
        "unique_rating_keys": not rolling.duplicated(["PLAYER_ID", "window_seasons", "window_end"]).any(),
        "complete_names": not rolling["PLAYER_NAME"].isna().any(),
        "lambda_training_matrices": (not store_lambda_matrices)
        or len(matrix_catalog) == len(window_ends),
        "lambda_evaluation_matrices": (not store_lambda_matrices)
        or int(matrix_catalog["evaluation_status"].eq("complete").sum())
        == sum(window_end < season_end for window_end in window_ends),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Rolling RAPM acceptance failed: {gates}")

    _write_parquet_atomic(rolling, output / "rolling_ratings.parquet")
    _write_parquet_atomic(peaks, output / "player_peaks.parquet")
    _write_parquet_atomic(quality_frame, output / "window_quality.parquet")
    _write_parquet_atomic(comparison, output / "reference_comparison.parquet")
    if store_lambda_matrices:
        _write_parquet_atomic(matrix_catalog, output / "lambda_matrix_catalog.parquet")
    latest = rolling.loc[rolling["window_end"].eq(max(window_ends)) & rolling["peak_eligible"]]
    latest_top = latest.nlargest(10, "net")[
        ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net", "Poss_Off", "Poss_Def"]
    ]
    run = {
        "run_id": run_id,
        "experiment_id": contract["experiment_id"],
        "status": "research_build_complete" if is_full_run else "research_pilot_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": contract["estimand"],
        "config": {
            "season_range": [season_start, season_end],
            "window_seasons": window_seasons,
            "window_ends": list(window_ends),
            "transition_season": transition_season,
            "lambda_off": float(model["lambda_off"]),
            "lambda_def": float(model["lambda_def"]),
            "lambda_home": float(model["lambda_home"]),
            "minimum_possessions_per_side_per_season": threshold,
            "sources": contract["sources"],
            "source_hashes": hashes,
            "source_code_hash": identity_payload["source_code"],
            "sufficient_statistics_code_hash": identity_payload[
                "sufficient_statistics_code"
            ],
            "name_source_hashes": {
                _relative(Path(path)): digest for path, digest in name_hashes.items()
            },
            "lambda_research": contract["lambda_research"],
        },
        "quality": {
            "windows": int(len(quality_frame)),
            "rating_rows": int(len(rolling)),
            "players": int(rolling["PLAYER_ID"].nunique()),
            "peak_rows": int(len(peaks)),
            "maximum_component_identity_error": max_identity_error,
            "maximum_unchanged_reference_error": max_reference_error,
            "gates": gates,
        },
        "latest_window_top_ten_net": latest_top.to_dict(orient="records"),
        "paths": {
            "artifact": _relative(output),
            "rolling_ratings": _relative(output / "rolling_ratings.parquet"),
            "player_peaks": _relative(output / "player_peaks.parquet"),
            "window_quality": _relative(output / "window_quality.parquet"),
            "reference_comparison": _relative(output / "reference_comparison.parquet"),
            "lambda_matrix_catalog": (
                _relative(output / "lambda_matrix_catalog.parquet")
                if store_lambda_matrices
                else None
            ),
        },
        "evidence_status": contract["evidence_status"],
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, completed)
    print(latest_top.to_string(index=False, float_format=lambda value: f"{value:+.2f}"), flush=True)
    print(f"wrote {run_id} -> {output}", flush=True)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-ends", nargs="+", type=int)
    parser.add_argument("--skip-lambda-matrices", action="store_true")
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()
    run = build_rolling_rapm(
        args.contract,
        output_root=args.output_root,
        requested_window_ends=tuple(args.window_ends) if args.window_ends else None,
        store_lambda_matrices=not args.skip_lambda_matrices,
        resume_run_dir=args.resume_run_dir,
    )
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
