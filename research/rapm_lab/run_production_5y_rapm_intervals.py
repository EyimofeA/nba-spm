"""Attach fast fixed-window analytic intervals to rolling five-year RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_homoskedastic_ridge_intervals


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/production_5y_rapm_intervals_v1.json"
SOURCE = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77"
OUTPUT = ROOT / "research/rapm_lab/outputs/production_5y_rapm_intervals"


def run(window_ends: tuple[int, ...] | None = None) -> dict:
    contract = json.loads(CONTRACT.read_text())
    requested = window_ends or tuple(contract["window_ends"])
    if invalid := sorted(set(requested) - set(contract["window_ends"])):
        raise ValueError(f"Window ends are outside the contract: {invalid}")
    identity = hashlib.sha256(json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"production_5y_rapm_intervals_v1_{identity}"
    checkpoint = output / "window_checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    source_ratings = pd.read_parquet(SOURCE / "rolling_ratings.parquet")
    penalties = contract["penalties"]
    quality_rows = []
    frames = []
    started = time.perf_counter()
    for end in requested:
        path = checkpoint / f"window_end_{end}.parquet"
        quality_path = checkpoint / f"window_end_{end}.json"
        if path.exists() and quality_path.exists():
            frame = pd.read_parquet(path)
            quality = json.loads(quality_path.read_text())
        else:
            matrix = SOURCE / "lambda_matrices" / f"5y_end_{end}"
            fit_started = time.perf_counter()
            frame, quality = stored_homoskedastic_ridge_intervals(
                matrix,
                lambda_off=penalties["lambda_off"],
                lambda_def=penalties["lambda_def"],
                lambda_home=penalties["lambda_home"],
            )
            reference = source_ratings.loc[
                source_ratings["window_end"].eq(end),
                ["PLAYER_ID", "PLAYER_NAME", "offense", "defense", "net"],
            ].rename(
                columns={
                    "offense": "reference_offense",
                    "defense": "reference_defense",
                    "net": "reference_net",
                }
            )
            frame = frame.merge(reference, on="PLAYER_ID", validate="one_to_one")
            maximum_reference_error = max(
                float((frame[component] - frame[f"reference_{component}"]).abs().max())
                for component in ("offense", "defense", "net")
            )
            if maximum_reference_error > 1e-6:
                raise AssertionError(
                    "Interval point estimates do not reproduce rolling RAPM."
                )
            frame = frame.drop(
                columns=["reference_offense", "reference_defense", "reference_net"]
            )
            frame["window_start"] = end - contract["window_seasons"] + 1
            frame["window_end"] = end
            quality["elapsed_seconds"] = time.perf_counter() - fit_started
            quality["maximum_reference_rating_error"] = maximum_reference_error
            frame.to_parquet(path, index=False)
            write_json_atomic(quality, quality_path)
        if "maximum_reference_rating_error" not in quality:
            reference = source_ratings.loc[
                source_ratings["window_end"].eq(end)
            ].set_index("PLAYER_ID")
            indexed = frame.set_index("PLAYER_ID")
            maximum_reference_error = max(
                float(
                    (indexed[component] - reference[component])
                    .abs()
                    .max()
                )
                for component in ("offense", "defense", "net")
            )
            if maximum_reference_error > 1e-6:
                raise AssertionError(
                    "Interval point estimates do not reproduce rolling RAPM."
                )
            quality["maximum_reference_rating_error"] = maximum_reference_error
            write_json_atomic(quality, quality_path)
        print(f"intervals window end {end}: {len(frame)} players", flush=True)
        frames.append(frame)
        quality_rows.append({"window_end": end, **quality})
    ratings = pd.concat(frames, ignore_index=True)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_parquet(output / "quality.parquet", index=False)
    complete = set(requested) == set(contract["window_ends"])
    manifest = {
        "run_id": output.name,
        "status": "research_fixed_window_intervals_complete" if complete else "research_pilot_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "quality": {
            "windows": len(requested),
            "rating_rows": int(len(ratings)),
            "maximum_component_identity_error": float(quality_frame["maximum_component_identity_error"].max()),
            "maximum_reference_rating_error": float(quality_frame["maximum_reference_rating_error"].max()),
        },
        "paths": {"ratings": "ratings.parquet", "quality": "quality.parquet"},
        "forbidden_interpretation": "Bootstrap uncertainty, selected-peak uncertainty, causal ability, or Season 2027 confirmation.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-ends", nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(run(tuple(args.window_ends) if args.window_ends else None), indent=2))
