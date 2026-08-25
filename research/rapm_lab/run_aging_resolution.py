"""Run the bounded aging-resolution and trailing-window experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.aging_resolution import evaluate_aging_resolutions, load_age_panel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = (
    REPO_ROOT
    / "artifacts/models/canonical_annual_target_panel"
    / "canonical_annual_target_panel_v1_4586bd2f72/targets.parquet"
)
DEFAULT_AGES = REPO_ROOT / "data/raw/playersheets/year_totals"
DEFAULT_OUTPUT = REPO_ROOT / "research/rapm_lab/outputs/aging_resolution"


def run(targets_path: Path, age_dir: Path, output_root: Path) -> dict:
    targets = pd.read_parquet(targets_path)
    if targets["Season"].max() >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    seasons = tuple(sorted(targets["Season"].astype(int).unique()))
    ages = load_age_panel(age_dir, seasons)
    result = evaluate_aging_resolutions(targets, ages)
    identity = hashlib.sha256(
        json.dumps(
            {
                "targets": sha256_file(targets_path),
                "ages": {str(season): sha256_file(age_dir / f"{season}.csv") for season in seasons},
                "bandwidths": [0.1, 0.5, 1.0, 2.0],
                "windows": [1, 3, 5],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"aging_resolution_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    result.metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    result.curves.to_parquet(output / "curves.parquet", index=False)
    summary = (
        result.metrics.groupby(
            ["trailing_window_seasons", "component", "variant", "bandwidth_years"],
            dropna=False,
            as_index=False,
        )
        .agg(
            folds=("origin_season", "nunique"),
            rows=("rows", "sum"),
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
        )
        .sort_values(["component", "trailing_window_seasons", "mean_weighted_rmse"])
    )
    summary.to_parquet(output / "summary.parquet", index=False)
    winners = (
        summary.sort_values(
            ["component", "trailing_window_seasons", "mean_weighted_rmse", "variant"],
            kind="stable",
        )
        .groupby(["component", "trailing_window_seasons"], as_index=False)
        .head(1)
    )
    run_manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "model_family": "walk_forward_kernel_aging_resolution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next_season_annual_rapm_from_trailing_annual_rapm_plus_expected_age_change",
        "config": {
            "bandwidth_years": [0.1, 0.5, 1.0, 2.0],
            "trailing_window_seasons": [1, 3, 5],
            "minimum_training_origins": 3,
            "source_hashes": {
                "targets": sha256_file(targets_path),
                "source_code": sha256_file(
                    REPO_ROOT / "src/nba_impact/models/aging_resolution.py"
                ),
            },
        },
        "quality": result.quality,
        "winners": winners.to_dict(orient="records"),
        "paths": {
            "fold_metrics": str((output / "fold_metrics.parquet").relative_to(REPO_ROOT)),
            "summary": str((output / "summary.parquet").relative_to(REPO_ROOT)),
            "predictions": str((output / "predictions.parquet").relative_to(REPO_ROOT)),
            "curves": str((output / "curves.parquet").relative_to(REPO_ROOT)),
        },
        "caveats": [
            "Stored ages are season-level integers, so subannual bandwidths do not create subannual age evidence.",
            "The 0.1-year and 0.5-year candidates can be numerically equivalent when only integer ages are available.",
            "The trailing-window baseline averages annual RAPM observations; it is not a refit of a pooled possession matrix.",
            "All scored seasons are reused diagnostics and Season 2027 is excluded.",
        ],
    }
    write_json_atomic(run_manifest, output / "run.json")
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--age-dir", type=Path, default=DEFAULT_AGES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.targets, args.age_dir, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
