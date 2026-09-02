#!/usr/bin/env python3
"""Fit annual and nine-year normal RAPM from canonical lineup stints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import (
    build_stint_design,
    fit_stint_center_path,
    load_canonical_stints,
    stint_ratings,
)


ROOT = Path(__file__).resolve().parents[1]
STINT_ROOT = ROOT / "data/lake/silver/canonical_lineup_stints"
OUTPUT_ROOT = ROOT / "artifacts/models/canonical_stint_rapm"


def fit_window(seasons: tuple[int, ...], lambda_def: float) -> pd.DataFrame:
    frame = load_canonical_stints(STINT_ROOT, seasons)
    design = build_stint_design(frame)
    config = RapmConfig(
        seasons,
        lambda_off=3000,
        lambda_def=lambda_def,
        lambda_home=300,
        data_scope="canonical_score_conserving_lineup_stints",
    )
    beta, _ = fit_stint_center_path(
        design,
        config,
        np.zeros(design.X.shape[1]),
        center_scales=(0.0,),
    )[0.0]
    return stint_ratings(design, beta)


def main() -> None:
    manifest_path = STINT_ROOT / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if not source_manifest["passed"]:
        raise ValueError("Canonical lineup stint contract has not passed.")
    source_hashes = {
        str(season): sha256_file(STINT_ROOT / f"season={season}" / "regular.parquet")
        for season in range(1997, 2027)
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "sources": source_hashes,
                "annual": [3000, 4500, 300],
                "target": [3000, 3000, 300],
                "horizon": 9,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"canonical_stint_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    annual_rows = []
    for season in range(1997, 2027):
        path = checkpoints / f"annual_{season}.parquet"
        if not path.exists():
            ratings = fit_window((season,), 4500)
            ratings["Season"] = season
            ratings.to_parquet(path, index=False)
            print(f"Annual RAPM {season}: complete", flush=True)
        annual_rows.append(pd.read_parquet(path))

    target_rows = []
    for end in range(2005, 2027):
        path = checkpoints / f"target_9y_{end}.parquet"
        if not path.exists():
            seasons = tuple(range(end - 8, end + 1))
            ratings = fit_window(seasons, 3000).rename(
                columns={
                    "offense": "target_offense",
                    "defense": "target_defense",
                    "net": "target_net",
                }
            )
            ratings["Window_End"] = end
            ratings["horizon"] = 9
            ratings["target_variant"] = "normal"
            ratings.to_parquet(path, index=False)
            print(f"Nine-year RAPM ending {end}: complete", flush=True)
        target_rows.append(pd.read_parquet(path))

    annual = pd.concat(annual_rows, ignore_index=True)
    targets = pd.concat(target_rows, ignore_index=True)
    annual.to_parquet(output / "annual_rapm.parquet", index=False)
    targets.to_parquet(output / "targets.parquet", index=False)
    point_only = []
    for season in range(1997, 2027):
        frame = pd.read_parquet(
            STINT_ROOT / f"season={season}" / "regular.parquet",
            columns=["home_possessions", "away_possessions", "home_points", "away_points"],
        )
        point_only.append(
            {
                "season": season,
                "point_only_rows": int(
                    ((frame["home_possessions"].eq(0)) & frame["home_points"].gt(0)).sum()
                    + ((frame["away_possessions"].eq(0)) & frame["away_points"].gt(0)).sum()
                ),
                "point_only_points": int(
                    frame.loc[frame["home_possessions"].eq(0), "home_points"].sum()
                    + frame.loc[frame["away_possessions"].eq(0), "away_points"].sum()
                ),
            }
        )
    pd.DataFrame(point_only).to_parquet(output / "point_only_audit.parquet", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_hashes": source_hashes,
        "annual_penalties": {"offense": 3000, "defense": 4500, "home": 300},
        "target_penalties": {"offense": 3000, "defense": 3000, "home": 300},
        "target_horizon": 9,
        "annual_rows": int(len(annual)),
        "target_rows": int(len(targets)),
        "annual_identity_max_error": float(
            (annual["offense"] + annual["defense"] - annual["net"]).abs().max()
        ),
        "target_identity_max_error": float(
            (targets["target_offense"] + targets["target_defense"] - targets["target_net"])
            .abs()
            .max()
        ),
        "point_only_observations": "included_in_score_rhs_with_zero_possession_weight",
        "artifacts": {
            "annual_rapm": "annual_rapm.parquet",
            "targets": "targets.parquet",
            "point_only_audit": "point_only_audit.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
