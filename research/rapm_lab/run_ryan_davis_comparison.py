"""Compare the 2019 normal RAPM panel with Ryan Davis's public tutorial output."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data/lake/bronze/external_rapm/ryan_davis/rapm.csv"
TARGETS = ROOT / (
    "artifacts/research/historical_official_preferred_normal_rapm/"
    "project_season=2019/models/current_single_season_rapm_targets/"
    "current_single_season_rapm_targets_v1_31aa665391/targets.parquet"
)
OUTPUT = ROOT / "research/rapm_lab/outputs/ryan_davis_comparison"
SOURCE_URL = "https://github.com/rd11490/NBA_Tutorials/tree/master/rapm"


def component_metrics(frame: pd.DataFrame, ours: str, external: str) -> dict:
    return {
        "pearson_correlation": float(frame[ours].corr(frame[external])),
        "spearman_correlation": float(frame[ours].corr(frame[external], method="spearman")),
        "mean_absolute_difference": float(np.abs(frame[ours] - frame[external]).mean()),
        "ours_standard_deviation": float(frame[ours].std()),
        "external_standard_deviation": float(frame[external].std()),
        "ours_on_external_scale_slope": float(np.polyfit(frame[external], frame[ours], 1)[0]),
    }


def run() -> dict:
    external = pd.read_csv(EXTERNAL)
    targets = pd.read_parquet(TARGETS)
    comparison = targets.merge(
        external,
        left_on="PLAYER_ID",
        right_on="playerId",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "playerName": "PLAYER_NAME",
            "RAPM__Off": "ryan_offense",
            "RAPM__Def": "ryan_defense",
            "RAPM": "ryan_net",
        }
    )
    comparison["net_difference"] = comparison["target_net"] - comparison["ryan_net"]
    metrics = {
        "offense": component_metrics(comparison, "target_offense", "ryan_offense"),
        "defense": component_metrics(comparison, "target_defense", "ryan_defense"),
        "net": component_metrics(comparison, "target_net", "ryan_net"),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "runner": sha256_file(Path(__file__)),
                "external": sha256_file(EXTERNAL),
                "targets": sha256_file(TARGETS),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"ryan_davis_comparison_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "Poss_Off",
        "Poss_Def",
        "target_offense",
        "target_defense",
        "target_net",
        "ryan_offense",
        "ryan_defense",
        "ryan_net",
        "net_difference",
    ]
    comparison[columns].to_parquet(output / "comparison.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "external_reproduction_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season": 2019,
        "source": SOURCE_URL,
        "matched_players": int(len(comparison)),
        "external_rows": int(len(external)),
        "internal_rows": int(len(targets)),
        "metrics": metrics,
        "method_difference": (
            "Both are possession-level offense/defense ridge RAPM. CourtSignal adds a home term, "
            "uses canonical terminal-lineup QA, fixed 3000/3000 penalties, and possession-weighted "
            "component recentering. The tutorial uses its own parser, no home term, RidgeCV, and rounded output."
        ),
        "forbidden_interpretation": (
            "An independent predictive validation or exact source-data reproduction; the possession parsers "
            "and included games differ."
        ),
        "paths": {"comparison": "comparison.parquet"},
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
