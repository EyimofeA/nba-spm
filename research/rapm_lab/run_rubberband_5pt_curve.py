"""Fit the descriptive five-point score-state curve after its rating test fails."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    load_unified_terminal_possessions,
)
from nba_impact.models.rubberband_score_state import (
    annotate_offense_margin_before,
    fit_score_state_rapm,
    score_state_curve,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/rubberband_5pt_lambda_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/rubberband_5pt_curve"


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    baseline = next(c for c in contract["lambda_candidates"] if c["name"] == "baseline")
    source = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        contract["seasons"],
        transition_season=2024,
        game_types=("regular",),
    )
    source = annotate_offense_margin_before(source)
    design = build_design(source, include_home=True)
    bucket = contract["score_bucket"]
    fit = fit_score_state_rapm(
        design,
        source["offense_margin_before"].to_numpy(dtype=float),
        RapmConfig(
            seasons=tuple(contract["seasons"]),
            lambda_off=float(baseline["lambda_off"]),
            lambda_def=float(baseline["lambda_def"]),
            lambda_home=float(contract["lambda_home"]),
            data_scope="descriptive_five_point_score_bucket_curve",
        ),
        minimum=int(bucket["minimum"]),
        maximum=int(bucket["maximum"]),
        bucket_width=int(bucket["width"]),
        state_penalty=float(bucket["penalty"]),
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(CONTRACT),
                "runner": sha256_file(Path(__file__)),
                "model": sha256_file(ROOT / "src/nba_impact/models/rubberband_score_state.py"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"rubberband_5pt_curve_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    curve = score_state_curve(fit)
    curve.to_parquet(output / "score_bucket_curve.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "descriptive_rejected_adjustment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": contract["seasons"],
        "score_bucket": bucket,
        "possessions": int(len(source)),
        "curve": curve.to_dict("records"),
        "decision": "display the effect; do not alter production player ratings",
        "artifact": "score_bucket_curve.parquet",
    }
    write_json_atomic(run, output / "run.json")
    return run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
