#!/usr/bin/env python3
"""Validate the CourtSignal PULSE release candidate and write its manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a"
DECOMPOSITION = ROOT / "artifacts/models/pulse_decomposition/pulse_decomposition_v1_7c856fd94b"
CANONICAL = ROOT / "data/lake/silver/canonical_lineup_stints"
PUBLIC = ROOT / "web/public/data"


def main() -> None:
    data = json.loads((CANONICAL / "manifest.json").read_text())
    pulse = json.loads((PULSE / "run.json").read_text())
    decomposition = json.loads((DECOMPOSITION / "run.json").read_text())
    ratings = pd.read_parquet(PULSE / "ratings.parquet")
    folds = pd.read_parquet(PULSE / "validation_folds.parquet")
    subgroups = pd.read_parquet(PULSE / "validation_subgroups.parquet")
    summary = folds.pivot(index="outcome_season", columns="candidate", values="mse")
    validation = pulse["validation"]
    pulse_summary = next(row for row in validation["summary"] if row["candidate"] == "pulse")
    rapm_summary = next(row for row in validation["summary"] if row["candidate"] == "rapm")
    bootstrap = validation["paired_bootstrap"]
    identity = {
        "prior": float((ratings["pulse_prior_offense"] + ratings["pulse_prior_defense"] - ratings["pulse_prior_net"]).abs().max()),
        "update": float((ratings["lineup_update_offense"] + ratings["lineup_update_defense"] - ratings["lineup_update_net"]).abs().max()),
        "pulse": float((ratings["pulse_prior_net"] + ratings["lineup_update_net"] - ratings["pulse_net"]).abs().max()),
    }
    initial = [PUBLIC / "catalog.json", PUBLIC / "players.json", PUBLIC / "leaderboard-2026.json"]
    initial_gzip = sum(len(gzip.compress(path.read_bytes(), compresslevel=9)) for path in initial)
    rapm_catalog = json.loads((PUBLIC / "rapm/catalog.json").read_text())
    gates = {
        "canonical_data": bool(
            data["passed"]
            and data["minimum_native_score_reconciliation"] >= 0.99
            and data["minimum_valid_lineup_fraction"] >= 0.95
            and data["unmatched_technical_free_throws"] == 0
        ),
        "pulse_identity": max(identity.values()) <= 1e-9,
        "rmse_improvement": rapm_summary["equal_season_rmse"] - pulse_summary["equal_season_rmse"] >= 0.05,
        "paired_interval": bootstrap["upper_95"] < 0,
        "correlation": pulse_summary["mean_correlation"] >= rapm_summary["mean_correlation"] - 0.01,
        "season_wins": int((summary["pulse"] < summary["rapm"]).sum()) > len(summary) / 2,
        "subgroups": bool(subgroups["equal_season_mse_delta_pulse_minus_rapm"].le(0).all()),
        "decomposition": bool(all(decomposition["identity_checks"].values())),
        "public_matchup_absent": not (PUBLIC / "matchup-lab.json").exists(),
        "initial_payload": initial_gzip < 250_000,
        "rapm_sweeps_absent": not any(
            token in json.dumps(rapm_catalog).casefold() for token in ("lambda sweep", "tuning sweep")
        ),
    }
    payload = {
        "pulse": sha256_file(PULSE / "run.json"),
        "canonical": sha256_file(CANONICAL / "manifest.json"),
        "decomposition": sha256_file(DECOMPOSITION / "run.json"),
        "web": sha256_file(PUBLIC / "snapshot-manifest.json"),
        "rapm": sha256_file(PUBLIC / "rapm/catalog.json"),
    }
    run_id = "courtsignal_pulse_release_v1_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/releases" / run_id
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(all(gates.values())),
        "gates": gates,
        "data_quality": data,
        "validation": {
            "pulse_rmse": pulse_summary["equal_season_rmse"],
            "rapm_rmse": rapm_summary["equal_season_rmse"],
            "rmse_improvement": rapm_summary["equal_season_rmse"] - pulse_summary["equal_season_rmse"],
            "mse_delta": bootstrap["pulse_minus_rapm_mse"],
            "mse_interval_95": [bootstrap["lower_95"], bootstrap["upper_95"]],
            "pulse_season_wins": int((summary["pulse"] < summary["rapm"]).sum()),
            "seasons": int(len(summary)),
            "subgroups": subgroups.to_dict("records"),
        },
        "initial_payload_gzip_bytes": initial_gzip,
        "source_hashes": payload,
    }
    write_json_atomic(report, output / "release_manifest.json")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("CourtSignal release candidate failed acceptance.")


if __name__ == "__main__":
    main()
