#!/usr/bin/env python3
"""Build the selected PULSE factor ledger."""

from pathlib import Path

from nba_impact.models.pulse_decomposition import build_pulse_decomposition


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    print(build_pulse_decomposition(
        pulse_run=ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a",
        factor_target_run=ROOT / "artifacts/research/historical_factor_targets/historical_factor_targets_v2_6cd7e959eb",
        factor_prediction_run=ROOT / "artifacts/research/historical_factor_residual_tournament/historical_factor_residual_tournament_v2_c06bdebcd5",
        artifact_root=ROOT / "artifacts",
    ))
