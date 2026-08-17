"""Forward-only smoothing for descriptive player-role memberships."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


DEFAULT_CURRENT_WEIGHTS = (0.40, 0.55, 0.70, 0.85, 1.00)
SIDE_CONFIG = {
    "offense": {"prefix": "off_role", "development_seasons": (2014, 2015, 2016, 2017, 2018)},
    "defense": {"prefix": "def_role", "development_seasons": (2018, 2019, 2020, 2021)},
}


def _smooth_assignments(
    frame: pd.DataFrame,
    prefix: str,
    current_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth consecutive seasons and reset after a missing season."""
    if not 0 < current_weight <= 1:
        raise ValueError("current_weight must be in (0, 1].")
    affinity_columns = sorted(
        column for column in frame if column.startswith(f"{prefix}_affinity_")
    )
    if not affinity_columns:
        raise ValueError(f"No {prefix} affinity columns were found.")
    work = frame.sort_values(["PLAYER_ID", "Season"], kind="stable").reset_index(drop=True)
    stable = np.empty((len(work), len(affinity_columns)), dtype=float)
    prediction_rows = []
    for _, rows in work.groupby("PLAYER_ID", sort=False).groups.items():
        previous: np.ndarray | None = None
        previous_season: int | None = None
        for row_index in rows:
            season = int(work.at[row_index, "Season"])
            current = work.loc[row_index, affinity_columns].to_numpy(dtype=float)
            if previous is None or previous_season is None or season != previous_season + 1:
                value = current
            else:
                prediction_rows.append(
                    {
                        "PLAYER_ID": int(work.at[row_index, "PLAYER_ID"]),
                        "Season": season,
                        "squared_error": float(np.mean((current - previous) ** 2)),
                    }
                )
                value = current_weight * current + (1.0 - current_weight) * previous
            stable[row_index] = value / value.sum()
            previous = stable[row_index]
            previous_season = season

    output = work.copy()
    for index, column in enumerate(affinity_columns):
        stable_column = column.replace("_affinity_", "_stable_affinity_")
        output[stable_column] = stable[:, index]
    stable_labels = stable.argmax(axis=1)
    output[f"{prefix}_stable_cluster"] = [
        f"{prefix}_{index}" for index in stable_labels
    ]
    output[f"{prefix}_stable_confidence"] = stable.max(axis=1)
    return output, pd.DataFrame(prediction_rows)


def _adjacent_metrics(frame: pd.DataFrame, prefix: str, development: tuple[int, ...]) -> dict:
    current = frame[
        ["PLAYER_ID", "Season", f"{prefix}_cluster", f"{prefix}_stable_cluster"]
    ].copy()
    previous = current.copy()
    previous["Season"] += 1
    previous = previous.rename(
        columns={
            f"{prefix}_cluster": "previous_raw_cluster",
            f"{prefix}_stable_cluster": "previous_stable_cluster",
        }
    )
    adjacent = current.merge(previous, on=["PLAYER_ID", "Season"], validate="one_to_one")
    later = adjacent.loc[~adjacent["Season"].isin(development)].copy()
    return {
        "later_adjacent_pairs": int(len(later)),
        "later_raw_same_role_rate": float(
            later[f"{prefix}_cluster"].eq(later["previous_raw_cluster"]).mean()
        ),
        "later_stable_same_role_rate": float(
            later[f"{prefix}_stable_cluster"].eq(later["previous_stable_cluster"]).mean()
        ),
        "later_stable_raw_disagreement_rate": float(
            later[f"{prefix}_stable_cluster"].ne(later[f"{prefix}_cluster"]).mean()
        ),
    }


def build_role_stabilization(
    side_roles_run_path: str | Path,
    *,
    artifact_root: str | Path,
    candidate_current_weights: tuple[float, ...] = DEFAULT_CURRENT_WEIGHTS,
) -> dict:
    """Select target-free smoothing on development seasons and freeze it later."""
    source = Path(side_roles_run_path)
    if not candidate_current_weights or any(
        value <= 0 or value > 1 for value in candidate_current_weights
    ):
        raise ValueError("candidate_current_weights must contain values in (0, 1].")

    outputs: dict[str, pd.DataFrame] = {}
    candidate_rows = []
    metrics = {}
    source_hashes = {"run": sha256_file(source / "run.json")}
    selected_weights = {}
    for side, side_config in SIDE_CONFIG.items():
        prefix = side_config["prefix"]
        development = side_config["development_seasons"]
        path = source / f"{side}_assignments.parquet"
        frame = pd.read_parquet(path)
        source_hashes[side] = sha256_file(path)
        candidates = []
        for current_weight in candidate_current_weights:
            stabilized, prediction = _smooth_assignments(frame, prefix, current_weight)
            development_prediction = prediction.loc[
                prediction["Season"].isin(development)
            ]
            mse = float(development_prediction["squared_error"].mean())
            candidates.append((current_weight, mse, stabilized))
            candidate_rows.append(
                {
                    "side": side,
                    "current_weight": current_weight,
                    "development_prediction_mse": mse,
                    "development_pairs": len(development_prediction),
                }
            )
        selected_weight, selected_mse, selected = min(
            candidates, key=lambda item: (item[1], -item[0])
        )
        selected_weights[side] = selected_weight
        outputs[side] = selected
        metrics[side] = {
            "selected_current_weight": selected_weight,
            "selected_development_prediction_mse": selected_mse,
            **_adjacent_metrics(selected, prefix, development),
        }

    identity_payload = {
        "source_hashes": source_hashes,
        "candidate_current_weights": candidate_current_weights,
        "selected_weights": selected_weights,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"role_stabilization_v1_{identity}"
    output = Path(artifact_root) / "features" / "role_stabilization" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for side, frame in outputs.items():
        frame.to_parquet(output / f"{side}_assignments.parquet", index=False)
    pd.DataFrame(candidate_rows).to_parquet(
        output / "candidate_metrics.parquet", index=False
    )
    run = {
        "run_id": run_id,
        "dataset": "forward_filtered_side_role_memberships_v1",
        "status": "validated_descriptive_stabilization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "source_side_roles_run_id": json.loads((source / "run.json").read_text())["run_id"],
            "candidate_current_weights": list(candidate_current_weights),
            "selection_rule": "minimum prior-state prediction MSE on development-season adjacent pairs",
            "gap_rule": "reset to current raw affinities after any missing season",
            "source_hashes": source_hashes,
            "source_code": sha256_file(Path(__file__)),
        },
        "metrics": metrics,
        "quality": {
            "offense_rows": len(outputs["offense"]),
            "defense_rows": len(outputs["defense"]),
            "nonfinite_values": int(
                sum(
                    (~np.isfinite(
                        frame.filter(like="_stable_affinity_").to_numpy(dtype=float)
                    )).sum()
                    for frame in outputs.values()
                )
            ),
        },
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Stabilization is a forward-only display layer and is not an SPM feature.",
            "A stable label can lag a real role change.",
            "The selection target is next-season raw role affinity, not player impact.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
