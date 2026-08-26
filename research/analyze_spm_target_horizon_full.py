"""Paired whole-game uncertainty for the frozen target-horizon development run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / (
    "artifacts/research/spm_target_horizon_full/"
    "spm_target_horizon_full_v1_f0777db1d4"
)
OUTPUT = ROOT / "research/audits/spm_target_horizon_full_v1"
SELECTED = "5y__zero_prior"


def _candidate_key(frame: pd.DataFrame) -> pd.Series:
    return frame["horizon"].astype(str) + "__" + frame["candidate"].astype(str)


def paired_rmse_bootstrap(
    games: pd.DataFrame,
    *,
    selected: str,
    repetitions: int = 10_000,
    seed: int = 20260826,
) -> pd.DataFrame:
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    frame = games.copy()
    frame["candidate_key"] = _candidate_key(frame)
    if frame.duplicated(["test_season", "game_id", "candidate_key"]).any():
        raise ValueError("Candidate game keys are not unique.")
    frame["squared_error"] = (
        frame["actual_margin"] - frame["predicted_margin"]
    ) ** 2
    wide = frame.pivot(
        index=["test_season", "game_id"],
        columns="candidate_key",
        values="squared_error",
    ).sort_index()
    if selected not in wide:
        raise ValueError(f"Selected candidate {selected!r} is missing.")
    if wide.isna().any().any():
        raise ValueError("Every candidate must score exactly the same games.")
    seasons = tuple(int(value) for value in wide.index.get_level_values(0).unique())
    rng = np.random.default_rng(seed)
    rows = []
    for challenger in sorted(column for column in wide if column != selected):
        observed_by_season = []
        samples = np.empty((repetitions, len(seasons)), dtype=float)
        for season_index, season in enumerate(seasons):
            season_frame = wide.xs(season, level="test_season")
            selected_error = season_frame[selected].to_numpy(dtype=float)
            challenger_error = season_frame[challenger].to_numpy(dtype=float)
            observed_by_season.append(
                np.sqrt(challenger_error.mean()) - np.sqrt(selected_error.mean())
            )
            indices = rng.integers(
                0, len(season_frame), size=(repetitions, len(season_frame))
            )
            samples[:, season_index] = np.sqrt(
                challenger_error[indices].mean(axis=1)
            ) - np.sqrt(selected_error[indices].mean(axis=1))
        draws = samples.mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "selected": selected,
                "challenger": challenger,
                "observed_equal_season_rmse_delta_challenger_minus_selected": float(
                    np.mean(observed_by_season)
                ),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "probability_challenger_better": float((draws < 0).mean()),
                "fold_wins_challenger": int(
                    sum(value < 0 for value in observed_by_season)
                ),
                "folds": len(seasons),
                "matched_games": int(len(wide)),
                "repetitions": repetitions,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "observed_equal_season_rmse_delta_challenger_minus_selected",
        kind="stable",
    )


def run() -> dict:
    manifest = json.loads((SOURCE_RUN / "run.json").read_text())
    if manifest["quality"]["maximum_loaded_season"] > 2024:
        raise ValueError("Development artifact used a post-2024 season.")
    games = pd.read_parquet(SOURCE_RUN / "games.parquet")
    comparisons = paired_rmse_bootstrap(games, selected=SELECTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons.to_parquet(OUTPUT / "paired_bootstrap.parquet", index=False)
    payload = {
        "experiment_id": "spm_target_horizon_full_v1_paired_bootstrap",
        "source_run_id": manifest["run_id"],
        "source_run_sha256": sha256_file(SOURCE_RUN / "run.json"),
        "source_games_sha256": sha256_file(SOURCE_RUN / "games.parquet"),
        "selected_candidate": SELECTED,
        "status": "development_uncertainty_complete",
        "repetitions": 10_000,
        "seed": 20260826,
        "comparisons": comparisons.to_dict("records"),
        "season_2027_loaded": False,
    }
    write_json_atomic(payload, OUTPUT / "bootstrap.json")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
