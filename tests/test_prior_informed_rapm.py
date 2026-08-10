from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.prior_informed_rapm import (
    build_prior_center,
    paired_confirmation_bootstrap,
    run_prior_informed_rapm_comparison,
)
from nba_impact.models.rapm import RapmConfig, build_design


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260810)
    players = np.arange(1, 17)
    offense = {int(player): (player - 8.5) * 0.004 for player in players}
    defense = {int(player): ((player * 5) % 13 - 6) * 0.003 for player in players}
    rows = []
    for season in range(2017, 2025):
        for game in range(5):
            for possession in range(40):
                lineup = rng.choice(players, size=10, replace=False)
                away, home = lineup[:5], lineup[5:]
                home_poss = bool(possession % 2)
                offense_lineup = home if home_poss else away
                defense_lineup = away if home_poss else home
                points = (
                    1.05
                    + sum(offense[int(player)] for player in offense_lineup)
                    - sum(defense[int(player)] for player in defense_lineup)
                    + rng.normal(0, 0.08)
                )
                rows.append(
                    {
                        "home_poss": home_poss,
                        "pts": points,
                        **{
                            f"a{index + 1}": int(player)
                            for index, player in enumerate(away)
                        },
                        **{
                            f"h{index + 1}": int(player)
                            for index, player in enumerate(home)
                        },
                        "season": season,
                        "date": f"{season - 1}-11-01",
                        "period": 1,
                        "num": possession + 1,
                        "gameid": f"002{season}{game:03d}",
                    }
                )
    priors = pd.DataFrame(
        [
            {
                "PLAYER_ID": int(player),
                "Window_End": window_end,
                "prior_offense_per_100": offense[int(player)] * 100,
                "prior_defense_per_100": defense[int(player)] * 100,
                "prior_net_per_100": (
                    offense[int(player)] + defense[int(player)]
                )
                * 100,
            }
            for window_end in range(2019, 2024)
            for player in players
        ]
    )
    return pd.DataFrame(rows), priors


def test_prior_center_uses_positive_good_defense_and_zero_weighted_means() -> None:
    frame, priors = _synthetic_inputs()
    design = build_design(frame)
    train_mask = np.isin(design.seasons, (2017, 2018, 2019))
    test_mask = design.seasons == 2020
    center, coverage = build_prior_center(
        design,
        priors,
        prior_window_end=2019,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    n_players = len(design.players)
    player_16 = int(np.flatnonzero(design.players == 16)[0])
    strong_defender = int(np.flatnonzero(design.players == 5)[0])
    assert center[player_16] > 0
    assert center[n_players + strong_defender] < 0
    assert abs(coverage["centered_offense_mean"]) < 1e-14
    assert abs(coverage["centered_defense_mean"]) < 1e-14
    assert coverage["test_lineup_slot_coverage"] == 1.0


def test_prior_informed_comparison_uses_matched_chronological_folds(
    tmp_path: Path,
) -> None:
    frame, priors = _synthetic_inputs()
    priors_path = tmp_path / "priors.parquet"
    priors.to_parquet(priors_path, index=False)
    run = run_prior_informed_rapm_comparison(
        frame,
        priors_path,
        RapmConfig(
            seasons=tuple(range(2017, 2025)),
            lambda_off=100.0,
            lambda_def=100.0,
            lambda_home=10.0,
        ),
        artifact_root=tmp_path,
        prior_scales=(0.5, 1.0),
    )
    output = Path(run["artifact_path"])
    folds = pd.read_parquet(output / "fold_results.parquet")
    assert set(folds["test_season"]) == {2020, 2021, 2022, 2023, 2024}
    assert set(folds["candidate"]) == {
        "zero_prior",
        "prior_only",
        "prior_scale_0p5",
        "prior_scale_1",
    }
    assert run["quality"]["games"] == 25
    assert run["quality"]["maximum_centered_block_mean"] < 1e-14
    assert run["metrics"]["selected_prior_scale"] in {0.0, 0.5, 1.0}
    bootstrap = run["metrics"]["paired_confirmation_bootstrap"]
    assert bootstrap["matched_games"] == 10
    assert bootstrap["repetitions"] == 2000
    assert 0 <= bootstrap["probability_selected_better"] <= 1
    assert (output / "confirmation_summary.parquet").exists()
    assert json.loads((output / "run.json").read_text())["run_id"] == run["run_id"]


def test_paired_confirmation_bootstrap_detects_consistent_gain() -> None:
    games = pd.DataFrame(
        [
            {
                "test_season": season,
                "game_id": f"{season}-{game}",
                "candidate": candidate,
                "squared_error": error,
            }
            for season in (2023, 2024)
            for game in range(20)
            for candidate, error in (("zero_prior", 4.0), ("prior_scale_1", 1.0))
        ]
    )
    result = paired_confirmation_bootstrap(
        games,
        selected_candidate="prior_scale_1",
        confirmation_test_seasons=(2023, 2024),
        repetitions=100,
        seed=7,
    )
    assert result["observed_mse_delta"] == -3.0
    assert result["ci_95_upper"] == -3.0
    assert result["probability_selected_better"] == 1.0


def test_exact_full_prior_comparison_handles_zero_prior_selection(
    tmp_path: Path,
) -> None:
    frame, priors = _synthetic_inputs()
    priors.loc[:, ["prior_offense_per_100", "prior_defense_per_100"]] *= -100
    priors["prior_net_per_100"] = (
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    )
    priors_path = tmp_path / "bad_priors.parquet"
    priors.to_parquet(priors_path, index=False)
    run = run_prior_informed_rapm_comparison(
        frame,
        priors_path,
        RapmConfig(
            seasons=tuple(range(2017, 2025)),
            lambda_off=100.0,
            lambda_def=100.0,
            lambda_home=10.0,
        ),
        artifact_root=tmp_path,
        prior_scales=(1.0,),
    )
    assert run["metrics"]["selected_candidate"] == "zero_prior"
    assert run["metrics"]["bootstrap_candidate"] == "prior_scale_1"
    assert run["status"] == "prior_center_rejected_in_selection"
