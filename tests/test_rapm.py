from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_possessions,
    ratings_table,
    run_regularization_comparison,
    run_walk_forward_comparison,
)


def test_current_adapter_makes_lineup_policy_explicit(tmp_path) -> None:
    possessions = pd.DataFrame(
        {
            "possession_id": ["g:001"], "game_id": ["g"], "possession_number": [1],
            "season_end": [2026], "season_type": ["regular"], "game_date": pd.to_datetime(["2026-01-01"]),
            "period": [1], "offense_is_home": [True], "points": [2.0],
        }
    )
    segment_rows = []
    for number, offset in ((1, 0), (2, 20)):
        segment_rows.append(
            {
                "possession_id": "g:001", "segment_number": number,
                **{f"home_player_{i}": i + offset for i in range(1, 6)},
                **{f"away_player_{i}": i + 10 + offset for i in range(1, 6)},
            }
        )
    possession_path = tmp_path / "possessions.parquet"
    segment_path = tmp_path / "segments.parquet"
    possessions.to_parquet(possession_path, index=False)
    pd.DataFrame(segment_rows).to_parquet(segment_path, index=False)
    start = load_current_possessions(possession_path, segment_path, lineup_policy="start")
    terminal = load_current_possessions(possession_path, segment_path, lineup_policy="terminal")
    assert start.loc[0, "h1"] == 1
    assert terminal.loc[0, "h1"] == 21
    assert start.loc[0, "home_poss"] == 1


def test_synthetic_offense_and_defense_signs() -> None:
    rng = np.random.default_rng(7)
    players = np.arange(1, 15)
    offense_skill = {int(player): 0.0 for player in players}
    defense_skill = {int(player): 0.0 for player in players}
    offense_skill[1] = 0.08
    defense_skill[14] = 0.08
    rows = []
    for possession in range(3000):
        lineup = rng.choice(players, size=10, replace=False)
        away, home = lineup[:5], lineup[5:]
        home_poss = bool(possession % 2)
        offense = home if home_poss else away
        defense = away if home_poss else home
        mean = 1.05 + sum(offense_skill[int(p)] for p in offense) - sum(
            defense_skill[int(p)] for p in defense
        )
        points = max(0.0, mean + rng.normal(0.0, 0.2))
        rows.append(
            {
                "home_poss": home_poss,
                "pts": points,
                **{f"a{i + 1}": int(player) for i, player in enumerate(away)},
                **{f"h{i + 1}": int(player) for i, player in enumerate(home)},
                "season": 2024,
                "date": "2024-01-01",
                "period": 1,
                "num": possession + 1,
                "gameid": f"g{possession // 100}",
            }
        )
    design = build_design(pd.DataFrame(rows))
    config = RapmConfig(seasons=(2024,), lambda_off=1.0, lambda_def=1.0, lambda_home=1.0)
    beta, _ = fit_coefficients(design, config)
    ratings = ratings_table(design, beta).set_index("player_id")
    assert ratings.loc[1, "offense_per_100"] > 0
    assert ratings.loc[14, "defense_per_100"] > 0


def test_centering_preserves_predictions_and_sets_weighted_average_to_zero() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_poss": bool(i % 2),
                "pts": float(i % 4),
                **{f"a{j + 1}": j + 1 for j in range(5)},
                **{f"h{j + 1}": j + 6 for j in range(5)},
                "season": 2024,
                "date": "2024-01-01",
                "period": 1,
                "num": i + 1,
                "gameid": "0020000001",
            }
            for i in range(100)
        ]
    )
    design = build_design(frame)
    beta, intercept = fit_coefficients(
        design,
        RapmConfig(seasons=(2024,), lambda_off=10.0, lambda_def=20.0, lambda_home=5.0),
    )
    ratings = ratings_table(design, beta)
    assert abs(np.average(ratings["offense_per_100"], weights=ratings["off_possessions"])) < 1e-10
    assert abs(np.average(ratings["defense_per_100"], weights=ratings["def_possessions"])) < 1e-10
    assert np.isfinite(np.asarray(design.X @ beta).ravel() + intercept).all()


def test_regularization_comparison_is_diagnostic(tmp_path) -> None:
    rows = []
    for season in (2023, 2024):
        for game in range(2):
            for i in range(20):
                rows.append(
                    {
                        "home_poss": bool(i % 2),
                        "pts": float((i + game) % 4),
                        **{f"a{j + 1}": j + 1 for j in range(5)},
                        **{f"h{j + 1}": j + 6 for j in range(5)},
                        "season": season,
                        "date": f"{season - 1}-11-01",
                        "period": 1,
                        "num": i + 1,
                        "gameid": f"002{season}{game}",
                    }
                )
    run = run_regularization_comparison(
        pd.DataFrame(rows),
        RapmConfig(seasons=(2023, 2024)),
        ((10.0, 10.0), (20.0, 30.0)),
        artifact_root=tmp_path,
    )
    assert run["status"] == "research_diagnostic_unverified"
    assert run["metrics"]["candidate_count"] == 2
    assert (tmp_path / "models" / "rapm_comparisons" / run["run_id"] / "results.parquet").exists()


def test_walk_forward_requires_multiple_folds_for_evidence(tmp_path) -> None:
    rows = []
    for season in (2021, 2022, 2023, 2024):
        for game in range(2):
            for i in range(20):
                rows.append(
                    {
                        "home_poss": bool(i % 2),
                        "pts": float((i + game + season) % 4),
                        **{f"a{j + 1}": j + 1 for j in range(5)},
                        **{f"h{j + 1}": j + 6 for j in range(5)},
                        "season": season,
                        "date": f"{season - 1}-11-01",
                        "period": 1,
                        "num": i + 1,
                        "gameid": f"002{season}{game}",
                    }
                )
    run = run_walk_forward_comparison(
        pd.DataFrame(rows),
        RapmConfig(seasons=(2021, 2022, 2023, 2024)),
        ((10.0, 10.0), (20.0, 30.0)),
        (2024,),
        train_window=3,
        artifact_root=tmp_path,
        bootstrap_repetitions=20,
    )
    statuses = {item["candidate"]: item["evidence_status"] for item in run["metrics"]["summary"]}
    assert statuses["off10_def10"] == "baseline"
    assert statuses["off20_def30"] == "insufficient_folds"
