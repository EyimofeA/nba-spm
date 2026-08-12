from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.rapm import RapmConfig
from nba_impact.models.rolling_rapm_peaks import (
    extract_player_peaks,
    fit_rolling_rapm_window,
    load_peak_player_names,
    run_selection_aware_peak_bootstrap,
)


def test_peak_names_fill_old_ids_from_player_sheets(tmp_path: Path) -> None:
    names = tmp_path / "names.csv"
    pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Known"]}).to_csv(names, index=False)
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    pd.DataFrame(
        {"PLAYER_ID": [1, 390], "PLAYER_NAME": ["Known", "Sasha Danilovic"]}
    ).to_csv(sheets / "1997.csv", index=False)
    resolved, hashes = load_peak_player_names(names, sheets, (1997,))
    lookup = resolved.set_index("PLAYER_ID")
    assert lookup.loc[390, "PLAYER_NAME"] == "Sasha Danilovic"
    assert lookup.loc[390, "name_source"] == "annual_player_sheet_fallback"
    assert len(hashes) == 2


def test_rolling_window_and_peak_selection_are_decomposed_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    players = np.arange(1, 15)
    rows = []
    for season in (2022, 2023, 2024):
        for possession in range(80):
            lineup = rng.choice(players, size=10, replace=False)
            rows.append(
                {
                    "home_poss": bool(possession % 2),
                    "pts": 1.05 + float(lineup[5:].sum() - lineup[:5].sum()) * 0.001,
                    **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                    **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                    "season": season,
                    "date": f"{season - 1}-11-01",
                    "period": 1,
                    "num": possession + 1,
                    "gameid": f"002{season}{possession // 20:03d}",
                }
            )
    ratings, quality = fit_rolling_rapm_window(
        pd.DataFrame(rows),
        RapmConfig(seasons=(2022, 2023, 2024), lambda_off=50, lambda_def=50),
        window_start=2022,
        window_end=2024,
        minimum_possessions_per_window_season=1,
    )
    ratings["PLAYER_NAME"] = ratings["PLAYER_ID"].map(lambda value: f"P{value}")
    np.testing.assert_allclose(ratings["net"], ratings["offense"] + ratings["defense"])
    assert quality["max_component_identity_error"] < 1e-12
    shifted = ratings.copy()
    shifted["window_start"] = 2021
    shifted["window_end"] = 2023
    shifted.loc[shifted["PLAYER_ID"].eq(1), "net"] += 10
    peaks = extract_player_peaks(pd.concat([ratings, shifted], ignore_index=True))
    player_one_net = peaks.loc[
        peaks["PLAYER_ID"].eq(1) & peaks["peak_component"].eq("net")
    ].iloc[0]
    assert player_one_net["window_end"] == 2023
    assert not peaks.duplicated(["PLAYER_ID", "window_seasons", "peak_component"]).any()


def test_season_scoring_environment_does_not_change_player_ratings() -> None:
    rng = np.random.default_rng(23)
    players = np.arange(1, 13)
    rows = []
    for season in (2022, 2023, 2024):
        for possession in range(100):
            lineup = rng.choice(players, size=10, replace=False)
            rows.append(
                {
                    "home_poss": bool(possession % 2),
                    "pts": 1.0 + float(lineup[5:].sum() - lineup[:5].sum()) * 0.001,
                    **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                    **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                    "season": season,
                    "date": f"{season - 1}-11-01",
                    "period": 1,
                    "num": possession + 1,
                    "gameid": f"002{season}{possession // 20:03d}",
                }
            )
    frame = pd.DataFrame(rows)
    config = RapmConfig(seasons=(2022, 2023, 2024), lambda_off=50, lambda_def=50)
    baseline, _ = fit_rolling_rapm_window(
        frame,
        config,
        window_start=2022,
        window_end=2024,
        minimum_possessions_per_window_season=1,
    )
    shifted = frame.copy()
    shifted.loc[shifted["season"].eq(2024), "pts"] += 0.25
    challenger, _ = fit_rolling_rapm_window(
        shifted,
        config,
        window_start=2022,
        window_end=2024,
        minimum_possessions_per_window_season=1,
    )
    baseline = baseline.sort_values("PLAYER_ID")
    challenger = challenger.sort_values("PLAYER_ID")
    np.testing.assert_allclose(
        baseline[["offense", "defense", "net"]],
        challenger[["offense", "defense", "net"]],
        atol=1e-10,
    )


def test_peak_eligibility_requires_threshold_in_every_window_season() -> None:
    rows = []
    for season in (2022, 2023, 2024):
        home = [1, 2, 3, 4, 5] if season != 2023 else [11, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        for possession in range(4):
            rows.append(
                {
                    "home_poss": bool(possession % 2),
                    "pts": 1.0,
                    **{f"a{i + 1}": value for i, value in enumerate(away)},
                    **{f"h{i + 1}": value for i, value in enumerate(home)},
                    "season": season,
                    "date": f"{season - 1}-11-01",
                    "period": 1,
                    "num": possession + 1,
                    "gameid": f"002{season}001",
                }
            )

    ratings, quality = fit_rolling_rapm_window(
        pd.DataFrame(rows),
        RapmConfig(seasons=(2022, 2023, 2024), lambda_off=50, lambda_def=50),
        window_start=2022,
        window_end=2024,
        minimum_possessions_per_window_season=1,
    )
    player = ratings.loc[ratings["PLAYER_ID"].eq(1)].iloc[0]

    assert player["Poss_Off"] >= 3
    assert player["Poss_Def"] >= 3
    assert player["minimum_season_off_possessions"] == 0
    assert player["minimum_season_def_possessions"] == 0
    assert not player["peak_eligible"]
    assert quality["minimum_peak_possessions_per_side_per_season"] == 1


def test_selection_aware_peak_bootstrap_reselects_and_checkpoints(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sheets = tmp_path / "sheets"
    cache.mkdir()
    sheets.mkdir()
    for season in (2022, 2023, 2024):
        rows = []
        for game in range(2):
            for possession in range(8):
                home = [6, 7, 8, 9, 10] if season != 2023 else [11, 7, 8, 9, 10]
                rows.append(
                    {
                        "home_poss": bool(possession % 2),
                        "pts": float((game + possession) % 3),
                        **{f"a{i}": i for i in range(1, 6)},
                        **{f"h{i}": value for i, value in enumerate(home, start=1)},
                        "season": season,
                        "date": f"{season - 1}-11-01",
                        "period": 1,
                        "num": possession + 1,
                        "gameid": f"002{season}{game:04d}",
                    }
                )
        pd.DataFrame(rows).to_parquet(cache / f"matchups_{season}.parquet", index=False)
        pd.DataFrame(
            {"PLAYER_ID": range(1, 11), "PLAYER_NAME": [f"P{i}" for i in range(1, 11)]}
        ).to_csv(sheets / f"{season}.csv", index=False)
    names = tmp_path / "names.csv"
    pd.DataFrame(
        {"PLAYER_ID": range(1, 11), "PLAYER_NAME": [f"P{i}" for i in range(1, 11)]}
    ).to_csv(names, index=False)
    contract = tmp_path / "contract.json"
    contract.write_text(
        __import__("json").dumps(
            {
                "status": "frozen_research_contract",
                "contract_version": "test",
                "estimand": "test",
                "season_range": [2022, 2024],
                "window_lengths": [3],
                "model": {"prior": "zero", "lambda_off": 10, "lambda_def": 10, "lambda_home": 2},
                "peak_eligibility": {
                    "minimum_offensive_possessions_per_window_season": 1,
                    "minimum_defensive_possessions_per_window_season": 1,
                },
                "caveats": [],
            }
        )
    )
    run = run_selection_aware_peak_bootstrap(
        cache, names, sheets, contract, artifact_root=tmp_path / "artifacts", draws=2, seed=4
    )
    output = Path(run["artifact_path"])
    draws = pd.read_parquet(output / "selected_draws" / "draw_0000.parquet")
    assert not draws.duplicated(["PLAYER_ID", "window_seasons", "peak_component"]).any()
    window = pd.read_parquet(output / "window_draws" / "draw_0000" / "3y_end_2024.parquet")
    # Players 6 and 11 are absent from one of the three constituent seasons.
    # The bootstrap must represent that absence as zero exposure, not drop the
    # season from the eligibility minimum.
    assert not window.loc[window["PLAYER_ID"].isin([6, 11]), "peak_eligible"].any()
    summary = pd.read_parquet(output / "selection_aware_peaks.parquet")
    assert summary["draw_coverage"].eq(2).all()
    assert summary["uncertainty_status"].eq("selection_aware_bootstrap_complete").all()
