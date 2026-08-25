import numpy as np
import pandas as pd

from nba_impact.models.age_adjusted_rapm import (
    age_curve,
    build_age_design,
    current_age_player_coefficients,
    fit_age_adjusted_rapm,
    predict_age_adjusted_rapm,
    season_decay_weights,
)
from nba_impact.models.rapm import RapmConfig, build_design


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(80):
        season = 2024 + index // 40
        rows.append(
            {
                "gameid": str(index // 10),
                "season": season,
                "home_poss": index % 2,
                "pts": 1.05 + 0.01 * (season - 2024),
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            }
        )
    return pd.DataFrame(rows)


def _ages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PLAYER_ID": player, "Season": season, "AGE": 24 + season - 2024}
            for season in (2024, 2025)
            for player in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
        ]
    )


def test_age_design_uses_reference_for_missing_slots() -> None:
    frame = _frame()
    ages = _ages().loc[lambda x: x["PLAYER_ID"].ne(15)]
    design = build_age_design(frame, ages, minimum_age=19, maximum_age=43, reference_age=27)
    assert design.total_slots == len(frame) * 10
    assert design.known_slots == len(frame) * 9


def test_age_fit_decomposes_conditional_and_age_27_predictions() -> None:
    frame = _frame()
    player_design = build_design(frame)
    ages = build_age_design(frame, _ages(), reference_age=27)
    fit = fit_age_adjusted_rapm(
        player_design,
        ages,
        RapmConfig(seasons=(2024, 2025), lambda_off=10, lambda_def=10),
        age_penalty=1,
    )
    neutral = predict_age_adjusted_rapm(
        fit, player_design, ages, include_age=False
    )
    conditional = predict_age_adjusted_rapm(fit, player_design, ages)
    np.testing.assert_allclose(
        conditional - neutral,
        np.asarray(ages.X @ fit.age_coefficients).ravel(),
        atol=1e-14,
    )
    curve = age_curve(fit)
    reference = curve.loc[curve["age"].eq(27)].iloc[0]
    assert reference[["offense", "defense", "net"]].eq(0).all()


def test_row_weights_match_integer_row_duplication() -> None:
    frame = _frame().head(20).copy()
    weights = np.where(frame.index % 2 == 0, 2.0, 1.0)
    ages_panel = _ages().loc[lambda value: value["Season"].eq(2024)]
    design = build_design(frame)
    ages = build_age_design(frame, ages_panel, reference_age=27)
    config = RapmConfig(seasons=(2024,), lambda_off=10, lambda_def=10)
    weighted = fit_age_adjusted_rapm(
        design, ages, config, age_penalty=2, row_weights=weights
    )

    duplicated = frame.loc[frame.index.repeat(weights.astype(int))].reset_index(drop=True)
    duplicate_design = build_design(duplicated)
    duplicate_ages = build_age_design(duplicated, ages_panel, reference_age=27)
    repeated = fit_age_adjusted_rapm(
        duplicate_design, duplicate_ages, config, age_penalty=2
    )
    np.testing.assert_allclose(
        weighted.player_coefficients, repeated.player_coefficients, atol=1e-10
    )
    np.testing.assert_allclose(
        weighted.age_coefficients, repeated.age_coefficients, atol=1e-10
    )
    assert abs(weighted.intercept - repeated.intercept) < 1e-10


def test_current_age_coefficients_add_observed_age_effects() -> None:
    frame = _frame()
    design = build_design(frame)
    panel = _ages()
    age_design = build_age_design(frame, panel, reference_age=27)
    fit = fit_age_adjusted_rapm(
        design,
        age_design,
        RapmConfig(seasons=(2024, 2025), lambda_off=10, lambda_def=10),
        age_penalty=1,
    )
    current, coverage = current_age_player_coefficients(
        fit, design, panel, season=2025
    )
    age = 25
    age_column = list(fit.ages).index(age)
    np.testing.assert_allclose(
        current[: len(design.players)] - fit.player_coefficients[: len(design.players)],
        fit.age_coefficients[age_column],
    )
    assert coverage == 1.0


def test_season_decay_halves_each_half_life() -> None:
    seasons = np.asarray([2020, 2021, 2022, 2023, 2024])
    np.testing.assert_allclose(
        season_decay_weights(seasons, window_end=2024, half_life_years=2),
        [0.25, 2 ** -1.5, 0.5, 2 ** -0.5, 1.0],
    )
    np.testing.assert_array_equal(
        season_decay_weights(seasons, window_end=2024, half_life_years=None),
        np.ones(5),
    )
