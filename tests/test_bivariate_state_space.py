import numpy as np
import pandas as pd

from nba_impact.models.bivariate_state_space import filter_bivariate_annual_rapm


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 2, 2],
            "Season": [2022, 2023, 2022, 2023],
            "target_offense": [1.0, 1.5, -1.0, -0.5],
            "target_defense": [0.5, 0.7, -0.2, -0.4],
            "observation_variance_offense": [0.5] * 4,
            "observation_variance_defense": [0.7] * 4,
            "observation_covariance_offense_defense": [0.1] * 4,
            "Poss_Off": [1000.0] * 4,
            "Poss_Def": [1000.0] * 4,
        }
    )


def test_bivariate_filter_is_causal_and_additive() -> None:
    original = filter_bivariate_annual_rapm(
        _observations(),
        phi=0.9,
        process_sd=0.25,
        process_correlation=0.25,
        use_observation_covariance=True,
    )
    changed = _observations()
    changed.loc[changed["Season"].eq(2023), "target_offense"] = 100.0
    rerun = filter_bivariate_annual_rapm(
        changed,
        phi=0.9,
        process_sd=0.25,
        process_correlation=0.25,
        use_observation_covariance=True,
    )
    left = original.loc[original["Season"].eq(2022), "filtered_net"].to_numpy()
    right = rerun.loc[rerun["Season"].eq(2022), "filtered_net"].to_numpy()
    assert np.allclose(left, right)
    assert np.allclose(
        original["filtered_offense"] + original["filtered_defense"],
        original["filtered_net"],
    )


def test_zero_covariance_matches_diagonal_observation_model() -> None:
    observations = _observations()
    observations["observation_covariance_offense_defense"] = 0.0
    with_covariance = filter_bivariate_annual_rapm(
        observations,
        phi=0.9,
        process_sd=0.25,
        process_correlation=0.0,
        use_observation_covariance=True,
    )
    diagonal = filter_bivariate_annual_rapm(
        observations,
        phi=0.9,
        process_sd=0.25,
        process_correlation=0.0,
        use_observation_covariance=False,
    )
    assert np.allclose(with_covariance["filtered_net"], diagonal["filtered_net"])
