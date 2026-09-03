from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.rapm_lab.run_pulse_external_common import common_coefficients


def test_common_support_centering_sign_and_zeroed_missing_players():
    source = SimpleNamespace(players=np.array([1, 2, 3]), X=np.zeros((1, 7)),
        off_possessions=np.array([100, 300, 200]), def_possessions=np.array([300, 100, 200]))
    ratings = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "offense": [8., 4., 99.], "defense": [6., 2., 99.]})
    result = common_coefficients(ratings, source, {1, 2}, .02)
    np.testing.assert_allclose(result, [.03, -.01, 0, -.01, .03, 0, .02])
    ratings[["offense", "defense"]] += 10
    np.testing.assert_allclose(common_coefficients(ratings, source, {1, 2}, .02), result)
