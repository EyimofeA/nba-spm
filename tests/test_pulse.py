from __future__ import annotations

import pandas as pd

from nba_impact.models.pulse import _identity_errors


def test_pulse_identities() -> None:
    frame = pd.DataFrame(
        {
            "pulse_prior_offense": [1.0],
            "pulse_prior_defense": [2.0],
            "pulse_prior_net": [3.0],
            "lineup_update_offense": [0.5],
            "lineup_update_defense": [-0.25],
            "lineup_update_net": [0.25],
            "pulse_offense": [1.5],
            "pulse_defense": [1.75],
            "pulse_net": [3.25],
            "rapm_offense": [0.4],
            "rapm_defense": [-0.1],
            "rapm_net": [0.3],
        }
    )
    assert max(_identity_errors(frame).values()) < 1e-12
