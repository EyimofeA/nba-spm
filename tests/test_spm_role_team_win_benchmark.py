from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_impact.models.spm_role_team_win_benchmark import (
    load_bbref_player_team_minutes,
)


def test_bbref_minutes_keep_real_team_stints_and_drop_synthetic_total(tmp_path: Path) -> None:
    identity = tmp_path / "2022.parquet"
    pd.DataFrame(
        {"PLAYER_ID": [1], "PLAYER_NAME": ["Player One"]}
    ).to_parquet(identity, index=False)
    (tmp_path / "nba_2022_totals.html").write_text(
        """
        <table id="totals_stats"><tbody>
          <tr><td data-stat="name_display">Player One</td><td data-stat="team_name_abbr">2TM</td><td data-stat="mp">100</td></tr>
          <tr><td data-stat="name_display">Player One</td><td data-stat="team_name_abbr">BRK</td><td data-stat="mp">60</td></tr>
          <tr><td data-stat="name_display">Player One</td><td data-stat="team_name_abbr">BOS</td><td data-stat="mp">40</td></tr>
        </tbody></table>
        """,
        encoding="utf-8",
    )
    team_games = pd.DataFrame(
        {
            "Season": [2022, 2022],
            "team": ["BKN", "BOS"],
            "team_id": [10, 20],
            "won": [True, False],
        }
    )

    minutes, unmatched, coverage = load_bbref_player_team_minutes(
        tmp_path, {2022: identity}, team_games
    )

    assert minutes.sort_values("team_id")["minutes"].tolist() == [60.0, 40.0]
    assert unmatched.empty
    assert coverage.loc[0, "minute_match_rate"] == 1.0
