import pandas as pd

from nba_impact.models.coach_rapm import build_coach_game_ledger, parse_bbref_coaches


def test_parse_coach_rows() -> None:
    html = """
    <table><tbody>
    <tr><th data-stat="coach"><a href="/coaches/alpha01c.html">A Coach</a></th>
    <td data-stat="team">ATL</td><td data-stat="cur_g">10</td></tr>
    </tbody></table>
    """
    parsed = parse_bbref_coaches(html, season=2026)
    assert parsed.iloc[0]["coach_id"] == "alpha01c"
    assert parsed.iloc[0]["games"] == 10


def test_coach_game_assignment_uses_game_counts() -> None:
    coaches = pd.DataFrame(
        {
            "season": [2026, 2026],
            "team_tricode": ["ATL", "ATL"],
            "coach_id": ["a", "b"],
            "coach": ["A", "B"],
            "games": [1, 2],
            "source_row": [0, 1],
        }
    )
    games = pd.DataFrame(
        {
            "project_season": [2026, 2026, 2026],
            "game_id": ["g1", "g2", "g3"],
            "game_date": ["2025-10-01", "2025-10-02", "2025-10-03"],
            "home_team_id": [1, 1, 1],
            "away_team_id": [2, 2, 2],
        }
    )
    team_dim = pd.DataFrame(
        {"team_id": [1, 2], "canonical_tricode": ["ATL", "BOS"]}
    )
    ledger, audit = build_coach_game_ledger(coaches, games, team_dim)
    atl = ledger.loc[ledger["team_id"].eq(1)]
    assert atl["coach_id"].tolist() == ["a", "b", "b"]
    assert audit.loc[audit["team_id"].eq(1), "difference"].iloc[0] == 0
