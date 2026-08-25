from nba_impact.data.historical_game_schedule import parse_league_game_log, season_label


def test_season_label_handles_century_boundary() -> None:
    assert season_label(1997) == "1996-97"
    assert season_label(2000) == "1999-00"


def test_parse_league_game_log_resolves_home_and_away() -> None:
    payload = {
        "resultSets": [{
            "headers": ["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP"],
            "rowSet": [
                ["1", "1996-11-01", 10, "AAA vs. BBB"],
                ["1", "1996-11-01", 20, "BBB @ AAA"],
            ],
        }]
    }
    games = parse_league_game_log(payload, season_end=1997)
    assert games.iloc[0]["game_id"] == "0000000001"
    assert games.iloc[0]["home_team_id"] == 10
    assert games.iloc[0]["away_team_id"] == 20
