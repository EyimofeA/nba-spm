from __future__ import annotations

import pandas as pd

from nba_impact.data.scoring_events import extract_scoring_events


def test_extract_scoring_events_preserves_score_corrections() -> None:
    frame = pd.DataFrame(
        [
            {"gameId": 21600001, "actionId": 1, "actionNumber": 1, "period": 1,
             "clock": "PT12M00.00S", "teamId": 0, "personId": 0,
             "actionType": "period", "subType": None, "scoreHome": 0,
             "scoreAway": 0, "pointsTotal": 0, "description": "start",
             "shotResult": None, "isFieldGoal": 0, "_season": 2016,
             "_season_type": "rg"},
            {"gameId": 21600001, "actionId": 2, "actionNumber": 2, "period": 1,
             "clock": "PT11M40.00S", "teamId": 10, "personId": 100,
             "actionType": "Made Shot", "subType": "Jump Shot", "scoreHome": 2,
             "scoreAway": 0, "pointsTotal": 2, "description": "made",
             "shotResult": "Made", "isFieldGoal": 1, "_season": 2016,
             "_season_type": "rg"},
            {"gameId": 21600001, "actionId": 3, "actionNumber": 3, "period": 1,
             "clock": "PT11M30.00S", "teamId": 20, "personId": 200,
             "actionType": "Missed Shot", "subType": "Jump Shot", "scoreHome": 0,
             "scoreAway": 0, "pointsTotal": 0, "description": "missed",
             "shotResult": "Missed", "isFieldGoal": 1, "_season": 2016,
             "_season_type": "rg"},
            {"gameId": 21600001, "actionId": 4, "actionNumber": 4, "period": 1,
             "clock": "PT11M20.00S", "teamId": 20, "personId": 200,
             "actionType": "Free Throw", "subType": "1 of 1", "scoreHome": 2,
             "scoreAway": 1, "pointsTotal": 3, "description": "free throw",
             "shotResult": None, "isFieldGoal": 0, "_season": 2016,
             "_season_type": "rg"},
            {"gameId": 21600001, "actionId": 5, "actionNumber": 5, "period": 1,
             "clock": "PT11M10.00S", "teamId": 0, "personId": 0,
             "actionType": "Instant Replay", "subType": None, "scoreHome": 2,
             "scoreAway": 0, "pointsTotal": 2, "description": "correction",
             "shotResult": None, "isFieldGoal": 0, "_season": 2016,
             "_season_type": "rg"},
        ]
    )

    output, metrics = extract_scoring_events(
        frame, project_season=2017, season_type="regular"
    )

    assert output["points_delta"].tolist() == [2, 1, -1]
    assert output["is_score_correction"].tolist() == [False, False, True]
    assert metrics["score_conservation_failures"] == 0
    assert metrics["points_total_mismatches"] == 0
    assert metrics["zero_score_sentinel_rows"] == 2
    assert metrics["structural_passed"]


def test_extract_scoring_events_rejects_wrong_source_season() -> None:
    frame = pd.DataFrame(
        [{column: 0 for column in (
            "gameId", "actionId", "actionNumber", "period", "teamId", "personId",
            "scoreHome", "scoreAway", "pointsTotal",
        )}]
    )
    frame["clock"] = "PT12M00.00S"
    frame["actionType"] = "period"
    frame["description"] = "start"
    frame["_season"] = 2015
    frame["_season_type"] = "rg"

    try:
        extract_scoring_events(frame, project_season=2017, season_type="regular")
    except ValueError as exc:
        assert "Expected source season 2016" in str(exc)
    else:
        raise AssertionError("wrong source season should fail")


def test_extract_scoring_events_discards_only_stale_tail_after_verified_final() -> None:
    rows = []
    for action_id, home, away, action_type in (
        (1, 2, 0, "Made Shot"),
        (2, 2, 3, "Made Shot"),
        (3, 4, 3, "Made Shot"),
        (4, 2, 3, "period"),
    ):
        rows.append(
            {
                "gameId": 22300001,
                "actionId": action_id,
                "actionNumber": action_id,
                "period": 1,
                "clock": "PT00M00.00S",
                "teamId": 10,
                "personId": 100,
                "actionType": action_type,
                "subType": None,
                "scoreHome": home,
                "scoreAway": away,
                "pointsTotal": home + away,
                "description": action_type,
                "shotResult": "Made",
                "isFieldGoal": 1,
                "_season": 2023,
                "_season_type": "rg",
            }
        )

    output, metrics = extract_scoring_events(
        pd.DataFrame(rows),
        project_season=2024,
        season_type="regular",
        expected_final_scores={"0022300001": (4, 3)},
    )

    assert tuple(output.iloc[-1][["score_home", "score_away"]]) == (4, 3)
    assert metrics["source_tail_rows_removed"] == 1
    assert metrics["repaired_final_score_games"] == 1
    assert metrics["structural_passed"]
