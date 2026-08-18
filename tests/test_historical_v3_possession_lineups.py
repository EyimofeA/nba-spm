from __future__ import annotations

import pandas as pd

from nba_impact.data.historical_v3_possession_lineups import (
    build_historical_v3_possession_lineup_candidate,
)
from nba_impact.data.historical_v3_possessions import infer_v3_possession_owners


HOME, AWAY = 1610612737, 1610612738
GAME = "0022200001"


def _raw() -> pd.DataFrame:
    rows = [
        (1, "PT12M00.00S", 0, 0, "Period", "start", "Period Start"),
        (2, "PT11M00.00S", HOME, 1, "Jump Ball", "", "Jump Ball"),
        (3, "PT10M00.00S", HOME, 1, "Substitution", "", "SUB: Home Six FOR Home One"),
        (4, "PT10M00.00S", AWAY, 101, "Substitution", "", "SUB: Away Six FOR Away One"),
        (5, "PT09M00.00S", HOME, 2, "Turnover", "Bad Pass", "Turnover"),
        (6, "PT08M00.00S", AWAY, 102, "Made Shot", "Jump Shot", "Away 18' Jump Shot"),
        (7, "PT00M00.00S", 0, 0, "Period", "end", "Period End"),
    ]
    return pd.DataFrame(
        [
            {
                "gameId": 22200001,
                "actionId": action_id,
                "actionNumber": action_id,
                "period": 1,
                "clock": clock,
                "teamId": team_id,
                "personId": person_id,
                "actionType": action_type,
                "subType": subtype,
                "description": description,
                "scoreHome": 0,
                "scoreAway": 2 if action_id >= 6 else 0,
            }
            for action_id, clock, team_id, person_id, action_type, subtype, description in rows
        ]
    )


def _candidates(raw: pd.DataFrame) -> pd.DataFrame:
    inferred, _ = infer_v3_possession_owners(raw)
    inferred["home_team_id"] = HOME
    inferred["away_team_id"] = AWAY
    owned = inferred.loc[inferred["possession"].isin([HOME, AWAY])].copy()
    new = owned["possession"].ne(owned["possession"].shift())
    owned["possession_number"] = new.cumsum().astype(int)
    owned["possession_id"] = GAME + ":v3:" + owned["possession_number"].astype(str).str.zfill(3)
    owned["home_points"] = (owned["scoring_team_id"].eq(HOME) * owned["points_added"]).astype(int)
    owned["away_points"] = (owned["scoring_team_id"].eq(AWAY) * owned["points_added"]).astype(int)
    grouped = owned.groupby(["possession_id", "possession_number"], as_index=False).agg(
        game_id=("game_id", "first"),
        period=("period", "first"),
        start_order_number=("event_order", "first"),
        end_order_number=("event_order", "last"),
        start_action_number=("action_number", "first"),
        end_action_number=("action_number", "last"),
        offense_team_id=("possession", "first"),
        points=("points_added", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        action_count=("event_order", "size"),
    )
    grouped["season_start"] = 2022
    grouped["season_end"] = 2023
    grouped["season_label"] = "2022-23"
    grouped["season_type"] = "regular"
    grouped["game_date"] = "2022-10-18"
    grouped["home_team_id"] = HOME
    grouped["away_team_id"] = AWAY
    grouped["start_seconds_elapsed"] = 0.0
    grouped["end_seconds_elapsed"] = 0.0
    grouped["low_confidence_actions"] = 0
    grouped["defense_team_id"] = grouped["offense_team_id"].map({HOME: AWAY, AWAY: HOME})
    grouped["offense_is_home"] = grouped["offense_team_id"].eq(HOME)
    grouped["lineup_ready"] = False
    grouped["owner_source"] = "v3_forward_owner_v1"
    return grouped


def _stints() -> pd.DataFrame:
    def row(number: int, start: int, end: int, home: list[int], away: list[int]) -> dict:
        return {
            "stint_id": f"{GAME}_v3_{number:03d}", "game_id": GAME,
            "season_type": "regular", "start_action_id": start, "end_action_id_exclusive": end,
            **{f"home_player_{index}": player for index, player in enumerate(home, start=1)},
            **{f"away_player_{index}": player for index, player in enumerate(away, start=1)},
        }
    return pd.DataFrame([
        row(1, 1, 3, [1, 2, 3, 4, 5], [101, 102, 103, 104, 105]),
        row(2, 3, 4, [2, 3, 4, 5, 6], [101, 102, 103, 104, 105]),
        row(3, 4, 8, [2, 3, 4, 5, 6], [102, 103, 104, 105, 106]),
    ])


def test_historical_adapter_maps_every_owned_action_and_conserves_segments(tmp_path) -> None:
    root = tmp_path / "v3"
    source = root / "nbastatsv3" / "project_season=2023" / "regular.parquet"
    source.parent.mkdir(parents=True)
    raw = _raw()
    raw.to_parquet(source, index=False)
    candidates = _candidates(raw)
    candidate_path = tmp_path / "possessions.parquet"
    candidates.to_parquet(candidate_path, index=False)
    possession_quality = tmp_path / "possession_quality.parquet"
    pd.DataFrame([{"game_id": GAME, "project_season": 2023, "season_type": "regular", "passed": True}]).to_parquet(possession_quality, index=False)
    lineup_stints = tmp_path / "lineup_stints.parquet"
    _stints().to_parquet(lineup_stints, index=False)
    lineup_quality = tmp_path / "lineup_quality.parquet"
    pd.DataFrame([{"game_id": GAME, "season_type": "regular", "passed": True}]).to_parquet(lineup_quality, index=False)
    report = build_historical_v3_possession_lineup_candidate(
        root, candidate_path, possession_quality, lineup_stints, lineup_quality,
        tmp_path / "output_possessions.parquet", tmp_path / "segments.parquet",
        tmp_path / "assigned.parquet", tmp_path / "quality.parquet", tmp_path / "report.json", tmp_path / "manifests",
        project_season=2023,
    )
    assigned = pd.read_parquet(tmp_path / "assigned.parquet")
    output = pd.read_parquet(tmp_path / "output_possessions.parquet")
    segments = pd.read_parquet(tmp_path / "segments.parquet")
    assert report["passed"]
    assert report["emitted_game_count"] == 1
    assert len(assigned) == assigned[["game_id", "event_order"]].drop_duplicates().shape[0]
    assert len(assigned) == int(candidates["action_count"].sum())
    assert output["lineup_ready"].all()
    assert segments.groupby("possession_id")["points"].sum().equals(output.set_index("possession_id")["points"])
    assert segments.groupby("possession_id")["action_count"].sum().equals(output.set_index("possession_id")["action_count"])
