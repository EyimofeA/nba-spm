"""Observable play-by-play outcome channels for exploratory RAPM.

These are event outcomes and shot-finish labels.  They are deliberately not
called Synergy play types: basic play-by-play does not identify pick-and-roll,
isolation, post-up, or handoff possessions with acceptable fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_impact.models.channel_rapm import lineup_sides
from nba_impact.models.possession_outcome_rapm import assign_events_to_possessions


POSSESSION_CHANNELS = (
    "rim_assists",
    "transition_points",
    "three_point_points",
    "free_throw_points",
    "midrange_attempts",
    "rim_points",
    "playtype_transition_points",
    "playtype_putback_points",
    "playtype_cut_points",
    "playtype_drive_points",
    "playtype_pullup_points",
    "playtype_post_points",
    "playtype_spotup_points",
    "playtype_other_points",
)


@dataclass(frozen=True)
class ObservableChannelLedger:
    possessions: pd.DataFrame
    offensive_contributions: pd.DataFrame
    steal_contributions: pd.DataFrame
    shots: pd.DataFrame
    block_contributions: pd.DataFrame
    rebounds: pd.DataFrame
    rebound_contributions: pd.DataFrame
    quality: dict[str, int | float]


def normalize_gabriel_events(frame: pd.DataFrame, *, season: int) -> pd.DataFrame:
    """Normalize and deduplicate the team-mirrored merged play-by-play files."""
    required = {
        "game_id", "actionNumber", "actionType", "description", "qualifier",
        "shotResult", "person_id", "assister_id", "stealPersonId",
        "blockPersonId", "teamId",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Gabriel play-by-play is missing columns: {missing}")
    output = frame.copy()
    output["game_id"] = output["game_id"].map(
        lambda value: f"{int(float(value)):010d}"
    )
    output["actionNumber"] = pd.to_numeric(
        output["actionNumber"], errors="raise"
    ).astype(np.int64)
    output["actionType"] = output["actionType"].astype(str).str.strip().str.casefold()
    output["description"] = output["description"].astype("string")
    output["qualifier"] = output["qualifier"].astype("string")
    output["shotResult"] = output["shotResult"].astype("string")
    output["season_end"] = int(season)
    # Each game is present once in each team's season file. The rows are exact
    # mirrors for the fields consumed below.
    conflicts = (
        output.groupby(["game_id", "actionNumber"], sort=False)["actionType"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        raise ValueError("Mirrored play-by-play has conflicting action types.")
    return output.drop_duplicates(["game_id", "actionNumber"], keep="first")


def _in_lineup(players: np.ndarray, actors: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(actors, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    return (players == numeric[:, None]).any(axis=1)


def _shot_finish(description: pd.Series, qualifier: pd.Series) -> pd.Series:
    text = description.fillna("").str.casefold()
    fastbreak = qualifier.fillna("").str.casefold().str.contains("fastbreak")
    result = pd.Series("other", index=text.index, dtype="string")
    result.loc[text.str.contains("jump shot") | text.str.contains("3pt")] = "spotup"
    result.loc[text.str.contains("turnaround|hook|fadeaway", regex=True)] = "post"
    result.loc[text.str.contains("pullup|pull-up|step back", regex=True)] = "pullup"
    result.loc[text.str.contains("driving")] = "drive"
    result.loc[text.str.contains("cutting|alley oop", regex=True)] = "cut"
    result.loc[text.str.contains("putback|tip ", regex=True)] = "putback"
    result.loc[fastbreak] = "transition"
    return result


def build_observable_channel_ledger(
    possessions: pd.DataFrame,
    events: pd.DataFrame,
) -> ObservableChannelLedger:
    """Create valid possession, shot, and rebound opportunity ledgers."""
    relevant = events.loc[
        events["actionType"].isin(
            ["2pt", "3pt", "freethrow", "turnover", "rebound", "steal", "block"]
        )
    ].copy()
    mapped = assign_events_to_possessions(possessions, relevant)
    mapped = mapped.merge(
        possessions[
            [
                "possession_id", "offense_team_id", "home_poss",
                "home_team_id", "away_team_id",
                *[f"a{slot}" for slot in range(1, 6)],
                *[f"h{slot}" for slot in range(1, 6)],
            ]
        ],
        on="possession_id",
        validate="many_to_one",
    )
    offense, defense = lineup_sides(mapped)
    event_team = pd.to_numeric(mapped["teamId"], errors="coerce")
    offense_event = event_team.eq(mapped["offense_team_id"]).fillna(False)
    action = mapped["actionType"]
    shot = action.isin(["2pt", "3pt"])
    made = mapped["shotResult"].fillna("").str.casefold().eq("made").fillna(False)
    field_goal_points = np.where(
        shot & made, np.where(action.eq("3pt"), 3.0, 2.0), 0.0
    )
    free_throw = action.eq("freethrow")
    free_throw_points = (free_throw & made).astype(float)
    mapped["event_points"] = field_goal_points + free_throw_points

    distance = pd.to_numeric(
        mapped["description"].str.extract(r"(?:^|\s)(\d+)'", expand=False),
        errors="coerce",
    )
    finish_text = mapped["description"].fillna("").str.casefold()
    rim = (action.eq("2pt") & (
        distance.le(4)
        | finish_text.str.contains("dunk|layup|alley oop|tip ", regex=True)
    )).fillna(False)
    midrange = action.eq("2pt") & ~rim
    fastbreak = mapped["qualifier"].fillna("").str.casefold().str.contains(
        "fastbreak"
    ).fillna(False)
    assisted = (
        pd.to_numeric(mapped["assister_id"], errors="coerce").notna() & shot & made
    ).fillna(False)
    finish = _shot_finish(mapped["description"], mapped["qualifier"])

    channel_events = pd.DataFrame(
        {
            "possession_id": mapped["possession_id"],
            "rim_assists": (offense_event & assisted & rim).astype(float),
            "transition_points": np.where(offense_event & fastbreak, mapped["event_points"], 0.0),
            "three_point_points": np.where(offense_event & action.eq("3pt"), mapped["event_points"], 0.0),
            "free_throw_points": np.where(offense_event & free_throw, mapped["event_points"], 0.0),
            "midrange_attempts": (offense_event & midrange).astype(float),
            "rim_points": np.where(offense_event & rim, mapped["event_points"], 0.0),
            **{
                f"playtype_{label}_points": np.where(
                    offense_event & shot & finish.eq(label), mapped["event_points"], 0.0
                )
                for label in (
                    "transition", "putback", "cut", "drive", "pullup", "post",
                    "spotup", "other",
                )
            },
        }
    )
    channels = channel_events.groupby("possession_id", as_index=False).sum()
    possession_targets = possessions.merge(
        channels, on="possession_id", how="left", validate="one_to_one"
    )
    possession_targets[list(POSSESSION_CHANNELS)] = possession_targets[
        list(POSSESSION_CHANNELS)
    ].fillna(0.0)

    offense_events = mapped.loc[
        offense_event & action.isin(["2pt", "3pt", "freethrow", "turnover"])
    ].copy()
    offense_players, _ = lineup_sides(offense_events)
    actor = pd.to_numeric(offense_events["person_id"], errors="coerce")
    actor_present = _in_lineup(offense_players, actor)
    offense_events["player_id"] = actor
    offense_events["teammate_scoring"] = np.where(
        actor_present, offense_events["event_points"], 0.0
    )
    offense_events["teammate_turnovers"] = np.where(
        actor_present & offense_events["actionType"].eq("turnover"), 1.0, 0.0
    )
    offense_events["teammate_assists"] = 0.0
    assist_rows = mapped.loc[offense_event & assisted].copy()
    assist_players, _ = lineup_sides(assist_rows)
    assist_actor = pd.to_numeric(assist_rows["assister_id"], errors="coerce")
    assist_present = _in_lineup(assist_players, assist_actor)
    assist_contributions = pd.DataFrame(
        {
            "possession_id": assist_rows["possession_id"],
            "player_id": assist_actor,
            "teammate_scoring": 0.0,
            "teammate_turnovers": 0.0,
            "teammate_assists": assist_present.astype(float),
        }
    )
    offensive_contributions = pd.concat(
        [
            offense_events[
                [
                    "possession_id", "player_id", "teammate_scoring",
                    "teammate_turnovers", "teammate_assists",
                ]
            ],
            assist_contributions,
        ],
        ignore_index=True,
    ).dropna(subset=["player_id"])
    offensive_contributions["player_id"] = offensive_contributions["player_id"].astype(int)
    offensive_contributions = offensive_contributions.groupby(
        ["possession_id", "player_id"], as_index=False
    )[["teammate_scoring", "teammate_turnovers", "teammate_assists"]].sum()
    offensive_totals = mapped.loc[offense_event].groupby("possession_id").agg(
        teammate_turnovers=("actionType", lambda value: float(value.eq("turnover").any())),
    )
    assist_totals = assist_rows.groupby("possession_id").size().astype(float)
    possession_targets = possession_targets.merge(
        offensive_totals.reset_index(), on="possession_id", how="left", validate="one_to_one"
    )
    # The canonical score change is the exact team-scoring target. Mapped event
    # points are used only to remove the focal player's identifiable own points.
    possession_targets["teammate_scoring"] = possession_targets["pts"].astype(float)
    possession_targets["teammate_assists"] = possession_targets["possession_id"].map(
        assist_totals
    )
    possession_targets[["teammate_scoring", "teammate_turnovers", "teammate_assists"]] = (
        possession_targets[["teammate_scoring", "teammate_turnovers", "teammate_assists"]]
        .fillna(0.0)
    )

    turnover_rows = mapped.loc[offense_event & action.eq("turnover")].copy()
    _, turnover_defense = lineup_sides(turnover_rows)
    stealer = pd.to_numeric(turnover_rows["stealPersonId"], errors="coerce")
    stealer_present = _in_lineup(turnover_defense, stealer)
    steal_contributions = pd.DataFrame(
        {
            "possession_id": turnover_rows["possession_id"],
            "player_id": stealer,
            "teammate_steals": stealer_present.astype(float),
        }
    ).dropna(subset=["player_id"])
    steal_contributions["player_id"] = steal_contributions["player_id"].astype(int)
    steal_contributions = steal_contributions.groupby(
        ["possession_id", "player_id"], as_index=False
    )["teammate_steals"].sum()
    steal_totals = turnover_rows.groupby("possession_id")["stealPersonId"].apply(
        lambda value: float(pd.to_numeric(value, errors="coerce").notna().any())
    )
    possession_targets["teammate_steals"] = possession_targets["possession_id"].map(
        steal_totals
    ).fillna(0.0)

    shots = mapped.loc[offense_event & shot].copy()
    shots["source_possession_id"] = shots["possession_id"]
    shots["possession_id"] = (
        shots["source_possession_id"].astype(str)
        + ":shot:"
        + shots["actionNumber"].astype(str)
    )
    _, shot_defense = lineup_sides(shots)
    blocker = pd.to_numeric(shots["blockPersonId"], errors="coerce")
    blocker_present = _in_lineup(shot_defense, blocker)
    shots["teammate_blocks"] = blocker.notna().astype(float)
    block_contributions = pd.DataFrame(
        {
            "possession_id": shots["possession_id"],
            "player_id": blocker,
            "teammate_blocks": blocker_present.astype(float),
        }
    ).dropna(subset=["player_id"])
    block_contributions["player_id"] = block_contributions["player_id"].astype(int)

    rebounds = mapped.loc[action.eq("rebound")].copy()
    rebound_text = rebounds["description"].fillna("").str.casefold()
    counts = rebound_text.str.extract(r"off:(\d+)\s+def:(\d+)").apply(
        pd.to_numeric, errors="coerce"
    )
    order = rebounds.sort_values(["game_id", "actionNumber"], kind="stable").index
    ordered_counts = counts.loc[order]
    ordered_keys = rebounds.loc[order, ["game_id", "person_id"]]
    off_increment = ordered_counts[0].groupby(
        [ordered_keys["game_id"], ordered_keys["person_id"]], dropna=False
    ).diff().fillna(ordered_counts[0])
    def_increment = ordered_counts[1].groupby(
        [ordered_keys["game_id"], ordered_keys["person_id"]], dropna=False
    ).diff().fillna(ordered_counts[1])
    offensive_rebound = pd.Series(False, index=rebounds.index)
    offensive_rebound.loc[order] = off_increment.gt(def_increment).fillna(False).to_numpy()
    offensive_rebound.loc[rebound_text.str.contains("team offensive")] = True
    offensive_rebound.loc[rebound_text.str.contains("team defensive")] = False
    classified_rebound = (
        counts.notna().all(axis=1)
        | rebound_text.str.contains("team offensive|team defensive", regex=True)
    )
    rebounds = rebounds.loc[classified_rebound].copy()
    offensive_rebound = offensive_rebound.loc[rebounds.index]
    event_team = pd.to_numeric(rebounds["teamId"], errors="coerce")
    other_team = pd.Series(
        np.where(
            event_team.eq(rebounds["home_team_id"]),
            rebounds["away_team_id"],
            rebounds["home_team_id"],
        ),
        index=rebounds.index,
    )
    shot_offense_team = event_team.where(offensive_rebound, other_team)
    rebounds["offense_team_id"] = shot_offense_team
    rebounds["home_poss"] = shot_offense_team.eq(rebounds["home_team_id"]).astype(int)
    rebounds["source_possession_id"] = rebounds["possession_id"]
    rebounds["possession_id"] = (
        rebounds["source_possession_id"].astype(str)
        + ":rebound:"
        + rebounds["actionNumber"].astype(str)
    )
    rebound_offense, rebound_defense = lineup_sides(rebounds)
    rebounder = pd.to_numeric(rebounds["person_id"], errors="coerce")
    identifiable_rebounder = rebounder.between(1, 10_000_000)
    offense_rebound = offensive_rebound
    offense_rebounder_present = _in_lineup(rebound_offense, rebounder)
    defense_rebounder_present = _in_lineup(rebound_defense, rebounder)
    rebounds["teammate_oreb"] = offense_rebound.astype(float)
    rebounds["teammate_dreb"] = (~offense_rebound).astype(float)
    rebound_contributions = pd.DataFrame(
        {
            "possession_id": rebounds["possession_id"],
            "player_id": rebounder.where(identifiable_rebounder),
            "teammate_oreb": (
                offense_rebound & offense_rebounder_present & identifiable_rebounder
            ).astype(float),
            "teammate_dreb": (
                ~offense_rebound & defense_rebounder_present & identifiable_rebounder
            ).astype(float),
        }
    ).dropna(subset=["player_id"])
    rebound_contributions["player_id"] = rebound_contributions["player_id"].astype(int)

    event_points = float(mapped.loc[offense_event, "event_points"].sum())
    official_points = float(possessions["pts"].sum())
    quality = {
        "possessions": int(len(possessions)),
        "source_events": int(len(events)),
        "relevant_events": int(len(relevant)),
        "mapped_events": int(len(mapped)),
        "event_mapping_rate": float(len(mapped) / len(relevant)) if len(relevant) else float("nan"),
        "official_points": official_points,
        "mapped_points": event_points,
        "scorer_point_attribution_rate": float(event_points / official_points),
        "shots": int(len(shots)),
        "rebounds": int(len(rebounds)),
        "classified_rebound_rate": float(len(rebounds) / action.eq("rebound").sum()),
        "assists": int(assisted.sum()),
        "steals": int(pd.to_numeric(turnover_rows["stealPersonId"], errors="coerce").notna().sum()),
        "blocks": int(blocker.notna().sum()),
        "offense_actor_lineup_coverage": float(actor_present.mean()),
        "assist_actor_lineup_coverage": float(assist_present.mean()),
        "steal_actor_lineup_coverage": float(stealer_present[stealer.notna()].mean()),
        "block_actor_lineup_coverage": float(blocker_present[blocker.notna()].mean()),
        "rebound_actor_lineup_coverage": float(
            np.where(offense_rebound, offense_rebounder_present, defense_rebounder_present)[
                identifiable_rebounder
            ].mean()
        ),
    }
    return ObservableChannelLedger(
        possessions=possession_targets,
        offensive_contributions=offensive_contributions,
        steal_contributions=steal_contributions,
        shots=shots,
        block_contributions=block_contributions,
        rebounds=rebounds,
        rebound_contributions=rebound_contributions,
        quality=quality,
    )
