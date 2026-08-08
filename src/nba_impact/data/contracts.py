"""Stable contracts for source and canonical tables."""
from __future__ import annotations


POSSESSION_COLUMNS = (
    "home_poss",
    "pts",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "season",
    "date",
    "period",
    "num",
    "gameid",
)

AWAY_PLAYER_COLUMNS = ("a1", "a2", "a3", "a4", "a5")
HOME_PLAYER_COLUMNS = ("h1", "h2", "h3", "h4", "h5")
PLAYER_COLUMNS = AWAY_PLAYER_COLUMNS + HOME_PLAYER_COLUMNS
POSSESSION_KEY = ("gameid", "period", "num")

