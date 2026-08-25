"""Build the local-only raw-versus-shot-quality matchup comparison file.

The output preserves the two model scales. It uses rank movement as the
comparison surface rather than subtracting unrelated Elo and shot-residual
units. This script does not change the public ratings snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT.parent / (
    "artifacts/models/lineup_shot_residual/"
    "lineup_shot_residual_v1_aeb57da06b/ratings.parquet"
)
OUTPUT = ROOT / "local-data/shot-quality-lineup-2026.json"
RAW = ROOT / "local-data/matchup-elo-2026.json"
LEADERBOARD = ROOT / "public/data/leaderboard-2026.json"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    quality = pd.read_parquet(QUALITY)
    quality = quality.loc[quality["shot_class"].eq("all")].copy()
    raw = pd.DataFrame(json.loads(RAW.read_text()))
    identity = pd.DataFrame(json.loads(LEADERBOARD.read_text()))[
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
    ]
    merged = quality.merge(
        raw[["PLAYER_ID", "net_elo"]],
        left_on="player_id",
        right_on="PLAYER_ID",
        how="inner",
        validate="one_to_one",
    ).merge(
        identity,
        on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["PLAYER_NAME"]).copy()
    merged["raw_net_rank"] = merged["net_elo"].rank(
        method="min", ascending=False
    ).astype(int)
    merged["shot_quality_net_rank"] = merged[
        "lineup_net_residual_per_100_shots"
    ].rank(method="min", ascending=False).astype(int)
    merged["rank_change"] = merged["raw_net_rank"] - merged["shot_quality_net_rank"]
    records = []
    for row in merged.itertuples(index=False):
        records.append(
            {
                "PLAYER_ID": int(row.PLAYER_ID),
                "PLAYER_NAME": str(row.PLAYER_NAME),
                "TEAM_ABBREVIATION": (
                    str(row.TEAM_ABBREVIATION)
                    if pd.notna(row.TEAM_ABBREVIATION)
                    else None
                ),
                "Season": 2026,
                "raw_net_rank": int(row.raw_net_rank),
                "shot_quality_net_rank": int(row.shot_quality_net_rank),
                "rank_change": int(row.rank_change),
                "lineup_offense_shotmaking_per_100_shots": round(
                    float(row.lineup_offense_shotmaking_per_100_shots), 6
                ),
                "lineup_defense_contest_per_100_shots": round(
                    float(row.lineup_defense_contest_per_100_shots), 6
                ),
                "lineup_net_residual_per_100_shots": round(
                    float(row.lineup_net_residual_per_100_shots), 6
                ),
                "lineup_shots": int(
                    min(row.offense_lineup_shots, row.defense_lineup_shots)
                ),
                "shooter_quality_above_league_per_100_shots": (
                    round(float(row.shooter_quality_above_league_per_100_shots), 6)
                    if pd.notna(row.shooter_quality_above_league_per_100_shots)
                    else None
                ),
                "shooter_shotmaking_above_quality_per_100_shots": (
                    round(float(row.shooter_shotmaking_above_quality_per_100_shots), 6)
                    if pd.notna(row.shooter_shotmaking_above_quality_per_100_shots)
                    else None
                ),
            }
        )
    OUTPUT.write_text(json.dumps(records, separators=(",", ":")))


if __name__ == "__main__":
    main()
