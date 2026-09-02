"""Reconstruct PIPM from its published coefficients and CourtSignal data.

The public PIPM table combines regular season and playoffs and uses a private
luck-adjustment layer. CourtSignal has a regular-season canonical lineup ledger.
This runner therefore fixes the published box and blend coefficients, rebuilds
raw offensive and defensive on/off from lineup stints, and reports agreement
against the source instead of claiming numerical identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GABRIEL_SHEETS = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
HISTORICAL_SHEETS = ROOT / "data/lake/bronze/historical_player_sheets/year_totals"
STINTS = ROOT / "data/lake/silver/canonical_lineup_stints"
SOURCE = Path("/Users/eadebayo/Downloads/PIPM Player Finder through 2021 - Database.csv")
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/pipm_reconstruction"
# CourtSignal's canonical lineup ledger begins in 1997. The historical player
# sheets expose the same published PIPM box inputs as the Gabriel-era sheets.
SEASONS = range(1997, 2027)

BOX_COEFFICIENTS = {
    "avg": (0.264, 0.116),
    "start_share_sq": (0.266, 0.549),
    "pts_p36": (0.394, -0.037),
    "orb_p36": (0.328, -0.230),
    "drb_p36": (0.051, 0.239),
    "ast_p36": (0.375, 0.033),
    "stl_p36": (0.263, 0.857),
    "blk_p36": (-0.022, 0.817),
    "tov_p36": (-0.701, -0.193),
    "pf_p36": (-0.185, 0.088),
    "fta_p36": (0.012, 0.037),
    "fg2a_p36": (-0.260, -0.047),
    "fg3a_p36": (-0.073, -0.082),
    "minutes_weight": (-0.092, -0.041),
}


def fold_name(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in text if not unicodedata.combining(c)).lower())


def lineup_context(season: int) -> pd.DataFrame:
    stints = pd.read_parquet(STINTS / f"season={season}/regular.parquet")
    totals = []
    on_rows = []
    for side, opponent in (("home", "away"), ("away", "home")):
        team = f"{side}_team_id"
        off_poss = f"{side}_possessions"
        def_poss = f"{opponent}_possessions"
        off_points = f"{side}_points"
        def_points = f"{opponent}_points"
        team_totals = stints.groupby(["game_id", team], as_index=False).agg(
            team_off_poss=(off_poss, "sum"), team_def_poss=(def_poss, "sum"),
            team_off_points=(off_points, "sum"), team_def_points=(def_points, "sum"),
        ).rename(columns={team: "TEAM_ID"})
        totals.append(team_totals)
        for slot in range(1, 6):
            rows = stints[["game_id", team, f"{side}_player_{slot}", off_poss, def_poss, off_points, def_points]].copy()
            rows.columns = ["game_id", "TEAM_ID", "PLAYER_ID", "on_off_poss", "on_def_poss", "on_off_points", "on_def_points"]
            on_rows.append(rows)
    total = pd.concat(totals, ignore_index=True)
    on = pd.concat(on_rows, ignore_index=True).groupby(
        ["game_id", "TEAM_ID", "PLAYER_ID"], as_index=False
    ).sum(numeric_only=True)
    player_games = on.merge(total, on=["game_id", "TEAM_ID"], validate="many_to_one")
    for stem in ("off_poss", "def_poss", "off_points", "def_points"):
        player_games[f"off_{stem}"] = player_games[f"team_{stem}"] - player_games[f"on_{stem}"]
    summed = player_games.groupby("PLAYER_ID", as_index=False).sum(numeric_only=True)
    rate = lambda points, poss: 100 * summed[points] / summed[poss].replace(0, np.nan)
    on_ortg = rate("on_off_points", "on_off_poss")
    off_ortg = rate("off_off_points", "off_off_poss")
    on_drtg = rate("on_def_points", "on_def_poss")
    off_drtg = rate("off_def_points", "off_def_poss")
    league_ortg = 100 * total["team_off_points"].sum() / total["team_off_poss"].sum()
    league_drtg = 100 * total["team_def_points"].sum() / total["team_def_poss"].sum()
    return pd.DataFrame({
        "PLAYER_ID": summed["PLAYER_ID"].astype(int),
        "o_onoff": on_ortg - off_ortg,
        "d_onoff": off_drtg - on_drtg,
        "o_oncourt_vs_avg": on_ortg - league_ortg,
        "d_oncourt_vs_avg": league_drtg - on_drtg,
        "on_off_poss": summed["on_off_poss"],
        "on_def_poss": summed["on_def_poss"],
    })


def starter_counts(season: int) -> pd.DataFrame:
    stints = pd.read_parquet(STINTS / f"season={season}/regular.parquet")
    opening = stints.loc[stints["period"].eq(1)].sort_values(["game_id", "stint"]).drop_duplicates("game_id")
    player_columns = [f"{side}_player_{slot}" for side in ("home", "away") for slot in range(1, 6)]
    return (
        opening[player_columns].stack().astype(int).value_counts()
        .rename_axis("PLAYER_ID").rename("GS").reset_index()
    )


def season_features(season: int) -> pd.DataFrame:
    gabriel = GABRIEL_SHEETS / f"{season}.parquet"
    historical = HISTORICAL_SHEETS / f"{season}.csv"
    sheet = pd.read_parquet(gabriel) if gabriel.exists() else pd.read_csv(historical)
    required = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "GP", "PACE",
        "PTS", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "FTA", "FG2A", "FG3A",
    ]
    frame = sheet[required].copy().merge(lineup_context(season), on="PLAYER_ID", how="inner")
    frame = frame.merge(starter_counts(season), on="PLAYER_ID", how="left")
    frame["GS"] = frame["GS"].fillna(0)
    league_pace = np.average(frame["PACE"].fillna(frame["PACE"].median()), weights=frame["MIN"].clip(lower=1))
    pace_factor = league_pace / frame["PACE"].replace(0, np.nan)
    for source, target in {
        "PTS": "pts_p36", "OREB": "orb_p36", "DREB": "drb_p36", "AST": "ast_p36",
        "STL": "stl_p36", "BLK": "blk_p36", "TOV": "tov_p36", "PF": "pf_p36",
        "FTA": "fta_p36", "FG2A": "fg2a_p36", "FG3A": "fg3a_p36",
    }.items():
        frame[target] = 36 * frame[source] / frame["MIN"].replace(0, np.nan) * pace_factor
    frame["start_share_sq"] = (frame["GS"] / frame["GP"].replace(0, np.nan)).clip(0, 1) ** 2
    frame["minutes_weight"] = np.sqrt((frame["MIN"] / 3936).clip(0, 1))
    frame["minute_share"] = (frame["MIN"] / (48 * frame["GP"].replace(0, np.nan))).clip(0, 1)
    frame["o_avg"] = frame["o_oncourt_vs_avg"] * frame["minute_share"]
    frame["d_avg"] = frame["d_oncourt_vs_avg"] * frame["minute_share"]

    # Goldstein's table has no published intercept. Center each box input on
    # its minutes-weighted season mean, as the original regression target was
    # league-relative RAPM.
    for feature in BOX_COEFFICIENTS:
        source = "o_avg" if feature == "avg" else feature
        values = frame[source].replace([np.inf, -np.inf], np.nan).fillna(0)
        center = np.average(values, weights=frame["MIN"].clip(lower=1))
        frame[f"centered_{feature}"] = values - center
    frame["box_offense"] = sum(
        frame[f"centered_{feature}"] * coefficient[0]
        for feature, coefficient in BOX_COEFFICIENTS.items()
    )
    frame["centered_avg"] = frame["d_avg"] - np.average(frame["d_avg"], weights=frame["MIN"].clip(lower=1))
    frame["box_defense"] = sum(
        frame[f"centered_{feature}"] * coefficient[1]
        for feature, coefficient in BOX_COEFFICIENTS.items()
    )

    weight = frame["minutes_weight"]
    raw_offense = 0.302 * frame["o_onoff"] * weight + 0.672 * frame["box_offense"] + 0.151 * frame["o_avg"] * weight - 0.452 * weight
    raw_defense = 0.401 * frame["d_onoff"] * weight + 0.751 * frame["box_defense"] + 0.136 * frame["d_avg"] * weight + 0.023 * weight
    frame["pipm_offense"] = (raw_offense * frame["MIN"] - 1.7 * 350) / (frame["MIN"] + 350)
    frame["pipm_defense"] = (raw_defense * frame["MIN"] - 0.3 * 450) / (frame["MIN"] + 450)
    frame["pipm_net"] = frame["pipm_offense"] + frame["pipm_defense"]
    frame["season"] = season
    return frame[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "season", "MIN", "GP", "GS",
        "box_offense", "box_defense", "o_onoff", "d_onoff", "o_avg", "d_avg",
        "pipm_offense", "pipm_defense", "pipm_net",
    ]]


def metrics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE).rename(columns={"O-PIPM": "source_offense", "D-PIPM": "source_defense", "PIPM": "source_net"})
    source["season"] = source["Season"].str[:4].astype(int) + 1
    source["name_key"] = source["Player"].map(fold_name)
    candidate = panel.copy()
    candidate["name_key"] = candidate["PLAYER_NAME"].map(fold_name)
    matched = candidate.merge(source[["season", "name_key", "source_offense", "source_defense", "source_net", "MP"]], on=["season", "name_key"], how="inner")
    rows = []
    for component in ("offense", "defense", "net"):
        x = matched[f"pipm_{component}"]
        y = matched[f"source_{component}"]
        rows.append({
            "component": component,
            "matched_rows": int(len(matched)),
            "pearson": float(x.corr(y)),
            "r_squared": float(1 - ((x - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "source_scope": "regular season plus playoffs",
            "reconstruction_scope": "regular season",
        })
    return pd.DataFrame(rows), matched


def main() -> None:
    panel = pd.concat([season_features(season) for season in SEASONS], ignore_index=True)
    agreement, matched = metrics(panel)
    digest = hashlib.sha256(json.dumps({"seasons": list(SEASONS), "box": BOX_COEFFICIENTS}, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"pipm_reconstruction_v1_{digest}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output / "reconstructions.parquet", index=False)
    matched.to_parquet(output / "source_matches.parquet", index=False)
    agreement.to_parquet(output / "agreement.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(SEASONS),
        "status": "methodology_aligned_reconstruction",
        "source_scope": "PIPM database combines regular season and playoffs",
        "reconstruction_scope": "CourtSignal regular-season lineup ledger",
        "metrics": agreement.to_dict("records"),
        "limitations": [
            "The private PIPM luck adjustment is unavailable.",
            "The public source and reconstruction use different season scopes.",
            "The published coefficient table omits its fitted intercept and exact team adjustment.",
        ],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(agreement.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
