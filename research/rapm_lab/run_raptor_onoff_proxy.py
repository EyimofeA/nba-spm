"""Build and validate a transparent RAPTOR-on/off-inspired courtmate chain.

FiveThirtyEight disclosed the feature families but not the fitted coefficients,
the exact opposition adjustment, or the final second-order weighting.  This
runner therefore builds a named proxy and evaluates it out of season.  It must
never be described as an exact RAPTOR reproduction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.rapm_lab.run_external_reproduction_benchmark import (  # noqa: E402
    comparison_metrics,
    normalize_name,
    sha256_file,
)
from research.rapm_lab.run_wowy_raptor_reproduction import (  # noqa: E402
    RAPTOR_TEAM_URL,
    download_official_raptor,
)


STINTS = ROOT / "data/lake/silver/canonical_lineup_stints"
OFFICIAL = (
    ROOT
    / "research/rapm_lab/data/external/fivethirtyeight_raptor/modern_RAPTOR_by_team.csv"
)
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/raptor_onoff_proxy"
SEASONS = tuple(range(2014, 2027))
TRAIN_SEASONS = tuple(range(2014, 2019))
TEST_SEASONS = tuple(range(2019, 2023))
PLAYER_COLUMNS = [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]
FEATURE_COLUMNS = ["own_on", "courtmates_without", "second_order"]


def _side_arrays(frame: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = frame[PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    home_possession = frame["home_poss"].to_numpy(dtype=bool)
    offense = np.where(home_possession[:, None], values[:, 5:], values[:, :5])
    defense = np.where(home_possession[:, None], values[:, :5], values[:, 5:])
    offense_team = np.where(
        home_possession,
        frame["home_team_id"].to_numpy(dtype=np.int64),
        frame["away_team_id"].to_numpy(dtype=np.int64),
    )
    defense_team = np.where(
        home_possession,
        frame["away_team_id"].to_numpy(dtype=np.int64),
        frame["home_team_id"].to_numpy(dtype=np.int64),
    )
    if side == "offense":
        return offense, offense_team, defense_team, frame["pts"].to_numpy(float)
    if side == "defense":
        return defense, defense_team, offense_team, frame["pts"].to_numpy(float)
    raise ValueError(side)


def _player_totals(
    lineups: np.ndarray,
    own_team: np.ndarray,
    points: np.ndarray,
    opponent_adjustment: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat = lineups.ravel()
    keep = flat > 0
    repeats = lineups.shape[1]
    rows = pd.DataFrame(
        {
            "PLAYER_ID": flat[keep],
            "team_id": np.repeat(own_team, repeats)[keep],
            "points": np.repeat(points, repeats)[keep],
            "opponent_adjustment": np.repeat(opponent_adjustment, repeats)[keep],
        }
    )
    rows["possessions"] = 1.0
    totals = rows.groupby("PLAYER_ID", as_index=False).agg(
        possessions=("possessions", "sum"),
        points=("points", "sum"),
        opponent_adjustment=("opponent_adjustment", "sum"),
    )
    stints = rows.groupby(["PLAYER_ID", "team_id"], as_index=False).agg(
        possessions=("possessions", "sum"),
        points=("points", "sum"),
        opponent_adjustment=("opponent_adjustment", "sum"),
    )
    return totals, stints


def _rating(
    points: pd.Series,
    possessions: pd.Series,
    adjustment: pd.Series,
    league_rating: float,
    side: str,
) -> pd.Series:
    raw = 100.0 * points / possessions
    opponent = adjustment / possessions
    if side == "offense":
        return raw - league_rating + opponent
    return league_rating - raw + opponent


def _pair_totals(
    lineups: np.ndarray,
    own_team: np.ndarray,
    points: np.ndarray,
    opponent_adjustment: np.ndarray,
) -> pd.DataFrame:
    parts = []
    for left, right in combinations(range(lineups.shape[1]), 2):
        player_left = lineups[:, left]
        player_right = lineups[:, right]
        keep = (player_left > 0) & (player_right > 0) & (player_left != player_right)
        parts.append(
            pd.DataFrame(
                {
                    "player_low": np.minimum(player_left[keep], player_right[keep]),
                    "player_high": np.maximum(player_left[keep], player_right[keep]),
                    "team_id": own_team[keep],
                    "shared_points": points[keep],
                    "shared_adjustment": opponent_adjustment[keep],
                    "shared_possessions": 1.0,
                }
            )
        )
    pairs = pd.concat(parts, ignore_index=True)
    return pairs.groupby(["player_low", "player_high", "team_id"], as_index=False).sum()


def _courtmate_chain(
    stints: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    side: str,
    league_rating: float,
) -> pd.DataFrame:
    low = pairs.rename(columns={"player_low": "PLAYER_ID", "player_high": "courtmate"})
    high = pairs.rename(columns={"player_high": "PLAYER_ID", "player_low": "courtmate"})
    directed = pd.concat([low, high], ignore_index=True)
    mate_stints = stints.rename(
        columns={
            "PLAYER_ID": "courtmate",
            "possessions": "mate_possessions",
            "points": "mate_points",
            "opponent_adjustment": "mate_adjustment",
        }
    )
    directed = directed.merge(
        mate_stints,
        on=["courtmate", "team_id"],
        how="left",
        validate="many_to_one",
    )
    directed["without_possessions"] = (
        directed["mate_possessions"] - directed["shared_possessions"]
    )
    directed["without_points"] = directed["mate_points"] - directed["shared_points"]
    directed["without_adjustment"] = (
        directed["mate_adjustment"] - directed["shared_adjustment"]
    )
    directed = directed.loc[directed["without_possessions"].gt(0)].copy()
    directed["courtmate_without_rating"] = _rating(
        directed["without_points"],
        directed["without_possessions"],
        directed["without_adjustment"],
        league_rating,
        side,
    )
    directed["official_style_weight"] = (
        directed["shared_possessions"] * directed["without_possessions"]
    )
    directed["weighted_rating"] = (
        directed["courtmate_without_rating"] * directed["official_style_weight"]
    )
    first = directed.groupby("PLAYER_ID", as_index=False).agg(
        courtmate_weight=("official_style_weight", "sum"),
        weighted_rating=("weighted_rating", "sum"),
    )
    first["courtmates_without"] = first["weighted_rating"] / first["courtmate_weight"]

    first_by_team = directed.groupby(["PLAYER_ID", "team_id"], as_index=False).agg(
        first_weight=("official_style_weight", "sum"),
        first_numerator=("weighted_rating", "sum"),
    )
    mate_first = first_by_team.rename(
        columns={
            "PLAYER_ID": "courtmate",
            "first_weight": "mate_first_weight",
            "first_numerator": "mate_first_numerator",
        }
    )
    reverse = directed[
        ["PLAYER_ID", "courtmate", "team_id", "official_style_weight", "weighted_rating"]
    ].rename(
        columns={
            "PLAYER_ID": "courtmate",
            "courtmate": "PLAYER_ID",
            "official_style_weight": "focal_contribution_weight",
            "weighted_rating": "focal_contribution_numerator",
        }
    )
    second_rows = directed.merge(
        mate_first,
        on=["courtmate", "team_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        reverse,
        on=["PLAYER_ID", "courtmate", "team_id"],
        how="left",
        validate="one_to_one",
    )
    second_rows["mate_excluding_focal_weight"] = (
        second_rows["mate_first_weight"] - second_rows["focal_contribution_weight"]
    )
    second_rows["mate_excluding_focal_numerator"] = (
        second_rows["mate_first_numerator"] - second_rows["focal_contribution_numerator"]
    )
    second_rows = second_rows.loc[second_rows["mate_excluding_focal_weight"].gt(0)].copy()
    second_rows["mate_first_order"] = (
        second_rows["mate_excluding_focal_numerator"]
        / second_rows["mate_excluding_focal_weight"]
    )
    second_rows["weighted_second"] = (
        second_rows["mate_first_order"] * second_rows["official_style_weight"]
    )
    second = second_rows.groupby("PLAYER_ID", as_index=False).agg(
        second_weight=("official_style_weight", "sum"),
        weighted_second=("weighted_second", "sum"),
    )
    second["second_order"] = second["weighted_second"] / second["second_weight"]
    return first[["PLAYER_ID", "courtmates_without"]].merge(
        second[["PLAYER_ID", "second_order"]],
        on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )


def season_side_features(frame: pd.DataFrame, *, season: int, side: str) -> pd.DataFrame:
    lineups, own_team, opponent_team, points = _side_arrays(frame, side)
    league_rating = 100.0 * float(points.sum()) / len(points)

    offense_team = np.where(
        frame["home_poss"].to_numpy(dtype=bool),
        frame["home_team_id"].to_numpy(dtype=np.int64),
        frame["away_team_id"].to_numpy(dtype=np.int64),
    )
    defense_team = np.where(
        frame["home_poss"].to_numpy(dtype=bool),
        frame["away_team_id"].to_numpy(dtype=np.int64),
        frame["home_team_id"].to_numpy(dtype=np.int64),
    )
    team_rows = pd.DataFrame(
        {"offense_team": offense_team, "defense_team": defense_team, "points": points}
    )
    team_offense = team_rows.groupby("offense_team")["points"].agg(["sum", "count"])
    team_defense = team_rows.groupby("defense_team")["points"].agg(["sum", "count"])
    team_offense["rating"] = 100.0 * team_offense["sum"] / team_offense["count"]
    team_defense["rating"] = 100.0 * team_defense["sum"] / team_defense["count"]
    if side == "offense":
        opponent_adjustment = league_rating - pd.Series(opponent_team).map(
            team_defense["rating"]
        ).to_numpy(float)
    else:
        opponent_adjustment = pd.Series(opponent_team).map(team_offense["rating"]).to_numpy(float) - league_rating

    totals, stints = _player_totals(lineups, own_team, points, opponent_adjustment)
    totals["own_on"] = _rating(
        totals["points"],
        totals["possessions"],
        totals["opponent_adjustment"],
        league_rating,
        side,
    )

    team_totals = pd.DataFrame(
        {
            "team_id": own_team,
            "points": points,
            "opponent_adjustment": opponent_adjustment,
            "possessions": 1.0,
        }
    ).groupby("team_id", as_index=False).sum()
    off_stints = stints.merge(team_totals, on="team_id", suffixes=("_on", "_team"))
    off_stints["off_possessions"] = (
        off_stints["possessions_team"] - off_stints["possessions_on"]
    )
    off_stints = off_stints.loc[off_stints["off_possessions"].gt(0)].copy()
    off_stints["off_points"] = off_stints["points_team"] - off_stints["points_on"]
    off_stints["off_adjustment"] = (
        off_stints["opponent_adjustment_team"] - off_stints["opponent_adjustment_on"]
    )
    on_rating = _rating(
        off_stints["points_on"],
        off_stints["possessions_on"],
        off_stints["opponent_adjustment_on"],
        league_rating,
        side,
    )
    off_rating = _rating(
        off_stints["off_points"],
        off_stints["off_possessions"],
        off_stints["off_adjustment"],
        league_rating,
        side,
    )
    off_stints["raw_onoff"] = on_rating - off_rating
    off_stints["weighted_raw_onoff"] = (
        off_stints["raw_onoff"] * off_stints["possessions_on"]
    )
    raw = off_stints.groupby("PLAYER_ID", as_index=False).agg(
        raw_weight=("possessions_on", "sum"),
        weighted_raw_onoff=("weighted_raw_onoff", "sum"),
    )
    raw["raw_onoff"] = raw["weighted_raw_onoff"] / raw["raw_weight"]

    pairs = _pair_totals(lineups, own_team, points, opponent_adjustment)
    chain = _courtmate_chain(stints, pairs, side=side, league_rating=league_rating)
    output = totals[["PLAYER_ID", "possessions", "own_on"]].merge(
        raw[["PLAYER_ID", "raw_onoff"]], on="PLAYER_ID", how="left", validate="one_to_one"
    )
    output = output.merge(chain, on="PLAYER_ID", how="left", validate="one_to_one")
    output["season"] = season
    output["side"] = side
    output["league_rating"] = league_rating
    return output


def load_season(season: int) -> pd.DataFrame:
    source = pd.read_parquet(STINTS / f"season={season}/regular.parquet")
    rows = []
    away = [f"away_player_{slot}" for slot in range(1, 6)]
    home = [f"home_player_{slot}" for slot in range(1, 6)]
    for home_poss, possession_column, point_column in (
        (False, "away_possessions", "away_points"),
        (True, "home_possessions", "home_points"),
    ):
        part = source.loc[source[possession_column].gt(0)].copy()
        counts = part[possession_column].astype(int)
        part["pts"] = part[point_column] / counts
        part = part.loc[part.index.repeat(counts)].reset_index(drop=True)
        part["home_poss"] = home_poss
        part = part.rename(
            columns={
                **{column: f"a{slot}" for slot, column in enumerate(away, 1)},
                **{column: f"h{slot}" for slot, column in enumerate(home, 1)},
            }
        )
        rows.append(part[["game_id", "home_poss", "pts", "home_team_id", "away_team_id", *PLAYER_COLUMNS]])
    return pd.concat(rows, ignore_index=True)


def build_feature_panel(checkpoint: Path) -> pd.DataFrame:
    checkpoint.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        path = checkpoint / f"season_{season}.parquet"
        if path.exists():
            season_features = pd.read_parquet(path)
        else:
            source = load_season(season)
            season_features = pd.concat(
                [
                    season_side_features(source, season=season, side="offense"),
                    season_side_features(source, season=season, side="defense"),
                ],
                ignore_index=True,
            )
            season_features.to_parquet(path, index=False)
        frames.append(season_features)
        print(f"features {season}: {len(season_features):,} side rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def official_regular_season_panel() -> pd.DataFrame:
    source = pd.read_csv(OFFICIAL)
    source = source.loc[source["season_type"].eq("RS")].copy()
    source["name_key"] = source["player_name"].map(normalize_name)
    components = ["raptor_onoff_offense", "raptor_onoff_defense"]
    source = source.dropna(subset=["poss", *components])
    player_keys = source[["player_id", "name_key", "season"]].drop_duplicates()
    if player_keys[["name_key", "season"]].duplicated().any():
        raise ValueError("Official RAPTOR has ambiguous normalized player-name seasons")
    for column in components:
        source[f"weighted_{column}"] = source[column] * source["poss"]
    grouped = source.groupby(["name_key", "season"], as_index=False).agg(
        player_name=("player_name", "first"),
        official_possessions=("poss", "sum"),
        official_minutes=("mp", "sum"),
        weighted_offense=("weighted_raptor_onoff_offense", "sum"),
        weighted_defense=("weighted_raptor_onoff_defense", "sum"),
    )
    grouped["target_offense"] = grouped["weighted_offense"] / grouped["official_possessions"]
    grouped["target_defense"] = grouped["weighted_defense"] / grouped["official_possessions"]
    grouped["target_net"] = grouped["target_offense"] + grouped["target_defense"]
    return grouped


def attach_names(features: pd.DataFrame) -> pd.DataFrame:
    names = pd.DataFrame(
        [(row["id"], row["full_name"]) for row in nba_players.get_players()],
        columns=["PLAYER_ID", "local_name"],
    ).drop_duplicates("PLAYER_ID")
    features = features.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    features["name_key"] = features["local_name"].map(normalize_name)
    return features


def fit_and_score(features: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    wide = features.pivot(
        index=["PLAYER_ID", "name_key", "local_name", "season"],
        columns="side",
        values=["possessions", "raw_onoff", *FEATURE_COLUMNS],
    )
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
    wide = wide.reset_index()
    # FiveThirtyEight uses Basketball Reference IDs, while the possession data
    # uses NBA IDs.  Names are the only shared public key.  Exclude ambiguous
    # same-name seasons instead of guessing which player is which.
    unique_name_season = ~wide.duplicated(["name_key", "season"], keep=False)
    wide = wide.loc[unique_name_season].copy()
    merged = wide.merge(official, on=["name_key", "season"], how="left", validate="one_to_one")

    train_rows = []
    for side in ("offense", "defense"):
        part = merged.loc[
            merged["season"].isin(TRAIN_SEASONS) & merged[f"target_{side}"].notna()
        ].copy()
        part["target"] = part[f"target_{side}"]
        part["side"] = side
        for feature in FEATURE_COLUMNS:
            part[feature] = part[f"{feature}_{side}"]
        part["fit_weight"] = np.sqrt(part[f"possessions_{side}"].clip(lower=1))
        train_rows.append(part)
    train = pd.concat(train_rows, ignore_index=True).dropna(
        subset=["target", "fit_weight", *FEATURE_COLUMNS]
    )
    model = LinearRegression().fit(
        train[FEATURE_COLUMNS], train["target"], sample_weight=train["fit_weight"]
    )
    own_model = LinearRegression().fit(
        train[["own_on"]], train["target"], sample_weight=train["fit_weight"]
    )

    for side in ("offense", "defense"):
        side_features = merged[[f"{feature}_{side}" for feature in FEATURE_COLUMNS]].copy()
        side_features.columns = FEATURE_COLUMNS
        complete = side_features.notna().all(axis=1)
        merged[f"proxy_{side}"] = np.nan
        merged.loc[complete, f"proxy_{side}"] = model.predict(side_features.loc[complete])
        merged[f"own_only_{side}"] = np.nan
        own_complete = side_features["own_on"].notna()
        merged.loc[own_complete, f"own_only_{side}"] = own_model.predict(
            side_features.loc[own_complete, ["own_on"]]
        )
    merged["proxy_net"] = merged["proxy_offense"] + merged["proxy_defense"]
    merged["own_only_net"] = merged["own_only_offense"] + merged["own_only_defense"]
    merged["raw_onoff_net"] = merged["raw_onoff_offense"] + merged["raw_onoff_defense"]

    metric_rows = []
    for cohort, mask in {
        "all": merged["season"].isin(TEST_SEASONS),
        "1000+ minutes": merged["season"].isin(TEST_SEASONS) & merged["official_minutes"].ge(1000),
    }.items():
        for model_name, prefix in {
            "raw on/off": "raw_onoff",
            "opposition-adjusted own-on": "own_only",
            "three-family courtmate proxy": "proxy",
        }.items():
            for component in ("offense", "defense", "net"):
                frame = merged.loc[mask, [f"target_{component}", f"{prefix}_{component}"]].rename(
                    columns={f"target_{component}": "reference", f"{prefix}_{component}": "courtsignal"}
                )
                metric_rows.append(
                    {
                        "cohort": cohort,
                        "model": model_name,
                        "component": component,
                        **comparison_metrics(frame),
                    }
                )
    coefficients = {
        "intercept": float(model.intercept_),
        "features": {feature: float(value) for feature, value in zip(FEATURE_COLUMNS, model.coef_)},
        "own_only_intercept": float(own_model.intercept_),
        "own_only_coefficient": float(own_model.coef_[0]),
    }
    return pd.DataFrame(metric_rows), merged, coefficients


def main() -> int:
    if not OFFICIAL.exists():
        download_official_raptor(OFFICIAL, url=RAPTOR_TEAM_URL)
    source_hashes = {
        str(OFFICIAL.relative_to(ROOT)): sha256_file(OFFICIAL),
        str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve()),
    }
    for season in SEASONS:
        path = STINTS / f"season={season}/regular.parquet"
        source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    identity_payload = {
        "seasons": SEASONS,
        "train": TRAIN_SEASONS,
        "test": TEST_SEASONS,
        "features": FEATURE_COLUMNS,
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"raptor_onoff_proxy_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    features = build_feature_panel(output / "feature_checkpoints")
    features = attach_names(features)
    official = official_regular_season_panel()
    metrics, matches, coefficients = fit_and_score(features, official)
    features.to_parquet(output / "features.parquet", index=False)
    matches.to_parquet(output / "matches.parquet", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_proxy_not_replication",
        "estimand": "regular-season RAPTOR-on/off-inspired courtmate chain",
        "target_source": (
            "FiveThirtyEight modern_RAPTOR_by_team.csv, regular-season team stints "
            "aggregated to player-season with possession weights"
        ),
        "development_seasons": list(TRAIN_SEASONS),
        "held_out_seasons": list(TEST_SEASONS),
        "published_seasons": list(SEASONS),
        "coefficients": coefficients,
        "source_hashes": source_hashes,
        "metrics": metrics.to_dict(orient="records"),
        "known_mismatches": [
            "FiveThirtyEight did not publish its three coefficients.",
            "The exact opposition adjustment was not published.",
            "The proxy uses full-season opponent team ratings, not player-leave-out ratings.",
            "The exact second-order courtmate weights were not published.",
            "The calibration fit weights each player-side by square-root possessions.",
            "This proxy learns against public RAPTOR on/off in development seasons.",
            "The proxy target is regular-season only; the player CSV identity check combines "
            "regular season and playoffs.",
        ],
        "forbidden_interpretation": (
            "This is not an independent validation target or an exact FiveThirtyEight replication. "
            "The correlated feature coefficients are not causal or individually identified effects."
        ),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(coefficients, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
