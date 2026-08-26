"""Downstream team-win evaluation for five-year SPM role challengers."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.external_impact_benchmark import normalize_player_name
from nba_impact.models.public_aio_benchmark import build_team_win_benchmark


EXPERIMENT_ID = "spm_role_team_win_benchmark_v1"
TEAM_ALIASES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}


class _TotalsParser(HTMLParser):
    """Extract the three required fields without an optional HTML dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.cell_stat: str | None = None
        self.cell_text: list[str] = []
        self.row: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "totals_stats":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = {}
        elif self.in_row and tag in {"th", "td"}:
            self.cell_stat = attributes.get("data-stat")
            self.cell_text = []

    def handle_data(self, data: str) -> None:
        if self.cell_stat is not None:
            self.cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and tag in {"th", "td"} and self.cell_stat is not None:
            self.row[self.cell_stat] = "".join(self.cell_text).strip()
            self.cell_stat = None
            self.cell_text = []
        elif self.in_row and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False


def _parse_bbref_totals(path: Path) -> pd.DataFrame:
    parser = _TotalsParser()
    parser.feed(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(parser.rows)
    required = {"name_display", "team_name_abbr", "mp"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Basketball-Reference table {path} is missing {missing}.")
    return frame.rename(
        columns={"name_display": "Player", "team_name_abbr": "Team", "mp": "MP"}
    )


def _load_team_games(schedule_root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        path = schedule_root / f"leaguegamelog_{season}.json.gz"
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        result = payload["resultSets"][0]
        frame = pd.DataFrame(result["rowSet"], columns=result["headers"])
        if frame.groupby("GAME_ID").size().ne(2).any() or frame["WL"].isna().any():
            raise ValueError(f"Season {season} has an invalid team-game schedule.")
        rows.append(
            pd.DataFrame(
                {
                    "Season": season,
                    "team_id": pd.to_numeric(frame["TEAM_ID"], errors="raise").astype(int),
                    "team": frame["TEAM_ABBREVIATION"].replace(TEAM_ALIASES),
                    "won": frame["WL"].eq("W"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_bbref_player_team_minutes(
    html_root: str | Path,
    identity_paths: dict[int, str | Path],
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve Basketball-Reference team-stint minutes to NBA player/team IDs."""
    identity_rows = []
    for season, path in sorted(identity_paths.items()):
        source = pd.read_parquet(path, columns=["PLAYER_ID", "PLAYER_NAME"])
        source = source.drop_duplicates().copy()
        source["Season"] = int(season)
        source["name_key"] = source["PLAYER_NAME"].map(normalize_player_name)
        identity_rows.append(source)
    identity = pd.concat(identity_rows, ignore_index=True)
    ambiguous = identity.groupby(["Season", "name_key"])["PLAYER_ID"].nunique()
    if ambiguous.gt(1).any():
        raise ValueError("Player identity table has an ambiguous season/name key.")
    identity = identity.drop_duplicates(["Season", "name_key"])

    team_map = team_games[["Season", "team", "team_id"]].drop_duplicates()
    if team_map.duplicated(["Season", "team"]).any():
        raise ValueError("Team abbreviation does not resolve to one NBA team ID.")

    minute_rows = []
    unmatched_rows = []
    coverage_rows = []
    root = Path(html_root)
    for season in sorted(identity_paths):
        path = root / f"nba_{season}_totals.html"
        frame = _parse_bbref_totals(path)
        frame = frame.loc[frame["Player"].ne("Player")].copy()
        frame["minutes"] = pd.to_numeric(frame["MP"], errors="coerce").fillna(0.0)
        frame = frame.loc[frame["minutes"].gt(0)].copy()
        # Traded players have a synthetic 2TM/3TM total plus one row per team.
        # Keep the real team rows so next-season allocation remains exact.
        frame = frame.loc[~frame["Team"].astype(str).str.endswith("TM")].copy()
        frame["Season"] = int(season)
        frame["team"] = frame["Team"].replace(TEAM_ALIASES)
        frame["name_key"] = frame["Player"].map(normalize_player_name)
        frame = frame.merge(
            identity.loc[identity["Season"].eq(season), ["PLAYER_ID", "name_key"]],
            on="name_key",
            how="left",
            validate="many_to_one",
        ).merge(
            team_map.loc[team_map["Season"].eq(season), ["team", "team_id"]],
            on="team",
            how="left",
            validate="many_to_one",
        )
        unmatched = frame.loc[
            frame["PLAYER_ID"].isna() | frame["team_id"].isna(),
            ["Season", "Player", "Team", "minutes"],
        ].copy()
        unmatched_rows.append(unmatched)
        matched = frame.dropna(subset=["PLAYER_ID", "team_id"]).copy()
        matched[["PLAYER_ID", "team_id"]] = matched[["PLAYER_ID", "team_id"]].astype(int)
        minute_rows.append(matched[["PLAYER_ID", "Season", "team_id", "minutes"]])
        coverage_rows.append(
            {
                "Season": int(season),
                "source_minutes": float(frame["minutes"].sum()),
                "matched_minutes": float(matched["minutes"].sum()),
                "minute_match_rate": float(matched["minutes"].sum() / frame["minutes"].sum()),
                "unmatched_rows": int(len(unmatched)),
            }
        )
    minutes = pd.concat(minute_rows, ignore_index=True)
    minutes = minutes.groupby(["PLAYER_ID", "Season", "team_id"], as_index=False)["minutes"].sum()
    coverage = pd.DataFrame(coverage_rows)
    if coverage["minute_match_rate"].min() < 0.999:
        raise ValueError("Basketball-Reference minute identity coverage is below 99.9%.")
    return minutes, pd.concat(unmatched_rows, ignore_index=True), coverage


def run_spm_role_team_win_benchmark(
    predictions_path: str | Path,
    *,
    html_root: str | Path,
    identity_paths: dict[int, str | Path],
    schedule_root: str | Path,
    artifact_root: str | Path,
    rating_seasons: tuple[int, ...] = (2020, 2021, 2022),
    minimum_metric_minutes: float = 250.0,
    replacement_value: float = -2.0,
) -> dict:
    predictions_path = Path(predictions_path)
    predictions = pd.read_parquet(predictions_path)
    required = {
        "PLAYER_ID", "Window_End", "variant", "prediction_offense",
        "prediction_defense", "prediction_net",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Role predictions are missing {missing}.")
    ratings = predictions.rename(
        columns={
            "Window_End": "Season",
            "prediction_offense": "offense",
            "prediction_defense": "defense",
            "prediction_net": "net",
        }
    )[["PLAYER_ID", "Season", "variant", "offense", "defense", "net"]].copy()
    ratings = ratings.loc[ratings["Season"].isin(rating_seasons)]
    if ratings.duplicated(["PLAYER_ID", "Season", "variant"]).any():
        raise ValueError("Role predictions contain duplicate player-season variants.")
    ratings["metric"] = ratings["variant"]
    ratings["metric_label"] = ratings["variant"].str.replace("_", " ").str.title()

    all_seasons = tuple(range(min(rating_seasons), max(rating_seasons) + 2))
    team_games = _load_team_games(Path(schedule_root), all_seasons)
    minutes, unmatched, source_coverage = load_bbref_player_team_minutes(
        html_root, identity_paths, team_games
    )
    folds, summary, metric_coverage = build_team_win_benchmark(
        ratings,
        minutes,
        team_games,
        rating_seasons=rating_seasons,
        minimum_metric_minutes=minimum_metric_minutes,
        replacement_values=(replacement_value,),
    )
    baseline = summary.loc[summary["metric"].eq("baseline")].iloc[0]
    summary["mean_r_squared_delta_vs_baseline"] = summary["mean_r_squared"] - float(
        baseline["mean_r_squared"]
    )
    summary["pooled_r_squared_delta_vs_baseline"] = summary["pooled_r_squared"] - float(
        baseline["pooled_r_squared"]
    )

    sources = {
        "predictions": predictions_path,
        **{f"identity_{season}": Path(path) for season, path in identity_paths.items()},
        **{
            f"bbref_totals_{season}": Path(html_root) / f"nba_{season}_totals.html"
            for season in all_seasons
        },
        **{
            f"team_games_{season}": Path(schedule_root) / f"leaguegamelog_{season}.json.gz"
            for season in all_seasons
        },
    }
    config = {
        "rating_seasons": list(rating_seasons),
        "outcome_seasons": [season + 1 for season in rating_seasons],
        "minimum_metric_minutes": minimum_metric_minutes,
        "replacement_value": replacement_value,
        "team_rating_formula": "5 * next-season-minute-weighted player rating",
        "source_hashes": {name: sha256_file(path) for name, path in sorted(sources.items())},
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "research" / "spm_role_team_win_benchmark" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    folds.to_parquet(output / "folds.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    metric_coverage.to_parquet(output / "metric_coverage.parquet", index=False)
    source_coverage.to_parquet(output / "source_coverage.parquet", index=False)
    unmatched.to_parquet(output / "unmatched_identities.parquet", index=False)
    run = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "association between season-t player ratings, season-t+1 observed minutes, and season-t+1 team win percentage",
        "config": config,
        "quality": {
            "folds": int(folds["rating_season"].nunique()),
            "team_seasons_per_variant": int(folds["teams"].sum() / folds["metric"].nunique()),
            "minimum_source_minute_match_rate": float(source_coverage["minute_match_rate"].min()),
            "minimum_rating_minute_coverage": float(metric_coverage["minute_coverage"].min()),
            "unmatched_identity_rows": int(len(unmatched)),
        },
        "caveats": [
            "Observed next-season minutes make this an oracle-minutes retrodiction, not a preseason forecast.",
            "All variants use the same role-known player cohort; missing and sub-250-minute ratings receive replacement value.",
            "Three outcome seasons provide only 90 team-seasons, so this is a selection diagnostic rather than confirmation.",
        ],
        "paths": {
            "folds": "folds.parquet",
            "summary": "summary.parquet",
            "metric_coverage": "metric_coverage.parquet",
            "source_coverage": "source_coverage.parquet",
            "unmatched_identities": "unmatched_identities.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    return run
