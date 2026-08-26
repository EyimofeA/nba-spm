#!/usr/bin/env python3
"""Build the localhost public all-in-one comparison artifact."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import write_json_atomic
from nba_impact.models.public_aio_benchmark import (
    build_public_aio_benchmark,
    load_epm_ratings,
    load_lebron_ratings,
    load_mamba_ratings,
    load_site_aio_ratings,
    map_named_metric,
)


ROOT = Path(__file__).resolve().parents[1]


METRIC_DEFINITIONS = [
    {
        "metric": "new_aio",
        "metric_label": "New AIO",
        "included": True,
        "kind": "CourtSignal hybrid",
        "how_it_works": "Five-year-target SPM prior with stabilized feature additions, updated by one season of possession RAPM.",
        "interpretation": "Retrospective impact estimate; research challenger.",
    },
    {
        "metric": "site_aio",
        "metric_label": "Website AIO",
        "included": True,
        "kind": "CourtSignal annual hybrid",
        "how_it_works": "One-season statistical prior, updated by that season's possession RAPM.",
        "interpretation": "The exact annual AIO rows currently shipped by the website.",
    },
    {
        "metric": "rapm",
        "metric_label": "RAPM",
        "included": True,
        "kind": "CourtSignal adjusted plus-minus",
        "how_it_works": "Zero-prior ridge regression on possession outcomes with separate offensive and defensive player coefficients.",
        "interpretation": "Noisy one-season lineup-adjusted reference, not a forecast.",
    },
    {
        "metric": "annual_spm",
        "metric_label": "Annual SPM",
        "included": True,
        "kind": "CourtSignal statistical model",
        "how_it_works": "Box, play-by-play, and tracking features predict annual zero-prior RAPM without using lineup outcomes as inputs.",
        "interpretation": "Stat-only estimate of annual impact.",
    },
    {
        "metric": "box_pipm_style",
        "metric_label": "BoxPIPM-style",
        "included": True,
        "kind": "CourtSignal box baseline",
        "how_it_works": "Leave-one-season-out ridge on 15 traditional per-100 box rates trained against annual RAPM.",
        "interpretation": "Transparent box-only baseline; not full PIPM.",
    },
    {
        "metric": "bpm",
        "metric_label": "BPM 2.0",
        "included": True,
        "kind": "Public box metric",
        "how_it_works": "Published box-score coefficients vary by estimated position and offensive role, then receive a team-efficiency adjustment.",
        "interpretation": "Official Basketball-Reference values; box only.",
    },
    {
        "metric": "xrapm",
        "metric_label": "xRAPM",
        "included": True,
        "kind": "Public hybrid",
        "how_it_works": "Statistical prior plus regularized adjusted plus-minus; defense is converted to positive-good here.",
        "interpretation": "Prior-informed impact comparator, not independent of RAPM.",
    },
    {
        "metric": "lebron",
        "metric_label": "LEBRON",
        "included": True,
        "kind": "Public hybrid",
        "how_it_works": "Role-stabilized BoxPIPM prior combined with luck-adjusted RAPM.",
        "interpretation": "Retrospective impact estimate from BBall Index.",
    },
    {
        "metric": "mamba",
        "metric_label": "MAMBA",
        "included": True,
        "kind": "Public research hybrid",
        "how_it_works": "Current-season statistical prior combined with time-decayed multi-year RAPM and small shooting-luck adjustments.",
        "interpretation": "Author-described proof of concept.",
    },
    {
        "metric": "epm",
        "metric_label": "EPM",
        "included": True,
        "kind": "Public predictive hybrid",
        "how_it_works": "Stat-specific estimated skills feed an SPM prior, which is combined with RAPM.",
        "interpretation": "Supplied all-season snapshot; historical values may reflect the current model rather than archived season-end vintages.",
    },
]


def load_player_minutes(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        required = {"player_id", "season_end", "team_id", "minutes_seconds", "season_type"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Player games {path} are missing {missing}.")
        frame = frame.loc[
            frame["season_type"].eq("regular")
            & frame["minutes_seconds"].fillna(0).gt(0),
            ["player_id", "season_end", "team_id", "minutes_seconds"],
        ].copy()
        frame = frame.rename(
            columns={"player_id": "PLAYER_ID", "season_end": "Season"}
        )
        frame["minutes"] = frame["minutes_seconds"] / 60.0
        frames.append(frame[["PLAYER_ID", "Season", "team_id", "minutes"]])
    minutes = pd.concat(frames, ignore_index=True)
    if minutes.duplicated(["PLAYER_ID", "Season", "team_id"]).all():
        raise ValueError("Player-minute inputs unexpectedly contain only duplicate keys.")
    return minutes


def load_team_games(schedule_root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = schedule_root / f"leaguegamelog_{season}.json.gz"
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        result = payload["resultSets"][0]
        frame = pd.DataFrame(result["rowSet"], columns=result["headers"])
        if frame.groupby("GAME_ID").size().ne(2).any():
            raise ValueError(f"Season {season} has a game without two team rows.")
        if frame["WL"].isna().any():
            raise ValueError(f"Season {season} has missing win/loss outcomes.")
        rows.append(
            pd.DataFrame(
                {
                    "Season": int(season),
                    "team_id": frame["TEAM_ID"].astype(int),
                    "won": frame["WL"].eq("W"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def internal_ratings(
    site_data_root: Path,
    new_aio_path: Path,
    annual_spm_path: Path,
    box_pipm_path: Path,
) -> list[pd.DataFrame]:
    frames = [load_site_aio_ratings(site_data_root, (2021, 2022, 2023, 2024))]
    rapm_rows = []
    for season in (2021, 2022, 2023, 2024):
        source = pd.read_json(site_data_root / f"leaderboard-{season}.json")
        rapm_rows.append(
            source.rename(
                columns={
                    "normal_rapm_offense": "offense",
                    "normal_rapm_defense": "defense",
                    "normal_rapm_net": "net",
                }
            )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
        )
    rapm = pd.concat(rapm_rows, ignore_index=True)
    rapm["metric"] = "rapm"
    rapm["metric_label"] = "RAPM"
    rapm["category"] = "CourtSignal adjusted plus-minus"
    frames.append(rapm)
    new = pd.read_parquet(new_aio_path)
    new = new.loc[
        new["variant"].eq("selected_combined")
        & new["Poss_Off"].gt(0)
        & new["Poss_Def"].gt(0),
        ["PLAYER_ID", "rating_season", "offense", "defense", "net"],
    ].rename(columns={"rating_season": "Season"})
    new["metric"] = "new_aio"
    new["metric_label"] = "New AIO"
    new["category"] = "CourtSignal hybrid"
    frames.append(new)
    spm = pd.read_parquet(annual_spm_path).rename(
        columns={
            "spm_offense": "offense",
            "spm_defense": "defense",
            "spm_net": "net",
        }
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    spm["metric"] = "annual_spm"
    spm["metric_label"] = "Annual SPM"
    spm["category"] = "CourtSignal statistical model"
    frames.append(spm)
    pipm = pd.read_parquet(box_pipm_path).rename(
        columns={
            "box_pipm_style_offense": "offense",
            "box_pipm_style_defense": "defense",
            "box_pipm_style_net": "net",
        }
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    pipm["metric"] = "box_pipm_style"
    pipm["metric_label"] = "BoxPIPM-style"
    pipm["category"] = "CourtSignal box baseline"
    frames.append(pipm)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epm", type=Path, required=True)
    parser.add_argument("--lebron", type=Path, required=True)
    parser.add_argument("--mamba", type=Path, required=True)
    parser.add_argument("--external-annual", type=Path, required=True)
    parser.add_argument("--annual-spm", type=Path, required=True)
    parser.add_argument("--box-pipm", type=Path, required=True)
    parser.add_argument("--historical-player-games", type=Path, required=True)
    parser.add_argument("--recent-player-games", type=Path, required=True)
    parser.add_argument("--site-data-root", type=Path, default=ROOT / "web/public/data")
    parser.add_argument(
        "--new-aio",
        type=Path,
        default=ROOT
        / "artifacts/models/five_year_spm_feature_research/five_year_spm_feature_research_v1_93c148510e/aio_ratings.parquet",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=ROOT / "data/lake/bronze/official_game_schedule_1997_2026",
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    epm = load_epm_ratings(args.epm)
    lebron, identity = load_lebron_ratings(args.lebron)
    mamba, mamba_unmatched = load_mamba_ratings(args.mamba, identity)
    external = pd.read_parquet(args.external_annual)
    bpm, bpm_unmatched = map_named_metric(
        external.dropna(subset=["player_name_bpm"]),
        identity,
        metric="bpm",
        metric_label="BPM 2.0",
        category="public box metric",
        season_column="season",
        name_column="player_name_bpm",
        offense_column="bpm_offense",
        defense_column="bpm_defense",
        net_column="bpm_net",
    )
    xrapm, xrapm_unmatched = map_named_metric(
        external.dropna(subset=["player_name_xrapm"]),
        identity,
        metric="xrapm",
        metric_label="xRAPM",
        category="public hybrid",
        season_column="season",
        name_column="player_name_xrapm",
        offense_column="xrapm_offense",
        defense_column="xrapm_defense",
        net_column="xrapm_net",
    )
    ratings = pd.concat(
        [
            *internal_ratings(
                args.site_data_root,
                args.new_aio,
                args.annual_spm,
                args.box_pipm,
            ),
            epm,
            lebron,
            mamba,
            bpm,
            xrapm,
        ],
        ignore_index=True,
    )
    minutes = load_player_minutes(
        [args.historical_player_games, args.recent_player_games]
    )
    team_games = load_team_games(args.schedule_root, (2022, 2023, 2024, 2025))
    source_files = {
        "epm": args.epm,
        "lebron": args.lebron,
        "mamba": args.mamba,
        "external_annual": args.external_annual,
        "annual_spm": args.annual_spm,
        "box_pipm": args.box_pipm,
        "historical_player_games": args.historical_player_games,
        "recent_player_games": args.recent_player_games,
        "new_aio": args.new_aio,
        **{
            f"site_aio_{season}": args.site_data_root / f"leaderboard-{season}.json"
            for season in (2021, 2022, 2023, 2024)
        },
        **{
            f"team_games_{season}": args.schedule_root
            / f"leaguegamelog_{season}.json.gz"
            for season in (2022, 2023, 2024, 2025)
        },
    }
    run = build_public_aio_benchmark(
        ratings,
        minutes,
        team_games,
        artifact_root=args.artifact_root,
        definitions=METRIC_DEFINITIONS,
        source_files=source_files,
    )
    output = (
        args.artifact_root
        / "research"
        / "public_aio_benchmark"
        / run["run_id"]
    )
    unmatched = pd.concat(
        [
            mamba_unmatched.assign(metric="mamba"),
            bpm_unmatched.assign(metric="bpm"),
            xrapm_unmatched.assign(metric="xrapm"),
        ],
        ignore_index=True,
    )
    unmatched.to_parquet(output / "unmatched_identities.parquet", index=False)
    run["quality"]["unmatched_identity_rows"] = int(len(unmatched))
    run["quality"]["unmatched_identity_rows_by_metric"] = {
        str(metric): int(count)
        for metric, count in unmatched.groupby("metric").size().items()
    }
    run["paths"]["unmatched_identities"] = "unmatched_identities.parquet"
    write_json_atomic(run, output / "run.json")
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "rating_rows": run["quality"]["rating_rows"],
                "metrics": run["quality"]["metrics"],
                "unmatched_identity_rows": len(unmatched),
                "projected_minutes": run["projected_minutes_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
