"""Matched public all-in-one agreement and next-season team-win benchmarks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.external_impact_benchmark import normalize_player_name


COMPONENTS = ("offense", "defense", "net")
RATING_COLUMNS = (
    "PLAYER_ID",
    "Season",
    "metric",
    "metric_label",
    "category",
    *COMPONENTS,
)


def load_lebron_ratings(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BBall Index LEBRON and return ratings plus an identity crosswalk."""
    source = pd.read_csv(path)
    required = {
        "nba_id",
        "Player",
        "Season",
        "LEBRON",
        "O-LEBRON",
        "D-LEBRON",
    }
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"LEBRON source is missing {missing}.")
    source["PLAYER_ID"] = pd.to_numeric(source["nba_id"], errors="raise").astype(int)
    source["Season"] = pd.to_numeric(source["Season"], errors="raise").astype(int)
    source["normalized_name"] = source["Player"].map(normalize_player_name)
    if source.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("LEBRON has duplicate player-season IDs.")
    ratings = source.rename(
        columns={
            "O-LEBRON": "offense",
            "D-LEBRON": "defense",
            "LEBRON": "net",
        }
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    ratings["metric"] = "lebron"
    ratings["metric_label"] = "LEBRON"
    ratings["category"] = "public hybrid"
    identity = source[["PLAYER_ID", "Season", "normalized_name"]].copy()
    return ratings, identity


def load_mamba_ratings(
    path: str | Path, identity: pd.DataFrame, *, maximum_season: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the author's public MAMBA workbook and resolve NBA IDs by season/name."""
    path = Path(path)
    source = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    required = {"Player", "Offense", "Defense", "Ovr", "Season"}
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"MAMBA source is missing {missing}.")
    source["Season"] = pd.to_numeric(source["Season"], errors="raise").astype(int)
    source = source.loc[source["Season"].le(maximum_season)].copy()
    source["normalized_name"] = source["Player"].map(normalize_player_name)
    if source.duplicated(["normalized_name", "Season"]).any():
        raise ValueError("MAMBA has duplicate normalized player-season names.")
    unique_identity = identity.loc[
        ~identity.duplicated(["normalized_name", "Season"], keep=False)
    ]
    matched = source.merge(
        unique_identity,
        on=["normalized_name", "Season"],
        how="left",
        validate="one_to_one",
    )
    unmatched = matched.loc[
        matched["PLAYER_ID"].isna(), ["Player", "Season", "normalized_name"]
    ].copy()
    matched = matched.dropna(subset=["PLAYER_ID"]).copy()
    matched["PLAYER_ID"] = matched["PLAYER_ID"].astype(int)
    ratings = matched.rename(
        columns={"Offense": "offense", "Defense": "defense", "Ovr": "net"}
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    ratings["metric"] = "mamba"
    ratings["metric_label"] = "MAMBA"
    ratings["category"] = "public research hybrid"
    return ratings, unmatched


def map_named_metric(
    source: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    category: str,
    season_column: str,
    name_column: str,
    offense_column: str,
    defense_column: str,
    net_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve a name-keyed metric against a season-specific NBA ID crosswalk."""
    frame = source.copy()
    frame["Season"] = pd.to_numeric(frame[season_column], errors="raise").astype(int)
    frame["normalized_name"] = frame[name_column].map(normalize_player_name)
    if frame.duplicated(["normalized_name", "Season"]).any():
        raise ValueError(f"{metric_label} has duplicate normalized name-season keys.")
    unique_identity = identity.loc[
        ~identity.duplicated(["normalized_name", "Season"], keep=False)
    ]
    matched = frame.merge(
        unique_identity,
        on=["normalized_name", "Season"],
        how="left",
        validate="one_to_one",
    )
    unmatched = matched.loc[
        matched["PLAYER_ID"].isna(), [name_column, "Season", "normalized_name"]
    ].copy()
    matched = matched.dropna(subset=["PLAYER_ID"]).copy()
    matched["PLAYER_ID"] = matched["PLAYER_ID"].astype(int)
    ratings = matched.rename(
        columns={
            offense_column: "offense",
            defense_column: "defense",
            net_column: "net",
        }
    )[["PLAYER_ID", "Season", "offense", "defense", "net"]]
    ratings["metric"] = metric
    ratings["metric_label"] = metric_label
    ratings["category"] = category
    return ratings, unmatched


def validate_rating_panel(ratings: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a long player-season metric panel."""
    missing = sorted(set(RATING_COLUMNS) - set(ratings.columns))
    if missing:
        raise ValueError(f"Rating panel is missing {missing}.")
    panel = ratings.loc[:, RATING_COLUMNS].copy()
    panel["PLAYER_ID"] = pd.to_numeric(panel["PLAYER_ID"], errors="raise").astype(int)
    panel["Season"] = pd.to_numeric(panel["Season"], errors="raise").astype(int)
    if panel.duplicated(["PLAYER_ID", "Season", "metric"]).any():
        raise ValueError("Rating panel has duplicate player-season-metric keys.")
    if panel[["metric", "metric_label", "category"]].isna().any().any():
        raise ValueError("Rating metadata cannot be missing.")
    if not np.isfinite(panel.loc[:, COMPONENTS].to_numpy(dtype=float)).all():
        raise ValueError("Rating components must be finite.")
    identity_error = (panel["offense"] + panel["defense"] - panel["net"]).abs()
    if identity_error.max() > 0.11:
        raise ValueError("Rating offense plus defense does not equal net.")
    return panel.sort_values(["Season", "metric", "PLAYER_ID"], kind="stable")


def build_pairwise_correlations(
    ratings: pd.DataFrame,
    player_minutes: pd.DataFrame,
    *,
    seasons: tuple[int, ...],
    minimum_minutes: float = 250.0,
) -> pd.DataFrame:
    """Compute pairwise-complete metric agreement on identical qualified rows."""
    if minimum_minutes < 0:
        raise ValueError("Minimum minutes cannot be negative.")
    minute_columns = {"PLAYER_ID", "Season", "minutes"}
    if missing := sorted(minute_columns - set(player_minutes.columns)):
        raise ValueError(f"Player minutes are missing {missing}.")
    minutes = (
        player_minutes.groupby(["PLAYER_ID", "Season"], as_index=False)["minutes"]
        .sum()
        .query("minutes >= @minimum_minutes")
    )
    eligible = ratings.loc[ratings["Season"].isin(seasons)].merge(
        minutes[["PLAYER_ID", "Season"]],
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="many_to_one",
    )
    metric_order = eligible[["metric", "metric_label"]].drop_duplicates()
    metric_order = metric_order.sort_values("metric_label", kind="stable")
    rows: list[dict] = []
    for component in COMPONENTS:
        wide = eligible.pivot(
            index=["PLAYER_ID", "Season"], columns="metric", values=component
        )
        for left in metric_order["metric"]:
            for right in metric_order["metric"]:
                if left not in wide or right not in wide:
                    continue
                pair = wide[[left, right]].dropna()
                if left == right:
                    pearson = 1.0
                    spearman = 1.0
                elif len(pair) >= 3:
                    pearson = float(pair[left].corr(pair[right], method="pearson"))
                    spearman = float(pair[left].corr(pair[right], method="spearman"))
                else:
                    pearson = np.nan
                    spearman = np.nan
                rows.append(
                    {
                        "component": component,
                        "left_metric": left,
                        "right_metric": right,
                        "rows": int(len(pair)),
                        "seasons": int(pair.reset_index()["Season"].nunique()),
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return pd.DataFrame(rows)


def _team_wins(team_games: pd.DataFrame) -> pd.DataFrame:
    required = {"Season", "team_id", "won"}
    if missing := sorted(required - set(team_games.columns)):
        raise ValueError(f"Team games are missing {missing}.")
    outcomes = team_games.copy()
    outcomes["won"] = outcomes["won"].astype(bool)
    return outcomes.groupby(["Season", "team_id"], as_index=False).agg(
        games=("won", "size"), wins=("won", "sum")
    ).assign(win_pct=lambda frame: frame["wins"] / frame["games"])


def build_team_win_benchmark(
    ratings: pd.DataFrame,
    player_minutes: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    rating_seasons: tuple[int, ...],
    minimum_metric_minutes: float = 250.0,
    replacement_values: tuple[float, ...] = (-2.0,),
    minimum_teams: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict next-season team win rate with observed next-season minutes.

    This is an oracle-minutes retrodiction, not a preseason forecast. A player
    receives the replacement value when absent from the metric or when their
    rating-season minutes are below ``minimum_metric_minutes``.
    """
    required_minutes = {"PLAYER_ID", "Season", "team_id", "minutes"}
    if missing := sorted(required_minutes - set(player_minutes.columns)):
        raise ValueError(f"Player minutes are missing {missing}.")
    if minimum_metric_minutes < 0:
        raise ValueError("Minimum metric-year minutes cannot be negative.")
    if not replacement_values:
        raise ValueError("At least one replacement value is required.")
    if minimum_teams < 2:
        raise ValueError("At least two teams are required for a correlation.")
    minutes = (
        player_minutes.groupby(
            ["PLAYER_ID", "Season", "team_id"], as_index=False
        )["minutes"]
        .sum()
        .query("minutes > 0")
    )
    metric_minutes = minutes.groupby(["PLAYER_ID", "Season"], as_index=False)[
        "minutes"
    ].sum().rename(columns={"minutes": "metric_year_minutes"})
    outcomes = _team_wins(team_games)
    fold_rows: list[dict] = []
    team_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    for rating_season in rating_seasons:
        next_season = int(rating_season) + 1
        next_minutes = minutes.loc[minutes["Season"].eq(next_season)].copy()
        next_outcomes = outcomes.loc[outcomes["Season"].eq(next_season)].copy()
        if next_minutes.empty or next_outcomes.empty:
            raise ValueError(f"Missing next-season inputs for {next_season}.")
        season_ratings = ratings.loc[ratings["Season"].eq(rating_season)].copy()
        for metric, metric_frame in season_ratings.groupby("metric", sort=False):
            label = str(metric_frame["metric_label"].iloc[0])
            allocation = next_minutes.merge(
                metric_frame[["PLAYER_ID", "net"]],
                on="PLAYER_ID",
                how="left",
                validate="many_to_one",
            ).merge(
                metric_minutes.loc[metric_minutes["Season"].eq(rating_season), [
                    "PLAYER_ID",
                    "metric_year_minutes",
                ]],
                on="PLAYER_ID",
                how="left",
                validate="many_to_one",
            )
            allocation["qualified_rating"] = allocation["net"].notna() & allocation[
                "metric_year_minutes"
            ].ge(minimum_metric_minutes)
            total_minutes = float(allocation["minutes"].sum())
            covered_minutes = float(
                allocation.loc[allocation["qualified_rating"], "minutes"].sum()
            )
            coverage_rows.append(
                {
                    "rating_season": int(rating_season),
                    "outcome_season": next_season,
                    "metric": metric,
                    "metric_label": label,
                    "next_season_minutes": total_minutes,
                    "qualified_metric_minutes": covered_minutes,
                    "minute_coverage": covered_minutes / total_minutes,
                    "replacement_minutes": total_minutes - covered_minutes,
                }
            )
            for replacement in replacement_values:
                scored = allocation.copy()
                scored["adjusted_rating"] = scored["net"].where(
                    scored["qualified_rating"], float(replacement)
                )
                teams = scored.groupby("team_id", as_index=False).apply(
                    lambda group: pd.Series(
                        {
                            "team_rating": 5.0
                            * np.average(group["adjusted_rating"], weights=group["minutes"]),
                            "minutes": group["minutes"].sum(),
                            "replacement_minute_share": 1.0
                            - group.loc[group["qualified_rating"], "minutes"].sum()
                            / group["minutes"].sum(),
                        }
                    ),
                    include_groups=False,
                ).reset_index(drop=True)
                teams = teams.merge(
                    next_outcomes,
                    on="team_id",
                    how="inner",
                    validate="one_to_one",
                )
                if len(teams) < minimum_teams:
                    raise ValueError(
                        f"Metric {metric} season {rating_season} matched only {len(teams)} teams."
                    )
                pearson = float(teams["team_rating"].corr(teams["win_pct"]))
                spearman = float(
                    teams["team_rating"].corr(teams["win_pct"], method="spearman")
                )
                fold_rows.append(
                    {
                        "rating_season": int(rating_season),
                        "outcome_season": next_season,
                        "metric": metric,
                        "metric_label": label,
                        "replacement_value": float(replacement),
                        "teams": int(len(teams)),
                        "pearson": pearson,
                        "spearman": spearman,
                        "r_squared": pearson**2,
                        "mean_minute_coverage": covered_minutes / total_minutes,
                    }
                )
                teams["rating_season"] = int(rating_season)
                teams["outcome_season"] = next_season
                teams["metric"] = metric
                teams["metric_label"] = label
                teams["replacement_value"] = float(replacement)
                team_rows.append(teams)
    folds = pd.DataFrame(fold_rows)
    team_predictions = pd.concat(team_rows, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    summaries: list[dict] = []
    for (metric, label, replacement), group in folds.groupby(
        ["metric", "metric_label", "replacement_value"], sort=False
    ):
        predictions = team_predictions.loc[
            team_predictions["metric"].eq(metric)
            & team_predictions["replacement_value"].eq(replacement)
        ]
        pooled_pearson = float(
            predictions["team_rating"].corr(predictions["win_pct"])
        )
        pooled_spearman = float(
            predictions["team_rating"].corr(
                predictions["win_pct"], method="spearman"
            )
        )
        summaries.append(
            {
                "metric": metric,
                "metric_label": label,
                "replacement_value": float(replacement),
                "folds": int(group["rating_season"].nunique()),
                "team_seasons": int(len(predictions)),
                "mean_pearson": float(group["pearson"].mean()),
                "mean_spearman": float(group["spearman"].mean()),
                "mean_r_squared": float(group["r_squared"].mean()),
                "pooled_pearson": pooled_pearson,
                "pooled_spearman": pooled_spearman,
                "pooled_r_squared": pooled_pearson**2,
                "minimum_minute_coverage": float(group["mean_minute_coverage"].min()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["replacement_value", "mean_r_squared"], ascending=[True, False]
    )
    return folds, summary, coverage


def build_public_aio_benchmark(
    ratings: pd.DataFrame,
    player_minutes: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    artifact_root: str | Path,
    definitions: list[dict],
    source_files: dict[str, str | Path],
    common_seasons: tuple[int, ...] = (2021, 2022, 2023, 2024),
    minimum_minutes: float = 250.0,
    replacement_values: tuple[float, ...] = (-3.0, -2.5, -2.0, -1.5),
    projected_minutes_path: str | Path | None = None,
) -> dict:
    """Build a content-addressed public AIO comparison artifact."""
    panel = validate_rating_panel(ratings)
    pairwise = build_pairwise_correlations(
        panel,
        player_minutes,
        seasons=common_seasons,
        minimum_minutes=minimum_minutes,
    )
    folds, summary, coverage = build_team_win_benchmark(
        panel,
        player_minutes,
        team_games,
        rating_seasons=common_seasons,
        minimum_metric_minutes=minimum_minutes,
        replacement_values=replacement_values,
    )
    source_hashes = {
        name: sha256_file(path) for name, path in sorted(source_files.items())
    }
    config = {
        "common_seasons": list(common_seasons),
        "minimum_metric_year_minutes": minimum_minutes,
        "replacement_values": list(replacement_values),
        "default_replacement_value": -2.0,
        "team_rating_formula": "5 * weighted_average(metric_Y, player_minutes_Y_plus_1)",
        "source_hashes": source_hashes,
        "builder_sha256": sha256_file(Path(__file__)),
        "projected_minutes": {
            "status": "available" if projected_minutes_path else "unavailable",
            "path_sha256": sha256_file(projected_minutes_path)
            if projected_minutes_path
            else None,
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = (
        Path(artifact_root)
        / "research"
        / "public_aio_benchmark"
        / f"public_aio_benchmark_v1_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    pairwise.to_parquet(output / "pairwise_correlations.parquet", index=False)
    folds.to_parquet(output / "team_win_folds.parquet", index=False)
    summary.to_parquet(output / "team_win_summary.parquet", index=False)
    coverage.to_parquet(output / "coverage.parquet", index=False)
    pd.DataFrame(definitions).to_parquet(output / "metric_definitions.parquet", index=False)
    run = {
        "run_id": output.name,
        "model_family": "public_all_in_one_benchmark",
        "status": "research_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rating_rows": int(len(panel)),
            "metrics": sorted(panel["metric"].unique().tolist()),
            "common_seasons": list(common_seasons),
            "pairwise_rows": int(len(pairwise)),
            "team_prediction_rows": int(
                summary.loc[summary["replacement_value"].eq(-2.0), "team_seasons"].sum()
            ),
            "duplicate_rating_keys": 0,
        },
        "projected_minutes_status": (
            "not_run_no_archived_projection_file"
            if projected_minutes_path is None
            else "available_not_implemented_in_v1"
        ),
        "caveats": [
            "The next-season team-win benchmark uses observed next-season minutes and is therefore an oracle-minutes retrodiction, not a preseason forecast.",
            "Metric agreement is not evidence that either metric is correct.",
            "EPM was not scored because a complete historical export was not available.",
            "BoxPIPM-style is a transparent box-score-only baseline, not a full PIPM reproduction.",
        ],
        "paths": {
            "pairwise_correlations": "pairwise_correlations.parquet",
            "team_win_folds": "team_win_folds.parquet",
            "team_win_summary": "team_win_summary.parquet",
            "coverage": "coverage.parquet",
            "metric_definitions": "metric_definitions.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    return run
