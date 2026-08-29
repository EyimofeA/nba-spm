"""Build the localhost-only SPM feature-research payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_impact.models.public_aio_benchmark import (
    build_pairwise_correlations,
    load_epm_ratings,
    load_lebron_ratings,
    load_mamba_ratings,
    load_site_aio_ratings,
    map_named_metric,
    validate_rating_panel,
)


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "web/local-data/spm-lab.json"
BOX15_RUN = (
    ROOT
    / "artifacts/research/final_box_feature_ladder"
    / "final_box_feature_ladder_v1_8bb26f12e7"
)
FULL_SPM_AIO = (
    ROOT
    / "artifacts/models/five_year_spm_feature_research"
    / "five_year_spm_feature_research_v1_93c148510e/aio_ratings.parquet"
)
ANNUAL_BOX_PIPM = (
    ROOT
    / "artifacts/models/box_pipm_style"
    / "box_pipm_style_v1_1768252352/oof_predictions.parquet"
)
EXTERNAL_ANNUAL = (
    ROOT
    / "artifacts/models/external_impact_benchmark"
    / "external_impact_benchmark_v1_bab43a4087/external_annual.parquet"
)
HISTORICAL_PLAYER_GAMES = ROOT / "data/lake/silver/historical_espn_player_games.parquet"
RECENT_PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
DOWNLOADS = Path.home() / "Downloads"
EPM_SOURCE = DOWNLOADS / "EPM_All_Seasons.csv"
LEBRON_SOURCE = DOWNLOADS / (
    "lebron-data-2026-2025-2024-2023-2022-2021-2020-2019-2018-2017-"
    "2016-2015-2014-2013-2012-2011-2010.csv"
)
MAMBA_SOURCE = DOWNLOADS / "MAMBAVALUES.xlsx - Sheet1.csv"
COMPARISON_SEASONS = (2021, 2022, 2023, 2024)


def latest_complete_run() -> Path:
    runs = sorted(
        (ROOT / "artifacts/models/five_year_spm_feature_research").glob(
            "five_year_spm_feature_research_v1_*/run.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    complete = [path.parent for path in runs if (path.parent / "aio_metrics.parquet").exists()]
    if not complete:
        raise FileNotFoundError("No complete five-year SPM feature-research run exists.")
    return complete[-1]


def latest_comparison_run() -> Path:
    runs = sorted(
        (ROOT / "artifacts/research/public_aio_benchmark").glob(
            "public_aio_benchmark_v1_*/run.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    required = (
        "pairwise_correlations.parquet",
        "team_win_folds.parquet",
        "team_win_summary.parquet",
        "coverage.parquet",
        "metric_definitions.parquet",
    )
    complete = [
        path.parent
        for path in runs
        if all((path.parent / name).exists() for name in required)
    ]
    if not complete:
        raise FileNotFoundError("No complete public AIO benchmark exists.")
    return complete[-1]


def latest_weight_ablation_run() -> Path:
    runs = sorted(
        (ROOT / "artifacts/research/spm_weight_ablation").glob(
            "spm_weight_ablation_v1_*/run.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    complete = [
        path.parent
        for path in runs
        if (path.parent / "summary.parquet").exists()
        and (path.parent / "feature_catalog.parquet").exists()
    ]
    if not complete:
        raise FileNotFoundError("No complete SPM sample-weight ablation exists.")
    return complete[-1]


def clean(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def player_minutes() -> pd.DataFrame:
    frames = []
    for path in (HISTORICAL_PLAYER_GAMES, RECENT_PLAYER_GAMES):
        frame = pd.read_parquet(path)
        frame = frame.loc[
            frame["season_type"].eq("regular")
            & frame["minutes_seconds"].fillna(0).gt(0),
            ["player_id", "season_end", "minutes_seconds"],
        ].rename(columns={"player_id": "PLAYER_ID", "season_end": "Season"})
        frame["minutes"] = frame["minutes_seconds"] / 60.0
        frames.append(frame[["PLAYER_ID", "Season", "minutes"]])
    return pd.concat(frames, ignore_index=True)


def site_metric(metric: str, label: str, columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for season in COMPARISON_SEASONS:
        source = pd.read_json(ROOT / f"web/public/data/leaderboard-{season}.json")
        rows.append(
            source.rename(columns=columns)[
                ["PLAYER_ID", "Season", "offense", "defense", "net"]
            ]
        )
    ratings = pd.concat(rows, ignore_index=True)
    ratings["metric"] = metric
    ratings["metric_label"] = label
    ratings["category"] = "CourtSignal"
    return ratings


def box15_payload(run_path: Path) -> dict:
    manifest = json.loads((run_path / "run.json").read_text())
    ratings = pd.read_parquet(run_path / "ratings.parquet")
    ratings = ratings.loc[
        ratings["candidate"].isin(("box_15", "box_15_aio"))
        & ratings["Poss_Off"].gt(0)
        & ratings["Poss_Def"].gt(0)
    ].copy()
    names = {
        int(row["id"]): row["name"]
        for row in json.loads((ROOT / "web/public/data/players.json").read_text())
    }
    ratings["PLAYER_NAME"] = ratings["PLAYER_ID"].map(names)
    if ratings["PLAYER_NAME"].isna().any():
        raise ValueError("Box15 active leaderboard has an unresolved player name.")

    key = ["PLAYER_ID", "rating_season", "PLAYER_NAME"]
    values = ["offense", "defense", "net"]
    prior = ratings.loc[ratings["candidate"].eq("box_15"), key + values].rename(
        columns={value: f"prior_{value}" for value in values}
    )
    posterior = ratings.loc[
        ratings["candidate"].eq("box_15_aio"), key + values
    ].rename(columns={value: f"posterior_{value}" for value in values})
    leaderboard = prior.merge(posterior, on=key, validate="one_to_one")
    leaderboard = leaderboard.rename(columns={"rating_season": "Season"})
    leaderboard["change_net"] = leaderboard["posterior_net"] - leaderboard["prior_net"]
    leaderboard = leaderboard.sort_values(
        ["Season", "posterior_net"], ascending=[True, False], kind="stable"
    )

    box_panel = []
    for candidate, metric, label in (
        ("box_15", "box15_prior", "Box15 prior"),
        ("box_15_aio", "box15_aio", "Box15 AIO"),
    ):
        frame = ratings.loc[
            ratings["candidate"].eq(candidate),
            ["PLAYER_ID", "rating_season", *values],
        ].rename(columns={"rating_season": "Season"})
        frame["metric"] = metric
        frame["metric_label"] = label
        frame["category"] = "CourtSignal"
        box_panel.append(frame)

    site_aio = load_site_aio_ratings(ROOT / "web/public/data", COMPARISON_SEASONS)
    site_aio["metric_label"] = "Website AIO"
    site_aio["category"] = "CourtSignal"
    full_spm = pd.read_parquet(FULL_SPM_AIO)
    full_spm = full_spm.loc[
        full_spm["variant"].eq("selected_combined")
        & full_spm["Poss_Off"].gt(0)
        & full_spm["Poss_Def"].gt(0),
        ["PLAYER_ID", "rating_season", *values],
    ].rename(columns={"rating_season": "Season"})
    full_spm["metric"] = "full_spm_aio"
    full_spm["metric_label"] = "Former full-SPM AIO"
    full_spm["category"] = "CourtSignal"

    annual_box = pd.read_parquet(ANNUAL_BOX_PIPM).rename(
        columns={
            "box_pipm_style_offense": "offense",
            "box_pipm_style_defense": "defense",
            "box_pipm_style_net": "net",
        }
    )[["PLAYER_ID", "Season", *values]]
    annual_box["metric"] = "annual_box_pipm"
    annual_box["metric_label"] = "Annual BoxPIPM-style"
    annual_box["category"] = "CourtSignal"

    epm = load_epm_ratings(EPM_SOURCE)
    lebron, identity = load_lebron_ratings(LEBRON_SOURCE)
    mamba, _ = load_mamba_ratings(MAMBA_SOURCE, identity)
    external = pd.read_parquet(EXTERNAL_ANNUAL)
    bpm, _ = map_named_metric(
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
    xrapm, _ = map_named_metric(
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
    rapm = site_metric(
        "rapm",
        "RAPM",
        {
            "normal_rapm_offense": "offense",
            "normal_rapm_defense": "defense",
            "normal_rapm_net": "net",
        },
    )
    annual_spm = site_metric(
        "annual_spm",
        "Annual SPM",
        {"spm_offense": "offense", "spm_defense": "defense", "spm_net": "net"},
    )
    panel = validate_rating_panel(
        pd.concat(
            [
                *box_panel,
                site_aio,
                full_spm,
                annual_box,
                rapm,
                annual_spm,
                epm,
                lebron,
                mamba,
                bpm,
                xrapm,
            ],
            ignore_index=True,
        )
    )
    pairwise = build_pairwise_correlations(
        panel,
        player_minutes(),
        seasons=COMPARISON_SEASONS,
        minimum_minutes=250.0,
    )
    labels = (
        panel[["metric", "metric_label"]]
        .drop_duplicates()
        .set_index("metric")["metric_label"]
        .to_dict()
    )
    comparators = (
        "site_aio",
        "full_spm_aio",
        "annual_box_pipm",
        "annual_spm",
        "rapm",
        "bpm",
        "xrapm",
        "lebron",
        "mamba",
        "epm",
    )
    correlations = []
    for component in ("net", "offense", "defense"):
        for comparator in comparators:
            prior_row = pairwise.loc[
                pairwise["component"].eq(component)
                & pairwise["left_metric"].eq("box15_prior")
                & pairwise["right_metric"].eq(comparator)
            ].iloc[0]
            posterior_row = pairwise.loc[
                pairwise["component"].eq(component)
                & pairwise["left_metric"].eq("box15_aio")
                & pairwise["right_metric"].eq(comparator)
            ].iloc[0]
            correlations.append(
                {
                    "component": component,
                    "metric": comparator,
                    "metric_label": labels[comparator],
                    "prior_rows": int(prior_row["rows"]),
                    "posterior_rows": int(posterior_row["rows"]),
                    "prior_pearson": float(prior_row["pearson"]),
                    "posterior_pearson": float(posterior_row["pearson"]),
                    "prior_spearman": float(prior_row["spearman"]),
                    "posterior_spearman": float(posterior_row["spearman"]),
                }
            )
    return {
        "run_id": manifest["run_id"],
        "seasons": sorted(leaderboard["Season"].unique().astype(int).tolist()),
        "correlation_seasons": list(COMPARISON_SEASONS),
        "minimum_minutes": 250,
        "leaderboard": clean(leaderboard),
        "correlations": correlations,
    }


def rating_rows(run_path: Path) -> list[dict]:
    spm = pd.read_parquet(run_path / "spm_predictions.parquet")
    aio = pd.read_parquet(run_path / "aio_ratings.parquet")
    active = aio.loc[
        aio["variant"].eq("selected_combined")
        & aio["Poss_Off"].gt(0)
        & aio["Poss_Def"].gt(0),
        ["PLAYER_ID", "rating_season", "PLAYER_NAME"],
    ].rename(columns={"rating_season": "Season"})
    active = active.dropna(subset=["PLAYER_NAME"]).drop_duplicates(
        ["PLAYER_ID", "Season"], keep="last"
    )
    outputs = []
    for metric, frame, columns in (
        (
            "spm",
            spm,
            {
                "prior_offense_per_100": "offense",
                "prior_defense_per_100": "defense",
                "prior_net_per_100": "net",
            },
        ),
        ("aio", aio, {"offense": "offense", "defense": "defense", "net": "net"}),
    ):
        scope = frame.loc[frame["variant"].isin(("baseline", "selected_combined"))].copy()
        scope = scope.rename(columns={"Window_End": "Season", "rating_season": "Season"})
        if "PLAYER_NAME" in scope:
            scope = scope.drop(columns="PLAYER_NAME")
        scope = scope.merge(
            active, on=["PLAYER_ID", "Season"], how="inner", validate="many_to_one"
        )
        selected = scope.loc[scope["variant"].eq("selected_combined"), [
            "PLAYER_ID", "Season", "PLAYER_NAME", *columns
        ]].rename(columns={source: f"selected_{target}" for source, target in columns.items()})
        baseline = scope.loc[scope["variant"].eq("baseline"), [
            "PLAYER_ID", "Season", *columns
        ]].rename(columns={source: f"baseline_{target}" for source, target in columns.items()})
        merged = selected.merge(baseline, on=["PLAYER_ID", "Season"], validate="one_to_one")
        merged["metric"] = metric
        for side in ("offense", "defense", "net"):
            merged[f"delta_{side}"] = merged[f"selected_{side}"] - merged[f"baseline_{side}"]
        outputs.append(merged)
    return clean(pd.concat(outputs, ignore_index=True))


def comparison_payload(run_path: Path) -> dict:
    manifest = json.loads((run_path / "run.json").read_text())
    summary = pd.read_parquet(run_path / "team_win_summary.parquet")
    folds = pd.read_parquet(run_path / "team_win_folds.parquet")
    default_replacement = float(manifest["config"]["default_replacement_value"])
    summary = summary.loc[summary["replacement_value"].eq(default_replacement)].copy()
    summary = summary.sort_values("mean_r_squared", ascending=False)
    folds = folds.loc[folds["replacement_value"].eq(default_replacement)].copy()
    return {
        "run_id": manifest["run_id"],
        "common_seasons": manifest["config"]["common_seasons"],
        "minimum_metric_year_minutes": manifest["config"][
            "minimum_metric_year_minutes"
        ],
        "replacement_value": default_replacement,
        "team_rating_formula": manifest["config"]["team_rating_formula"],
        "minutes_mode": "observed_next_season",
        "projected_minutes_status": manifest["projected_minutes_status"],
        "team_win_summary": clean(summary),
        "team_win_folds": clean(folds.sort_values(["rating_season", "metric_label"])),
        "pairwise_correlations": clean(
            pd.read_parquet(run_path / "pairwise_correlations.parquet")
        ),
        "coverage": clean(pd.read_parquet(run_path / "coverage.parquet")),
        "definitions": clean(pd.read_parquet(run_path / "metric_definitions.parquet")),
        "caveats": manifest["caveats"],
    }


def weighting_payload(run_path: Path) -> dict:
    manifest = json.loads((run_path / "run.json").read_text())
    return {
        "run_id": manifest["run_id"],
        "summary": clean(pd.read_parquet(run_path / "summary.parquet")),
        "fold_metrics": clean(pd.read_parquet(run_path / "fold_metrics.parquet")),
        "feature_catalog": clean(pd.read_parquet(run_path / "feature_catalog.parquet")),
        "quality": manifest["quality"],
        "caveats": manifest["caveats"],
    }


def build(
    run_path: Path,
    comparison_run_path: Path | None = None,
    weight_run_path: Path | None = None,
    box15_run_path: Path = BOX15_RUN,
) -> dict:
    manifest = json.loads((run_path / "run.json").read_text())
    decisions = pd.read_parquet(run_path / "feature_group_decisions.parquet")
    aio = pd.read_parquet(run_path / "aio_metrics.parquet")
    wide = aio.pivot(index="test_season", columns="variant", values=["margin_rmse", "margin_correlation"])
    validation = pd.DataFrame(
        {
            "test_season": wide.index.astype(int),
            "baseline_rmse": wide[("margin_rmse", "baseline")].to_numpy(),
            "selected_rmse": wide[("margin_rmse", "selected_combined")].to_numpy(),
            "rmse_delta": (
                wide[("margin_rmse", "selected_combined")]
                - wide[("margin_rmse", "baseline")]
            ).to_numpy(),
            "baseline_correlation": wide[("margin_correlation", "baseline")].to_numpy(),
            "selected_correlation": wide[("margin_correlation", "selected_combined")].to_numpy(),
        }
    )
    payload = {
        "run_id": manifest["run_id"],
        "scope": "localhost_only",
        "seasons": [2021, 2022, 2023, 2024, 2025, 2026],
        "stabilization": manifest["stabilization_contract"],
        "selection_gate": manifest["selection_gate"],
        "decisions": clean(decisions),
        "validation": clean(validation),
        "ratings": rating_rows(run_path),
        "box15": box15_payload(box15_run_path),
        "comparison": comparison_payload(
            comparison_run_path or latest_comparison_run()
        ),
        "weighting": weighting_payload(weight_run_path or latest_weight_ablation_run()),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--weight-run", type=Path)
    parser.add_argument("--box15-run", type=Path, default=BOX15_RUN)
    args = parser.parse_args()
    payload = build(
        args.run or latest_complete_run(),
        args.comparison_run,
        args.weight_run,
        args.box15_run,
    )
    print(json.dumps({"run_id": payload["run_id"], "rows": len(payload["ratings"])}, indent=2))


if __name__ == "__main__":
    main()
