"""Build the localhost-only RAPM Lab payload from saved research runs."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "research" / "rapm_lab" / "outputs"
DESTINATION = REPO_ROOT / "web" / "local-data" / "rapm-lab.json"
PIPM_REFERENCE = (
    REPO_ROOT
    / "artifacts/research/pipm_reference_comparison"
    / "pipm_reference_comparison_v1_49a3c2c973"
)
BPM_REFERENCE = (
    REPO_ROOT
    / "artifacts/models/external_impact_benchmark"
    / "external_impact_benchmark_v1_bab43a4087"
    / "external_annual.parquet"
)


def latest_run(group: str, prefix: str) -> tuple[dict, Path]:
    paths = sorted(
        (OUTPUTS / group).glob(f"{prefix}*/run.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise FileNotFoundError(f"No saved RAPM Lab run for {group}/{prefix}")
    path = paths[-1]
    return json.loads(path.read_text()), path.parent


def pinned_run(group: str, run_id: str) -> tuple[dict, Path]:
    path = OUTPUTS / group / run_id / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing pinned RAPM Lab run {group}/{run_id}")
    return json.loads(path.read_text()), path.parent


def experiment(
    experiment_id: str,
    title: str,
    test: str,
    result: str,
    decision: str,
    status: str,
    run_id: str,
) -> dict:
    return {
        "id": experiment_id,
        "title": title,
        "test": test,
        "result": result,
        "decision": decision,
        "status": status,
        "run_id": run_id,
    }


def clean_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def leaderboard(
    board_id: str,
    experiment_id: str,
    title: str,
    columns: list[tuple[str, str]],
    rows: pd.DataFrame,
) -> dict:
    return {
        "id": board_id,
        "experiment_id": experiment_id,
        "title": title,
        "columns": [{"key": key, "label": label} for key, label in columns],
        "rows": clean_records(rows[[key for key, _ in columns]]),
    }


def unit_name(value: str, names: dict[int, str]) -> str:
    return " + ".join(names.get(int(player), str(player)) for player in value.split("|"))


def only_metric(
    metrics: pd.DataFrame, filters: dict[str, str], description: str
) -> pd.Series:
    rows = metrics
    for column, value in filters.items():
        rows = rows.loc[rows[column].eq(value)]
    if len(rows) != 1:
        raise ValueError(f"Expected one metric row for {description}")
    return rows.iloc[0]


def coefficient_of_determination(
    reference: pd.Series, reconstruction: pd.Series
) -> float:
    complete = pd.DataFrame(
        {"reference": reference, "reconstruction": reconstruction}
    ).dropna()
    residual = float(((complete["reference"] - complete["reconstruction"]) ** 2).sum())
    total = float(((complete["reference"] - complete["reference"].mean()) ** 2).sum())
    return 1.0 - residual / total


def boards_for(
    frame: pd.DataFrame,
    *,
    metric: str,
    slug: str,
    columns: list[dict],
    sort_key: str = "net",
    minimum: float = 250,
    minimum_key: str | None = None,
) -> list[dict]:
    boards = []
    numeric_keys = [column["key"] for column in columns]
    for season in sorted(frame["season"].dropna().astype(int).unique(), reverse=True):
        rows = frame.loc[frame["season"].eq(season)].copy()
        filter_key = minimum_key
        if filter_key is None:
            if "minutes" in rows and rows["minutes"].notna().any():
                filter_key = "minutes"
            elif "exposure" in rows and rows["exposure"].notna().any():
                filter_key = "exposure"
        if filter_key and filter_key in rows:
            rows = rows.loc[rows[filter_key].fillna(0).ge(minimum)]
        keep = [
            key for key in numeric_keys
            if key in rows and (key in {"player", "team"} or rows[key].notna().any())
        ]
        rows = rows.loc[rows["player"].notna(), keep].sort_values(sort_key, ascending=False)
        boards.append({
            "id": f"{slug}-{season}",
            "title": f"{metric} · {season}",
            "season": int(season),
            "metric": metric,
            "source": "CourtSignal reconstruction",
            "columns": [column for column in columns if column["key"] in keep],
            "rows": clean_records(rows),
        })
    return boards


def latest_repaired_wp(default: Path) -> Path:
    repaired = sorted(
        (OUTPUTS / "rolling_5y_wp_rapm").glob("*/ratings_repaired.parquet")
    )
    return repaired[-1] if repaired else default


def read_optional_parquet(path: Path, fallback: pd.DataFrame) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return fallback


def build_payload() -> dict:
    aging, _ = latest_run("aging_resolution", "aging_resolution_v1_")
    interactions, interactions_path = latest_run(
        "lineup_interactions", "lineup_interactions_v1_"
    )
    units, units_path = latest_run(
        "standalone_unit_rapm", "standalone_unit_rapm_v1_"
    )
    outcomes, outcomes_path = latest_run(
        "possession_outcomes", "possession_outcome_rapm_v1_"
    )
    wp, wp_path = latest_run("win_probability_rapm", "win_probability_rapm_v1_")
    points, points_path = latest_run("points_channel_rapm", "points_channel_rapm_v1_")
    rubberband, rubberband_path = latest_run("rubberband", "rubberband_adjustment_v1_")
    adjusted, adjusted_path = latest_run(
        "rubberband_progress_rapm", "rubberband_progress_rapm_v2_"
    )
    je, je_path = latest_run("rubberband_je", "rubberband_je_replication_v1_")
    joint_clock, joint_clock_path = latest_run(
        "rubberband_joint_clock", "rubberband_joint_clock_v1_"
    )
    score_signal, score_signal_path = pinned_run(
        "rubberband_score_signal", "rubberband_score_signal_v1_deac872ede"
    )
    age_score, age_score_path = pinned_run(
        "age_score_context", "age_score_context_v1_7e8689fee8"
    )
    age_adjusted, age_adjusted_path = latest_run(
        "age_adjusted_rapm", "age_adjusted_full_1997_2026_v1_"
    )
    factor_reconstruction, factor_reconstruction_path = latest_run(
        "factor_reconstruction", "factor_rapm_reconstruction_ts_v2_"
    )
    rubberband_5pt, rubberband_5pt_path = latest_run(
        "rubberband_5pt_lambda", "rubberband_5pt_lambda_v1_"
    )
    rubberband_5pt_curve, _ = latest_run(
        "rubberband_5pt_curve", "rubberband_5pt_curve_v1_"
    )
    wp_lambda, _ = latest_run("wp_rapm_lambda", "wp_rapm_lambda_v1_")
    rolling_wp, rolling_wp_path = latest_run(
        "rolling_5y_wp_rapm", "rolling_5y_wp_rapm_v1_"
    )
    wp_spm_aio, wp_spm_aio_path = latest_run("wp_spm_aio", "wp_spm_aio_v1_")
    ts_factors, ts_factors_path = latest_run(
        "ts_factor_rapm", "ts_factor_rapm_v1_"
    )
    full_coach, full_coach_path = latest_run(
        "full_coach_age_rapm", "coach_age_full_1997_2026_v1_"
    )
    pair_buckets, pair_buckets_path = latest_run(
        "pair_exposure_bucketing", "pair_exposure_bucketing_v1_"
    )
    production_5y, production_5y_path = latest_run(
        "production_5y_rapm_intervals", "production_5y_rapm_intervals_v1_"
    )
    horizons, horizons_path = latest_run(
        "rapm_target_horizon_bakeoff", "rapm_target_horizon_bakeoff_v1_"
    )
    shooting_luck, shooting_luck_path = latest_run(
        "luck_teammate_shooting_rapm", "luck_teammate_shooting_rapm_v1_"
    )
    ryan_davis, ryan_davis_path = latest_run(
        "ryan_davis_comparison", "ryan_davis_comparison_v1_"
    )
    teammate_channels, teammate_channels_path = latest_run(
        "teammate_play_channels", "teammate_play_channels_v1_"
    )
    time_age, time_age_path = latest_run(
        "time_decay_actual_age_5y_rapm", "time_decay_actual_age_5y_rapm_v1_"
    )
    external, external_path = latest_run(
        "external_reproduction_benchmark", "external_reproduction_benchmark_v1_"
    )
    single_lambda, single_lambda_path = latest_run(
        "single_season_defense_lambda", "single_season_defense_lambda_v1_"
    )
    single_sweep, _ = latest_run(
        "single_season_rapm_sweep", "single_season_rapm_sweep_v1_"
    )
    horizon_age, _ = latest_run(
        "rapm_horizon_age_sweep", "rapm_horizon_age_sweep_v1_"
    )
    single_lambda_summary = {
        row["variant"]: row for row in single_lambda["summary"]
    }
    external_metrics = pd.read_parquet(external_path / "comparison_metrics.parquet")
    external_matched = pd.read_parquet(external_path / "matched_rows.parquet")
    public_reproduction, public_reproduction_path = latest_run(
        "wowy_raptor_reproduction", "wowy_raptor_reproduction_v1_"
    )
    raptor, raptor_path = latest_run(
        "raptor_reconstruction", "raptor_reconstruction_v1_"
    )
    pipm, pipm_path = latest_run(
        "pipm_reconstruction", "pipm_reconstruction_v1_"
    )
    public_metrics = pd.DataFrame(public_reproduction["metrics"])
    raptor_metrics = pd.DataFrame(raptor["metrics"])
    pipm_metrics = pd.DataFrame(pipm["metrics"])
    raptor_net = raptor_metrics.loc[
        raptor_metrics["family"].eq("raptor")
        & raptor_metrics["component"].eq("net")
    ].iloc[0]
    pipm_net = pipm_metrics.loc[pipm_metrics["component"].eq("net")].iloc[0]
    darko_matches = pd.read_parquet(public_reproduction_path / "darko_matches.parquet")
    darko_reconstruction_path = public_reproduction_path / "darko_reconstructions.parquet"
    darko_reconstructions = read_optional_parquet(
        darko_reconstruction_path, darko_matches
    )
    raptor_ratings = pd.read_parquet(raptor_path / "reconstructions.parquet")

    darko_net_r2 = coefficient_of_determination(
        darko_matches["reference_net"], darko_matches["reproduced_net"]
    )
    replications = [
        {
            "metric": "CourtSignal DARKO WOWY reconstruction",
            "build": "Season average from public player-game files",
            "status": "exact_reconstruction",
            "matched_rows": int(only_metric(public_metrics, {"source": "DARKO WOWY", "component": "net"}, "DARKO WOWY/net")["matched_rows"]),
            "pearson": float(only_metric(public_metrics, {"source": "DARKO WOWY", "component": "net"}, "DARKO WOWY/net")["pearson"]),
            "r_squared": darko_net_r2,
            "maximum_absolute_error": float(
                only_metric(public_metrics, {"source": "DARKO WOWY", "component": "net"}, "DARKO WOWY/net")["maximum_absolute_error"]
            ),
            "decision": "Exact reconstruction of the published player-game aggregation. This does not reproduce DARKO's private model.",
            "run_id": public_reproduction["run_id"],
        },
        {
            "metric": "CourtSignal RAPTOR reconstruction",
            "build": "Fitted box component plus reconstructed RAPTOR on/off",
            "status": raptor["status"],
            "matched_rows": int(raptor_net["matched_rows"]),
            "pearson": float(raptor_net["pearson"]),
            "r_squared": float(raptor_net["r_squared"]),
            "maximum_absolute_error": None,
            "decision": "Methodology-aligned reconstruction. FiveThirtyEight did not publish the fitted box or on/off coefficients, so exact identity is not claimed.",
            "run_id": raptor["run_id"],
        },
        {
            "metric": "CourtSignal PIPM reconstruction",
            "build": "Published PIPM box and blend coefficients with CourtSignal raw on/off",
            "status": pipm["status"],
            "matched_rows": int(pipm_net["matched_rows"]),
            "pearson": float(pipm_net["pearson"]),
            "r_squared": float(pipm_net["r_squared"]),
            "maximum_absolute_error": None,
            "decision": "Methodology-aligned regular-season reconstruction. The source includes playoffs and uses a private luck adjustment, so exact identity is not claimed.",
            "run_id": pipm["run_id"],
        },
    ]
    darko_ratings = darko_reconstructions.rename(columns={
        "player_name": "player",
        "reproduced_offense": "offense",
        "reproduced_defense": "defense",
        "reproduced_net": "net",
    })
    raptor_ratings = raptor_ratings.rename(columns={
        "TEAM_ABBREVIATION": "team",
        "raptor_offense": "offense",
        "raptor_defense": "defense",
        "raptor_net": "net",
    })
    raptor_ratings["exposure"] = raptor_ratings[["OffPoss", "DefPoss"]].min(axis=1)
    pipm_ratings = pd.read_parquet(pipm_path / "reconstructions.parquet").rename(columns={
        "PLAYER_NAME": "player",
        "TEAM_ABBREVIATION": "team",
        "MIN": "minutes",
        "pipm_offense": "offense",
        "pipm_defense": "defense",
        "pipm_net": "net",
    })

    standard_columns = [
        {"key": "player", "label": "Player"},
        {"key": "team", "label": "Team"},
        {"key": "offense", "label": "Off"},
        {"key": "defense", "label": "Def"},
        {"key": "net", "label": "Net"},
        {"key": "minutes", "label": "Minutes"},
        {"key": "exposure", "label": "Exposure"},
    ]
    replication_leaderboards = []
    replication_leaderboards += boards_for(
        darko_ratings,
        metric="CourtSignal DARKO WOWY reconstruction",
        slug="darko-wowy-reconstruction",
        columns=standard_columns,
        minimum=10,
        minimum_key="reproduced_games",
    )
    replication_leaderboards += boards_for(
        raptor_ratings,
        metric="CourtSignal RAPTOR reconstruction",
        slug="raptor-reconstruction",
        columns=[
            {"key": "player", "label": "Player"},
            {"key": "team", "label": "Team"},
            {"key": "box_offense", "label": "Box Off"},
            {"key": "box_defense", "label": "Box Def"},
            {"key": "box_net", "label": "Box"},
            {"key": "onoff_offense", "label": "On/Off Off"},
            {"key": "onoff_defense", "label": "On/Off Def"},
            {"key": "onoff_net", "label": "On/Off"},
            {"key": "offense", "label": "RAPTOR Off"},
            {"key": "defense", "label": "RAPTOR Def"},
            {"key": "net", "label": "RAPTOR"},
            {"key": "exposure", "label": "Poss"},
        ],
        minimum=1000,
        minimum_key="exposure",
    )
    replication_leaderboards += boards_for(
        pipm_ratings,
        metric="CourtSignal PIPM reconstruction",
        slug="pipm-reconstruction",
        columns=standard_columns,
    )

    def external_metric(
        source: str, comparison: str, scope: str, component: str
    ) -> pd.Series:
        return only_metric(
            external_metrics,
            {
                "source": source,
                "comparison": comparison,
                "scope": scope,
                "component": component,
            },
            f"{source}/{comparison}/{scope}/{component}",
        )

    multinomial = next(
        row
        for row in outcomes["multinomial_metrics"]
        if row["stage"] == "diagnostic" and row["model"] == "multinomial"
    )
    linear = next(
        row
        for row in outcomes["multinomial_metrics"]
        if row["stage"] == "diagnostic" and row["model"] == "linear_points_rapm"
    )
    experiments = [
        experiment(
            "wp-spm-aio",
            "WP-RAPM statistical priors",
            "Train Box15 and rich statistical models on past rolling five-year progress-WP RAPM, update each with one season of WP possessions, then score the same next-season games.",
            f"Zero-centered WP-RAPM had {next(row['mean_rmse'] for row in wp_spm_aio['summary'] if row['candidate'] == 'zero_wp_rapm'):.3f} RMSE. Box15 WP-AIO had {next(row['mean_rmse'] for row in wp_spm_aio['summary'] if row['candidate'] == 'box15_aio'):.3f}; rich WP-AIO had {next(row['mean_rmse'] for row in wp_spm_aio['summary'] if row['candidate'] == 'rich_aio'):.3f}.",
            "Reject both priors. Each raised paired next-season game MSE on all five folds. Keep zero-centered WP-RAPM as the research control.",
            "lost",
            wp_spm_aio["run_id"],
        ),
        experiment(
            "single-season-rapm-sweep",
            "Single-season penalty and score-control sweep",
            "Select 20 offense/defense penalty pairs and five score-state choices on 2015 to 2022. Diagnose the frozen choices on the same 2023 to 2026 games.",
            f"The early folds selected 3000/{single_sweep['selected_lambda']['lambda_def']:.0f}, but that pair lost to 3000/3000 later. The score-state winner was {single_sweep['selected_score_control']}. Historical calibration reduced the selected model's predicted-margin spread and later MSE.",
            "Keep 3000/4500/300 as the robust single-season research setting. Do not increase defense shrinkage to 6000. Do not add score buckets.",
            "built",
            single_sweep["run_id"],
        ),
        experiment(
            "rapm-horizon-age-sweep",
            "One- to ten-year RAPM and age context",
            "Score every plain horizon on identical 2024 to 2026 games. Add fixed, jointly estimated lineup-age terms to the five- through ten-year variants.",
            f"Plain RAPM was best at 4 years ({next(row['mean_margin_rmse'] for row in horizon_age['summary'] if row['model'] == 'normal' and row['horizon'] == '4y'):.3f} RMSE). Age-conditional RAPM was best at 7 years ({next(row['mean_margin_rmse'] for row in horizon_age['summary'] if row['model'] == 'age_conditional' and row['horizon'] == '7y'):.3f}).",
            "Use age context only for predictive research. Keep neutral retrospective RAPM separate from ratings that know each future lineup's age.",
            "won",
            horizon_age["run_id"],
        ),
        experiment(
            "single-season-defense-lambda",
            "Single-season defense shrinkage",
            "Fit one-season RAPM at 3000/3000 and 3000/4500. Score both on the same next-season games across 12 folds from 2015 to 2026.",
            f"Mean RMSE fell from {single_lambda_summary['reference']['mean_margin_rmse']:.3f} to {single_lambda_summary['defense_4500']['mean_margin_rmse']:.3f}. Reference-minus-candidate MSE was {single_lambda['paired_bootstrap']['reference_minus_candidate_mse']:.3f}, with a 95% interval from {single_lambda['paired_bootstrap']['lower_95']:.3f} to {single_lambda['paired_bootstrap']['upper_95']:.3f}.",
            "Use 3000/4500/300 as the single-season research successor. Rebuild dependent AIO artifacts before public promotion.",
            "won",
            single_lambda["run_id"],
        ),
        experiment(
            "five-season-defense-lambda",
            "Five-season defense shrinkage",
            "Compare 3000/3000 and 3000/4500 on the stored five-season RAPM matrices. Use five earlier selection seasons and three later diagnostics.",
            "The split penalty slightly lowered selection RMSE but reduced correlation. On 2024 to 2026, RMSE rose from 14.936 to 14.950 and correlation fell from 0.366 to 0.361.",
            "Keep 3000/3000/300 for five-season RAPM.",
            "lost",
            "lambda_grid_v1_ef9f6a7a5f",
        ),
        experiment(
            "age-score-context",
            "Age and score controls",
            "Keep possession points unchanged. Fit smooth lineup-age controls and ten signed score-margin buckets jointly beside player and home columns. Select settings on 2025, then compare normal, age only, score only, and age plus score on the same reused 2026 games.",
            f"Player-only RMSE changes were age {age_score['diagnostic']['paired_game_bootstrap']['player_only']['age_only']['observed_rmse_delta']:+.3f}, score {age_score['diagnostic']['paired_game_bootstrap']['player_only']['score_only']['observed_rmse_delta']:+.3f}, and joint {age_score['diagnostic']['paired_game_bootstrap']['player_only']['age_plus_score']['observed_rmse_delta']:+.3f}. Keeping known age at prediction time changed age-only RMSE by {age_score['diagnostic']['paired_game_bootstrap']['pregame_context']['age_only']['observed_rmse_delta']:+.3f}.",
            "Do not promote. Every player-only rating loses; known age is neutral for pregame prediction; score buckets also lose.",
            "lost",
            age_score["run_id"],
        ),
        experiment(
            "same-age-rapm",
            "Full-span same-age RAPM",
            "Fit offense and defense age indicators jointly with 1997-2025 player RAPM. Select age shrinkage on 2025 and check ordinary, age-27, and actual-age predictions on the same reused 2026 games.",
            f"Age-27 player-only RMSE changed by {age_adjusted['bootstrap_vs_normal']['same_age_27']['observed_rmse_delta']:+.3f}. Adding actual lineup ages changed it by {age_adjusted['bootstrap_vs_normal']['age_conditional']['observed_rmse_delta']:+.3f}.",
            "Keep the conditional age control for research. The age-27 leaderboard is not a predictive upgrade.",
            "built",
            age_adjusted["run_id"],
        ),
        experiment(
            "rubberband-5pt",
            "Five-point score buckets",
            "Estimate 5-point lead buckets on 2014-24, select the score-control and offense/defense penalties on 2025, then compare the frozen winner with 3000/3000 RAPM on the same 2026 games.",
            f"The selected candidate changed 2026 RMSE by {rubberband_5pt['diagnostic']['rmse_delta_candidate_minus_baseline']:+.3f}; its paired 95% interval was {rubberband_5pt['diagnostic']['paired_game_bootstrap']['lower_95']:+.3f} to {rubberband_5pt['diagnostic']['paired_game_bootstrap']['upper_95']:+.3f}.",
            "Keep the empirical 5-point curve. Reject score buckets and differential penalties in production points RAPM.",
            "lost",
            rubberband_5pt["run_id"],
        ),
        experiment(
            "factor-reconstruction",
            "Factor RAPM reconstruction",
            "Fit annual true-shooting, turnover, and offensive-rebound RAPMs. Learn the factor-to-total mapping on 2024, select on 2025, and diagnose on qualified 2026 players.",
            f"2026 net correlation {factor_reconstruction['diagnostic_net_comparison'][0]['weighted_correlation']:.3f}, R squared {factor_reconstruction['diagnostic_net_comparison'][0]['weighted_r2']:.3f}, RMSE {factor_reconstruction['diagnostic_net_comparison'][0]['weighted_rmse']:.3f}.",
            "Successful reconstruction, not independent validation: factor and total RAPM share lineups and season outcomes.",
            "won",
            factor_reconstruction["run_id"],
        ),
        experiment(
            "rubberband-je",
            "JE categorical score-state RAPM",
            "Fit exact pre-possession score-margin indicators jointly with player RAPM on 2014-25, then check neutral player ratings on reused 2026 games.",
            f"Neutral 2026 margin RMSE changed by {je['bootstrap_vs_normal']['neutral_player_only']['observed_rmse_delta']:+.3f}; the paired 95% interval was {je['bootstrap_vs_normal']['neutral_player_only']['lower_95']:+.3f} to {je['bootstrap_vs_normal']['neutral_player_only']['upper_95']:+.3f}.",
            "Keep the score-state curve as context. Reject the adjusted player rating.",
            "lost",
            je["run_id"],
        ),
        experiment(
            "rubberband-joint-clock",
            "Joint actual-clock score-state RAPM",
            "Keep possession points unchanged. Fit eight actual six-minute signed-margin columns jointly beside player offense, player defense, and home; select only context shrinkage on 2025 and check neutral player ratings on reused 2026.",
            f"The selected context penalty was {joint_clock['selected_context_penalty']:g}. Neutral 2026 RMSE changed by {joint_clock['diagnostic']['rmse_delta_candidate_minus_baseline']:+.3f} and correlation by {joint_clock['diagnostic']['correlation_delta_candidate_minus_baseline']:+.3f}.",
            "Reject for production. It improves correlation slightly but worsens calibrated margin RMSE, and its paired interval crosses zero.",
            "lost",
            joint_clock["run_id"],
        ),
        experiment(
            "rubberband",
            "Rubber-band adjustment",
            "Five whole-game lineup cross-fit folds per season. Develop on 2024, select shape on 2025, check on reused 2026.",
            f"Six-minute bins with margin clipped at {rubberband['config']['selected_margin_clip']:.0f}. 2026 possession MSE improved 0.028%; all 2,000 game resamples improved. The 10 and 15-point caps were statistically tied.",
            "Retain the descriptive curve. Its adjusted-RAPM test appears below.",
            "estimate",
            rubberband["run_id"],
        ),
        experiment(
            "rubberband-rapm",
            "Rubber-band adjusted RAPM",
            "Fit the clock and fixed 25-possession progress curves on 2024-25. Remove each curve from the player target, then compare neutral player-only ratings with normal RAPM on the same 1,228 reused 2026 games.",
            "Clock-adjusted RMSE changed by +0.018 and correlation by +0.009. Possession-adjusted RMSE changed by +0.026 and correlation by +0.009. Both paired RMSE intervals cross zero.",
            "Keep the descriptive curve. Do not promote either adjusted player rating.",
            "lost",
            adjusted["run_id"],
        ),
        experiment(
            "aging",
            "Age translation",
            "Walk forward from trailing 1, 3, and 5-season annual ratings. Compare 0.1, 0.5, 1, and 2-year smoothing bandwidths.",
            "One year won net RMSE in every history window. Improvements were 0.014, 0.027, and 0.036 points.",
            "Use for forecasts only. Retrospective RAPM stays age-neutral.",
            "won",
            aging["run_id"],
        ),
        experiment(
            "residual-units",
            "Residual unit layers",
            "Fit each residual unit order after player RAPM. Select shrinkage on 2025 and check on reused 2026.",
            "2026 RMSE changes: pair +0.035, trio +0.070, four +0.036, five -0.004.",
            "Reject 2 to 4. Five-player result is too small to promote.",
            "lost",
            interactions["run_id"],
        ),
        experiment(
            "standalone-units",
            "Pair, trio, four-man, and lineup RAPM",
            "Fit raw possession points using only k-player offense units, k-player defense units, and home advantage. Use five training seasons, select shrinkage on 2025, then compare with one-player RAPM on the same 1,228 games in reused 2026.",
            "2026 RMSE changes versus one-player RAPM: pair +0.471, trio +0.780, four-man +1.022, lineup +1.226. Pair coverage was 41.4%; lineup coverage was 3.7%.",
            "Reject as player-impact replacements under the frozen exposure rules. Sparsity worsens sharply with unit size.",
            "lost",
            units["run_id"],
        ),
        experiment(
            "pair-bucketing",
            "Low-sample pair bucketing",
            "Compare a hard 500-possession pair cutoff with exposure-bucket and inverse-exposure shrinkage on 2025. Only advance the selected pair model to 2026; stop higher-order tests unless it beats one-player RAPM.",
            f"Rare-pair retention raised 2025 slot coverage to 48.7% from 43.0% but lost. Hard-floor pair RAPM then changed 2026 RMSE by {pair_buckets['diagnostic']['winner']['rmse_delta_vs_one_player']:+.3f}.",
            "Stop after pairs. More coverage did not improve prediction, so bucketed trio, quartet, and lineup fits are not justified.",
            "lost",
            pair_buckets["run_id"],
        ),
        experiment(
            "production-5y",
            "Rolling five-year RAPM with intervals",
            "Reuse the validated 3000/3000/300 sufficient statistics for every 2014-18 through 2022-26 window. Compute fixed-window homoskedastic analytic ridge sampling intervals and verify point-estimate parity.",
            f"{production_5y['quality']['rating_rows']:,} player-window rows across {production_5y['quality']['windows']} windows; maximum point-estimate mismatch {production_5y['quality']['maximum_reference_rating_error']:.2e}.",
            "Ready for the local production-candidate table. Intervals do not cover game clustering, ridge bias, or selecting a peak window.",
            "built",
            production_5y["run_id"],
        ),
        experiment(
            "target-horizons",
            "RAPM target horizon bake-off",
            "Fit 1-, 3-, 5-, and 6-year RAPM windows, score all four on the same seven next seasons from 2020 to 2026, and build one descriptive full 2014-2026 fit.",
            "Equal-season mean RMSE: 5y 14.140, 6y 14.177, 3y 14.183, 1y 14.376. Five years also had the highest mean margin correlation at 0.368.",
            "Use five-year RAPM as the predictive/stable SPM target challenger. Keep one-year RAPM as the retrospective single-season public estimand.",
            "won",
            horizons["run_id"],
        ),
        experiment(
            "factors",
            "Six-sided factor RAPM",
            "Fit offense and defense for shooting TS, turnovers, and offensive rebounds on exact opportunity rows from 2024 to 2026.",
            f"{ts_factors['quality']['possessions']:,} possessions; {100 * ts_factors['quality']['event_mapping_rate']:.2f}% of relevant events mapped.",
            "Keep as descriptive skill ratings. The factors do not add to total RAPM.",
            "built",
            ts_factors["run_id"],
        ),
        experiment(
            "shooting-luck",
            "FT and 3P luck-adjusted RAPM",
            "Replace realized offensive free throws and threes with leave-current-game-out player-season empirical-Bayes expected makes. Fit on 2024-25 and score actual 2026 game margins against normal RAPM.",
            f"RMSE changed by {shooting_luck['diagnostic']['paired_bootstrap']['observed_rmse_delta']:+.3f}; paired 95% {shooting_luck['diagnostic']['paired_bootstrap']['lower_95']:+.3f} to {shooting_luck['diagnostic']['paired_bootstrap']['upper_95']:+.3f}. Correlation rose from {shooting_luck['diagnostic']['normal']['margin_correlation']:.3f} to {shooting_luck['diagnostic']['luck_adjusted']['margin_correlation']:.3f}.",
            "Keep as a research challenger. The point estimate improves, but the paired RMSE interval crosses zero.",
            "estimate",
            shooting_luck["run_id"],
        ),
        experiment(
            "teammate-efg",
            "Teammate eFG RAPM",
            "On every field-goal attempt, replace the shooter in the offensive lineup with a universal dummy. Fit four-teammate offensive eFG effect and five-defender shot-defense effect on 2024-26.",
            f"{shooting_luck['quality']['teammate_shots']:,} shot rows fitted; {shooting_luck['quality']['source_shots_excluded_missing_shooter']:,} source shots lacked a matching terminal-lineup shooter and were excluded.",
            "Display as descriptive shooting context. It is not causal spacing or shot quality.",
            "built",
            shooting_luck["run_id"],
        ),
        experiment(
            "ryan-davis",
            "Ryan Davis RAPM comparison",
            "Match the public 2019 tutorial ratings to CourtSignal's 2019 terminal-lineup zero-prior RAPM by NBA player ID.",
            f"Across {ryan_davis['matched_players']} players, net Pearson correlation was {ryan_davis['metrics']['net']['pearson_correlation']:.3f} and rank correlation was {ryan_davis['metrics']['net']['spearman_correlation']:.3f}.",
            "The implementations agree strongly on ordering. CourtSignal is less shrunk; this is parser and specification agreement, not predictive validation.",
            "won",
            ryan_davis["run_id"],
        ),
        experiment(
            "external-reproductions",
            "External RAPM and plus-minus checks",
            "Match exact player IDs and windows where available. Reproduce the local AuPM formula and a game-level margin ridge. Report WOWY and on/off only as different-estimand agreement checks.",
            (
                f"Ryan Davis pooled annual net RAPM: r={external_metric('Ryan Davis annual RAPM', 'RAPM', '2014-2023 pooled', 'net')['pearson']:.3f}; "
                f"exact 5y windows: r={external_metric('Ryan Davis multi-year RAPM', '5-year RAPM', 'exact 5-year windows pooled', 'net')['pearson']:.3f}; "
                f"exact 3y windows: r={external_metric('Ryan Davis multi-year RAPM', '3-year RAPM', 'exact 3-year windows pooled', 'net')['pearson']:.3f}; "
                f"current xRAPM 3y: r={external_metric('xRAPM', 'Current three-year RAPM', '2024-2026', 'net')['pearson']:.3f}."
            ),
            "The exact-key checks show strong implementation agreement. Different-estimand correlations are diagnostics, not validation targets.",
            "won",
            external["run_id"],
        ),
        experiment(
            "multinomial",
            "Multinomial RAPM",
            "Fit 0, 1, 2, and 3-plus point probabilities. Select alpha on 2025 and compare with linear points RAPM on reused 2026.",
            f"2026 game RMSE {multinomial['margin_rmse']:.3f} versus {linear['margin_rmse']:.3f} for linear RAPM.",
            "Reject as a predictor. Keep only for outcome-shape research.",
            "lost",
            outcomes["run_id"],
        ),
        experiment(
            "win-probability",
            "Win-probability RAPM",
            "Tune offense and defense penalties on 2025 to 2026, then fit conserved rolling five-year WP credit from 2014 to 2026 with player-neutral surfaces trained only on prior seasons.",
            f"3000 offense / 10000 defense improved 2026 game-total RMSE by {wp_lambda['delta_selected_minus_baseline']['game_total_rmse']:+.4f}. All {len(rolling_wp['windows'])} rolling windows conserve game WP within {rolling_wp['quality']['maximum_conservation_error']:.2e}.",
            "Use the tuned penalties for WP research only. WP credit is leverage attribution, not points impact.",
            "built",
            rolling_wp["run_id"],
        ),
        experiment(
            "coach",
            "Full-span age-adjusted coach RAPM",
            "Fit player, lineup-age, coach, and home effects jointly from 1997 to 2025. Select coach shrinkage on 2025, diagnose on reused 2026, and compare the full fit with xRAPM's coach table.",
            f"Coach terms changed 2026 RMSE by {full_coach['diagnostic']['paired_bootstrap']['observed_rmse_delta']:+.4f}; the paired 95% interval was {full_coach['diagnostic']['paired_bootstrap']['lower_95']:+.4f} to {full_coach['diagnostic']['paired_bootstrap']['upper_95']:+.4f}. Net correlation with xRAPM was {full_coach['external_comparison']['net_correlation']:.3f} across {full_coach['external_comparison']['matched_coaches']} matched coaches.",
            "Reject coach terms for prediction. Keep the leaderboard as descriptive association after player and age controls.",
            "lost",
            full_coach["run_id"],
        ),
        experiment(
            "point-channels",
            "Conserved point channels",
            "Fit one-point, two-point, and three-plus targets with the same design and factorization as RAPM.",
            f"{points['quality']['possession_rows']:,} possessions; summed ratings match RAPM within {points['quality']['maximum_canonical_rapm_error']:.2e} points per 100.",
            "Keep as the additive scoring decomposition.",
            "won",
            points["run_id"],
        ),
        experiment(
            "infinite",
            "Infinite RAPM",
            "Walk a RAPM prior through 2012, 2015, 2018, and 2021, then compare the chained rating with the zero-prior reference.",
            "Depth-three chain correlation fell to 0.5933 and degradation grew with each update.",
            "Reject repeated prior chaining.",
            "lost",
            "research_log_2012_2021_infinite_chain",
        ),
        experiment(
            "learned-buckets",
            "Learned recency buckets",
            "Fit unrestricted age weights, inspect their shape, then compare with simple exponential decay across folds.",
            "The learned weights were non-monotone and failed to transfer. Exponential decay won both folds.",
            "Reject free buckets. Time decay itself remains parked.",
            "lost",
            "research_log_learned_decay_buckets",
        ),
    ]

    experiments.extend(
        [
            experiment(
                "teammate-effects",
                "Teammate outcome RAPM",
                "For each player-opportunity, remove that player's own event and control the other four teammates and five opponents with separate nuisance coefficients. Fit 2024-26 only.",
                f"{teammate_channels['quality']['possessions']:,} possessions. Named assister, stealer, blocker, and rebounder lineup coverage ranged from {100 * min(teammate_channels['quality'][key] for key in ['assist_actor_lineup_coverage', 'steal_actor_lineup_coverage', 'block_actor_lineup_coverage', 'rebound_actor_lineup_coverage']):.1f}% to {100 * max(teammate_channels['quality'][key] for key in ['assist_actor_lineup_coverage', 'steal_actor_lineup_coverage', 'block_actor_lineup_coverage', 'rebound_actor_lineup_coverage']):.1f}%.",
                "Keep as descriptive association. Role and lineup selection can still drive the coefficients.",
                "built",
                teammate_channels["run_id"],
            ),
            experiment(
                "observable-play-channels",
                "Observable play-channel RAPM",
                "Fit rim assists, transition points, three-point points, free-throw points, midrange attempts, rim points, and eight mutually exclusive shot-finish channels on 2024-26 possessions.",
                f"Mapped {100 * teammate_channels['quality']['event_mapping_rate']:.2f}% of eligible source events and attributed {100 * teammate_channels['quality']['scorer_point_attribution_rate']:.2f}% of official points to named scorers.",
                "Publish as finish and outcome context, not Synergy playtype RAPM.",
                "built",
                teammate_channels["run_id"],
            ),
            experiment(
                "time-decay-actual-age",
                "Five-year decay plus actual age",
                "Select half-life, age shrinkage, and shared player shrinkage on 2025. Refit 2021-25 and compare with ordinary five-year RAPM on the same reused 2026 games.",
                f"Selected a {time_age['selected']['half_life_years']:g}-year half-life, age penalty {time_age['selected']['age_penalty']:g}, and player penalty {time_age['selected']['player_penalty']:g}. It improved 2025 RMSE by {-time_age['selection_rmse_delta']:.3f}, then worsened 2026 RMSE by {time_age['diagnostic']['rmse_delta']:+.3f}.",
                "Do not promote. Keep ordinary five-year RAPM as the production reference.",
                "lost",
                time_age["run_id"],
            ),
        ]
    )

    coefficients = pd.read_parquet(rubberband_path / "coefficients.parquet")
    curve = pd.read_parquet(rubberband_path / "curve.parquet")
    comparison_coefficients = pd.read_parquet(
        adjusted_path / "coefficients.parquet"
    )
    adjusted_ratings = pd.read_parquet(adjusted_path / "ratings.parquet")
    adjusted_ratings = adjusted_ratings.loc[
        adjusted_ratings[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(5000)
    ].copy()
    je_curve = pd.read_parquet(je_path / "score_state_curve.parquet")
    je_ratings = pd.read_parquet(je_path / "ratings.parquet")
    je_qualified = je_ratings.loc[
        je_ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(5000)
    ].copy()
    joint_clock_ratings = pd.read_parquet(joint_clock_path / "ratings.parquet")
    joint_clock_qualified = joint_clock_ratings.loc[
        joint_clock_ratings[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ].copy()
    age_score_ratings = pd.read_parquet(age_score_path / "ratings.parquet")
    age_score_qualified = age_score_ratings.loc[
        age_score_ratings[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ].copy()

    points_ratings = pd.read_parquet(points_path / "ratings.parquet")
    all_names = pd.read_parquet(
        REPO_ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a/ratings.parquet",
        columns=["PLAYER_ID", "PLAYER_NAME", "Season"],
    ).sort_values("Season").drop_duplicates("PLAYER_ID", keep="last")
    name_lookup = dict(zip(all_names["PLAYER_ID"].astype(int), all_names["PLAYER_NAME"].astype(str)))
    lab_leaderboards = [
        leaderboard(
            "age-score-context-ratings",
            "age-score-context",
            "Age-27 and score-controlled RAPM, 2020-26",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("age_net", "Age-27 fit"),
                ("score_net", "Score-controlled"),
                ("combined_offense", "Joint offense"),
                ("combined_defense", "Joint defense"),
                ("combined_net", "Joint fit at age 27"),
                ("combined_net_change", "Change"),
            ],
            age_score_qualified.sort_values("combined_net", ascending=False),
        ),
        leaderboard(
            "rubberband-joint-clock-ratings",
            "rubberband-joint-clock",
            "Joint actual-clock score-state RAPM, 2024-26",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("joint_offense", "Offense"),
                ("joint_defense", "Defense"),
                ("joint_net", "Joint net"),
                ("joint_net_change", "Change"),
            ],
            joint_clock_qualified.sort_values("joint_net", ascending=False),
        ),
        leaderboard(
            "same-age-ratings",
            "same-age-rapm",
            "Same-age 27 RAPM, 1997-2026",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("age27_offense", "Offense"),
                ("age27_defense", "Defense"),
                ("age27_net", "Age 27 net"),
                ("age27_net_change", "Change"),
            ],
            pd.read_parquet(age_adjusted_path / "ratings.parquet")
            .loc[
                lambda frame: frame[["off_possessions", "def_possessions"]]
                .min(axis=1)
                .ge(10_000)
            ]
            .sort_values("age27_net", ascending=False),
        ),
        leaderboard(
            "factor-reconstruction-ratings",
            "factor-reconstruction",
            "Factor reconstruction, 2026",
            [
                ("PLAYER_NAME", "Player"),
                ("target_offense", "Actual offense"),
                ("predicted_offense", "Estimated offense"),
                ("target_defense", "Actual defense"),
                ("predicted_defense", "Estimated defense"),
                ("target_net", "Actual net"),
                ("predicted_net", "Estimated net"),
                ("residual_net", "Error"),
            ],
            pd.read_parquet(factor_reconstruction_path / "predictions_2026.parquet")
            .sort_values("predicted_net", ascending=False),
        ),
        leaderboard(
            "rubberband-ratings",
            "rubberband-rapm",
            "Rubber-band adjusted RAPM, 2024-26",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("clock_net", "Clock"),
                ("clock_net_change", "Clock change"),
                ("possession_net", "Possession"),
                ("possession_net_change", "Possession change"),
            ],
            adjusted_ratings.sort_values("clock_net", ascending=False),
        ),
        leaderboard(
            "rubberband-je-ratings",
            "rubberband-je",
            "JE score-state adjusted RAPM, 2014-26",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("je_net", "JE adjusted"),
                ("je_net_change", "Change"),
            ],
            je_qualified.sort_values("je_net", ascending=False),
        ),
        leaderboard(
            "rubberband-5pt-ratings",
            "rubberband-5pt",
            "Differential-penalty candidate, 2014-2026",
            [
                ("player_name", "Player"),
                ("baseline_net", "3000 / 3000"),
                ("candidate_offense", "Offense"),
                ("candidate_defense", "Defense"),
                ("candidate_net", "Candidate"),
                ("candidate_net_change", "Change"),
            ],
            pd.read_parquet(rubberband_5pt_path / "ratings.parquet")
            .loc[
                lambda frame: frame[["off_possessions", "def_possessions"]]
                .min(axis=1)
                .ge(5000)
            ]
            .sort_values("candidate_net", ascending=False),
        ),
    ]

    single_lambda_ratings = pd.read_parquet(
        single_lambda_path / "ratings_2026.parquet"
    )
    lambda_wide = single_lambda_ratings.pivot(
        index=["player_id", "player_name"],
        columns="variant",
        values=["offense_per_100", "defense_per_100", "net_per_100"],
    )
    lambda_wide.columns = [f"{component}_{variant}" for component, variant in lambda_wide.columns]
    lambda_wide = lambda_wide.reset_index()
    lambda_wide["net_change"] = (
        lambda_wide["net_per_100_defense_4500"]
        - lambda_wide["net_per_100_reference"]
    )
    lab_leaderboards.append(
        leaderboard(
            "single-season-defense-lambda-ratings",
            "single-season-defense-lambda",
            "2026 single-season defense-penalty comparison",
            [
                ("player_name", "Player"),
                ("offense_per_100_defense_4500", "Offense"),
                ("defense_per_100_defense_4500", "Defense"),
                ("net_per_100_defense_4500", "Net 4500"),
                ("net_per_100_reference", "Net 3000"),
                ("net_change", "Change"),
            ],
            lambda_wide.sort_values("net_per_100_defense_4500", ascending=False),
        )
    )

    wp_ratings = pd.read_parquet(wp_path / "ratings.parquet")
    wp_ratings["player_name"] = wp_ratings["player_id"].map(name_lookup).fillna(
        wp_ratings["player_id"].astype(str)
    )
    wp_qualified = wp_ratings.loc[
        wp_ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(3000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "wp-ratings",
            "win-probability",
            "Win-probability RAPM",
            [
                ("player_name", "Player"),
                ("offense_wp_percentage_points_per_100", "Offense"),
                ("defense_wp_percentage_points_per_100", "Defense"),
                ("net_wp_percentage_points_per_100", "Net"),
            ],
            wp_qualified.sort_values(
                "net_wp_percentage_points_per_100", ascending=False
            ),
        )
    )
    rolling_wp_ratings = pd.read_parquet(
        latest_repaired_wp(rolling_wp_path / "ratings.parquet")
    )
    rolling_wp_ratings["player_name"] = rolling_wp_ratings["player_id"].map(name_lookup).fillna(
        rolling_wp_ratings.get("player_name")
    )
    rolling_wp_qualified = rolling_wp_ratings.loc[
        rolling_wp_ratings[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "rolling-wp-ratings",
            "win-probability",
            "Rolling five-year WP-RAPM",
            [
                ("player_name", "Player"),
                ("window_start", "From"),
                ("window_end", "To"),
                ("offense_wp_percentage_points_per_100", "Offense"),
                ("defense_wp_percentage_points_per_100", "Defense"),
                ("net_wp_percentage_points_per_100", "Net"),
            ],
            rolling_wp_qualified.sort_values(
                ["window_end", "net_wp_percentage_points_per_100"],
                ascending=[False, False],
            ),
        )
    )

    wp_aio_ratings = pd.read_parquet(wp_spm_aio_path / "leaderboard_2026.parquet")
    wp_aio_ratings["candidate"] = wp_aio_ratings["candidate"].map({
        "zero_wp_rapm": "Zero WP-RAPM",
        "box15_aio": "WP-PULSE",
        "rich_aio": "Rich WP-AIO",
    }).fillna(wp_aio_ratings["candidate"])
    wp_aio_qualified = wp_aio_ratings.loc[
        wp_aio_ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(3000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "wp-spm-aio-2026",
            "wp-spm-aio",
            "WP-RAPM and statistical-prior posteriors · 2026",
            [
                ("player_name", "Player"),
                ("candidate", "Model"),
                ("offense_per_100", "Offense"),
                ("defense_per_100", "Defense"),
                ("net_per_100", "Net"),
                ("off_possessions", "Off poss"),
                ("def_possessions", "Def poss"),
            ],
            wp_aio_qualified.sort_values(["candidate", "net_per_100"], ascending=[True, False]),
        )
    )

    points_qualified = points_ratings.loc[
        points_ratings[["Poss_Off", "Poss_Def"]].min(axis=1).ge(5000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "point-channel-ratings",
            "point-channels",
            "Conserved point-channel RAPM",
            [
                ("PLAYER_NAME", "Player"),
                ("one_point_net", "1 point"),
                ("two_point_net", "2 points"),
                ("three_plus_net", "3 plus"),
                ("net", "Net"),
            ],
            points_qualified.sort_values("net", ascending=False),
        )
    )

    factor_ratings = pd.read_parquet(ts_factors_path / "factor_ratings.parquet")
    factor_ratings["player_name"] = factor_ratings["player_id"].map(name_lookup).fillna(
        factor_ratings["player_id"].astype(str)
    )
    factor_specs = [
        ("shooting", "Shooting TS factor", "shooting_ts", 2000),
        ("turnover", "Turnover factor", "turnover", 3000),
        ("rebound", "Offensive-rebound factor", "offensive_rebound", 1500),
    ]
    for key, title, prefix, floor in factor_specs:
        qualified = factor_ratings.loc[
            factor_ratings[[f"{prefix}_off_exposure", f"{prefix}_def_exposure"]]
            .min(axis=1)
            .ge(floor)
        ]
        lab_leaderboards.append(
            leaderboard(
                f"factor-{key}",
                "factors",
                title,
                [
                    ("player_name", "Player"),
                    (f"{prefix}_offense", "Offense"),
                    (f"{prefix}_defense", "Defense"),
                    (f"{prefix}_net", "Net"),
                ],
                qualified.sort_values(f"{prefix}_net", ascending=False),
            )
        )

    combined_factors = factor_ratings.loc[
        factor_ratings[
            [
                "shooting_ts_off_exposure",
                "shooting_ts_def_exposure",
                "turnover_off_exposure",
                "turnover_def_exposure",
                "offensive_rebound_off_exposure",
                "offensive_rebound_def_exposure",
            ]
        ]
        .min(axis=1)
        .ge(1500)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "factor-six-sides",
            "factors",
            "Six-sided TS, turnover, and rebounding RAPM",
            [
                ("player_name", "Player"),
                ("shooting_ts_offense", "TS off"),
                ("shooting_ts_defense", "TS def"),
                ("turnover_offense", "TOV off"),
                ("turnover_defense", "TOV def"),
                ("offensive_rebound_offense", "OREB off"),
                ("offensive_rebound_defense", "OREB def"),
            ],
            combined_factors.sort_values("shooting_ts_offense", ascending=False),
        )
    )

    pair_bucket_ratings = pd.read_parquet(pair_buckets_path / "pair_ratings.parquet")
    pair_bucket_tails = pd.concat(
        [
            pair_bucket_ratings.nlargest(100, "net_per_100").assign(sample="Top 100"),
            pair_bucket_ratings.nsmallest(100, "net_per_100").assign(sample="Bottom 100"),
        ],
        ignore_index=True,
    ).drop_duplicates("players")
    lab_leaderboards.append(
        leaderboard(
            "pair-bucketing-ratings",
            "pair-bucketing",
            "Selected pair model, top and bottom 100",
            [
                ("pair", "Pair"),
                ("sample", "Set"),
                ("offense_per_100", "Offense"),
                ("defense_per_100", "Defense"),
                ("net_per_100", "Net"),
            ],
            pair_bucket_tails.sort_values("net_per_100", ascending=False),
        )
    )

    production_ratings = pd.read_parquet(production_5y_path / "ratings.parquet")
    production_qualified = production_ratings.loc[
        production_ratings[["Poss_Off", "Poss_Def"]].min(axis=1).ge(3000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "production-5y-ratings",
            "production-5y",
            "Rolling five-year RAPM with 95% intervals",
            [
                ("PLAYER_NAME", "Player"),
                ("window_start", "From"),
                ("window_end", "To"),
                ("offense", "Offense"),
                ("defense", "Defense"),
                ("net", "Net"),
                ("net_ci95_low", "95% low"),
                ("net_ci95_high", "95% high"),
            ],
            production_qualified.sort_values(
                ["window_end", "net"], ascending=[False, False]
            ),
        )
    )

    horizon_ratings = pd.read_parquet(horizons_path / "ratings.parquet")
    horizon_qualified = horizon_ratings.loc[
        horizon_ratings[["Poss_Off", "Poss_Def"]].min(axis=1).ge(3000)
    ].copy()
    lab_leaderboards.append(
        leaderboard(
            "target-horizon-ratings",
            "target-horizons",
            "RAPM target panels",
            [
                ("PLAYER_NAME", "Player"),
                ("horizon", "Horizon"),
                ("window_start", "From"),
                ("window_end", "To"),
                ("offense", "Offense"),
                ("defense", "Defense"),
                ("net", "Net"),
            ],
            horizon_qualified.sort_values(
                ["window_end", "horizon", "net"], ascending=[False, True, False]
            ),
        )
    )

    luck_ratings = pd.read_parquet(
        shooting_luck_path / "luck_adjusted_ratings.parquet"
    ).loc[
        lambda frame: frame[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ]
    lab_leaderboards.append(
        leaderboard(
            "shooting-luck-ratings",
            "shooting-luck",
            "FT and 3P luck-adjusted RAPM, 2024-26",
            [
                ("player_name", "Player"),
                ("normal_net", "Normal"),
                ("luck_offense", "Offense"),
                ("luck_defense", "Defense"),
                ("luck_net", "Luck adjusted"),
                ("luck_net_change", "Change"),
            ],
            luck_ratings.sort_values("luck_net", ascending=False),
        )
    )
    teammate_ratings = pd.read_parquet(
        shooting_luck_path / "teammate_efg_ratings.parquet"
    ).loc[
        lambda frame: frame[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(2000)
    ]
    lab_leaderboards.append(
        leaderboard(
            "teammate-efg-ratings",
            "teammate-efg",
            "Teammate-only eFG RAPM, 2024-26",
            [
                ("player_name", "Player"),
                ("teammate_efg_offense", "Teammate eFG"),
                ("shot_defense", "Shot defense"),
                ("teammate_efg_net", "Net"),
            ],
            teammate_ratings.sort_values("teammate_efg_net", ascending=False),
        )
    )

    teammate_effects = pd.read_parquet(
        teammate_channels_path / "teammate_effect_ratings.parquet"
    ).loc[
        lambda frame: frame["possession_opportunities"].ge(3000)
        & frame["block_opportunities"].ge(1000)
        & frame["oreb_opportunities"].ge(1000)
        & frame["dreb_opportunities"].ge(1000)
    ]
    lab_leaderboards.append(
        leaderboard(
            "teammate-effect-ratings",
            "teammate-effects",
            "Teammate effects, 2024-26",
            [
                ("player_name", "Player"),
                ("teammate_scoring", "Scoring"),
                ("teammate_turnovers", "TOV prevention"),
                ("teammate_assists", "Assists"),
                ("teammate_steals", "Steals"),
                ("teammate_blocks", "Blocks"),
                ("teammate_oreb", "OREB"),
                ("teammate_dreb", "DREB"),
            ],
            teammate_effects.sort_values("teammate_scoring", ascending=False),
        )
    )

    play_channels = pd.read_parquet(
        teammate_channels_path / "observable_play_channel_ratings.parquet"
    ).loc[
        lambda frame: frame[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ]
    lab_leaderboards.append(
        leaderboard(
            "observable-scoring-channels",
            "observable-play-channels",
            "Scoring and shot channels, 2024-26",
            [
                ("player_name", "Player"),
                ("rim_assists_net", "Rim AST"),
                ("transition_points_net", "Transition"),
                ("three_point_points_net", "3PT"),
                ("free_throw_points_net", "FT"),
                ("midrange_attempts_net", "Mid freq"),
                ("rim_points_net", "Rim"),
            ],
            play_channels.sort_values("rim_points_net", ascending=False),
        )
    )
    lab_leaderboards.append(
        leaderboard(
            "observable-finish-channels",
            "observable-play-channels",
            "Observable shot-finish channels, 2024-26",
            [
                ("player_name", "Player"),
                ("playtype_putback_points_net", "Putback"),
                ("playtype_cut_points_net", "Cut"),
                ("playtype_drive_points_net", "Drive"),
                ("playtype_pullup_points_net", "Pull-up"),
                ("playtype_post_points_net", "Post-like"),
                ("playtype_spotup_points_net", "Jump shot"),
                ("playtype_other_points_net", "Other"),
            ],
            play_channels.sort_values("playtype_drive_points_net", ascending=False),
        )
    )

    current_age_ratings = pd.read_parquet(time_age_path / "ratings.parquet").loc[
        lambda frame: frame[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(3000)
    ]
    lab_leaderboards.append(
        leaderboard(
            "time-decay-actual-age-ratings",
            "time-decay-actual-age",
            "Five-year time-decay actual-age challenger, 2022-26",
            [
                ("player_name", "Player"),
                ("offense", "Offense"),
                ("defense", "Defense"),
                ("net", "Net"),
                ("age_net_adjustment", "Age adjustment"),
            ],
            current_age_ratings.sort_values("net", ascending=False),
        )
    )

    ryan_ratings = pd.read_parquet(ryan_davis_path / "comparison.parquet")
    lab_leaderboards.append(
        leaderboard(
            "ryan-davis-ratings",
            "ryan-davis",
            "2019 normal RAPM versus Ryan Davis",
            [
                ("PLAYER_NAME", "Player"),
                ("target_offense", "Our offense"),
                ("target_defense", "Our defense"),
                ("target_net", "Our net"),
                ("ryan_offense", "Ryan offense"),
                ("ryan_defense", "Ryan defense"),
                ("ryan_net", "Ryan net"),
                ("net_difference", "Difference"),
            ],
            ryan_ratings.sort_values("target_net", ascending=False),
        )
    )

    metric_rows = external_metrics.copy()
    metric_rows["n"] = metric_rows["n"].astype(int)
    lab_leaderboards.append(
        leaderboard(
            "external-comparison-metrics",
            "external-reproductions",
            "External comparison metrics",
            [
                ("source", "Source"),
                ("comparison", "Variant"),
                ("scope", "Scope"),
                ("component", "Side"),
                ("method_status", "Alignment"),
                ("n", "Matched"),
                ("pearson", "Pearson"),
                ("spearman", "Rank"),
                ("slope", "Scale"),
                ("rmse", "RMSE"),
            ],
            metric_rows.sort_values(
                ["method_status", "source", "scope", "component"], kind="stable"
            ),
        )
    )

    external_boards = [
        ("external-xrapm-3y", "xRAPM", "2024-2026", "Current 3y RAPM versus xRAPM"),
        ("external-ryan-annual", "Ryan Davis annual RAPM", "2023", "2023 RAPM versus Ryan Davis"),
        ("external-ryan-five", "Ryan Davis multi-year RAPM", "2019-2023", "2019-23 five-year RAPM versus Ryan Davis"),
        ("external-ryan-three", "Ryan Davis multi-year RAPM", "2021-2023", "2021-23 three-year RAPM versus Ryan Davis"),
        ("external-darko", "DARKO WOWY", "2026", "2026 RAPM versus DARKO WOWY"),
        ("external-raptor", "FiveThirtyEight RAPTOR", "2022", "2022 RAPM versus RAPTOR on/off"),
        ("external-aupm", "Local legacy AuPM", "2024", "2024 RAPM versus reproduced AuPM"),
        ("external-gpm", "CourtSignal reproduction", "2024-2026", "2024-26 RAPM versus game-level PM"),
        ("external-28y", "Downloaded 28-year RAPM", "1997-2024", "1997-2024 long-span reproduction"),
        ("external-weighted", "Downloaded weighted RAPM", "2022-2024", "2022-24 equal versus weighted RAPM"),
    ]
    for board_id, source, scope, title in external_boards:
        rows = external_matched.loc[
            external_matched["source"].eq(source)
            & external_matched["scope"].eq(scope)
            & external_matched["component"].eq("net")
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing external leaderboard rows for {source}/{scope}")
        lab_leaderboards.append(
            leaderboard(
                board_id,
                "external-reproductions",
                title,
                [
                    ("player_name", "Player"),
                    ("reference", "Reference"),
                    ("courtsignal", "CourtSignal"),
                ],
                rows.sort_values("courtsignal", ascending=False),
            )
        )

    unit_ratings = pd.read_parquet(units_path / "diagnostic_unit_ratings.parquet")
    unit_ratings["unit"] = unit_ratings["players"].map(
        lambda value: unit_name(value, name_lookup)
    )
    for order, title in (
        (2, "Pair RAPM, top and bottom 100"),
        (3, "Trio RAPM, top and bottom 100"),
        (4, "Four-man RAPM, top and bottom 100"),
        (5, "Lineup RAPM, top and bottom 100"),
    ):
        order_rows = unit_ratings.loc[unit_ratings["order"].eq(order)]
        rows = pd.concat(
            [
                order_rows.nlargest(100, "net_per_100").assign(sample="Top 100"),
                order_rows.nsmallest(100, "net_per_100").assign(
                    sample="Bottom 100"
                ),
            ],
            ignore_index=True,
        ).sort_values("net_per_100", ascending=False)
        rows = rows.drop_duplicates("players", keep="first")
        lab_leaderboards.append(
            leaderboard(
                f"unit-{order}",
                "standalone-units",
                title,
                [
                    ("unit", "Unit"),
                    ("sample", "Set"),
                    ("offense_per_100", "Offense"),
                    ("defense_per_100", "Defense"),
                    ("net_per_100", "Net"),
                ],
                rows,
            )
        )

    coach_ratings = pd.read_parquet(full_coach_path / "xrapm_comparison.parquet")
    lab_leaderboards.append(
        leaderboard(
            "coach-ratings",
            "coach",
            "Age-adjusted coach RAPM, 1997-2026",
            [
                ("coach", "Coach"),
                ("offense", "Offense"),
                ("defense", "Defense"),
                ("net", "Net"),
                ("xrapm_net", "xRAPM"),
                ("listed_games", "Games"),
            ],
            coach_ratings.sort_values("net", ascending=False),
        )
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "localhost_only",
        "experiments": experiments,
        "leaderboards": lab_leaderboards,
        "replications": replications,
        "replication_leaderboards": replication_leaderboards,
        "age": {
            "run_id": age_adjusted["run_id"],
            "curve": clean_records(
                pd.read_parquet(age_adjusted_path / "age_curve.parquet")
            ),
            "selection": age_adjusted["selection"],
            "evaluation": age_adjusted["evaluation"],
            "bootstrap_vs_normal": age_adjusted["bootstrap_vs_normal"],
            "rating_effect": age_adjusted["rating_effect"],
            "quality": age_adjusted["quality"],
        },
        "rubberband": {
            "run_id": rubberband["run_id"],
            "selected_spec": rubberband["config"]["selected_spec"],
            "margin_clip": rubberband["config"]["selected_margin_clip"],
            "possessions": rubberband["quality"]["possessions"],
            "games": rubberband["quality"]["games"],
            "expected_games": rubberband["quality"]["expected_regular_season_games"],
            "game_coverage_rate": rubberband["quality"]["game_coverage_rate"],
            "diagnostic": rubberband["diagnostic"],
            "bootstrap": rubberband["paired_diagnostic_bootstrap"],
            "minimum_season_correlation": rubberband["coefficient_stability"][
                "minimum_pairwise_season_correlation"
            ],
            "selection_winner_vs_runner_up": rubberband[
                "selection_winner_vs_runner_up"
            ],
            "coefficients": coefficients.to_dict("records"),
            "curve": curve.to_dict("records"),
            "comparison_run_id": adjusted["run_id"],
            "comparison_coefficients": clean_records(comparison_coefficients),
            "context_effect": adjusted["context_effect"],
            "rapm_evaluation": adjusted["rapm_evaluation"],
            "rapm_bootstrap_vs_normal": adjusted["rapm_bootstrap_vs_normal"],
            "conditional_rapm_bootstrap_vs_normal": adjusted[
                "conditional_rapm_bootstrap_vs_normal"
            ],
            "rating_effect": adjusted["rating_effect"],
            "je": {
                "run_id": je["run_id"],
                "fit_seasons": je["config"]["fit_seasons"],
                "curve": clean_records(je_curve),
                "effects": je["score_state_effects_points_per_100_vs_tie"],
                "evaluation": je["evaluation"],
                "bootstrap_vs_normal": je["bootstrap_vs_normal"],
                "rating_effect": je["rating_effect"],
            },
            "five_point": {
                "run_id": rubberband_5pt["run_id"],
                "curve_run_id": rubberband_5pt_curve["run_id"],
                "curve": rubberband_5pt_curve["curve"],
                "selection_winner": rubberband_5pt["selection_winner"],
                "diagnostic": rubberband_5pt["diagnostic"],
                "decision": rubberband_5pt["decision"],
            },
            "score_signal": {
                "run_id": score_signal["run_id"],
                "selection_winner": score_signal["selection_winner"],
                "diagnostic": score_signal["diagnostic"],
                "decision": score_signal["decision"],
                "curve": clean_records(
                    pd.read_parquet(score_signal_path / "curve.parquet")
                ),
            },
            "age_score": {
                "run_id": age_score["run_id"],
                "selection": age_score["selection"],
                "diagnostic": age_score["diagnostic"],
                "quality": age_score["quality"],
            },
            "ratings": clean_records(adjusted_ratings),
            "test": (
                "Lineup effects are estimated out of fold by whole game. The curve uses actual "
                "possession-start clock and score. 2024 develops, 2025 selects, and 2026 is a reused diagnostic."
            ),
            "decision": "Both curves are real possession-context estimates. Neither adjusted player rating improves 2026 RMSE enough to promote.",
        },
    }


def main() -> None:
    payload = json_safe(build_payload())
    DESTINATION.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    print(
        f"wrote {DESTINATION} with {len(payload['experiments'])} experiments and "
        f"{len(payload['rubberband']['curve'])} rubber-band curve rows"
    )


if __name__ == "__main__":
    main()
