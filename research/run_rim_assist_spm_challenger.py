#!/usr/bin/env python3
"""Test stabilized rim assists in the frozen five-year BoxPIPM-style prior."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from run_aio_prior_bakeoff import _paired_bootstrap
from run_pipm_breaker import (
    RATING_SEASONS,
    _fit_aio,
    _fit_direct_prior,
    _game_summary,
    _panel,
    _target_columns,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "rim_assist_spm_challenger_v1"
BASELINE = "box_courtsignal_ridge"
CHALLENGER = "box_courtsignal_rim_assists"
RIM_ASSIST_FEATURE = "rim_assists_p100_eb_5y"
PRIOR_POSSESSIONS = 500.0


def annual_rim_assists(
    player_sheet_root: Path,
    seasons: range,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual rim-assist rates with same-season empirical-Bayes centers."""
    rows = []
    coverage = []
    for season in seasons:
        path = player_sheet_root / f"{season}.parquet"
        frame = pd.read_parquet(
            path,
            columns=["PLAYER_ID", "AtRimAssists", "OffPoss"],
        ).drop_duplicates()
        for column in ("PLAYER_ID", "AtRimAssists", "OffPoss"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["PLAYER_ID"]).copy()
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        valid = frame["AtRimAssists"].notna() & frame["OffPoss"].gt(0)
        attempts = float(frame.loc[valid, "AtRimAssists"].sum())
        possessions = float(frame.loc[valid, "OffPoss"].sum())
        if possessions <= 0:
            raise ValueError(f"Season {season} has no rim-assist exposure.")
        center = attempts / possessions
        frame["rim_assists_p100_eb"] = 100.0 * (
            frame["AtRimAssists"].where(valid, 0.0) + PRIOR_POSSESSIONS * center
        ) / (frame["OffPoss"].where(valid, 0.0) + PRIOR_POSSESSIONS)
        frame.loc[~valid, "rim_assists_p100_eb"] = np.nan
        frame["Season"] = int(season)
        rows.append(
            frame[["PLAYER_ID", "Season", "OffPoss", "rim_assists_p100_eb"]]
        )
        coverage.append(
            {
                "Season": int(season),
                "player_rows": len(frame),
                "observed_rows": int(valid.sum()),
                "observed_row_rate": float(valid.mean()),
                "observed_offensive_possessions": possessions,
                "rim_assists": attempts,
                "league_rim_assists_per_100": 100.0 * center,
            }
        )
    annual = pd.concat(rows, ignore_index=True)
    if annual.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Rim-assist source has duplicate player-season keys.")
    return annual, pd.DataFrame(coverage)


def pool_rim_assists(
    annual: pd.DataFrame,
    window_ends: range,
) -> pd.DataFrame:
    """Pool five frozen annual rates with offensive-possession weights."""
    rows = []
    for end in window_ends:
        window = annual.loc[annual["Season"].between(end - 4, end)].copy()
        valid = window[RIM_ASSIST_FEATURE.removesuffix("_5y")].notna() & window[
            "OffPoss"
        ].gt(0)
        value = window[RIM_ASSIST_FEATURE.removesuffix("_5y")].where(valid, 0.0)
        weight = window["OffPoss"].where(valid, 0.0)
        numerator = (value * weight).groupby(window["PLAYER_ID"]).sum()
        denominator = weight.groupby(window["PLAYER_ID"]).sum()
        pooled = numerator / denominator.replace(0.0, np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "PLAYER_ID": pooled.index.astype(int),
                    "Window_End": int(end),
                    RIM_ASSIST_FEATURE: pooled.to_numpy(dtype=float),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Pooled rim-assist keys are duplicated.")
    return result


def fill_unobserved_rim_assists(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use the same-window league center without exposing a missingness feature."""
    output = panel.copy()
    observed = output[RIM_ASSIST_FEATURE].notna()
    weights = output["courtsignal_exposure"].clip(lower=0.0)
    observed_weight = float(weights.where(observed, 0.0).sum())
    total_weight = float(weights.sum())
    possession_coverage = observed_weight / total_weight
    if possession_coverage < 0.98:
        raise ValueError("Five-year rim-assist possession coverage is below 98%.")
    for end, group in output.groupby("Window_End", sort=False):
        valid = group[RIM_ASSIST_FEATURE].notna() & group[
            "courtsignal_exposure"
        ].gt(0)
        center = float(
            np.average(
                group.loc[valid, RIM_ASSIST_FEATURE],
                weights=group.loc[valid, "courtsignal_exposure"],
            )
        )
        missing = group[RIM_ASSIST_FEATURE].isna()
        output.loc[group.index[missing], RIM_ASSIST_FEATURE] = center
    if output[RIM_ASSIST_FEATURE].isna().any():
        raise ValueError("Neutral rim-assist fill left missing values.")
    return output, {
        "observed_row_rate": float(observed.mean()),
        "observed_possession_rate": possession_coverage,
        "neutral_fill": "same-window possession-weighted league center",
    }


def fit_challenger(
    panel: pd.DataFrame,
    baseline_priors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one additional offense field and preserve the baseline defense prior."""
    rows = [baseline_priors.assign(candidate=BASELINE)]
    selections = []
    features = (*BOX_PIPM_STYLE_FEATURES, RIM_ASSIST_FEATURE)
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        train["sample_weight"] = train["courtsignal_exposure"]
        test["sample_weight"] = test["courtsignal_exposure"]
        train, test = _target_columns(train, test, "offense")
        prediction, spec, scores = _fit_direct_prior(
            train,
            test,
            features,
            "courtsignal_offense",
            ridge_only=True,
        )
        baseline = baseline_priors.loc[
            baseline_priors["Window_End"].eq(season),
            ["PLAYER_ID", "Window_End", "prior_defense_per_100"],
        ]
        challenger = test[["PLAYER_ID", "Window_End"]].copy()
        challenger["prior_offense_per_100"] = prediction
        challenger = challenger.merge(
            baseline,
            on=["PLAYER_ID", "Window_End"],
            how="left",
            validate="one_to_one",
        )
        challenger["prior_net_per_100"] = (
            challenger["prior_offense_per_100"]
            + challenger["prior_defense_per_100"]
        )
        challenger["candidate"] = CHALLENGER
        rows.append(challenger)
        selections.append(
            scores.assign(
                rating_season=season,
                side="offense",
                selected=lambda frame, spec=spec: (
                    frame["family"].eq(spec.family)
                    & frame["config"].eq(json.dumps(spec.config, sort_keys=True))
                ),
                feature_count=len(features),
            )
        )
    return pd.concat(rows, ignore_index=True), pd.concat(selections, ignore_index=True)


def main() -> None:
    features_path = ROOT / "artifacts/research/spm_target_horizon_full/spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet"
    targets_path = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
    ryan_path = ROOT / "research/rapm_lab/data/external/user_downloads/ryan_davis_multi_rapm.csv"
    base_run = ROOT / "artifacts/research/pipm_breaker/pipm_breaker_v1_d154ebea55"
    player_sheet_root = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
    possession_source = ROOT / "research/rapm_lab/external/external/poss_data"
    schedule_root = ROOT / "data/lake/bronze/official_game_schedule_1997_2026"
    matrix_root = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"

    panel, _ = _panel(features_path, targets_path, ryan_path)
    annual, coverage = annual_rim_assists(player_sheet_root, range(2014, 2024))
    pooled = pool_rim_assists(annual, range(2018, 2024))
    panel = panel.merge(
        pooled,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    panel, pooled_coverage = fill_unobserved_rim_assists(panel)
    baseline_priors = pd.read_parquet(base_run / "prior_ratings.parquet")
    baseline_priors = baseline_priors.loc[
        baseline_priors["candidate"].eq(BASELINE)
    ].copy()
    priors, selections = fit_challenger(panel, baseline_priors)
    ratings, games, prior_coverage, possession_quality = _fit_aio(
        priors,
        possession_source=possession_source,
        schedule_root=schedule_root,
        matrix_root=matrix_root,
    )
    game_metrics, summary = _game_summary(games)
    bootstrap = _paired_bootstrap(
        games,
        baseline=BASELINE,
        draws=5_000,
        seed=20260827,
    )

    source_paths = {
        "features": features_path,
        "targets": targets_path,
        "ryan_ratings": ryan_path,
        "baseline_manifest": base_run / "run.json",
        "runner": Path(__file__),
        **{
            f"player_sheet_{season}": player_sheet_root / f"{season}.parquet"
            for season in range(2014, 2024)
        },
    }
    config = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "baseline": BASELINE,
        "challenger": CHALLENGER,
        "rim_assist_feature": {
            "annual": "100 * EB(AtRimAssists / OffPoss)",
            "same_season_prior_possessions": PRIOR_POSSESSIONS,
            "five_year_pool": "offensive-possession-weighted frozen annual estimates",
        },
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/rim_assist_spm_challenger" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "summary": summary,
        "game_metrics": game_metrics,
        "game_predictions": games,
        "prior_ratings": priors,
        "posterior_ratings": ratings,
        "prior_coverage": prior_coverage,
        "possession_source_quality": possession_quality,
        "model_selection": selections,
        "annual_feature_coverage": coverage,
        "paired_bootstrap": bootstrap,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_historical_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "five_year_feature_coverage": pooled_coverage,
            "identical_games": True,
            "component_identity_max_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "season_2027_rows": 0,
        },
        "paths": {name: f"{name}.parquet" for name in outputs},
        "decision_rule": "Promote only if paired future-game MSE improves without a material correlation loss.",
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
