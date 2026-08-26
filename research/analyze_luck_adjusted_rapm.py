"""Audit the four-arm luck target and reproduce the earlier FT/3P diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import write_json_atomic
from nba_impact.models.luck_adjusted_rapm import (
    ARMS,
    _game_metrics,
    build_expected_outcome_frame,
    compose_arm_beta,
    shooter_skill_bonus,
)
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficients,
    load_current_player_names,
    ratings_table,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "luck_adjusted_rapm_v1_8580bb30e9"
LEGACY_ARM = "ft3p_player_skill_adjusted_joint"


def _weighted_corr(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    weight = weight / weight.sum()
    actual_centered = actual - np.sum(weight * actual)
    predicted_centered = predicted - np.sum(weight * predicted)
    denominator = np.sqrt(
        np.sum(weight * actual_centered**2) * np.sum(weight * predicted_centered**2)
    )
    return float(np.sum(weight * actual_centered * predicted_centered) / denominator)


def _component_metrics(
    frame: pd.DataFrame,
    *,
    actual: str,
    predicted: str,
) -> dict:
    y = frame[actual].to_numpy(dtype=float)
    p = frame[predicted].to_numpy(dtype=float)
    w = frame["weight"].to_numpy(dtype=float)
    variance = float(np.average((p - np.average(p, weights=w)) ** 2, weights=w))
    covariance = float(
        np.average(
            (y - np.average(y, weights=w)) * (p - np.average(p, weights=w)),
            weights=w,
        )
    )
    slope = covariance / variance if variance > 0 else np.nan
    return {
        "players": int(len(frame)),
        "weighted_rmse": float(np.sqrt(np.average((y - p) ** 2, weights=w))),
        "weighted_correlation": _weighted_corr(y, p, w),
        "calibration_slope": slope,
        "calibration_intercept": float(
            np.average(y, weights=w) - slope * np.average(p, weights=w)
        ),
    }


def _paired_bootstrap(
    games: pd.DataFrame,
    *,
    challenger: str,
    test_season: int,
    draws: int = 5000,
    seed: int = 20260826,
) -> dict:
    window = games.loc[games["test_season"].eq(test_season)].copy()
    normal = window.loc[window["arm"].eq("normal_realized_points"), ["game_id", "actual_margin", "predicted_margin"]].rename(
        columns={"predicted_margin": "normal_prediction"}
    )
    candidate = window.loc[window["arm"].eq(challenger), ["game_id", "actual_margin", "predicted_margin"]].rename(
        columns={"actual_margin": "candidate_actual", "predicted_margin": "candidate_prediction"}
    )
    merged = normal.merge(candidate, on="game_id", validate="one_to_one")
    if not np.allclose(merged["actual_margin"], merged["candidate_actual"]):
        raise ValueError("Paired bootstrap arms do not share outcomes.")
    normal_sq = (merged["actual_margin"] - merged["normal_prediction"]).to_numpy(dtype=float) ** 2
    candidate_sq = (merged["actual_margin"] - merged["candidate_prediction"]).to_numpy(dtype=float) ** 2
    rng = np.random.default_rng(seed + test_season)
    deltas = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        indices = rng.integers(0, len(merged), size=(stop - start, len(merged)))
        deltas[start:stop] = np.sqrt(candidate_sq[indices].mean(axis=1)) - np.sqrt(
            normal_sq[indices].mean(axis=1)
        )
    observed = float(np.sqrt(candidate_sq.mean()) - np.sqrt(normal_sq.mean()))
    return {
        "test_season": test_season,
        "challenger": challenger,
        "games": int(len(merged)),
        "draws": draws,
        "rmse_delta": observed,
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_better": float(np.mean(deltas < 0)),
    }


def _ratings(design, beta: np.ndarray, names: pd.DataFrame) -> pd.DataFrame:
    return ratings_table(design, beta, names=names).rename(
        columns={
            "offense_per_100": "offense",
            "defense_per_100": "defense",
            "net_per_100": "net",
        }
    )


def analyze(run_id: str = DEFAULT_RUN) -> dict:
    artifact = ROOT / "artifacts/models/luck_adjusted_rapm" / run_id
    if not (artifact / "run.json").exists():
        raise FileNotFoundError(f"Missing luck-adjusted RAPM run {run_id}.")
    ledger = pd.read_parquet(artifact / "checkpoints/conversion_ledger.parquet")
    possessions = pd.read_parquet(ROOT / "data/lake/silver/possessions.parquet")
    segments = pd.read_parquet(ROOT / "data/lake/silver/possession_lineup_segments.parquet")
    base = canonical_terminal_frame(possessions, segments, seasons=(2024, 2025, 2026))
    expected = build_expected_outcome_frame(base, ledger)
    ft3p = ledger.loc[ledger["category"].isin(["ft", "corner_3", "above_break_3"])].copy()
    # The legacy diagnostic retained player EB shooting skill, rather than the
    # player-neutral expected value used by the broad four-arm experiment.
    grouped = ft3p.groupby("possession_id", as_index=False).agg(
        actual_conversion_points=("actual_points", "sum"),
        skill_expected_conversion_points=("skill_expected_points", "sum"),
    )
    legacy_frame = base.merge(grouped, on="possession_id", how="left", validate="one_to_one")
    legacy_frame[["actual_conversion_points", "skill_expected_conversion_points"]] = legacy_frame[
        ["actual_conversion_points", "skill_expected_conversion_points"]
    ].fillna(0.0)
    legacy_frame["legacy_pts"] = (
        legacy_frame["pts"]
        - legacy_frame["actual_conversion_points"]
        + legacy_frame["skill_expected_conversion_points"]
    )

    normal_design = build_design(base, include_home=True)
    expected_design = build_design(expected.assign(pts=expected["expected_pts"]), include_home=True)
    legacy_design = build_design(legacy_frame.assign(pts=legacy_frame["legacy_pts"]), include_home=True)
    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="luck_adjusted_rapm_spm_v1_audit",
    )
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet"
    )
    game_parts = [pd.read_parquet(artifact / "future_game_predictions.parquet")]
    future_rating_rows: list[dict] = []
    for test_season in (2025, 2026):
        train_mask = normal_design.seasons < test_season
        test_mask = normal_design.seasons == test_season
        normal_beta, normal_intercept = fit_coefficients(normal_design, config, row_mask=train_mask)
        expected_beta, expected_intercept = fit_coefficients(expected_design, config, row_mask=train_mask)
        legacy_beta, legacy_intercept = fit_coefficients(legacy_design, config, row_mask=train_mask)
        bonus = shooter_skill_bonus(normal_design, ledger, train_mask=train_mask)
        arm_betas = {
            arm: compose_arm_beta(
                normal_beta,
                expected_beta,
                n_players=len(normal_design.players),
                arm=arm,
                shooting_bonus=bonus,
            )
            for arm in ARMS
        }
        arm_betas[LEGACY_ARM] = legacy_beta
        legacy_games = _game_margin_frame(
            normal_design, legacy_beta, legacy_intercept, test_mask, train_mask
        )
        legacy_games["arm"] = LEGACY_ARM
        legacy_games["test_season"] = test_season
        game_parts.append(legacy_games)

        target_frame = base.loc[base["season"].eq(test_season)].reset_index(drop=True)
        target_design = build_design(target_frame, include_home=True)
        target_beta, _ = fit_coefficients(target_design, config)
        target = _ratings(target_design, target_beta, names)[
            ["player_id", "offense", "defense", "net", "off_possessions", "def_possessions"]
        ].rename(
            columns={
                "offense": "target_offense",
                "defense": "target_defense",
                "net": "target_net",
                "off_possessions": "target_off_possessions",
                "def_possessions": "target_def_possessions",
            }
        )
        train_exposure = np.asarray(
            normal_design.X[train_mask, : 2 * len(normal_design.players)].sum(axis=0)
        ).ravel()
        exposure = pd.DataFrame(
            {
                "player_id": normal_design.players,
                "train_off_possessions": train_exposure[: len(normal_design.players)],
                "train_def_possessions": train_exposure[len(normal_design.players) :],
            }
        )
        for arm, beta in arm_betas.items():
            prediction = _ratings(normal_design, beta, names)[
                ["player_id", "offense", "defense", "net"]
            ].merge(exposure, on="player_id", validate="one_to_one")
            comparison = prediction.merge(target, on="player_id", validate="one_to_one")
            comparison["target_exposure"] = comparison[
                ["target_off_possessions", "target_def_possessions"]
            ].min(axis=1)
            comparison["train_exposure"] = comparison[
                ["train_off_possessions", "train_def_possessions"]
            ].min(axis=1)
            comparison = comparison.loc[
                comparison["target_exposure"].gt(0) & comparison["train_exposure"].gt(0)
            ].copy()
            comparison["weight"] = np.sqrt(comparison["target_exposure"])
            bands = {
                "all": comparison,
                "low_exposure": comparison.loc[comparison["target_exposure"].lt(1500)],
                "high_exposure": comparison.loc[comparison["target_exposure"].ge(3000)],
            }
            for band, frame in bands.items():
                if len(frame) < 10:
                    continue
                for component in ("offense", "defense", "net"):
                    future_rating_rows.append(
                        {
                            "test_season": test_season,
                            "arm": arm,
                            "component": component,
                            "exposure_band": band,
                            **_component_metrics(
                                frame,
                                actual=f"target_{component}",
                                predicted=component,
                            ),
                        }
                    )

    games = pd.concat(game_parts, ignore_index=True)
    game_metrics = (
        games.groupby(["test_season", "arm"], sort=False)
        .apply(lambda frame: pd.Series(_game_metrics(frame)), include_groups=False)
        .reset_index()
    )
    bootstrap_rows = [
        _paired_bootstrap(games, challenger=arm, test_season=test_season)
        for test_season in (2025, 2026)
        for arm in (*ARMS[1:], LEGACY_ARM)
    ]
    future_ratings = pd.DataFrame(future_rating_rows)
    audit = ROOT / "research/audits/luck_adjusted_rapm_spm_v1"
    audit.mkdir(parents=True, exist_ok=True)
    game_metrics.to_parquet(audit / "future_game_metrics.parquet", index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(audit / "paired_game_bootstrap.parquet", index=False)
    future_ratings.to_parquet(audit / "future_normal_rapm_metrics.parquet", index=False)

    normal_game = game_metrics.loc[game_metrics["arm"].eq("normal_realized_points")].set_index("test_season")
    winners = game_metrics.sort_values(["test_season", "margin_rmse"], kind="stable").groupby(
        "test_season", as_index=False
    ).head(1)
    legacy_delta = game_metrics.loc[game_metrics["arm"].eq(LEGACY_ARM)].set_index("test_season")[
        "margin_rmse"
    ] - normal_game["margin_rmse"]
    decision = {
        "experiment_id": "luck_adjusted_rapm_spm_v1",
        "source_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "normal_realized_points_retained",
        "game_winners": winners[["test_season", "arm", "margin_rmse"]].to_dict(orient="records"),
        "legacy_ft3p_rmse_delta": {str(index): float(value) for index, value in legacy_delta.items()},
        "spm_verdict": (
            "blocked_by_label_history: only 2024-25 have complete shot-level expectations before the 2026 output"
        ),
        "promotion": False,
        "reason": (
            "Every broad luck arm loses future-game RMSE to normal RAPM in both reused diagnostics. "
            "Two luck target seasons are insufficient for chronological SPM model selection."
        ),
    }
    write_json_atomic(decision, audit / "decision.json")
    pivot = game_metrics.pivot(
        index="arm", columns="test_season", values="margin_rmse"
    ).reindex([*ARMS, LEGACY_ARM])
    metric_table = [
        "| Arm | 2025 | 2026 |",
        "| --- | ---: | ---: |",
        *(
            f"| {arm} | {row.get(2025, np.nan):.4f} | {row.get(2026, np.nan):.4f} |"
            for arm, row in pivot.iterrows()
        ),
    ]
    target_net = future_ratings.loc[
        future_ratings["component"].eq("net")
        & future_ratings["exposure_band"].eq("all")
    ].pivot(index="arm", columns="test_season", values=["weighted_rmse", "weighted_correlation"])
    target_table = [
        "| Arm | 2025 RMSE | 2025 corr | 2026 RMSE | 2026 corr |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                arm,
                target_net.loc[arm, ("weighted_rmse", 2025)],
                target_net.loc[arm, ("weighted_correlation", 2025)],
                target_net.loc[arm, ("weighted_rmse", 2026)],
                target_net.loc[arm, ("weighted_correlation", 2026)],
            )
            for arm in [*ARMS, LEGACY_ARM]
        ),
    ]
    report = [
        "# Luck-adjusted RAPM and SPM decision",
        "",
        "Normal realized-points RAPM remains the reference. None of the broad conversion-adjusted arms improves future-game prediction in either reused diagnostic season.",
        "",
        "## Future-game margin RMSE",
        "",
        *metric_table,
        "",
        "## Earlier FT/3P result",
        "",
        (
            f"The corrected pre-season player-skill FT/3P diagnostic changes RMSE by {legacy_delta.loc[2025]:+.4f} in 2025 and {legacy_delta.loc[2026]:+.4f} in 2026 versus normal RAPM. "
            "The paired whole-game intervals are stored in `paired_game_bootstrap.parquet`."
        ),
        "",
        "It nearly reproduces the earlier 2026 gain, but the 95% interval crosses zero and the same frozen arm loses in 2025. That is a null, not a promotion.",
        "",
        "## Prediction of future normal RAPM",
        "",
        *target_table,
        "",
        "Expected-outcome ratings often lower future-RAPM RMSE by compressing the rating spread, but they lose net correlation to normal RAPM in both seasons. The calibration slopes and low/high-exposure slices are stored in `future_normal_rapm_metrics.parquet`. Smoother labels are not automatically better AIO priors.",
        "",
        "## SPM stop",
        "",
        "A luck-adjusted SPM was not fit. Complete shot-level expected-outcome labels begin in 2024, leaving only 2024 and 2025 as legal training labels for a 2026 output. Two seasons cannot support the required chronological feature and learner selection. Forcing that model would be less defensible than recording the data limit.",
        "",
        "2025 and 2026 are reused diagnostics. Season 2027 was not loaded.",
    ]
    (audit / "report.md").write_text("\n".join(report) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_id), indent=2))


if __name__ == "__main__":
    main()
