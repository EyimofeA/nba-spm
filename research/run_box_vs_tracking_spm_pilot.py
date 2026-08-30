#!/usr/bin/env python3
"""Run one reused BoxSPM versus tracking-only SPM oracle-lineup pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.defensive_tracking_features import DEFENSIVE_TRACKING_FEATURES
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.matchup_defense_features import MATCHUP_DEFENSE_FEATURES
from nba_impact.data.statistical_features import RATIO_SPECS, TRACKING_RATE_SPECS
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit,
    _select_alpha,
)
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _game_metrics, _prior_frame
from run_aio_prior_canonical_followup import _center, _remap_annual, _solve

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "box_vs_tracking_spm_pilot_v1"
CONTRACT = ROOT / "research/experiments/box_vs_tracking_spm_pilot_v1.yml"
FEATURE_RUN = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_4ffd1e34df"
)
TARGETS = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
RATING_SEASON = 2025
MODEL_ORDER = (
    "box_spm",
    "tracking_spm",
    "zero_prior_rapm",
    "box_spm_aio",
    "tracking_spm_aio",
)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _tracking_features(
    selected: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Select source-tracking concepts without any Box15 field."""
    offense_selected = set(selected["offense"])
    defense_selected = set(selected["defense"])
    traditional_ratios = {"fg2_pct", "fg3_pct", "ft_pct"}

    offense = [
        feature
        for feature, (_, denominator) in TRACKING_RATE_SPECS.items()
        if denominator == "OffPoss" and feature in offense_selected
    ]
    for feature in RATIO_SPECS:
        if feature in traditional_ratios:
            continue
        stabilized = f"{feature}_eb"
        if stabilized in offense_selected:
            offense.append(stabilized)
        elif feature in offense_selected:
            offense.append(feature)
    offense.extend(
        feature
        for feature in ("shot_quality_average_relative",)
        if feature in offense_selected
    )

    defense = [
        feature
        for feature, (_, denominator) in TRACKING_RATE_SPECS.items()
        if denominator == "DefPoss" and feature in defense_selected
    ]
    defense.extend(
        feature
        for feature in (*DEFENSIVE_TRACKING_FEATURES, *MATCHUP_DEFENSE_FEATURES)
        if feature in defense_selected
    )
    defense.extend(
        feature
        for feature in (
            "has_hustle_tracking",
            "has_matchup_tracking",
            "has_dfg_tracking",
            "has_rim_defense_tracking",
        )
        if feature in defense_selected
    )
    result = {
        "offense": tuple(dict.fromkeys(offense)),
        "defense": tuple(dict.fromkeys(defense)),
    }
    for side, features in result.items():
        if not features or set(features) & set(BOX_PIPM_STYLE_FEATURES):
            raise ValueError(f"{side} tracking contract is empty or overlaps Box15.")
    return result


def _fit_priors(
    panel: pd.DataFrame,
    candidates: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel.loc[panel["Window_End"].lt(RATING_SEASON)].copy()
    test = panel.loc[panel["Window_End"].eq(RATING_SEASON)].copy()
    if train["Window_End"].nunique() < 3 or test.empty:
        raise ValueError("Pilot lacks chronological training or rating rows.")
    priors = []
    metric_rows = []
    alpha_rows = []
    for candidate, sides in candidates.items():
        prior = test[["PLAYER_ID", "Window_End"]].copy()
        for side in ("offense", "defense"):
            features = sides[side]
            target = f"target_{side}"
            alpha = _select_alpha(
                train.rename(columns={"Window_End": "Season"}),
                features,
                target,
                ALPHA_GRID,
            )
            model = _fit(train, features, target, alpha)
            prediction = model.predict(test.loc[:, features])
            prior[side] = prediction
            alpha_rows.append(
                {
                    "candidate": candidate,
                    "component": side,
                    "selected_alpha": alpha,
                    "feature_count": len(features),
                    "train_window_min": int(train["Window_End"].min()),
                    "train_window_max": int(train["Window_End"].max()),
                }
            )
            metric_rows.append(
                {
                    "candidate": candidate,
                    "component": side,
                    **_metrics(
                        test[target].to_numpy(dtype=float),
                        prediction,
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
        prior["net"] = prior["offense"] + prior["defense"]
        metric_rows.append(
            {
                "candidate": candidate,
                "component": "net",
                **_metrics(
                    test["target_net"].to_numpy(dtype=float),
                    prior["net"].to_numpy(dtype=float),
                    test["sample_weight"].to_numpy(dtype=float),
                ),
            }
        )
        priors.append(_prior_frame(prior, candidate))
    return (
        pd.concat(priors, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(alpha_rows),
    )


def _score(
    priors: pd.DataFrame,
    annual: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_dir = MATRIX_ROOT / f"5y_end_{RATING_SEASON}"
    players = np.load(matrix_dir / "player_ids.npy")
    bundle = _remap_annual(annual[RATING_SEASON], players)
    zero = np.zeros(2 * len(players) + 1)
    zero_beta, zero_intercept = _solve(bundle, zero, scale=0.0)
    models = [("zero_prior_rapm", zero_beta, zero_intercept)]
    coverage_rows = []
    for candidate in ("box_spm", "tracking_spm"):
        prior = priors.loc[priors["candidate"].eq(candidate)]
        center, coverage = _center(prior, bundle)
        standalone = center.copy()
        standalone[-1] = zero_beta[-1]
        posterior, posterior_intercept = _solve(bundle, center, scale=1.0)
        models.extend(
            [
                (candidate, standalone, zero_intercept),
                (f"{candidate}_aio", posterior, posterior_intercept),
            ]
        )
        coverage_rows.append({"candidate": candidate, **coverage})

    game_rows = []
    metric_rows = []
    for candidate, beta, intercept in models:
        games = stored_evaluation_predictions(matrix_dir, beta, intercept)
        games["candidate"] = candidate
        games["rating_season"] = RATING_SEASON
        games["test_season"] = RATING_SEASON + 1
        game_rows.append(games)
        metric_rows.append({"candidate": candidate, **_game_metrics(games)})
    all_games = pd.concat(game_rows, ignore_index=True)
    counts = all_games.groupby("candidate")["game_id"].nunique()
    if set(counts.index) != set(MODEL_ORDER) or counts.nunique() != 1:
        raise ValueError("Pilot candidates did not score identical games.")
    return all_games, pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows)


def _annual_2025() -> tuple[dict, pd.DataFrame]:
    annual = {}
    for season in range(2020, 2024):
        frame = base.load_legacy_possessions(
            POSSESSION_CACHE, (season,), game_types=("regular",)
        )
        annual[season] = base._annual_from_frame(frame, season)
    reconstruction = []
    for season in (2024, 2025):
        bundle, quality = base._recover_annual(
            MATRIX_ROOT / f"5y_end_{season}", season, annual
        )
        annual[season] = bundle
        reconstruction.append(quality)
    return annual, pd.DataFrame(reconstruction)


def main() -> None:
    contract = json.loads(json.dumps(yaml.safe_load(CONTRACT.read_text()), default=str))
    cutoff = contract["information_cutoff"]
    if contract["experiment_id"] != EXPERIMENT_ID or cutoff["season_2027"] != "forbidden":
        raise ValueError("Pilot contract changed.")
    panel, selected = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    candidates = {
        "box_spm": {
            "offense": BOX_PIPM_STYLE_FEATURES,
            "defense": BOX_PIPM_STYLE_FEATURES,
        },
        "tracking_spm": _tracking_features(selected),
    }
    used = sorted(
        set().union(
            *(set(features) for sides in candidates.values() for features in sides.values())
        )
    )
    if panel[used].isna().any().any() or not np.isfinite(
        panel[used].to_numpy(dtype=float)
    ).all():
        raise ValueError("Pilot inputs contain missing or nonfinite values.")

    priors, target_metrics, alpha_selection = _fit_priors(panel, candidates)
    annual, reconstruction = _annual_2025()
    games, game_metrics, coverage = _score(priors, annual)
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {
        frozenset(("box_spm", "tracking_spm")),
        frozenset(("box_spm_aio", "tracking_spm_aio")),
    }
    bootstrap_models, bootstrap_pairs = base.paired_game_bootstrap(
        games, draws=5000, seed=20260830
    )

    source_paths = {
        "contract": CONTRACT,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "runner": Path(__file__),
        **{
            f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json"
            for season in (2024, 2025)
        },
        **{
            f"possessions_{season}": POSSESSION_CACHE / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_season": RATING_SEASON,
        "test_season": RATING_SEASON + 1,
        "candidate_features": candidates,
        "alpha_grid": ALPHA_GRID,
        "bootstrap": {"draws": 5000, "seed": 20260830, "unit": "whole game"},
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=list).encode()
    ).hexdigest()[:10]
    output = (
        ROOT
        / "artifacts/research/box_vs_tracking_spm_pilot"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    feature_rows = [
        {"candidate": candidate, "component": side, "feature": feature}
        for candidate, sides in candidates.items()
        for side, features in sides.items()
        for feature in features
    ]
    outputs = {
        "candidate_features.parquet": pd.DataFrame(feature_rows),
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "alpha_selection.parquet": alpha_selection,
        "game_predictions.parquet": games,
        "game_metrics.parquet": game_metrics,
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "status": "reused_2026_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "input_missing_values": 0,
            "input_nonfinite_values": 0,
            "identical_games": True,
            "season_2027_loaded": False,
        },
        "files": {},
        "forbidden_interpretation": (
            "One reused oracle-lineup fold cannot promote either SPM."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(game_metrics.sort_values("margin_rmse").to_string(index=False))
    print(
        bootstrap_pairs.loc[bootstrap_pairs["primary_comparison"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
