#!/usr/bin/env python3
"""Test same-season mechanism features around the Box15 SPM prior."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.mechanism_features import (
    DEFENSE_MECHANISM_FEATURES,
    OFFENSE_MECHANISM_FEATURES,
)
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _prior_frame

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "mechanism_feature_challenger_v1"
CONTRACT = ROOT / "research/experiments/mechanism_feature_challenger_v1.yml"
FEATURE_RUN = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e"
)
TARGETS = ROOT / (
    "artifacts/models/five_year_target_spm/"
    "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = ROOT / (
    "research/rapm_lab/outputs/rolling_5y_2014_2026/"
    "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
PRIOR_CANDIDATES = (
    "box_15",
    "box_plus_offense_mechanisms",
    "box_plus_defense_mechanisms",
    "box_plus_all_mechanisms",
)
MODEL_ORDER = (
    *PRIOR_CANDIDATES,
    "zero_prior_rapm",
    *(f"{candidate}_aio" for candidate in PRIOR_CANDIDATES),
)
PRIMARY_PAIRS = {
    frozenset(("box_15", candidate))
    for candidate in PRIOR_CANDIDATES[1:]
} | {
    frozenset(("box_15_aio", f"{candidate}_aio"))
    for candidate in PRIOR_CANDIDATES[1:]
}


def _features() -> dict[str, dict[str, tuple[str, ...]]]:
    box = tuple(BOX_PIPM_STYLE_FEATURES)
    return {
        "box_15": {"offense": box, "defense": box},
        "box_plus_offense_mechanisms": {
            "offense": (*box, *OFFENSE_MECHANISM_FEATURES),
            "defense": box,
        },
        "box_plus_defense_mechanisms": {
            "offense": box,
            "defense": (*box, *DEFENSE_MECHANISM_FEATURES),
        },
        "box_plus_all_mechanisms": {
            "offense": (*box, *OFFENSE_MECHANISM_FEATURES),
            "defense": (*box, *DEFENSE_MECHANISM_FEATURES),
        },
    }


def _fit_priors(
    panel: pd.DataFrame,
    candidates: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidates = _features() if candidates is None else candidates
    prior_rows = []
    metric_rows = []
    selection_rows = []
    models = {}
    for season in base.RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Rating season {season} lacks chronological history.")
        for candidate, sides in candidates.items():
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                features = sides[side]
                target = f"target_{side}"
                alpha = base._select_box_alpha_rolling_origin(
                    train, features, target, ALPHA_GRID
                )
                model = _fit(train, features, target, alpha)
                prediction = model.predict(test.loc[:, features])
                prior[side] = prediction
                models[(season, candidate, side)] = model
                selection_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "selected_alpha": alpha,
                        "feature_count": len(features),
                        "train_window_min": int(train["Window_End"].min()),
                        "train_window_max": int(train["Window_End"].max()),
                    }
                )
                metric_rows.append(
                    {
                        "rating_season": season,
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
                    "rating_season": season,
                    "candidate": candidate,
                    "component": "net",
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["net"].to_numpy(dtype=float),
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prior_rows.append(_prior_frame(prior, candidate))
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        models,
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-run", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()

    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")
    if tuple(contract["candidates"]) != PRIOR_CANDIDATES:
        raise ValueError("Candidate order differs from the frozen contract.")

    panel, _ = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    mechanism_path = args.mechanism_run / "five_year_features.parquet"
    mechanism = pd.read_parquet(mechanism_path)
    panel = panel.merge(
        mechanism,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    all_features = {
        feature
        for sides in _features().values()
        for features in sides.values()
        for feature in features
    }
    if panel[list(all_features)].isna().any().any():
        raise ValueError("The challenger panel contains missing selected inputs.")

    priors, target_metrics, selections, models = _fit_priors(panel)
    base.PRIOR_CANDIDATES = PRIOR_CANDIDATES
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = PRIMARY_PAIRS
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games,
        draws=args.draws,
        seed=int(contract["evaluation"]["uncertainty"]["seed"]),
    )

    sources = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "mechanism_manifest": args.mechanism_run / "run.json",
        "mechanism_features": mechanism_path,
        "base_features": FEATURE_RUN / "five_year_features.parquet",
        "base_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "candidates": list(PRIOR_CANDIDATES),
        "candidate_features": {
            candidate: {side: list(features) for side, features in sides.items()}
            for candidate, sides in _features().items()
        },
        "rating_seasons": list(base.RATING_SEASONS),
        "evaluated_rating_seasons": list(base.EVALUATED_RATING_SEASONS),
        "bootstrap_draws": args.draws,
        "sources": {
            name: {"path": _relative(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / (
        "artifacts/research/mechanism_feature_challenger/"
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "model_selection.parquet": selections,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    model_root = output / "models"
    model_root.mkdir()
    for (season, candidate, side), model in models.items():
        joblib.dump(model, model_root / f"{season}_{candidate}_{side}.joblib")

    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "missing_selected_values": 0,
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(paired.loc[paired["primary_comparison"]].to_string(index=False))


if __name__ == "__main__":
    main()
