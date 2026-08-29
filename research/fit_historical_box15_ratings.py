#!/usr/bin/env python3
"""Fit the 1997-2026 RAPM and 2001-2026 Box15 SPM/AIO research panels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.annual_aio_ratings import fit_annual_aio_season
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit,
    _select_alpha,
)
from nba_impact.models.rapm import (
    RapmConfig,
    load_current_possessions,
    load_legacy_possessions,
)
from nba_impact.models.statistical_impact import _metrics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "historical_box15_ratings_v1"
INPUT_RUN = (
    ROOT
    / "artifacts/research/historical_box15_extension"
    / "historical_box15_extension_v1_08ff4c34ff"
)
VALIDATION_RUN = (
    ROOT
    / "artifacts/research/historical_box15_validation"
    / "historical_box15_validation_v1_fa08210f64"
)
ANNUAL_TARGETS = (
    ROOT
    / "artifacts/models/canonical_annual_target_panel"
    / "canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
)
CACHE = ROOT / "rapm/data/possession_cache"
CURRENT_POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
CURRENT_SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
OUTPUT_ROOT = ROOT / "artifacts/research/historical_box15_ratings"
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _panel() -> pd.DataFrame:
    features = pd.read_parquet(INPUT_RUN / "five_year_box15_features.parquet")
    targets = pd.read_parquet(INPUT_RUN / "five_year_targets.parquet")
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def _spm_priors(panel: pd.DataFrame):
    final_models = {}
    final_alphas = {}
    for side in ("offense", "defense"):
        target = f"target_{side}"
        alpha = _select_alpha(
            panel.rename(columns={"Window_End": "Season"}),
            BOX_PIPM_STYLE_FEATURES,
            target,
            ALPHA_GRID,
        )
        final_alphas[side] = alpha
        final_models[side] = _fit(panel, BOX_PIPM_STYLE_FEATURES, target, alpha)

    rows: list[pd.DataFrame] = []
    alpha_rows: list[dict] = []
    metric_rows: list[dict] = []
    for season in range(2001, 2027):
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        prior = test[["PLAYER_ID", "Window_End"]].copy()
        if season >= 2004:
            train = panel.loc[panel["Window_End"].lt(season)].copy()
            status = "chronological"
        else:
            train = None
            status = "descriptive_full_fit_backcast"
        for side in ("offense", "defense"):
            target = f"target_{side}"
            if train is None:
                alpha = final_alphas[side]
                model = final_models[side]
                training_min = 2001
                training_max = 2026
            else:
                alpha = _select_alpha(
                    train.rename(columns={"Window_End": "Season"}),
                    BOX_PIPM_STYLE_FEATURES,
                    target,
                    ALPHA_GRID,
                )
                model = _fit(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
                training_min = int(train["Window_End"].min())
                training_max = int(train["Window_End"].max())
            prediction = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
            prior[f"prior_{side}_per_100"] = prediction
            alpha_rows.append(
                {
                    "rating_season": season,
                    "side": side,
                    "information_status": status,
                    "selected_alpha": alpha,
                    "training_window_min": training_min,
                    "training_window_max": training_max,
                }
            )
            if status == "chronological":
                metric_rows.append(
                    {
                        "rating_season": season,
                        "component": side,
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prediction,
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
        prior["prior_net_per_100"] = (
            prior["prior_offense_per_100"] + prior["prior_defense_per_100"]
        )
        prior["information_status"] = status
        if status == "chronological":
            metric_rows.append(
                {
                    "rating_season": season,
                    "component": "net",
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["prior_net_per_100"].to_numpy(dtype=float),
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
        rows.append(prior)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(alpha_rows), pd.DataFrame(metric_rows)


def _unified_rapm() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = pd.read_parquet(INPUT_RUN / "historical_annual_rapm.parquet")
    current = pd.read_parquet(ANNUAL_TARGETS)
    checks = []
    for season in range(2014, 2019):
        left = historical.loc[
            historical["Season"].eq(season), ["PLAYER_ID", "offense", "defense", "net"]
        ]
        right = current.loc[
            current["Season"].eq(season),
            ["PLAYER_ID", "target_offense", "target_defense", "target_net"],
        ]
        matched = left.merge(right, on="PLAYER_ID", validate="one_to_one")
        for component in ("offense", "defense", "net"):
            checks.append(
                {
                    "season": season,
                    "component": component,
                    "matched_players": len(matched),
                    "maximum_absolute_error": float(
                        (
                            matched[component] - matched[f"target_{component}"]
                        ).abs().max()
                    ),
                }
            )
    overlap = pd.DataFrame(checks)
    if overlap["maximum_absolute_error"].max() > 1e-8:
        raise ValueError("Historical and canonical annual RAPM overlap differs.")
    old = historical.loc[historical["Season"].le(2013)].rename(
        columns={
            "offense": "rapm_offense",
            "defense": "rapm_defense",
            "net": "rapm_net",
        }
    )[
        ["PLAYER_ID", "Season", "rapm_offense", "rapm_defense", "rapm_net", "Poss_Off", "Poss_Def"]
    ]
    new = current.rename(
        columns={
            "target_offense": "rapm_offense",
            "target_defense": "rapm_defense",
            "target_net": "rapm_net",
        }
    )[
        ["PLAYER_ID", "Season", "rapm_offense", "rapm_defense", "rapm_net", "Poss_Off", "Poss_Def"]
    ]
    return pd.concat([old, new], ignore_index=True), overlap


def _aio(priors: pd.DataFrame):
    rows: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    current = load_current_possessions(
        CURRENT_POSSESSIONS,
        CURRENT_SEGMENTS,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    for season in range(2001, 2027):
        if season < 2024:
            frame = load_legacy_possessions(CACHE, (season,), game_types=("regular",))
        else:
            frame = current.loc[current["season"].eq(season)].copy()
        config = RapmConfig(
            seasons=(season,),
            lambda_off=3000,
            lambda_def=3000,
            lambda_home=300,
            game_types=("regular",),
            data_scope="historical_box15_annual_aio",
        )
        ratings, quality = fit_annual_aio_season(
            frame,
            priors,
            config,
            season=season,
        )
        status = str(
            priors.loc[priors["Window_End"].eq(season), "information_status"].iloc[0]
        )
        ratings["information_status"] = status
        rows.append(ratings)
        quality_rows.append({"information_status": status, **quality})
        print(f"AIO {season}: fitted", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(quality_rows)


def main() -> None:
    validation = json.loads((VALIDATION_RUN / "run.json").read_text())
    if not validation["quality"]["retention_gate_passed"]:
        raise ValueError("Historical Box15 expansion failed its retention gate.")
    panel = _panel()
    priors, alphas, target_metrics = _spm_priors(panel)
    rapm, rapm_overlap = _unified_rapm()
    aio, aio_quality = _aio(priors)
    if sorted(rapm["Season"].unique()) != list(range(1997, 2027)):
        raise ValueError("Unified RAPM does not cover 1997-2026.")
    if sorted(aio["Season"].unique()) != list(range(2001, 2027)):
        raise ValueError("Historical AIO does not cover 2001-2026.")

    source_paths = {
        "input_run": INPUT_RUN / "run.json",
        "validation_run": VALIDATION_RUN / "run.json",
        "annual_targets": ANNUAL_TARGETS,
        "current_possessions": CURRENT_POSSESSIONS,
        "current_segments": CURRENT_SEGMENTS,
        "runner": Path(__file__),
        **{
            f"legacy_possessions_{season}": CACHE / f"matchups_{season}.parquet"
            for season in range(2001, 2024)
        },
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "rapm_ratings.parquet": rapm,
        "spm_priors.parquet": priors,
        "aio_ratings.parquet": aio,
        "spm_alpha_selection.parquet": alphas,
        "spm_chronological_target_metrics.parquet": target_metrics,
        "annual_rapm_overlap_validation.parquet": rapm_overlap,
        "aio_season_quality.parquet": aio_quality,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    identity_error = float(
        (aio["aio_offense"] + aio["aio_defense"] - aio["aio_net"]).abs().max()
    )
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_leaderboard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimands": {
            "rapm": "single-season retrospective impact, 1997-2026",
            "spm": "five-year Box15 prior, complete windows 2001-2026",
            "aio": "Box15 prior updated by same-season single-season RAPM, 2001-2026",
        },
        "quality": {
            "retention_gate_passed": True,
            "rapm_seasons": [1997, 2026],
            "spm_and_aio_seasons": [2001, 2026],
            "descriptive_backcast_seasons": [2001, 2003],
            "chronological_spm_seasons": [2004, 2026],
            "maximum_component_identity_error": identity_error,
            "minimum_aio_prior_player_coverage": float(
                aio_quality["player_prior_coverage"].min()
            ),
            "season_2027_loaded": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashes[name],
            }
            for name, path in source_paths.items()
        },
        "files": {},
        "forbidden_interpretation": (
            "The 2001-03 SPM priors are descriptive full-fit backcasts. The "
            "2004-26 priors are chronological research ratings. No scored "
            "season is untouched confirmation, and Season 2027 was not used."
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
    print(json.dumps(run["quality"], indent=2))


if __name__ == "__main__":
    main()
