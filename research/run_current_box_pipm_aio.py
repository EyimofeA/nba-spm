#!/usr/bin/env python3
"""Fit the current BoxPIPM-style prior and use it in the 2026 AIO update.

This is a research leaderboard. It does not claim to reproduce Jacob
Goldstein's full PIPM, which also used luck-adjusted on/off information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit as _fit_box,
    _select_alpha,
)
from nba_impact.models.rapm import load_legacy_possessions
from run_aio_prior_bakeoff import ALPHA_GRID, _prior_frame
from run_aio_prior_canonical_followup import (
    _annual_from_frame,
    _center,
    _recover_annual,
    _solve,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "current_box_pipm_aio_v1"
RATING_SEASON = 2026


def _five_year_box_features(source_root: Path) -> pd.DataFrame:
    loaded = {
        season: _load_source(source_root / f"{season}.parquet", season)[0]
        for season in range(2014, RATING_SEASON + 1)
    }
    return pd.concat(
        [
            _aggregate_window(
                [loaded[season] for season in range(end - 4, end + 1)], end
            )
            for end in range(2018, RATING_SEASON + 1)
        ],
        ignore_index=True,
    )


def _box_prior(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    train = panel.loc[panel["Window_End"].lt(RATING_SEASON)].copy()
    test = panel.loc[panel["Window_End"].eq(RATING_SEASON)].copy()
    if train["Window_End"].nunique() < 3 or test.empty:
        raise ValueError("The current BoxPIPM-style fold lacks chronological history.")
    prior = test[["PLAYER_ID", "Window_End"]].copy()
    selected_alpha = {}
    for side in ("offense", "defense"):
        target = f"target_{side}"
        alpha = _select_alpha(
            train.rename(columns={"Window_End": "Season"}),
            BOX_PIPM_STYLE_FEATURES,
            target,
            ALPHA_GRID,
        )
        model = _fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
        prior[side] = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
        selected_alpha[side] = float(alpha)
    prior["net"] = prior["offense"] + prior["defense"]
    return _prior_frame(prior, "box_pipm_style_prior"), selected_alpha


def _annual_2026(matrix_root: Path, legacy_cache: Path):
    annual = {}
    for season in range(2020, 2024):
        frame = load_legacy_possessions(
            legacy_cache, (season,), game_types=("regular",)
        )
        annual[season] = _annual_from_frame(frame, season)
    reconstruction = []
    for season in (2024, 2025, 2026):
        annual[season], quality = _recover_annual(
            matrix_root / f"5y_end_{season}", season, annual
        )
        reconstruction.append(quality)
    return annual[2026], reconstruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument(
        "--player-sheet-root",
        type=Path,
        default=(
            ROOT
            / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
        ),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=(
            ROOT
            / "research/rapm_lab/outputs/rolling_5y_2014_2026"
            / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=(
            ROOT
            / "artifacts/models/five_year_target_spm"
            / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
        ),
    )
    parser.add_argument(
        "--reference-aio",
        type=Path,
        default=(
            ROOT
            / "artifacts/models/five_year_target_spm"
            / "five_year_target_spm_v1_65550acb79/aio_ratings.parquet"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    features = _five_year_box_features(args.player_sheet_root)
    targets = pd.read_parquet(args.targets)
    prior, selected_alpha = _box_prior(features, targets)
    annual, reconstruction = _annual_2026(args.matrix_root, args.legacy_cache)
    center, coverage = _center(prior, annual)
    beta, intercept = _solve(annual, center, scale=1.0)

    n = len(annual.players)
    ratings = pd.DataFrame(
        {
            "PLAYER_ID": annual.players,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": annual.off_possessions,
            "Poss_Def": annual.def_possessions,
        }
    )
    ratings["net"] = ratings["offense"] + ratings["defense"]
    current = pd.read_parquet(args.player_sheet_root / "2026.parquet")
    names = (
        current.sort_values(["PLAYER_ID", "POSS"], kind="stable")
        .drop_duplicates("PLAYER_ID", keep="last")
        [["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]
    )
    ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
    ratings["rating_season"] = RATING_SEASON
    ratings["candidate"] = "box_pipm_style_aio"
    ratings["min_possessions"] = np.minimum(ratings["Poss_Off"], ratings["Poss_Def"])
    ratings["rank"] = pd.Series(pd.NA, index=ratings.index, dtype="Int64")
    active = ratings["min_possessions"].gt(0)
    ratings.loc[active, "rank"] = (
        ratings.loc[active, "net"].rank(method="min", ascending=False).astype(int)
    )
    ratings["rank_500"] = pd.Series(pd.NA, index=ratings.index, dtype="Int64")
    eligible = ratings["min_possessions"].ge(500)
    ratings.loc[eligible, "rank_500"] = (
        ratings.loc[eligible, "net"].rank(method="min", ascending=False).astype(int)
    )
    ratings = ratings.sort_values(["net", "PLAYER_ID"], ascending=[False, True])

    reference = pd.read_parquet(args.reference_aio)
    reference = reference.loc[
        reference["rating_season"].eq(RATING_SEASON)
        & reference["candidate"].isin(["zero_prior_rapm", "five_year_target_aio"])
    ]
    comparisons = []
    for candidate, frame in reference.groupby("candidate"):
        matched = ratings.merge(
            frame[["PLAYER_ID", "offense", "defense", "net"]],
            on="PLAYER_ID",
            suffixes=("_box_aio", "_reference"),
            validate="one_to_one",
        )
        exposure = np.minimum(
            matched["Poss_Off"].to_numpy(), matched["Poss_Def"].to_numpy()
        )
        matched = matched.loc[exposure >= 1000]
        for side in ("offense", "defense", "net"):
            comparisons.append(
                {
                    "reference": candidate,
                    "component": side,
                    "players": len(matched),
                    "pearson": float(
                        matched[
                            [f"{side}_box_aio", f"{side}_reference"]
                        ].corr().iloc[0, 1]
                    ),
                    "spearman": float(
                        matched[
                            [f"{side}_box_aio", f"{side}_reference"]
                        ].corr(method="spearman").iloc[0, 1]
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)

    config = {
        "rating_season": RATING_SEASON,
        "box_features": list(BOX_PIPM_STYLE_FEATURES),
        "selected_alpha": selected_alpha,
        "aio": {
            "lambda_off": 3000.0,
            "lambda_def": 3000.0,
            "lambda_home": 300.0,
            "center_scale": 1.0,
            "likelihood_seasons": 1,
        },
        "source_hashes": {
            "targets": sha256_file(args.targets),
            "reference_aio": sha256_file(args.reference_aio),
            "runner": sha256_file(Path(__file__)),
            "matrix_2026_manifest": sha256_file(
                args.matrix_root / "5y_end_2026/manifest.json"
            ),
            "player_sheets": {
                str(season): sha256_file(args.player_sheet_root / f"{season}.parquet")
                for season in range(2014, RATING_SEASON + 1)
            },
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research/current_box_pipm_aio"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    prior.to_parquet(output / "prior.parquet", index=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    comparison_frame.to_parquet(output / "comparisons.parquet", index=False)
    pd.DataFrame(reconstruction).to_parquet(
        output / "matrix_reconstruction.parquet", index=False
    )
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_leaderboard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "ratings": len(ratings),
            "players_with_names": int(ratings["PLAYER_NAME"].notna().sum()),
            "active_players": int(active.sum()),
            "active_players_with_names": int(
                ratings.loc[active, "PLAYER_NAME"].notna().sum()
            ),
            "component_identity_max_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "prior_coverage": coverage,
            "matrix_reconstruction": reconstruction,
            "season_2027_rows": 0,
        },
        "paths": {
            "prior": "prior.parquet",
            "ratings": "ratings.parquet",
            "comparisons": "comparisons.parquet",
            "matrix_reconstruction": "matrix_reconstruction.parquet",
        },
        "caveats": [
            "BoxPIPM-style is the box-only reproducible baseline, not full PIPM.",
            "The 2026 leaderboard has no Season 2027 predictive evaluation.",
            "The prior uses a rolling five-year box feature window; the RAPM likelihood uses only 2026 possessions.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(
        ratings.loc[ratings["min_possessions"].ge(1000)]
        .sort_values("net", ascending=False)
        .head(25)
        .to_string(index=False)
    )
    print(comparison_frame.to_string(index=False))
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
