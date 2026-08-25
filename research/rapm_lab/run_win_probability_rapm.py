"""Run conserved possession-level win-probability RAPM on 2025-26."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.possession_outcome_rapm import canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, ratings_table
from nba_impact.models.win_probability_rapm import fit_win_probability_rapm


REPO_ROOT = Path(__file__).resolve().parents[2]
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
WP_2025 = (
    REPO_ROOT
    / "artifacts/models/win_probability_possession"
    / "wp_possession_start_v2_1db472e450/test_predictions.parquet"
)
WP_2026 = (
    REPO_ROOT
    / "artifacts/models/win_probability_possession"
    / "wp_possession_start_v2_0a5d626234/test_predictions.parquet"
)
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/win_probability_rapm"


def _fit_points(frame: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    design = build_design(frame, include_home=True)
    config = RapmConfig(
        seasons=seasons,
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="wp_rapm_exact_row_points_comparator",
    )
    beta, _ = fit_coefficients(design, config)
    return ratings_table(design, beta)


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    possessions = pd.read_parquet(POSSESSIONS)
    segments = pd.read_parquet(SEGMENTS)
    base = canonical_terminal_frame(
        possessions, segments, seasons=(2025, 2026)
    )
    wp_rows = []
    for path in (WP_2025, WP_2026):
        frame = pd.read_parquet(
            path,
            columns=[
                "possession_id",
                "season_end",
                "season_type",
                "home_win",
                "probability_context",
            ],
        )
        wp_rows.append(frame.loc[frame["season_type"].eq("regular")])
    wp = pd.concat(wp_rows, ignore_index=True)
    if wp.duplicated("possession_id").any():
        raise ValueError("Cross-fitted WP rows must be unique by possession.")
    frame = base.merge(
        wp[["possession_id", "home_win", "probability_context"]],
        on="possession_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(base):
        raise ValueError("WP RAPM must score every canonical regular-season possession.")
    if frame["season"].max() >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    config = RapmConfig(
        seasons=(2025, 2026),
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="cross_fitted_player_neutral_wp_credit",
    )
    combined = fit_win_probability_rapm(frame, config)
    points = _fit_points(frame, (2025, 2026))
    comparison = combined.ratings.merge(
        points[["player_id", "offense_per_100", "defense_per_100", "net_per_100"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    correlations = {
        component: float(
            comparison[
                [f"{component}_wp_percentage_points_per_100", f"{component}_per_100"]
            ].corr().iloc[0, 1]
        )
        for component in ("offense", "defense", "net")
    }
    annual_ratings = []
    annual_quality = []
    for season in (2025, 2026):
        result = fit_win_probability_rapm(
            frame.loc[frame["season"].eq(season)].copy(),
            RapmConfig(
                seasons=(season,),
                lambda_off=3000,
                lambda_def=3000,
                lambda_home=300,
                data_scope="annual_cross_fitted_player_neutral_wp_credit",
            ),
        )
        annual_ratings.append(result.ratings.assign(season=season))
        annual_quality.append({"season": season, **result.quality})
    annual = pd.concat(annual_ratings, ignore_index=True)
    stability = annual.loc[annual["season"].eq(2025), [
        "player_id", "offense_wp_percentage_points_per_100",
        "defense_wp_percentage_points_per_100", "net_wp_percentage_points_per_100",
    ]].merge(
        annual.loc[annual["season"].eq(2026), [
            "player_id", "offense_wp_percentage_points_per_100",
            "defense_wp_percentage_points_per_100", "net_wp_percentage_points_per_100",
        ]],
        on="player_id",
        suffixes=("_2025", "_2026"),
    )
    annual_stability = {
        component: float(
            stability[
                [
                    f"{component}_wp_percentage_points_per_100_2025",
                    f"{component}_wp_percentage_points_per_100_2026",
                ]
            ].corr().iloc[0, 1]
        )
        for component in ("offense", "defense", "net")
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "wp_2025": sha256_file(WP_2025),
                "wp_2026": sha256_file(WP_2026),
                "possessions": sha256_file(POSSESSIONS),
                "segments": sha256_file(SEGMENTS),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"win_probability_rapm_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    combined.ratings.to_parquet(output / "ratings.parquet", index=False)
    combined.game_conservation.to_parquet(output / "game_conservation.parquet", index=False)
    comparison.to_parquet(output / "points_comparison.parquet", index=False)
    annual.to_parquet(output / "annual_ratings.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "model_family": "conserved_cross_fitted_win_probability_rapm",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "lineup_adjusted_change_in_player_neutral_home_win_probability_per_possession",
        "config": {
            "seasons": [2025, 2026],
            "probability_surface": "context only; no lineup or player identity",
            "target": "next possession-start probability minus current; final jump to observed result",
            "sign": "offense perspective; positive offense, defense, and net are good",
            "penalties": {"offense": 3000, "defense": 3000, "home": 300},
        },
        "quality": combined.quality,
        "annual_quality": annual_quality,
        "correlation_with_exact_row_points_rapm": correlations,
        "annual_rating_stability": annual_stability,
        "paths": {
            "ratings": str((output / "ratings.parquet").relative_to(REPO_ROOT)),
            "annual_ratings": str((output / "annual_ratings.parquet").relative_to(REPO_ROOT)),
            "game_conservation": str((output / "game_conservation.parquet").relative_to(REPO_ROOT)),
            "points_comparison": str((output / "points_comparison.parquet").relative_to(REPO_ROOT)),
        },
        "caveats": [
            "The final possession receives the jump from the last model state to the observed game result.",
            "The rating is leverage-weighted credit, not expected future player strength or total points impact.",
            "The WP surfaces are prior-season-trained, but both scored seasons are reused research diagnostics.",
            "Season 2027 is not loaded.",
        ],
        "forbidden_interpretation": "Causal individual win credit, forecast value, or a replacement for normal RAPM.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
