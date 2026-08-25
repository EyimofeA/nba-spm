"""Previous-season RAPM persistence gate on the frozen forward folds.

For held-out season T, fit the same zero-prior coefficient specification on
T-1 only, then score the exact T games used by the bake-off. Unseen players
receive zero through the shared train/test player universe and ridge penalty.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from rapm_ridge import load_cache, load_canonical, player_universe
from tune_bakeoff import EVAL_SEASONS, SuffStats, design_with_intercept, game_eval

LAB_ROOT = Path(__file__).resolve().parent


def load_previous_season(season: int) -> pd.DataFrame:
    """Respect the lab's legacy-through-2023 and canonical-from-2024 boundary."""
    if season <= 2023:
        return load_cache([season])
    return load_canonical((season,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=EVAL_SEASONS,
        default=list(EVAL_SEASONS),
    )
    parser.add_argument(
        "--lambdas",
        nargs=2,
        type=float,
        metavar=("OFF", "DEF"),
        default=(500.0, 2000.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "outputs" / "persistence_last3.csv",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=LAB_ROOT / "outputs" / "bakeoff_last3.csv",
    )
    parser.add_argument("--candidate-model", default="ridge_tuned")
    args = parser.parse_args()

    folds = sorted(set(args.folds))
    lambda_off, lambda_def = args.lambdas
    print(
        "CONTRACT: previous-season-only ratings; identical held-out games; "
        "2024-2026 development/selection.",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists() and args.output.stat().st_size > 0:
        completed = set(pd.read_csv(args.output)["fold"].astype(int))

    for held in folds:
        if held in completed:
            print(f"  persistence end={held}: checkpointed", flush=True)
            continue

        train_season = held - 1
        train = load_previous_season(train_season)
        test = load_canonical((held,))
        players = player_universe(train, test)
        X_train, y_train, _ = design_with_intercept(train, players)
        X_test, _, test_meta = design_with_intercept(test, players)
        stats = SuffStats(X_train, y_train)
        beta = stats.beta(lambda_off, lambda_def)
        games, correlation, mae = game_eval(X_test @ beta, test_meta, test)

        row = {
            "model": "previous_season",
            "fold": held,
            "train_season": train_season,
            "games": games,
            "corr": correlation,
            "mae": mae,
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
        }
        header = not args.output.exists() or args.output.stat().st_size == 0
        pd.DataFrame([row]).to_csv(
            args.output,
            mode="a",
            index=False,
            header=header,
        )
        print(
            f"  persistence end={held}: corr={correlation:.3f} "
            f"mae={mae:.2f} games={games}  [saved]",
            flush=True,
        )
        del train, test, players, X_train, y_train, X_test, test_meta, stats, beta
        gc.collect()

    persistence = pd.read_csv(args.output)
    persistence = persistence[persistence["fold"].isin(folds)].copy()
    if set(persistence["fold"].astype(int)) != set(folds):
        raise RuntimeError("Persistence output does not cover every requested fold.")

    candidate = pd.read_csv(args.candidate)
    candidate = candidate[
        (candidate["model"] == args.candidate_model)
        & candidate["fold"].isin(folds)
    ].copy()
    if set(candidate["fold"].astype(int)) != set(folds):
        raise RuntimeError("Candidate output does not cover every requested fold.")

    comparison = candidate.merge(
        persistence,
        on="fold",
        suffixes=("_candidate", "_persistence"),
        validate="one_to_one",
    )
    if not np.array_equal(
        comparison["games_candidate"].to_numpy(),
        comparison["games_persistence"].to_numpy(),
    ):
        raise RuntimeError("Candidate and persistence game counts differ.")

    candidate_corr = float(comparison["corr_candidate"].mean())
    persistence_corr = float(comparison["corr_persistence"].mean())
    candidate_mae = float(comparison["mae_candidate"].mean())
    persistence_mae = float(comparison["mae_persistence"].mean())
    gate = "PASS" if candidate_corr > persistence_corr else "NULL"

    print("\nPERSISTENCE GATE (development/selection only):")
    print(
        comparison[
            [
                "fold",
                "games_candidate",
                "corr_candidate",
                "corr_persistence",
                "mae_candidate",
                "mae_persistence",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(
        f"\n{gate}: {args.candidate_model} mean corr {candidate_corr:.3f} "
        f"vs persistence {persistence_corr:.3f}; mean MAE {candidate_mae:.3f} "
        f"vs {persistence_mae:.3f}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
