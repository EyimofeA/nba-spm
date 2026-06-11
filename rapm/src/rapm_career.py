"""rapm_career.py — run 1-year RAPM for each season and build a career long-table.

This drives `rapm_core.run()` in a loop, one season at a time, then stitches
the resulting per-season leaderboards into a single long table:

    PLAYER_ID, Name, Season, Age, MPG, Tier, Off, Def, RAPM, VORP,
    Poss_Off, Poss_Def, [Off_SE, Def_SE, RAPM_SE if std errors enabled]

`Age` is pulled from `data/raw/playersheets/year_totals/{season}.csv`.

Typical use:

    python rapm/src/rapm_career.py \
        --start-season 2003 --end-season 2024 \
        --search-mode grid \
        --grid-off 500,1500,3000 --grid-def 500,1500,3000 \
        --grid-cv-folds 3 \
        --lam-home 200 --lam-rb 1500 --lam-playoff 500 \
        --no-std-errors

Skip `--no-std-errors` if you want per-season error bars in the career table
(slower). Use `--search-mode ridgecv` for quick sanity passes.

Output → `rapm/outputs/career/career_rapm_1year.csv`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from paths import (
    CAREER_META_JSON,
    CAREER_RAPM_CSV,
    PLAYERSHEETS_YEAR_TOTALS,
    ensure_dirs,
)
from rapm_core import RunConfig, TierEdges, run as rapm_core_run


# =============================================================================
# Helpers
# =============================================================================
def load_ages_for_season(season: int) -> dict[int, int]:
    """Return {PLAYER_ID -> AGE} for the given season.

    We prefer the regular-season totals file. Age is the listed age for that
    season; players who play for multiple teams appear with their max-minutes
    row (we take the first and trust nba_api's roll-up).
    """
    path = PLAYERSHEETS_YEAR_TOTALS / f"{season}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
    df = df.dropna(subset=["PLAYER_ID", "AGE"])
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["AGE"] = df["AGE"].astype(int)
    return dict(zip(df["PLAYER_ID"], df["AGE"]))


def collect_season_df(result_path: Path, season: int) -> pd.DataFrame:
    """Load a `Core_Rapm_*.csv` and return a trimmed frame for the career table."""
    df = pd.read_csv(result_path)
    df["Season"] = season
    ages = load_ages_for_season(season)
    df["Age"] = df["PLAYER_ID"].map(ages)
    keep = [
        "Season", "Age", "PLAYER_ID", "Name", "MPG", "Tier",
        "Off", "Def", "RAPM", "VORP_Off", "VORP_Def", "VORP",
        "Poss_Off", "Poss_Def",
    ]
    opt = ["Off_SE", "Def_SE", "RAPM_SE",
           "Off_CI_lo", "Off_CI_hi", "Def_CI_lo", "Def_CI_hi",
           "RAPM_CI_lo", "RAPM_CI_hi"]
    keep += [c for c in opt if c in df.columns]
    return df[keep]


# =============================================================================
# Main loop
# =============================================================================
def run_career(args: argparse.Namespace) -> None:
    ensure_dirs()
    seasons = list(range(args.start_season, args.end_season + 1))

    def _parse_grid(s: str) -> tuple[float, ...]:
        return tuple(float(x.strip()) for x in s.split(",") if x.strip())

    summaries: list[dict] = []
    frames: list[pd.DataFrame] = []
    t0 = time.time()

    for i, season in enumerate(seasons, start=1):
        print(f"\n{'=' * 72}")
        print(f"[career {i}/{len(seasons)}] season={season}  elapsed={time.time() - t0:.0f}s")
        print('=' * 72)

        cfg = RunConfig(
            seasons=(season,),
            window_label="1",
            suffix=f"career_{season}",
            use_home=args.use_home,
            use_rubberband=args.use_rubberband,
            use_playoff=args.use_playoff,
            use_b2b=args.use_b2b,
            use_coach=False,
            search_mode=args.search_mode,
            cv_folds=args.cv_folds,
            cv_method=args.cv_method,
            alpha_grid=_parse_grid(args.alpha_grid),
            ratio_home=args.ratio_home,
            ratio_rb=args.ratio_rb,
            ratio_playoff=args.ratio_playoff,
            ratio_b2b=args.ratio_b2b,
            grid_off=_parse_grid(args.grid_off),
            grid_def=_parse_grid(args.grid_def),
            grid_cv_folds=args.grid_cv_folds,
            lam_home=args.lam_home,
            lam_rb=args.lam_rb,
            lam_playoff=args.lam_playoff,
            lam_b2b=args.lam_b2b,
            lam_coach=args.lam_coach,
            replacement_mode=args.replacement_shrinkage,
            tier_edges=TierEdges(edges=_parse_grid(args.mpg_edges)),
            run_diagnostics=False,       # per-season diagnostics would be noisy
            compute_std_errors=args.compute_std_errors,
        )

        try:
            summary = rapm_core_run(cfg)
        except Exception as e:                               # noqa: BLE001
            print(f"  [SKIP] season {season} failed: {e}")
            continue

        result_path = Path(summary["result_path"])
        df = collect_season_df(result_path, season)
        print(f"  season {season}: {len(df):,} players")
        frames.append(df)
        summaries.append({
            "season": season,
            "run_id": summary["run_id"],
            "best_lambdas": summary["best_lambdas"],
            "result_path": str(result_path),
            "n_players": int(len(df)),
        })

    if not frames:
        print("\nNo seasons produced output — aborting before write.")
        sys.exit(1)

    career = pd.concat(frames, ignore_index=True)
    career = career.sort_values(["Season", "RAPM"], ascending=[True, False])
    career.to_csv(CAREER_RAPM_CSV, index=False)
    CAREER_META_JSON.write_text(json.dumps({
        "seasons": seasons,
        "runs": summaries,
        "n_rows": int(len(career)),
        "n_players": int(career["PLAYER_ID"].nunique()),
    }, indent=2, default=float))

    print(f"\n\nCareer table → {CAREER_RAPM_CSV}")
    print(f"  rows: {len(career):,}   unique players: {career['PLAYER_ID'].nunique():,}")
    print(f"  seasons covered: {sorted(career['Season'].unique().tolist())}")
    print(f"  meta → {CAREER_META_JSON}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 1-year RAPM per season and build a career table.")
    p.add_argument("--start-season", type=int, required=True)
    p.add_argument("--end-season", type=int, required=True)

    # Meta toggles (mirror rapm_core)
    p.add_argument("--home", dest="use_home", action="store_true", default=True)
    p.add_argument("--no-home", dest="use_home", action="store_false")
    p.add_argument("--rubberband", dest="use_rubberband", action="store_true", default=True)
    p.add_argument("--no-rubberband", dest="use_rubberband", action="store_false")
    p.add_argument("--playoff", dest="use_playoff", action="store_true", default=True)
    p.add_argument("--no-playoff", dest="use_playoff", action="store_false")
    p.add_argument("--b2b", dest="use_b2b", action="store_true", default=False)

    # Search mode
    p.add_argument("--search-mode", choices=["ridgecv", "grid"], default="grid",
                   help="Usually 'grid' once you know your preferred off/def grids.")

    # RidgeCV path
    p.add_argument("--alpha-grid", default="500,1000,2000,3000,5000")
    p.add_argument("--cv-method", choices=["gcv", "kfold"], default="gcv")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--ratio-home", type=float, default=0.05)
    p.add_argument("--ratio-rb", type=float, default=0.50)
    p.add_argument("--ratio-playoff", type=float, default=0.25)
    p.add_argument("--ratio-b2b", type=float, default=0.25)

    # Grid path (always off/def only — meta fixed)
    p.add_argument("--grid-off", default="1000,2000,3000")
    p.add_argument("--grid-def", default="1000,2000,3000")
    p.add_argument("--grid-cv-folds", type=int, default=3)
    p.add_argument("--lam-home", type=float, default=200.0)
    p.add_argument("--lam-rb", type=float, default=1500.0)
    p.add_argument("--lam-playoff", type=float, default=500.0)
    p.add_argument("--lam-b2b", type=float, default=500.0)
    p.add_argument("--lam-coach", type=float, default=2000.0)

    # Replacement
    p.add_argument("--replacement-shrinkage", choices=["off", "tier", "uniform"], default="tier")
    p.add_argument("--mpg-edges", default="5,10,15,20,25,30,35,40")

    # Standard errors — off by default for career runs (faster; age adjustment
    # doesn't need them for the aging curve itself).
    p.add_argument("--std-errors", dest="compute_std_errors", action="store_true", default=False)
    p.add_argument("--no-std-errors", dest="compute_std_errors", action="store_false")

    return p.parse_args(argv)


if __name__ == "__main__":
    run_career(parse_args())
