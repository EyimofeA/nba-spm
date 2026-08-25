#!/usr/bin/env python3
"""Lambda sweep for the winning RAPM spec, judged on rating-level metrics
(next-season margin retrodiction + one split-half repeat), not possession RMSE."""
from __future__ import annotations

import argparse
import time

import pandas as pd

from paths import STANDARD_RAPM_DIAGNOSTICS, ensure_dirs
from standard_rapm import fetch_possessions, format_season_list
from validate_ratings import make_cfg, next_season_retrodiction, split_half_reliability

ensure_dirs()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", type=int, default=2021)
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--next-season", type=int, default=2024)
    parser.add_argument("--spec", default="home")
    parser.add_argument("--lambdas", default="1000,2000,3000,5000,8000")
    parser.add_argument("--min-poss", type=int, default=2000)
    args = parser.parse_args(argv)

    train_seasons = list(range(args.train_start, args.train_end + 1))
    train_rows = fetch_possessions(train_seasons)
    next_rows = fetch_possessions([args.next_season])
    print(f"train {format_season_list(train_seasons)}: {len(train_rows):,} | next: {len(next_rows):,}", flush=True)

    results = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        t0 = time.time()
        # Reuse the arg namespace shape make_cfg expects.
        args.lambda_off = args.lambda_def = lam
        args.lambda_meta, args.lambda_season = 300.0, 100.0
        cfg = make_cfg(train_seasons, args.spec, args)
        retro = next_season_retrodiction(train_rows, next_rows, cfg)
        rel = split_half_reliability(train_rows, cfg, n_repeats=1, min_poss=args.min_poss)
        row = {"lambda": lam, **retro, **rel, "elapsed_s": round(time.time() - t0, 1)}
        results.append(row)
        print(f"lambda={lam:.0f}: next r={retro['next_margin_corr']:.3f} "
              f"rmse={retro['next_margin_rmse']:.2f} | split-half r={rel['split_half_corr']:.3f} "
              f"[{row['elapsed_s']}s]", flush=True)

    out = pd.DataFrame(results)
    path = STANDARD_RAPM_DIAGNOSTICS / f"lambda_sweep_{args.spec}_{format_season_list(train_seasons)}.csv"
    out.to_csv(path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
