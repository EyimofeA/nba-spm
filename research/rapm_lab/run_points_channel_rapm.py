"""Run one conserved scoring-channel RAPM pilot on the latest five-year window."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.points_channel_rapm import fit_points_channel_rapm
from nba_impact.models.rapm import (
    RapmConfig,
    load_current_player_names,
    load_unified_terminal_possessions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_points_channel_rapm_v1.json"
)
DEFAULT_CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
DEFAULT_PLAYER_GAMES = REPO_ROOT / "data" / "lake" / "silver" / "player_games.parquet"
DEFAULT_NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "points_channel_rapm"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def run_points_channel_pilot(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Points-channel RAPM requires a frozen research contract.")
    start, end = (int(value) for value in contract["window"])
    seasons = tuple(range(start, end + 1))
    if 2027 in seasons:
        raise ValueError("Season 2027 must remain untouched.")
    source_paths = [
        *(DEFAULT_CACHE / f"matchups_{season}.parquet" for season in seasons if season < 2024),
        DEFAULT_POSSESSIONS,
        DEFAULT_SEGMENTS,
        DEFAULT_PLAYER_GAMES,
        DEFAULT_NAMES,
    ]
    source_hashes = {_relative(path): sha256_file(path) for path in source_paths}
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "points_channel_rapm.py"
        ),
        "sources": source_hashes,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"points_channel_rapm_v1_{identity}"
    output = output_root / run_id
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)

    frame = load_unified_terminal_possessions(
        DEFAULT_CACHE,
        DEFAULT_POSSESSIONS,
        DEFAULT_SEGMENTS,
        seasons,
        game_types=("regular",),
    )
    names = load_current_player_names(DEFAULT_NAMES, DEFAULT_PLAYER_GAMES)
    model = contract["model"]
    result = fit_points_channel_rapm(
        frame,
        RapmConfig(
            seasons=seasons,
            lambda_off=float(model["lambda_off"]),
            lambda_def=float(model["lambda_def"]),
            lambda_home=float(model["lambda_home"]),
        ),
        names=names,
    )
    ratings = result.ratings
    ratings.to_parquet(output / "ratings.parquet", index=False)
    gates = {
        "target_recomposition": result.quality["maximum_target_recomposition_error"]
        <= float(contract["acceptance"]["maximum_target_recomposition_error"]),
        "rating_recomposition": result.quality["maximum_rating_recomposition_error"]
        <= float(contract["acceptance"]["maximum_rating_recomposition_error"]),
        "channel_net_identity": result.quality["maximum_channel_net_identity_error"]
        <= float(contract["acceptance"]["maximum_channel_net_identity_error"]),
        "canonical_rapm_match": result.quality["maximum_canonical_rapm_error"]
        <= float(contract["acceptance"]["maximum_canonical_rapm_error"]),
        "intercept_recomposition": result.quality["intercept_recomposition_error"]
        <= float(contract["acceptance"]["maximum_intercept_recomposition_error"]),
        "unique_player_ids": bool(not ratings["PLAYER_ID"].duplicated().any()),
        "complete_names": bool(ratings["PLAYER_NAME"].notna().all()),
    }
    if not all(gates.values()):
        raise ValueError(f"Points-channel RAPM quality gates failed: {gates}")
    qualified = ratings.loc[
        ratings[["Poss_Off", "Poss_Def"]].min(axis=1).ge(5000)
    ].copy()
    leaderboards = {}
    for column in ("one_point_net", "two_point_net", "three_plus_net", "net"):
        leaderboards[column] = qualified.nlargest(10, column)[
            ["PLAYER_ID", "PLAYER_NAME", column]
        ].to_dict("records")
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_diagnostic",
        "estimand": contract["estimand"],
        "evidence_status": contract["evidence_status"],
        "forbidden_interpretation": contract["forbidden_interpretation"],
        "config": contract["model"],
        "quality": {**result.quality, "gates": gates},
        "channel_intercepts": result.channel_intercepts,
        "channel_home_effects_per_100": result.channel_home_effects_per_100,
        "leaderboards_minimum_5000_possessions_per_side": leaderboards,
        "hashes": identity_payload,
        "paths": {
            "artifact": _relative(output),
            "ratings": _relative(output / "ratings.parquet"),
        },
    }
    write_json_atomic(run, run_path)
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = run_points_channel_pilot(args.contract, output_root=args.output_root)
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
