"""Command line entry point for the new data/model spine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_impact.data.download import run_ingest_manifest
from nba_impact.data.event_quality import build_event_snapshot
from nba_impact.data.manifest import build_possession_snapshot, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    load_legacy_possessions,
    run_rapm,
    run_regularization_comparison,
)
from nba_impact.paths import (
    ARTIFACT_ROOT,
    BRONZE_ROOT,
    LEGACY_POSSESSION_CACHE,
    MANIFEST_ROOT,
    PLAYER_NAMES,
    REGISTRY_PATH,
    ensure_owned_dirs,
)
from nba_impact.registry import register_model_run, register_snapshot


def _season_list(value: str) -> tuple[int, ...]:
    seasons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seasons:
        raise argparse.ArgumentTypeError("provide at least one comma-separated season")
    return seasons


def command_audit(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = list(args.seasons)
    snapshot = build_possession_snapshot(args.cache_dir, seasons)
    destination = Path(args.output or MANIFEST_ROOT / f"{snapshot['snapshot_id']}.json")
    write_json_atomic(snapshot, destination)
    register_snapshot(args.registry, snapshot)
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "passed": snapshot["passed"], "path": str(destination)}, indent=2))
    for item in snapshot["files"]:
        quality = item["quality"]
        print(
            f"{item['season']}: rows={quality['row_count']:,} games={quality['game_count']:,} "
            f"dates={quality['date_min']}..{quality['date_max']} passed={quality['passed']}"
        )
        for issue in quality["issues"]:
            print(f"  {issue['severity']:>8} {issue['code']}: {issue['count']} — {issue['message']}")
    return 0 if snapshot["passed"] else 2


def command_ingest(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    summary = run_ingest_manifest(args.manifest, root=args.root)
    print(json.dumps({key: summary[key] for key in ("succeeded", "failed", "completed_at")}, indent=2))
    return 0


def command_audit_events(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_event_snapshot(args.root)
    destination = Path(args.output or MANIFEST_ROOT / f"{snapshot['snapshot_id']}.json")
    write_json_atomic(snapshot, destination)
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "files": len(snapshot["files"]),
                "path": str(destination),
            },
            indent=2,
        )
    )
    for item in snapshot["files"]:
        print(
            f"{item['source']:<12} {item['season']} {item['season_type']:<8} "
            f"rows={item['row_count']:,} games={item['game_count']:,} passed={item['passed']}"
        )
    for item in snapshot["reconciliation"]:
        if not item["passed"]:
            print(
                f"  reconcile {item['season']} {item['season_type']} {item['source']}: "
                f"missing={item['missing_games']} extra={item['extra_games']}"
            )
    return 0 if snapshot["passed"] else 2


def command_fit_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = tuple(args.seasons)
    game_types = tuple(item.strip() for item in args.game_types.split(",") if item.strip())
    frame = load_legacy_possessions(args.cache_dir, seasons, game_types=game_types)
    names = pd.read_csv(args.names) if Path(args.names).exists() else None
    config = RapmConfig(
        seasons=seasons,
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
        include_home=not args.no_home,
        game_types=game_types,
    )
    run = run_rapm(frame, config, artifact_root=args.artifact_root, names=names)
    run["dataset_snapshot_id"] = args.snapshot_id
    write_json_atomic(run, Path(run["artifact_path"]) / "run.json")
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def _lambda_pairs(value: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    try:
        for item in value.split(","):
            offense, defense = item.strip().split(":", maxsplit=1)
            pairs.append((float(offense), float(defense)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated offense:def pairs") from exc
    if not pairs or any(offense <= 0 or defense <= 0 for offense, defense in pairs):
        raise argparse.ArgumentTypeError("lambda pairs must be positive")
    return tuple(pairs)


def command_compare_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = tuple(args.seasons)
    game_types = tuple(item.strip() for item in args.game_types.split(",") if item.strip())
    frame = load_legacy_possessions(args.cache_dir, seasons, game_types=game_types)
    config = RapmConfig(
        seasons=seasons,
        lambda_home=args.lambda_home,
        include_home=not args.no_home,
        game_types=game_types,
    )
    run = run_regularization_comparison(
        frame,
        config,
        args.lambda_pairs,
        artifact_root=args.artifact_root,
    )
    run["dataset_snapshot_id"] = args.snapshot_id
    write_json_atomic(run, Path(run["artifact_path"]) / "run.json")
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-impact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-possessions", help="Audit and register legacy possession partitions.")
    audit.add_argument("--seasons", type=_season_list, default=tuple(range(1997, 2026)))
    audit.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    audit.set_defaults(func=command_audit)

    ingest = subparsers.add_parser("ingest", help="Run a resumable HTTP ingestion manifest.")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--root", type=Path, default=BRONZE_ROOT)
    ingest.set_defaults(func=command_ingest)

    event_audit = subparsers.add_parser("audit-events", help="Audit and reconcile downloaded event sources.")
    event_audit.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    event_audit.add_argument("--output", type=Path)
    event_audit.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    event_audit.set_defaults(func=command_audit_events)

    rapm = subparsers.add_parser("fit-rapm", help="Fit the independent zero-prior RAPM baseline.")
    rapm.add_argument("--seasons", type=_season_list, required=True)
    rapm.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    rapm.add_argument("--names", type=Path, default=PLAYER_NAMES)
    rapm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    rapm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    rapm.add_argument("--snapshot-id")
    rapm.add_argument("--lambda-off", type=float, default=3000.0)
    rapm.add_argument("--lambda-def", type=float, default=3000.0)
    rapm.add_argument("--lambda-home", type=float, default=300.0)
    rapm.add_argument("--game-types", default="regular")
    rapm.add_argument("--no-home", action="store_true")
    rapm.set_defaults(func=command_fit_rapm)

    compare = subparsers.add_parser(
        "compare-rapm", help="Compare fixed RAPM penalties on one chronological diagnostic fold."
    )
    compare.add_argument("--seasons", type=_season_list, required=True)
    compare.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    compare.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    compare.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    compare.add_argument("--snapshot-id")
    compare.add_argument(
        "--lambda-pairs",
        type=_lambda_pairs,
        default=_lambda_pairs("500:500,1000:1000,2000:2000,3000:3000,6000:6000,2000:4500"),
    )
    compare.add_argument("--lambda-home", type=float, default=300.0)
    compare.add_argument("--game-types", default="regular")
    compare.add_argument("--no-home", action="store_true")
    compare.set_defaults(func=command_compare_rapm)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
