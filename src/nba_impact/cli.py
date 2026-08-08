"""Command line entry point for the new data/model spine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_impact.data.download import run_ingest_manifest
from nba_impact.data.event_quality import build_event_snapshot
from nba_impact.data.event_state import build_event_states
from nba_impact.data.game_dim import build_game_dimension
from nba_impact.data.lineups import build_lineup_stints
from nba_impact.data.manifest import build_possession_snapshot, write_json_atomic
from nba_impact.data.player_game import build_player_games
from nba_impact.models.rapm import (
    RapmConfig,
    load_legacy_possessions,
    run_rapm,
    run_regularization_comparison,
    run_walk_forward_comparison,
)
from nba_impact.models.win_probability import run_win_probability
from nba_impact.paths import (
    ARTIFACT_ROOT,
    BRONZE_ROOT,
    LEGACY_POSSESSION_CACHE,
    MANIFEST_ROOT,
    PLAYER_NAMES,
    REGISTRY_PATH,
    SILVER_ROOT,
    ensure_owned_dirs,
)
from nba_impact.registry import register_model_run, register_snapshot


def _season_list(value: str) -> tuple[int, ...]:
    seasons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seasons:
        raise argparse.ArgumentTypeError("provide at least one comma-separated season")
    return seasons


def _text_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return items


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


def command_build_game_dim(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_game_dimension(args.root, args.output, args.manifest_dir)
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "season_labels": snapshot["season_labels"],
                "coverage": snapshot["source_coverage"],
                "issues": snapshot["issues"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_build_event_states(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_event_states(args.root, args.game_dim, args.output, args.manifest_dir)
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "games": snapshot["game_count"],
                "issues": snapshot["issues"],
                "warnings": snapshot["warnings"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_build_player_games(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_player_games(
        args.box,
        args.espn,
        args.game_dim,
        args.output,
        args.manifest_dir,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "games": snapshot["game_count"],
                "espn_games": snapshot["espn_game_count"],
                "issues": snapshot["issues"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_build_lineups(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_lineup_stints(
        args.root,
        args.player_games,
        args.game_dim,
        args.output,
        args.quality_output,
        args.manifest_dir,
        minute_tolerance_seconds=args.minute_tolerance_seconds,
        max_quarantine_fraction=args.max_quarantine_fraction,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "stints": snapshot["row_count"],
                "passed_games": snapshot["passed_game_count"],
                "quarantined_games": snapshot["quarantined_game_count"],
                "quarantine_fraction": snapshot["quarantine_fraction"],
                "source_counts": snapshot["source_counts"],
                "issues": snapshot["issues"],
                "path": snapshot["path"],
                "quality_path": snapshot["quality_path"],
            },
            indent=2,
        )
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


def command_walk_forward_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    test_seasons = tuple(args.test_seasons)
    season_start = min(test_seasons) - args.train_window
    seasons = tuple(range(season_start, max(test_seasons) + 1))
    game_types = tuple(item.strip() for item in args.game_types.split(",") if item.strip())
    frame = load_legacy_possessions(args.cache_dir, seasons, game_types=game_types)
    config = RapmConfig(
        seasons=seasons,
        lambda_home=args.lambda_home,
        include_home=not args.no_home,
        game_types=game_types,
    )
    run = run_walk_forward_comparison(
        frame,
        config,
        args.lambda_pairs,
        test_seasons,
        train_window=args.train_window,
        artifact_root=args.artifact_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    run["dataset_snapshot_id"] = args.snapshot_id
    write_json_atomic(run, Path(run["artifact_path"]) / "run.json")
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_fit_win_probability(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability(
        args.event_states,
        train_season_labels=args.train_seasons,
        test_season_labels=args.test_seasons,
        artifact_root=args.artifact_root,
        interval_seconds=args.interval_seconds,
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

    game_dim = subparsers.add_parser("build-game-dim", help="Build the canonical silver game dimension.")
    game_dim.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    game_dim.add_argument("--output", type=Path, default=SILVER_ROOT / "game_dim.parquet")
    game_dim.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    game_dim.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    game_dim.set_defaults(func=command_build_game_dim)

    event_states = subparsers.add_parser(
        "build-event-states", help="Build canonical post-action score states from V3 events."
    )
    event_states.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    event_states.add_argument("--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet")
    event_states.add_argument("--output", type=Path, default=SILVER_ROOT / "event_states.parquet")
    event_states.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    event_states.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    event_states.set_defaults(func=command_build_event_states)

    player_games = subparsers.add_parser(
        "build-player-games", help="Build canonical player-game boxes and exact starter seeds."
    )
    player_games.add_argument(
        "--box",
        type=Path,
        default=BRONZE_ROOT / "llimllib_nba_data" / "player_game_logs.parquet",
    )
    player_games.add_argument(
        "--espn",
        type=Path,
        default=BRONZE_ROOT / "llimllib_nba_data" / "espn" / "player_box.parquet",
    )
    player_games.add_argument("--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet")
    player_games.add_argument("--output", type=Path, default=SILVER_ROOT / "player_games.parquet")
    player_games.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    player_games.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    player_games.set_defaults(func=command_build_player_games)

    lineups = subparsers.add_parser(
        "build-lineups", help="Reconstruct and minute-reconcile current five-player lineup stints."
    )
    lineups.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    lineups.add_argument("--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet")
    lineups.add_argument("--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet")
    lineups.add_argument("--output", type=Path, default=SILVER_ROOT / "lineup_stints.parquet")
    lineups.add_argument(
        "--quality-output", type=Path, default=SILVER_ROOT / "lineup_game_quality.parquet"
    )
    lineups.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    lineups.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    lineups.add_argument("--minute-tolerance-seconds", type=float, default=5.0)
    lineups.add_argument("--max-quarantine-fraction", type=float, default=0.005)
    lineups.set_defaults(func=command_build_lineups)

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

    walk = subparsers.add_parser(
        "walk-forward-rapm", help="Compare RAPM candidates across chronological outer seasons."
    )
    walk.add_argument("--test-seasons", type=_season_list, required=True)
    walk.add_argument("--train-window", type=int, default=3)
    walk.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    walk.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    walk.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    walk.add_argument("--snapshot-id")
    walk.add_argument(
        "--lambda-pairs",
        type=_lambda_pairs,
        default=_lambda_pairs("3000:3000,2000:4500,1000:1000"),
    )
    walk.add_argument("--lambda-home", type=float, default=300.0)
    walk.add_argument("--bootstrap-repetitions", type=int, default=2000)
    walk.add_argument("--seed", type=int, default=7)
    walk.add_argument("--game-types", default="regular")
    walk.add_argument("--no-home", action="store_true")
    walk.set_defaults(func=command_walk_forward_rapm)

    win_probability = subparsers.add_parser(
        "fit-win-probability", help="Fit the chronological state-only win-probability baseline."
    )
    win_probability.add_argument("--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet")
    win_probability.add_argument("--train-seasons", type=_text_list, default=("2024-25",))
    win_probability.add_argument("--test-seasons", type=_text_list, default=("2025-26",))
    win_probability.add_argument("--interval-seconds", type=int, default=30)
    win_probability.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    win_probability.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    win_probability.add_argument("--snapshot-id")
    win_probability.set_defaults(func=command_fit_win_probability)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
