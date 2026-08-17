"""Command line entry point for the new data/model spine."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.api.ratings import RatingsApiConfig, RatingsStore
from nba_impact.api.server import serve
from nba_impact.api.web_snapshot import build_web_snapshot
from nba_impact.data.assist_quality_features import build_assist_quality_features
from nba_impact.data.behavior_roles import build_behavior_roles
from nba_impact.data.side_roles import build_side_roles
from nba_impact.data.role_stabilization import build_role_stabilization
from nba_impact.data.download import plan_ingest_manifest, run_ingest_manifest
from nba_impact.data.defensive_tracking_features import build_defensive_tracking_features
from nba_impact.data.event_quality import build_event_snapshot
from nba_impact.data.event_state import build_event_states
from nba_impact.data.espn_win_probability import ingest_espn_win_probability
from nba_impact.data.game_dim import build_game_dimension
from nba_impact.data.identity import build_identity_dimensions
from nba_impact.data.lineups import build_lineup_stints
from nba_impact.data.matchup_defense_features import build_matchup_defense_features
from nba_impact.data.manifest import (
    build_possession_snapshot,
    sha256_file,
    write_json_atomic,
)
from nba_impact.data.official_boxscore import ingest_official_boxscores
from nba_impact.data.official_game_scores import build_official_game_scores
from nba_impact.data.player_game import build_player_games
from nba_impact.data.playtype_features import build_playtype_features
from nba_impact.data.player_skill_features import build_player_skill_features
from nba_impact.data.possession_context import build_possession_start_context
from nba_impact.data.role_context import build_role_context_features
from nba_impact.data.shot_defense import build_shot_defense_events
from nba_impact.data.possessions import build_possessions
from nba_impact.data.scoring_events import build_scoring_event_dataset
from nba_impact.data.statistical_features import build_statistical_feature_windows
from nba_impact.data.statistical_features_v2 import build_statistical_features_v2
from nba_impact.models.rapm import (
    RapmConfig,
    load_current_possessions,
    load_current_player_names,
    load_legacy_possessions,
    run_nested_normal_rapm_tuning,
    run_rapm,
    run_regularization_comparison,
    run_walk_forward_comparison,
)
from nba_impact.models.rapm_lineup_policy import run_rapm_lineup_policy_comparison
from nba_impact.models.rapm_uncertainty import (
    RapmUncertaintyConfig,
    run_rapm_uncertainty,
)
from nba_impact.models.shot_defense_pilot import run_shot_defense_team_pilot
from nba_impact.models.time_decayed_trajectory import (
    build_time_decayed_trajectory,
)
from nba_impact.models.dynamic_state_space import (
    build_annual_observation_variance,
    build_state_space_trajectory,
)
from nba_impact.models.expected_possession_points import (
    build_expected_possession_points,
)
from nba_impact.models.external_impact_benchmark import (
    acquire_external_impact_pages,
    build_external_impact_benchmark,
)
from nba_impact.models.prior_informed_rapm import (
    run_prior_informed_rapm_comparison,
)
from nba_impact.models.precision_aware_prior import (
    run_precision_aware_prior_comparison,
)
from nba_impact.models.annual_spm_priors import (
    build_forward_chained_annual_spm_priors,
    build_leave_one_season_out_annual_spm_priors,
)
from nba_impact.models.annual_aio_ratings import build_annual_aio_ratings
from nba_impact.models.annual_defense_ridge_nested import run_annual_defense_ridge_nested
from nba_impact.models.annual_defense_features_nested import run_annual_defense_features_nested
from nba_impact.models.defense_role_challenger import run_defense_role_challenger
from nba_impact.models.current_spm_confirmation import run_current_spm_confirmation
from nba_impact.models.current_spm_diagnostics import run_current_spm_diagnostics
from nba_impact.models.rolling_rapm_peaks import (
    build_rolling_rapm_peaks,
    run_selection_aware_peak_bootstrap,
)
from nba_impact.models.statistical_impact import run_statistical_impact_baseline
from nba_impact.models.single_season_spm import (
    build_single_season_rapm_targets,
    fit_single_season_spm,
)
from nba_impact.models.current_single_season_rapm import (
    build_current_single_season_rapm_targets,
)
from nba_impact.models.annual_target_transition import (
    build_canonical_annual_target_panel,
)
from nba_impact.models.aging_balanced_validation import (
    run_aging_balanced_validation,
)
from nba_impact.models.aging_projection import build_aging_projection
from nba_impact.models.statistical_direct_net import (
    run_statistical_direct_net_comparison,
)
from nba_impact.models.statistical_feature_ablation import (
    fit_optimized_statistical_aio,
    run_statistical_feature_ablation,
)
from nba_impact.models.statistical_model_comparison import (
    run_statistical_model_comparison,
)
from nba_impact.models.statistical_priors import (
    build_cross_fitted_statistical_priors,
)
from nba_impact.models.statistical_interpretability import (
    run_statistical_interpretability,
)
from nba_impact.models.statistical_feature_v2 import (
    run_statistical_feature_v2_comparison,
)
from nba_impact.models.inpredictable_benchmark import (
    run_inpredictable_surface_benchmark,
)
from nba_impact.models.win_probability import run_win_probability
from nba_impact.models.win_probability_ablation import run_win_probability_elo_ablation
from nba_impact.models.win_probability_benchmark import (
    run_espn_win_probability_benchmark,
)
from nba_impact.models.win_probability_lineup import run_win_probability_lineup_ablation
from nba_impact.models.win_probability_mlp import run_win_probability_mlp_comparison
from nba_impact.models.win_probability_possession import (
    run_win_probability_possession_ablation,
)
from nba_impact.models.win_probability_stage1 import (
    run_win_probability_stage1_comparison,
)
from nba_impact.paths import (
    ARTIFACT_ROOT,
    BRONZE_ROOT,
    LEGACY_PLAYER_SHEETS,
    LEGACY_POSSESSION_CACHE,
    MANIFEST_ROOT,
    OFFICIAL_BOXSCORE_ROOT,
    PLAYER_NAMES,
    PROJECT_ROOT,
    REGISTRY_PATH,
    SILVER_ROOT,
    ensure_owned_dirs,
)
from nba_impact.registry import register_model_run, register_snapshot
from nba_impact.research.control_plane import validate_pinned_artifacts
from nba_impact.research.release import build_local_release_bundle


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


def _float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "provide comma-separated numeric values"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("provide at least one numeric value")
    return values


def command_audit(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = list(args.seasons)
    snapshot = build_possession_snapshot(args.cache_dir, seasons)
    destination = Path(args.output or MANIFEST_ROOT / f"{snapshot['snapshot_id']}.json")
    write_json_atomic(snapshot, destination)
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "path": str(destination),
            },
            indent=2,
        )
    )
    for item in snapshot["files"]:
        quality = item["quality"]
        print(
            f"{item['season']}: rows={quality['row_count']:,} games={quality['game_count']:,} "
            f"dates={quality['date_min']}..{quality['date_max']} passed={quality['passed']}"
        )
        for issue in quality["issues"]:
            print(
                f"  {issue['severity']:>8} {issue['code']}: {issue['count']} — {issue['message']}"
            )
    return 0 if snapshot["passed"] else 2


def command_validate_research_control(args: argparse.Namespace) -> int:
    """Validate the release/model gate before new public-impact work begins."""
    issues = validate_pinned_artifacts(args.contract, args.artifact_root)
    payload = {
        "contract": str(args.contract),
        "artifact_root": str(args.artifact_root),
        "passed": not issues,
        "issues": [issue.__dict__ for issue in issues],
    }
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 2


def command_build_local_release(args: argparse.Namespace) -> int:
    release = build_local_release_bundle(
        args.api_config,
        args.artifact_root,
        args.contract,
        release_root=args.release_root,
    )
    print(json.dumps(release, indent=2))
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    if args.dry_run:
        plan = plan_ingest_manifest(args.manifest, root=args.root)
        print(json.dumps(plan, indent=2))
        return 0
    summary = run_ingest_manifest(args.manifest, root=args.root)
    print(
        json.dumps(
            {key: summary[key] for key in ("succeeded", "failed", "completed_at")},
            indent=2,
        )
    )
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
    snapshot = build_event_states(
        args.root, args.game_dim, args.output, args.manifest_dir
    )
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
        official_box_dir=args.official_box_dir,
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


def command_build_identity_dimensions(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_identity_dimensions(
        args.game_dim,
        args.player_games,
        args.output_dir,
        args.manifest_dir,
        event_states_path=args.event_states,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "row_counts": snapshot["row_counts"],
                "issues": snapshot["issues"],
                "artifacts": snapshot["artifacts"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_ingest_official_boxscores(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    quality = pd.read_parquet(args.quality)
    failed = quality.loc[~quality["passed"]].copy()
    if args.seasons:
        failed = failed.loc[failed["season_label"].isin(args.seasons)]
    snapshot = ingest_official_boxscores(
        failed["game_id"].astype(str).tolist(),
        args.output_root,
        args.manifest_dir,
        max_attempts=args.max_attempts,
        minimum_delay_seconds=args.minimum_delay_seconds,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "requested_games": snapshot["requested_game_count"],
                "downloaded_games": snapshot["downloaded_game_count"],
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


def command_build_possessions(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_possessions(
        args.root,
        args.event_states,
        args.game_dim,
        args.player_games,
        args.lineup_quality,
        args.output,
        args.segments_output,
        args.manifest_dir,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "segments": snapshot["segment_row_count"],
                "games": snapshot["game_count"],
                "coverage_rate": snapshot["coverage_rate"],
                "point_reconciliation": snapshot["point_reconciliation"],
                "issues": snapshot["issues"],
                "warnings": snapshot["warnings"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_build_possession_start_context(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_possession_start_context(
        args.possessions, args.output, args.manifest_dir
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "games": snapshot["game_count"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_build_shot_defense_events(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = build_shot_defense_events(
        args.root,
        args.event_states,
        args.game_dim,
        args.segments,
        args.output,
        args.manifest_dir,
        seasons=args.seasons,
    )
    register_snapshot(args.registry, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "passed": snapshot["passed"],
                "rows": snapshot["row_count"],
                "games": snapshot["game_count"],
                "seasons": snapshot["seasons"],
                "path": snapshot["path"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["passed"] else 2


def command_run_shot_defense_pilot(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_shot_defense_team_pilot(
        args.panel,
        artifact_root=args.artifact_root,
        season=args.season,
        train_fraction=args.train_fraction,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_scoring_events(args: argparse.Namespace) -> int:
    run = build_scoring_event_dataset(
        args.root,
        args.output,
        project_seasons=tuple(args.seasons),
        game_dim_path=args.game_dim,
        legacy_cache_dir=args.legacy_cache,
        official_game_scores_path=args.official_game_scores,
        require_reference_coverage=not args.allow_missing_reference,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_download_official_game_scores(args: argparse.Namespace) -> int:
    run = build_official_game_scores(
        args.output,
        project_seasons=tuple(args.seasons),
        max_attempts=args.max_attempts,
        request_delay_seconds=args.request_delay,
    )
    print(json.dumps(run, indent=2))
    return 0 if run["passed"] else 2


def command_fit_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = tuple(args.seasons)
    game_types = tuple(
        item.strip() for item in args.game_types.split(",") if item.strip()
    )
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


def command_fit_current_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    game_types = tuple(
        item.strip() for item in args.game_types.split(",") if item.strip()
    )
    frame = load_current_possessions(
        args.possessions,
        args.segments,
        lineup_policy=args.lineup_policy,
        game_types=game_types,
    )
    seasons = tuple(sorted(int(value) for value in frame["season"].unique()))
    names = load_current_player_names(args.names, args.player_games)
    config = RapmConfig(
        seasons=seasons,
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
        include_home=not args.no_home,
        game_types=game_types,
        data_scope=f"current_cdn_{args.lineup_policy}_lineup",
    )
    run = run_rapm(frame, config, artifact_root=args.artifact_root, names=names)
    ratings_path = Path(run["artifact_path"]) / "ratings.parquet"
    ratings = pd.read_parquet(ratings_path, columns=["player_name"])
    run["status"] = (
        "research_frozen_baseline"
        if args.lineup_policy == "terminal"
        else "research_lineup_sensitivity"
    )
    run["config"]["lineup_policy"] = args.lineup_policy
    run["config"]["possessions_path"] = str(args.possessions.resolve())
    run["config"]["segments_path"] = str(args.segments.resolve())
    run["config"]["source_hashes"] = {
        "possessions": sha256_file(args.possessions),
        "segments": sha256_file(args.segments),
        "player_games": sha256_file(args.player_games),
        "legacy_names": sha256_file(args.names) if Path(args.names).exists() else None,
    }
    run["metrics"]["missing_player_names"] = int(ratings["player_name"].isna().sum())
    run["caveats"] = [
        "The estimand is descriptive and uses observed lineups, not a pregame forecast.",
        "Source games that fail lineup quality gates are excluded rather than repaired.",
        "The latest-season retrodiction contains players absent from earlier training seasons.",
        "Uncertainty is not estimated in this version.",
    ]
    run["dataset_snapshot_id"] = args.snapshot_id
    write_json_atomic(run["config"], Path(run["artifact_path"]) / "config.json")
    write_json_atomic(run["metrics"], Path(run["artifact_path"]) / "metrics.json")
    write_json_atomic(run, Path(run["artifact_path"]) / "run.json")
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_quantify_rapm_uncertainty(args: argparse.Namespace) -> int:
    """Run the frozen terminal-lineup whole-game RAPM bootstrap."""
    ensure_owned_dirs()
    seasons = tuple(args.seasons)
    if 2027 in seasons:
        raise ValueError("Season 2027 is reserved and cannot enter this command.")
    game_types = tuple(item.strip() for item in args.game_types.split(",") if item.strip())
    if args.source == "legacy":
        frame = load_legacy_possessions(args.cache_dir, seasons, game_types=game_types)
        names = pd.read_csv(args.names) if Path(args.names).exists() else None
        hashes = {
            str(Path(args.cache_dir) / f"matchups_{season}.parquet"): sha256_file(
                Path(args.cache_dir) / f"matchups_{season}.parquet"
            )
            for season in seasons
        }
        data_scope = "legacy_terminal_lineup"
    else:
        frame = load_current_possessions(
            args.possessions,
            args.segments,
            lineup_policy="terminal",
            game_types=game_types,
        )
        frame = frame.loc[frame["season"].isin(seasons)].copy()
        if set(seasons) != set(int(value) for value in frame["season"].unique()):
            raise ValueError("Requested current uncertainty seasons are not all available.")
        names = load_current_player_names(args.names, args.player_games)
        hashes = {
            "possessions": sha256_file(args.possessions),
            "segments": sha256_file(args.segments),
            "player_games": sha256_file(args.player_games),
        }
        data_scope = "current_terminal_lineup"
    config = RapmConfig(
        seasons=seasons,
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
        include_home=True,
        game_types=game_types,
        data_scope=data_scope,
    )
    uncertainty = RapmUncertaintyConfig(draws=args.draws, seed=args.seed)
    run = run_rapm_uncertainty(
        frame,
        config,
        uncertainty,
        artifact_root=args.artifact_root,
        names=names,
        source_hashes=hashes,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_quantify_rolling_peak_uncertainty(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_selection_aware_peak_bootstrap(
        args.cache_dir,
        args.names,
        args.player_sheets_dir,
        args.contract,
        artifact_root=args.artifact_root,
        draws=args.draws,
        seed=args.seed,
    )
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
        raise argparse.ArgumentTypeError(
            "use comma-separated offense:def pairs"
        ) from exc
    if not pairs or any(offense <= 0 or defense <= 0 for offense, defense in pairs):
        raise argparse.ArgumentTypeError("lambda pairs must be positive")
    return tuple(pairs)


def _penalty_triples(value: str) -> tuple[tuple[float, float, float], ...]:
    triples: list[tuple[float, float, float]] = []
    try:
        for item in value.split(","):
            offense, defense, home = item.strip().split(":", maxsplit=2)
            triples.append((float(offense), float(defense), float(home)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated offense:defense:home triples"
        ) from exc
    if not triples or any(min(triple) <= 0 for triple in triples):
        raise argparse.ArgumentTypeError("penalty triples must be positive")
    return tuple(triples)


def command_tune_normal_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    frame = load_current_possessions(
        args.possessions,
        args.segments,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    seasons = (2024, 2025, 2026)
    frame = frame.loc[frame["season"].isin(seasons)].copy()
    names = pd.read_csv(args.names) if Path(args.names).exists() else None
    run = run_nested_normal_rapm_tuning(
        frame,
        RapmConfig(seasons=seasons, data_scope="current_cdn_normal_rapm"),
        args.penalty_candidates,
        selection_train_seasons=(2024,),
        selection_test_season=2025,
        confirmation_train_seasons=(2024, 2025),
        confirmation_test_season=2026,
        artifact_root=args.artifact_root,
        names=names,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_fit_statistical_impact(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_impact_baseline(
        args.features,
        args.targets,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_statistical_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_statistical_feature_windows(
        args.source_dir,
        artifact_root=args.artifact_root,
        window_ends=args.window_ends,
        window_seasons=args.window_seasons,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_statistical_features_v2(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_statistical_features_v2(
        args.source_dir,
        args.base_features,
        artifact_root=args.artifact_root,
        window_ends=args.window_ends,
        pooled_window_seasons=args.pooled_window_seasons,
        playtype_features_path=args.playtype_features,
        defensive_tracking_features_path=args.defensive_tracking_features,
        assist_quality_features_path=args.assist_quality_features,
        matchup_defense_features_path=args.matchup_defense_features,
        player_skill_features_path=args.player_skill_features,
        behavior_roles_path=args.behavior_roles,
        offense_roles_path=args.offense_roles,
        defense_roles_path=args.defense_roles,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_playtype_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_playtype_features(
        args.playtype_source, args.box_source_dir,
        artifact_root=args.artifact_root, seasons=args.seasons,
        minimum_minutes=args.minimum_minutes,
        minimum_player_playtype_possessions=args.minimum_synergy_possessions,
        minimum_league_row_possessions=args.minimum_league_row_possessions,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_defensive_tracking_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_defensive_tracking_features(
        args.dfg_source, args.rim_dfg_source, args.hustle_source,
        args.box_source_dir, artifact_root=args.artifact_root, seasons=args.seasons,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_matchup_defense_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_matchup_defense_features(
        args.archive_root,
        args.box_source_dir,
        artifact_root=args.artifact_root,
        seasons=args.seasons,
        defender_prior_possessions=args.defender_prior_possessions,
        shooting_prior_attempts=args.shooting_prior_attempts,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_assist_quality_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_assist_quality_features(
        args.source, artifact_root=args.artifact_root, seasons=args.seasons
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_role_context_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_role_context_features(
        args.shooting_by_dribble_source,
        args.jump_shot_by_dribble_source,
        artifact_root=args.artifact_root,
        seasons=args.seasons,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_player_skill_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_player_skill_features(
        args.shooting_source,
        args.passing_source,
        args.hustle_source,
        args.shotzone_source,
        artifact_root=args.artifact_root,
        seasons=args.seasons,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_behavior_roles(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_behavior_roles(
        args.annual_features,
        args.role_context,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_build_side_roles(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_side_roles(
        args.annual_features,
        args.dribble_context,
        args.playtype_source,
        args.defensive_tracking,
        args.matchup_archive_root,
        artifact_root=args.artifact_root,
        offense_seasons=args.offense_seasons,
        defense_seasons=args.defense_seasons,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_stabilize_roles(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_role_stabilization(
        args.side_roles_run,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_run_defense_role_challenger(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_defense_role_challenger(
        args.features,
        args.targets,
        args.frozen_spm_run,
        args.contract,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(run, indent=2))
    return 0


def command_compare_statistical_models(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_model_comparison(
        args.features,
        args.targets,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_compare_statistical_direct_net(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_direct_net_comparison(
        args.features,
        args.targets,
        args.component_run,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_ablate_statistical_features(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_feature_ablation(
        args.features,
        args.targets,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_fit_optimized_statistical_aio(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = fit_optimized_statistical_aio(
        args.features,
        args.targets,
        args.ablation_run,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_compare_statistical_features_v2(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_feature_v2_comparison(
        args.features,
        args.targets,
        args.baseline_run,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_interpret_statistical_aio(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_statistical_interpretability(
        args.features,
        args.targets,
        args.reference_run,
        artifact_root=args.artifact_root,
        test_window_end=args.test_window_end,
        group_repeats=args.group_repeats,
        individual_repeats=args.individual_repeats,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_validate_aging_balance(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_aging_balanced_validation(
        args.predictions,
        args.age_source_dir,
        artifact_root=args.artifact_root,
        minimum_training_origins=args.minimum_training_origins,
        aging_ridge_alpha=args.aging_ridge_alpha,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_statistical_priors(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_cross_fitted_statistical_priors(
        args.features,
        args.targets,
        args.reference_run,
        artifact_root=args.artifact_root,
        prediction_window_ends=args.window_ends,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_compare_prior_informed_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    test_seasons = tuple(args.test_seasons)
    seasons = tuple(
        range(min(test_seasons) - args.train_window, max(test_seasons) + 1)
    )
    frame = load_legacy_possessions(
        args.cache_dir, seasons, game_types=("regular",)
    )
    run = run_prior_informed_rapm_comparison(
        frame,
        args.priors,
        RapmConfig(
            seasons=seasons,
            lambda_off=args.lambda_off,
            lambda_def=args.lambda_def,
            lambda_home=args.lambda_home,
            game_types=("regular",),
            data_scope="legacy_regular_prior_informed_comparison",
        ),
        artifact_root=args.artifact_root,
        test_seasons=test_seasons,
        train_window=args.train_window,
        prior_scales=args.prior_scales,
        selection_test_seasons=tuple(args.selection_test_seasons),
        confirmation_test_seasons=tuple(args.confirmation_test_seasons),
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.seed,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_compare_precision_aware_prior(args: argparse.Namespace) -> int:
    """Run and retain the preregistered four-model prior comparison."""
    ensure_owned_dirs()
    test_seasons = tuple(args.test_seasons)
    selection_seasons = tuple(args.selection_seasons)
    diagnostic_seasons = tuple(args.diagnostic_seasons)
    if 2027 in test_seasons:
        raise ValueError("Season 2027 is reserved and cannot enter this comparison.")
    seasons = tuple(range(min(test_seasons) - args.train_window, max(test_seasons) + 1))
    frame = load_legacy_possessions(args.cache_dir, seasons, game_types=("regular",))
    priors = pd.read_parquet(args.priors)
    calibration = pd.read_parquet(args.calibration)
    config = RapmConfig(
        seasons=seasons,
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
        game_types=("regular",),
        data_scope="legacy_regular_precision_aware_prior_comparison",
    )
    folds, calibration_rows, paired_bootstrap = run_precision_aware_prior_comparison(
        frame,
        priors,
        calibration,
        config,
        test_seasons=test_seasons,
        train_window=args.train_window,
        selection_seasons=selection_seasons,
        diagnostic_seasons=diagnostic_seasons,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.seed,
    )
    expected = {
        "zero_prior",
        "statistical_prior_only",
        "fixed_center_prior",
        "precision_aware_side_specific_prior",
    }
    by_season = folds.groupby("test_season")["candidate"].agg(set)
    invalid = {
        int(season): sorted(expected - candidates)
        for season, candidates in by_season.items()
        if candidates != expected
    }
    if invalid:
        raise ValueError(f"Each scored season must include exactly four models: {invalid}")
    zero = folds.loc[folds["candidate"].eq("zero_prior"), [
        "test_season", "margin_rmse", "margin_correlation"
    ]].rename(columns={
        "margin_rmse": "zero_margin_rmse",
        "margin_correlation": "zero_margin_correlation",
    })
    comparison = folds.merge(zero, on="test_season", validate="many_to_one")
    comparison["rmse_improvement_vs_zero"] = (
        comparison["zero_margin_rmse"] - comparison["margin_rmse"]
    )
    comparison["correlation_change_vs_zero"] = (
        comparison["margin_correlation"] - comparison["zero_margin_correlation"]
    )
    diagnostic_candidate = comparison.loc[
        comparison["test_season"].isin(diagnostic_seasons)
        & comparison["candidate"].eq("precision_aware_side_specific_prior")
    ]
    gate = {
        "minimum_rmse_improvement": 0.05,
        "observed_mean_rmse_improvement": float(
            diagnostic_candidate["rmse_improvement_vs_zero"].mean()
        ),
        "mean_correlation_change": float(
            diagnostic_candidate["correlation_change_vs_zero"].mean()
        ),
        "paired_bootstrap": paired_bootstrap,
        "status": "unreviewed_research_result",
    }
    run_id = f"precision_aware_prior_rapm_v1_{uuid.uuid4().hex[:10]}"
    output = Path(args.artifact_root) / "models" / "precision_aware_prior_rapm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    folds.to_parquet(output / "folds.parquet", index=False)
    calibration_rows.to_parquet(output / "precision_calibration.parquet", index=False)
    comparison.to_parquet(output / "fold_comparison.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "precision_aware_side_specific_prior_rapm",
        "estimand_id": "annual_retrospective_impact_v1",
        "estimand": "regular-season held-out game-margin prediction from trailing three-season normal RAPM",
        "status": "research_challenger_unreviewed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "contract_path": str(args.contract),
            "test_seasons": list(test_seasons),
            "selection_seasons": list(selection_seasons),
            "diagnostic_seasons": list(diagnostic_seasons),
            "train_window": args.train_window,
            "rapm": {
                "lambda_off": args.lambda_off,
                "lambda_def": args.lambda_def,
                "lambda_home": args.lambda_home,
            },
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.seed,
            "source_hashes": {
                "contract": sha256_file(args.contract),
                "priors": sha256_file(args.priors),
                "calibration": sha256_file(args.calibration),
                "source_code": sha256_file(
                    PROJECT_ROOT / "src/nba_impact/models/precision_aware_prior.py"
                ),
            },
        },
        "quality": {
            "folds": int(len(folds)),
            "all_scored_seasons_have_exactly_four_models": True,
            "identical_game_count_by_candidate_and_season": bool(
                folds.groupby("test_season")["games"].nunique().eq(1).all()
            ),
        },
        "metrics": {
            "folds": folds.to_dict(orient="records"),
            "promotion_gate": gate,
        },
        "promotion_gate": gate,
        "caveats": [
            "This is a preregistered research challenger, not a production replacement.",
            "Season 2027 remains untouched confirmation and is not part of this run.",
            "A promotion decision requires the separate frozen review after results exist.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_ingest_external_impact(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    manifest = acquire_external_impact_pages(
        args.raw_root,
        seasons=tuple(args.seasons),
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_benchmark_external_impact(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_external_impact_benchmark(
        args.priors,
        args.features,
        args.names,
        args.raw_root,
        artifact_root=args.artifact_root,
        window_ends=tuple(args.window_ends),
        minimum_window_possessions_per_side=args.minimum_possessions,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_single_season_rapm_targets(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_single_season_rapm_targets(
        args.cache_dir,
        artifact_root=args.artifact_root,
        seasons=tuple(args.seasons),
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_current_single_season_rapm_targets(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_current_single_season_rapm_targets(
        args.possessions,
        args.segments,
        args.names,
        args.player_games,
        artifact_root=args.artifact_root,
        seasons=tuple(args.seasons),
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_canonical_annual_target_panel(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_canonical_annual_target_panel(
        args.legacy_targets,
        args.canonical_targets,
        args.names,
        args.player_games,
        artifact_root=args.artifact_root,
        transition_season=args.transition_season,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_time_decayed_trajectory(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_time_decayed_trajectory(
        args.targets,
        args.names,
        artifact_root=args.artifact_root,
        candidate_decays=tuple(args.candidate_decays),
        candidate_exposure_powers=tuple(args.candidate_exposure_powers),
        selection_origins=tuple(args.selection_origins),
        diagnostic_origins=tuple(args.diagnostic_origins),
        minimum_side_possessions=args.minimum_side_possessions,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_annual_observation_variance(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_annual_observation_variance(
        args.targets,
        args.cache_dir,
        args.possessions,
        args.segments,
        artifact_root=args.artifact_root,
        transition_season=args.transition_season,
        seasons=tuple(args.seasons) if args.seasons else None,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_state_space_trajectory(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_state_space_trajectory(
        args.targets,
        args.observation_variance,
        args.names,
        args.time_decay_trajectories,
        artifact_root=args.artifact_root,
        candidate_phis=tuple(args.candidate_phis),
        candidate_process_sds=tuple(args.candidate_process_sds),
        selection_origins=tuple(args.selection_origins),
        diagnostic_origins=tuple(args.diagnostic_origins),
        minimum_side_possessions=args.minimum_side_possessions,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_aging_projection(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_aging_projection(
        args.trajectories,
        args.targets,
        args.player_sheets_dir,
        artifact_root=args.artifact_root,
        selection_origins=tuple(args.selection_origins),
        diagnostic_origins=tuple(args.diagnostic_origins),
        projection_origin=args.projection_origin,
        minimum_side_possessions=args.minimum_side_possessions,
        alpha=args.alpha,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_expected_possession_points(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_expected_possession_points(
        args.context,
        artifact_root=args.artifact_root,
        test_seasons=tuple(args.test_seasons),
        alpha=args.alpha,
        max_iter=args.max_iter,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_fit_single_season_spm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = fit_single_season_spm(
        args.features,
        args.targets,
        args.reference_run,
        args.names,
        args.external_raw_root,
        artifact_root=args.artifact_root,
        output_seasons=tuple(args.output_seasons),
        minimum_possessions_per_side=args.minimum_possessions,
        additional_offense_features=args.additional_offense_features,
        additional_defense_features=args.additional_defense_features,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_forward_annual_spm_priors(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_forward_chained_annual_spm_priors(
        args.features,
        args.targets,
        args.reference_run,
        args.contract,
        artifact_root=args.artifact_root,
        output_seasons=tuple(args.output_seasons),
        minimum_training_seasons=args.minimum_training_seasons,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_oof_annual_spm_priors(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_leave_one_season_out_annual_spm_priors(
        args.spm_run,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_annual_aio_ratings(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_annual_aio_ratings(
        args.cache_dir,
        args.priors,
        args.names,
        artifact_root=args.artifact_root,
        seasons=tuple(args.seasons),
        lambda_off=args.lambda_off,
        lambda_def=args.lambda_def,
        lambda_home=args.lambda_home,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_build_rolling_rapm_peaks(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = build_rolling_rapm_peaks(
        args.cache_dir,
        args.names,
        args.player_sheets_dir,
        args.contract,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_confirm_current_spm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_current_spm_confirmation(
        args.features,
        args.frozen_spm_run,
        args.possessions,
        args.segments,
        args.names,
        args.player_games,
        artifact_root=args.artifact_root,
        season=args.season,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_diagnose_current_spm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_current_spm_diagnostics(
        args.confirmation_run,
        args.current_features,
        args.reference_features,
        args.frozen_spm_run,
        args.possessions,
        args.segments,
        artifact_root=args.artifact_root,
        season=args.season,
        comparison_season=args.comparison_season,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_run_annual_defense_ridge_nested(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_annual_defense_ridge_nested(
        args.features,
        args.targets,
        args.frozen_spm_run,
        args.contract,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_run_annual_defense_features_nested(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_annual_defense_features_nested(
        args.features,
        args.targets,
        args.frozen_spm_run,
        args.contract,
        artifact_root=args.artifact_root,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_serve_ratings(args: argparse.Namespace) -> int:
    config = RatingsApiConfig.from_json(args.config)
    store = RatingsStore(config, args.artifact_root)
    serve(store, args.host, args.port)
    return 0


def command_build_web_snapshot(args: argparse.Namespace) -> int:
    result = build_web_snapshot(
        args.config,
        args.artifact_root,
        args.aging_curve,
        args.output_dir,
        spm_run_path=args.spm_run,
        player_sheets_dir=args.player_sheets_dir,
        features_path=args.features,
        walk_forward_run_path=args.walk_forward_run,
        walk_backward_run_path=args.walk_backward_run,
        aging_projection_run_path=args.aging_projection_run,
        win_probability_run_path=args.win_probability_run,
        shards=args.shards,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_compare_rapm(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    seasons = tuple(args.seasons)
    game_types = tuple(
        item.strip() for item in args.game_types.split(",") if item.strip()
    )
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
    game_types = tuple(
        item.strip() for item in args.game_types.split(",") if item.strip()
    )
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


def command_compare_win_probability(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability_elo_ablation(
        args.event_states,
        args.game_dim,
        train_season_labels=args.train_seasons,
        test_season_labels=args.test_seasons,
        artifact_root=args.artifact_root,
        interval_seconds=args.interval_seconds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    run["dataset_snapshot_id"] = args.snapshot_id
    write_json_atomic(run, Path(run["artifact_path"]) / "run.json")
    register_model_run(args.registry, run)
    summary = {
        "run_id": run["run_id"],
        "status": run["status"],
        "variants": run["metrics"]["nonterminal_variants"],
        "paired_game_bootstrap": run["metrics"]["paired_game_bootstrap"],
        "artifact_path": run["artifact_path"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_ingest_espn_win_probability(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    snapshot = ingest_espn_win_probability(
        args.game_dim,
        season_labels=args.seasons,
        raw_root=args.raw_root,
        index_output=args.output,
        manifest_dir=args.manifest_dir,
        max_workers=args.max_workers,
    )
    register_snapshot(args.registry, snapshot)
    print(json.dumps(snapshot, indent=2))
    return 0 if snapshot["games_ready"] == snapshot["games_expected"] else 2


def command_benchmark_win_probability(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_espn_win_probability_benchmark(
        args.event_states,
        args.game_dim,
        args.espn_index,
        args.model_run,
        artifact_root=args.artifact_root,
        clock_tolerance_seconds=args.clock_tolerance_seconds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    register_model_run(args.registry, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "coverage": run["coverage"],
                "metrics": run["metrics"],
                "paired_game_bootstrap": run["paired_game_bootstrap"],
                "checkpoints": run["checkpoints"],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def command_compare_wp_lineup_strength(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability_lineup_ablation(
        args.event_states,
        args.game_dim,
        args.player_games,
        args.possessions,
        args.segments,
        args.legacy_cache,
        args.espn_index,
        artifact_root=args.artifact_root,
        train_season=args.train_season,
        test_season=args.test_season,
        include_espn=not args.skip_espn,
        interval_seconds=args.interval_seconds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    register_model_run(args.registry, run)
    metrics = run["metrics"]
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "variants": metrics["variants"],
                "paired_game_bootstrap": metrics["paired_game_bootstrap"],
                "team_context_paired_game_bootstrap": metrics[
                    "team_context_paired_game_bootstrap"
                ],
                "starter_free_context_vs_starter_context": metrics[
                    "starter_free_context_vs_starter_context"
                ],
                "starter_free_context_vs_elo": metrics["starter_free_context_vs_elo"],
                "espn_game_start": metrics["espn_game_start"],
                "espn_game_start_paired": metrics["espn_game_start_paired"],
                "starter_rating_coverage": metrics["starter_rating_coverage"],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def command_benchmark_inpredictable(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_inpredictable_surface_benchmark(
        args.model_run, artifact_root=args.artifact_root, max_workers=args.max_workers
    )
    register_model_run(args.registry, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "metrics": run["metrics"],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def command_compare_wp_possession(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability_possession_ablation(
        args.possessions,
        args.game_dim,
        artifact_root=args.artifact_root,
        train_season=args.train_season,
        test_season=args.test_season,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    register_model_run(args.registry, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "variants": run["metrics"]["variants"],
                "paired_game_bootstrap": run["metrics"]["paired_game_bootstrap"],
                "close_last_2m_margin_le_3": run["metrics"][
                    "close_last_2m_margin_le_3"
                ],
                "possession_effects": run["metrics"]["possession_effects"],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def command_compare_wp_stage1(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability_stage1_comparison(
        args.event_states,
        args.game_dim,
        artifact_root=args.artifact_root,
        interval_seconds=args.interval_seconds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    register_model_run(args.registry, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "folds": run["metrics"]["folds"],
                "pooled_paired_vs_logistic": run["metrics"][
                    "pooled_paired_vs_logistic"
                ],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def command_compare_wp_mlp(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_win_probability_mlp_comparison(
        args.event_states,
        args.game_dim,
        artifact_root=args.artifact_root,
        interval_seconds=args.interval_seconds,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    register_model_run(args.registry, run)
    print(json.dumps(run, indent=2))
    return 0


def command_compare_rapm_lineups(args: argparse.Namespace) -> int:
    ensure_owned_dirs()
    run = run_rapm_lineup_policy_comparison(
        args.possessions,
        args.segments,
        artifact_root=args.artifact_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    register_model_run(args.registry, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                **run["metrics"],
                "artifact_path": run["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-impact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-possessions", help="Audit and register legacy possession partitions."
    )
    audit.add_argument("--seasons", type=_season_list, default=tuple(range(1997, 2026)))
    audit.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    audit.set_defaults(func=command_audit)

    control = subparsers.add_parser(
        "validate-research-control",
        help="Reject incomplete pinned lineage before model or release work.",
    )
    control.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "research" / "pinned_artifact_contracts.json",
    )
    control.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    control.set_defaults(func=command_validate_research_control)

    release = subparsers.add_parser(
        "build-local-release",
        help="Build a validated local bundle of schemas and derived ratings only.",
    )
    release.add_argument(
        "--api-config", type=Path, default=PROJECT_ROOT / "configs" / "api" / "ratings_v2.json"
    )
    release.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    release.add_argument(
        "--contract", type=Path, default=PROJECT_ROOT / "research" / "pinned_artifact_contracts.json"
    )
    release.add_argument("--release-root", type=Path, default=ARTIFACT_ROOT / "releases")
    release.set_defaults(func=command_build_local_release)

    ingest = subparsers.add_parser(
        "ingest", help="Run a resumable HTTP ingestion manifest."
    )
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--root", type=Path, default=BRONZE_ROOT)
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Report verified and remaining files without downloading.",
    )
    ingest.set_defaults(func=command_ingest)

    event_audit = subparsers.add_parser(
        "audit-events", help="Audit and reconcile downloaded event sources."
    )
    event_audit.add_argument(
        "--root", type=Path, default=BRONZE_ROOT / "nba_data_archive"
    )
    event_audit.add_argument("--output", type=Path)
    event_audit.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    event_audit.set_defaults(func=command_audit_events)

    scoring_events = subparsers.add_parser(
        "build-scoring-events",
        help="Build and verify compact 2017-2026 score-change events.",
    )
    scoring_events.add_argument(
        "--root",
        type=Path,
        default=BRONZE_ROOT
        / "nba_data_archive_scoring"
        / "revision=dfa8fa43"
        / "nbastatsv3",
    )
    scoring_events.add_argument(
        "--output",
        type=Path,
        default=SILVER_ROOT / "scoring_events_2017_2026",
    )
    scoring_events.add_argument(
        "--seasons",
        type=_season_list,
        default=tuple(range(2017, 2027)),
    )
    scoring_events.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    scoring_events.add_argument(
        "--legacy-cache", type=Path, default=LEGACY_POSSESSION_CACHE
    )
    scoring_events.add_argument(
        "--official-game-scores",
        type=Path,
        default=BRONZE_ROOT / "official_game_scores" / "official_game_scores.parquet",
    )
    scoring_events.add_argument("--allow-missing-reference", action="store_true")
    scoring_events.set_defaults(func=command_build_scoring_events)

    official_scores = subparsers.add_parser(
        "download-official-game-scores",
        help="Download minimal 2017-2026 official NBA final scores.",
    )
    official_scores.add_argument(
        "--output", type=Path, default=BRONZE_ROOT / "official_game_scores"
    )
    official_scores.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2017, 2027))
    )
    official_scores.add_argument("--max-attempts", type=int, default=20)
    official_scores.add_argument("--request-delay", type=float, default=0.6)
    official_scores.set_defaults(func=command_download_official_game_scores)

    game_dim = subparsers.add_parser(
        "build-game-dim", help="Build the canonical silver game dimension."
    )
    game_dim.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    game_dim.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    game_dim.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    game_dim.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    game_dim.set_defaults(func=command_build_game_dim)

    event_states = subparsers.add_parser(
        "build-event-states",
        help="Build canonical post-action score states from V3 events.",
    )
    event_states.add_argument(
        "--root", type=Path, default=BRONZE_ROOT / "nba_data_archive"
    )
    event_states.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    event_states.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    event_states.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    event_states.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    event_states.set_defaults(func=command_build_event_states)

    player_games = subparsers.add_parser(
        "build-player-games",
        help="Build canonical player-game boxes and exact starter seeds.",
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
    player_games.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    player_games.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    player_games.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    player_games.add_argument(
        "--official-box-dir", type=Path, default=OFFICIAL_BOXSCORE_ROOT
    )
    player_games.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    player_games.set_defaults(func=command_build_player_games)

    identity_dimensions = subparsers.add_parser(
        "build-identity-dimensions",
        help="Build canonical player/team aliases and observed player-team stints.",
    )
    identity_dimensions.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    identity_dimensions.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    identity_dimensions.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    identity_dimensions.add_argument("--output-dir", type=Path, default=SILVER_ROOT)
    identity_dimensions.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    identity_dimensions.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    identity_dimensions.set_defaults(func=command_build_identity_dimensions)

    official_boxes = subparsers.add_parser(
        "ingest-official-boxscores",
        help="Repair quarantined player-game boxes from NBA Stats V3.",
    )
    official_boxes.add_argument(
        "--quality", type=Path, default=SILVER_ROOT / "lineup_game_quality.parquet"
    )
    official_boxes.add_argument("--seasons", type=_text_list, default=())
    official_boxes.add_argument(
        "--output-root", type=Path, default=OFFICIAL_BOXSCORE_ROOT
    )
    official_boxes.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    official_boxes.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    official_boxes.add_argument("--max-attempts", type=int, default=20)
    official_boxes.add_argument("--minimum-delay-seconds", type=float, default=0.6)
    official_boxes.set_defaults(func=command_ingest_official_boxscores)

    lineups = subparsers.add_parser(
        "build-lineups",
        help="Reconstruct and minute-reconcile current five-player lineup stints.",
    )
    lineups.add_argument("--root", type=Path, default=BRONZE_ROOT / "nba_data_archive")
    lineups.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    lineups.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    lineups.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "lineup_stints.parquet"
    )
    lineups.add_argument(
        "--quality-output",
        type=Path,
        default=SILVER_ROOT / "lineup_game_quality.parquet",
    )
    lineups.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    lineups.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    lineups.add_argument("--minute-tolerance-seconds", type=float, default=5.0)
    lineups.add_argument("--max-quarantine-fraction", type=float, default=0.005)
    lineups.set_defaults(func=command_build_lineups)

    possessions = subparsers.add_parser(
        "build-possessions",
        help="Build current possessions and ordinal lineup segments.",
    )
    possessions.add_argument(
        "--root", type=Path, default=BRONZE_ROOT / "nba_data_archive"
    )
    possessions.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    possessions.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    possessions.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    possessions.add_argument(
        "--lineup-quality",
        type=Path,
        default=SILVER_ROOT / "lineup_game_quality.parquet",
    )
    possessions.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    possessions.add_argument(
        "--segments-output",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    possessions.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    possessions.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    possessions.set_defaults(func=command_build_possessions)

    possession_context = subparsers.add_parser(
        "build-possession-start-context",
        help="Build player-neutral context known before each canonical possession.",
    )
    possession_context.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    possession_context.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "possession_start_context.parquet"
    )
    possession_context.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    possession_context.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    possession_context.set_defaults(func=command_build_possession_start_context)

    shot_defense = subparsers.add_parser(
        "build-shot-defense-events",
        help="Build research-only official FGAs with exact ordinal 5v5 lineups.",
    )
    shot_defense.add_argument(
        "--root", type=Path, default=BRONZE_ROOT / "nba_data_archive"
    )
    shot_defense.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    shot_defense.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    shot_defense.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    shot_defense.add_argument(
        "--output", type=Path, default=SILVER_ROOT / "shot_defense_events.parquet"
    )
    shot_defense.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    shot_defense.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    shot_defense.add_argument(
        "--seasons", type=_season_list, default=(2023, 2024, 2025)
    )
    shot_defense.set_defaults(func=command_build_shot_defense_events)

    shot_defense_pilot = subparsers.add_parser(
        "run-shot-defense-pilot",
        help="Run the cost-bounded defense-team shot-model feasibility gate.",
    )
    shot_defense_pilot.add_argument(
        "--panel", type=Path, default=SILVER_ROOT / "shot_defense_events.parquet"
    )
    shot_defense_pilot.add_argument("--season", type=int, default=2024)
    shot_defense_pilot.add_argument("--train-fraction", type=float, default=0.70)
    shot_defense_pilot.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    shot_defense_pilot.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    shot_defense_pilot.set_defaults(func=command_run_shot_defense_pilot)

    rapm = subparsers.add_parser(
        "fit-rapm", help="Fit the independent zero-prior RAPM baseline."
    )
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

    current_rapm = subparsers.add_parser(
        "fit-current-rapm",
        help="Fit current RAPM from canonical CDN possessions and ordinal lineups.",
    )
    current_rapm.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    current_rapm.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    current_rapm.add_argument(
        "--lineup-policy", choices=("start", "terminal"), default="start"
    )
    current_rapm.add_argument("--names", type=Path, default=PLAYER_NAMES)
    current_rapm.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    current_rapm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    current_rapm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    current_rapm.add_argument("--snapshot-id")
    current_rapm.add_argument("--lambda-off", type=float, default=3000.0)
    current_rapm.add_argument("--lambda-def", type=float, default=3000.0)
    current_rapm.add_argument("--lambda-home", type=float, default=300.0)
    current_rapm.add_argument("--game-types", default="regular")
    current_rapm.add_argument("--no-home", action="store_true")
    current_rapm.set_defaults(func=command_fit_current_rapm)

    uncertainty = subparsers.add_parser(
        "quantify-rapm-uncertainty",
        help="Fit frozen terminal-lineup normal RAPM with whole-game bootstrap intervals.",
    )
    uncertainty.add_argument("--source", choices=("legacy", "current"), required=True)
    uncertainty.add_argument("--seasons", type=_season_list, required=True)
    uncertainty.add_argument("--draws", type=int, default=1000)
    uncertainty.add_argument("--seed", type=int, default=20260812)
    uncertainty.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    uncertainty.add_argument("--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet")
    uncertainty.add_argument("--segments", type=Path, default=SILVER_ROOT / "possession_lineup_segments.parquet")
    uncertainty.add_argument("--names", type=Path, default=PLAYER_NAMES)
    uncertainty.add_argument("--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet")
    uncertainty.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    uncertainty.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    uncertainty.add_argument("--lambda-off", type=float, default=3000.0)
    uncertainty.add_argument("--lambda-def", type=float, default=3000.0)
    uncertainty.add_argument("--lambda-home", type=float, default=300.0)
    uncertainty.add_argument("--game-types", default="regular")
    uncertainty.set_defaults(func=command_quantify_rapm_uncertainty)

    normal_rapm = subparsers.add_parser(
        "tune-normal-rapm",
        help="Select normal RAPM penalties on 2024-25 and confirm on 2025-26.",
    )
    normal_rapm.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    normal_rapm.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    normal_rapm.add_argument("--names", type=Path, default=PLAYER_NAMES)
    normal_rapm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    normal_rapm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    normal_rapm.add_argument(
        "--penalty-candidates",
        type=_penalty_triples,
        default=_penalty_triples(
            "1000:1000:100,1000:1000:300,1000:1000:1000,"
            "2000:2000:100,2000:2000:300,2000:2000:1000,"
            "3000:3000:100,3000:3000:300,3000:3000:1000,"
            "4500:4500:100,4500:4500:300,4500:4500:1000,"
            "6000:6000:100,6000:6000:300,6000:6000:1000,"
            "1000:3000:300,2000:4500:300,3000:6000:300,"
            "4500:2000:300,6000:3000:300"
        ),
    )
    normal_rapm.set_defaults(func=command_tune_normal_rapm)

    statistical_features = subparsers.add_parser(
        "build-statistical-features",
        help="Build pooled three-season box and tracking features.",
    )
    statistical_features.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/playersheets/year_totals"),
    )
    statistical_features.add_argument(
        "--window-ends",
        type=_season_list,
        default=tuple(range(2016, 2025)),
    )
    statistical_features.add_argument("--window-seasons", type=int, default=3)
    statistical_features.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_features.set_defaults(func=command_build_statistical_features)

    statistical_features_v2 = subparsers.add_parser(
        "build-statistical-features-v2",
        help="Add stabilized, era-relative, temporal, role, and interaction features.",
    )
    statistical_features_v2.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/playersheets/year_totals"),
    )
    statistical_features_v2.add_argument(
        "--pooled-window-seasons", type=int, default=3
    )
    statistical_features_v2.add_argument("--base-features", type=Path, required=True)
    statistical_features_v2.add_argument("--playtype-features", type=Path)
    statistical_features_v2.add_argument("--defensive-tracking-features", type=Path)
    statistical_features_v2.add_argument("--assist-quality-features", type=Path)
    statistical_features_v2.add_argument("--matchup-defense-features", type=Path)
    statistical_features_v2.add_argument("--player-skill-features", type=Path)
    statistical_features_v2.add_argument("--behavior-roles", type=Path)
    statistical_features_v2.add_argument("--offense-roles", type=Path)
    statistical_features_v2.add_argument("--defense-roles", type=Path)
    statistical_features_v2.add_argument(
        "--window-ends",
        type=_season_list,
        default=tuple(range(2016, 2025)),
    )
    statistical_features_v2.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_features_v2.set_defaults(func=command_build_statistical_features_v2)

    playtype_features = subparsers.add_parser(
        "build-playtype-features",
        help="Build annual zTS and playtype points-over-expectation features.",
    )
    playtype_features.add_argument("--playtype-source", type=Path, required=True)
    playtype_features.add_argument(
        "--box-source-dir", type=Path,
        default=Path("data/raw/playersheets/year_totals"),
    )
    playtype_features.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    playtype_features.add_argument("--minimum-minutes", type=float, default=250.0)
    playtype_features.add_argument("--minimum-synergy-possessions", type=float, default=50.0)
    playtype_features.add_argument("--minimum-league-row-possessions", type=float, default=20.0)
    playtype_features.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    playtype_features.set_defaults(func=command_build_playtype_features)

    defensive_tracking = subparsers.add_parser(
        "build-defensive-tracking-features",
        help="Build annual DFG, rim-defense, and hustle features.",
    )
    defensive_tracking.add_argument("--dfg-source", type=Path, required=True)
    defensive_tracking.add_argument("--rim-dfg-source", type=Path, required=True)
    defensive_tracking.add_argument("--hustle-source", type=Path, required=True)
    defensive_tracking.add_argument(
        "--box-source-dir", type=Path,
        default=Path("data/raw/playersheets/year_totals"),
    )
    defensive_tracking.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    defensive_tracking.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    defensive_tracking.set_defaults(func=command_build_defensive_tracking_features)

    matchup_defense = subparsers.add_parser(
        "build-matchup-defense-features",
        help="Build opponent-adjusted annual primary-defender matchup features.",
    )
    matchup_defense.add_argument(
        "--archive-root",
        type=Path,
        default=(
            BRONZE_ROOT
            / "shufinskiy_nba_data"
            / "revision=e829d46"
            / "matchups"
        ),
    )
    matchup_defense.add_argument(
        "--box-source-dir", type=Path, default=LEGACY_PLAYER_SHEETS
    )
    matchup_defense.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2018, 2026))
    )
    matchup_defense.add_argument(
        "--defender-prior-possessions", type=float, default=500.0
    )
    matchup_defense.add_argument(
        "--shooting-prior-attempts", type=float, default=200.0
    )
    matchup_defense.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    matchup_defense.set_defaults(func=command_build_matchup_defense_features)

    assist_quality = subparsers.add_parser(
        "build-assist-quality-features",
        help="Build corrected annual free-throw-adjusted assist-quality features.",
    )
    assist_quality.add_argument("--source", type=Path, required=True)
    assist_quality.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    assist_quality.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    assist_quality.set_defaults(func=command_build_assist_quality_features)

    role_context = subparsers.add_parser(
        "build-role-context-features",
        help="Build research-only annual shooting-context features by dribble bucket.",
    )
    role_context.add_argument("--shooting-by-dribble-source", type=Path, required=True)
    role_context.add_argument("--jump-shot-by-dribble-source", type=Path, required=True)
    role_context.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2026))
    )
    role_context.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    role_context.set_defaults(func=command_build_role_context_features)

    player_skill = subparsers.add_parser(
        "build-player-skill-features",
        help="Build annual shot-making, passing, screening, and hustle skill features.",
    )
    player_skill.add_argument("--shooting-source", type=Path, required=True)
    player_skill.add_argument("--passing-source", type=Path, required=True)
    player_skill.add_argument("--hustle-source", type=Path, required=True)
    player_skill.add_argument("--shotzone-source", type=Path, required=True)
    player_skill.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    player_skill.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    player_skill.set_defaults(func=command_build_player_skill_features)

    behavior_roles = subparsers.add_parser(
        "build-behavior-roles",
        help="Build behavior-only annual role axes, affinities, and stability diagnostics.",
    )
    behavior_roles.add_argument("--annual-features", type=Path, required=True)
    behavior_roles.add_argument("--role-context", type=Path, required=True)
    behavior_roles.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    behavior_roles.set_defaults(func=command_build_behavior_roles)

    side_roles = subparsers.add_parser(
        "build-side-roles",
        help="Build separate offense and defense deployment-role maps.",
    )
    side_roles.add_argument("--annual-features", type=Path, required=True)
    side_roles.add_argument("--dribble-context", type=Path, required=True)
    side_roles.add_argument("--playtype-source", type=Path, required=True)
    side_roles.add_argument("--defensive-tracking", type=Path, required=True)
    side_roles.add_argument("--matchup-archive-root", type=Path, required=True)
    side_roles.add_argument(
        "--offense-seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    side_roles.add_argument(
        "--defense-seasons", type=_season_list, default=tuple(range(2018, 2025))
    )
    side_roles.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    side_roles.set_defaults(func=command_build_side_roles)

    stable_roles = subparsers.add_parser(
        "stabilize-roles",
        help="Select and apply forward-only role-membership smoothing.",
    )
    stable_roles.add_argument("--side-roles-run", type=Path, required=True)
    stable_roles.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    stable_roles.set_defaults(func=command_stabilize_roles)

    defense_role_challenger = subparsers.add_parser(
        "run-defense-role-challenger",
        help="Select new defense information on fixed older seasons and score diagnostics.",
    )
    defense_role_challenger.add_argument("--features", type=Path, required=True)
    defense_role_challenger.add_argument("--targets", type=Path, required=True)
    defense_role_challenger.add_argument("--frozen-spm-run", type=Path, required=True)
    defense_role_challenger.add_argument(
        "--contract", type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "defense_role_challenger_v1.json",
    )
    defense_role_challenger.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    defense_role_challenger.set_defaults(func=command_run_defense_role_challenger)

    statistical_impact = subparsers.add_parser(
        "fit-statistical-impact",
        help="Fit the first purged three-season normal-RAPM statistical baseline.",
    )
    statistical_impact.add_argument(
        "--features",
        type=Path,
        default=Path("rapm/data/spm_features_windows.parquet"),
    )
    statistical_impact.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_impact.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_impact.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_impact.set_defaults(func=command_fit_statistical_impact)

    statistical_models = subparsers.add_parser(
        "compare-statistical-models",
        help="Compare ridge, elastic net, and histogram GBM on purged folds.",
    )
    statistical_models.add_argument("--features", type=Path, required=True)
    statistical_models.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_models.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_models.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_models.set_defaults(func=command_compare_statistical_models)

    statistical_interpretability = subparsers.add_parser(
        "interpret-statistical-aio",
        help="Measure grouped feature reliance on a frozen diagnostic fold.",
    )
    statistical_interpretability.add_argument("--features", type=Path, required=True)
    statistical_interpretability.add_argument("--reference-run", type=Path, required=True)
    statistical_interpretability.add_argument(
        "--targets", type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_interpretability.add_argument("--test-window-end", type=int, default=2024)
    statistical_interpretability.add_argument("--group-repeats", type=int, default=20)
    statistical_interpretability.add_argument("--individual-repeats", type=int, default=3)
    statistical_interpretability.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_interpretability.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_interpretability.set_defaults(func=command_interpret_statistical_aio)

    aging_balance = subparsers.add_parser(
        "validate-aging-balance",
        help="Score annual ratings forward and backward with earlier-only age adjustment.",
    )
    aging_balance.add_argument("--predictions", type=Path, required=True)
    aging_balance.add_argument(
        "--age-source-dir", type=Path, default=LEGACY_PLAYER_SHEETS
    )
    aging_balance.add_argument("--minimum-training-origins", type=int, default=3)
    aging_balance.add_argument("--aging-ridge-alpha", type=float, default=10.0)
    aging_balance.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    aging_balance.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    aging_balance.set_defaults(func=command_validate_aging_balance)

    statistical_direct_net = subparsers.add_parser(
        "compare-statistical-direct-net",
        help="Compare direct nonlinear net RAPM with decomposed predictions.",
    )
    statistical_direct_net.add_argument("--features", type=Path, required=True)
    statistical_direct_net.add_argument("--component-run", type=Path, required=True)
    statistical_direct_net.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_direct_net.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    statistical_direct_net.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_direct_net.set_defaults(func=command_compare_statistical_direct_net)

    statistical_ablation = subparsers.add_parser(
        "ablate-statistical-features",
        help="Ablate feature families with frozen offense and defense models.",
    )
    statistical_ablation.add_argument("--features", type=Path, required=True)
    statistical_ablation.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_ablation.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_ablation.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_ablation.set_defaults(func=command_ablate_statistical_features)

    optimized_statistical_aio = subparsers.add_parser(
        "fit-optimized-statistical-aio",
        help="Fit frozen optimized offense and defense statistical AIO models.",
    )
    optimized_statistical_aio.add_argument("--features", type=Path, required=True)
    optimized_statistical_aio.add_argument("--ablation-run", type=Path, required=True)
    optimized_statistical_aio.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    optimized_statistical_aio.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    optimized_statistical_aio.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    optimized_statistical_aio.set_defaults(func=command_fit_optimized_statistical_aio)

    statistical_features_v2_comparison = subparsers.add_parser(
        "compare-statistical-features-v2",
        help="Select v2 feature blocks with the frozen offense and defense learners.",
    )
    statistical_features_v2_comparison.add_argument("--features", type=Path, required=True)
    statistical_features_v2_comparison.add_argument("--baseline-run", type=Path, required=True)
    statistical_features_v2_comparison.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_features_v2_comparison.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    statistical_features_v2_comparison.add_argument(
        "--registry", type=Path, default=REGISTRY_PATH
    )
    statistical_features_v2_comparison.set_defaults(
        func=command_compare_statistical_features_v2
    )

    statistical_priors = subparsers.add_parser(
        "build-statistical-priors",
        help="Build purged cross-fitted statistical priors for historical RAPM windows.",
    )
    statistical_priors.add_argument("--features", type=Path, required=True)
    statistical_priors.add_argument("--reference-run", type=Path, required=True)
    statistical_priors.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "rapm/outputs/rapm_results/final_20260703_hl250/rapm_all_windows.csv"
        ),
    )
    statistical_priors.add_argument(
        "--window-ends", type=_season_list, default=None
    )
    statistical_priors.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    statistical_priors.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    statistical_priors.set_defaults(func=command_build_statistical_priors)

    prior_informed_rapm = subparsers.add_parser(
        "compare-prior-informed-rapm",
        help="Compare zero-prior, prior-only, and prior-centered normal RAPM.",
    )
    prior_informed_rapm.add_argument("--priors", type=Path, required=True)
    prior_informed_rapm.add_argument(
        "--test-seasons", type=_season_list, default=(2020, 2021, 2022, 2023, 2024)
    )
    prior_informed_rapm.add_argument(
        "--selection-test-seasons", type=_season_list, default=(2020, 2021, 2022)
    )
    prior_informed_rapm.add_argument(
        "--confirmation-test-seasons", type=_season_list, default=(2023, 2024)
    )
    prior_informed_rapm.add_argument("--train-window", type=int, default=3)
    prior_informed_rapm.add_argument(
        "--prior-scales", type=_float_list, default=(0.25, 0.5, 0.75, 1.0)
    )
    prior_informed_rapm.add_argument("--lambda-off", type=float, default=3000.0)
    prior_informed_rapm.add_argument("--lambda-def", type=float, default=3000.0)
    prior_informed_rapm.add_argument("--lambda-home", type=float, default=300.0)
    prior_informed_rapm.add_argument("--bootstrap-repetitions", type=int, default=2000)
    prior_informed_rapm.add_argument("--seed", type=int, default=20260810)
    prior_informed_rapm.add_argument(
        "--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE
    )
    prior_informed_rapm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    prior_informed_rapm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    prior_informed_rapm.set_defaults(func=command_compare_prior_informed_rapm)

    precision_aware_prior = subparsers.add_parser(
        "compare-precision-aware-prior",
        help="Run the frozen four-model precision-aware RAPM prior comparison.",
    )
    precision_aware_prior.add_argument("--priors", type=Path, required=True)
    precision_aware_prior.add_argument("--calibration", type=Path, required=True)
    precision_aware_prior.add_argument(
        "--contract", type=Path,
        default=Path("research/experiments/precision_aware_prior_rapm_v1.yml"),
    )
    precision_aware_prior.add_argument(
        "--test-seasons", type=_season_list, default=(2021, 2022, 2023, 2024)
    )
    precision_aware_prior.add_argument(
        "--selection-seasons", type=_season_list, default=(2021, 2022)
    )
    precision_aware_prior.add_argument(
        "--diagnostic-seasons", type=_season_list, default=(2023, 2024)
    )
    precision_aware_prior.add_argument("--train-window", type=int, default=3)
    precision_aware_prior.add_argument("--lambda-off", type=float, default=3000.0)
    precision_aware_prior.add_argument("--lambda-def", type=float, default=3000.0)
    precision_aware_prior.add_argument("--lambda-home", type=float, default=300.0)
    precision_aware_prior.add_argument("--bootstrap-repetitions", type=int, default=2000)
    precision_aware_prior.add_argument("--seed", type=int, default=20260812)
    precision_aware_prior.add_argument(
        "--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE
    )
    precision_aware_prior.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    precision_aware_prior.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    precision_aware_prior.set_defaults(func=command_compare_precision_aware_prior)

    external_ingest = subparsers.add_parser(
        "ingest-external-impact",
        help="Download versioned Basketball Reference BPM and xRAPM pages.",
    )
    external_ingest.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2017, 2025))
    )
    external_ingest.add_argument(
        "--raw-root", type=Path, default=BRONZE_ROOT / "external_impact"
    )
    external_ingest.set_defaults(func=command_ingest_external_impact)

    external_benchmark = subparsers.add_parser(
        "benchmark-external-impact",
        help="Compare three-season statistical priors with BPM and xRAPM.",
    )
    external_benchmark.add_argument("--priors", type=Path, required=True)
    external_benchmark.add_argument("--features", type=Path, required=True)
    external_benchmark.add_argument("--names", type=Path, default=PLAYER_NAMES)
    external_benchmark.add_argument(
        "--raw-root", type=Path, default=BRONZE_ROOT / "external_impact"
    )
    external_benchmark.add_argument(
        "--window-ends", type=_season_list, default=(2019, 2020, 2021, 2022, 2023, 2024)
    )
    external_benchmark.add_argument("--minimum-possessions", type=float, default=3000.0)
    external_benchmark.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    external_benchmark.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    external_benchmark.set_defaults(func=command_benchmark_external_impact)

    annual_rapm_targets = subparsers.add_parser(
        "build-single-season-rapm-targets",
        help="Fit one zero-prior normal RAPM target table per regular season.",
    )
    annual_rapm_targets.add_argument(
        "--seasons", type=_season_list, default=tuple(range(2014, 2025))
    )
    annual_rapm_targets.add_argument(
        "--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE
    )
    annual_rapm_targets.add_argument("--lambda-off", type=float, default=3000.0)
    annual_rapm_targets.add_argument("--lambda-def", type=float, default=3000.0)
    annual_rapm_targets.add_argument("--lambda-home", type=float, default=300.0)
    annual_rapm_targets.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    annual_rapm_targets.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    annual_rapm_targets.set_defaults(func=command_build_single_season_rapm_targets)

    current_annual_rapm_targets = subparsers.add_parser(
        "build-current-single-season-rapm-targets",
        help="Fit canonical annual terminal-lineup zero-prior normal-RAPM targets.",
    )
    current_annual_rapm_targets.add_argument(
        "--seasons", type=_season_list, default=(2024, 2025, 2026)
    )
    current_annual_rapm_targets.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    current_annual_rapm_targets.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    current_annual_rapm_targets.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    current_annual_rapm_targets.add_argument("--names", type=Path, default=PLAYER_NAMES)
    current_annual_rapm_targets.add_argument("--lambda-off", type=float, default=3000.0)
    current_annual_rapm_targets.add_argument("--lambda-def", type=float, default=3000.0)
    current_annual_rapm_targets.add_argument("--lambda-home", type=float, default=300.0)
    current_annual_rapm_targets.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    current_annual_rapm_targets.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    current_annual_rapm_targets.set_defaults(
        func=command_build_current_single_season_rapm_targets
    )

    annual_target_panel = subparsers.add_parser(
        "build-canonical-annual-target-panel",
        help="Audit and join legacy and canonical annual normal-RAPM targets.",
    )
    annual_target_panel.add_argument("--legacy-targets", type=Path, required=True)
    annual_target_panel.add_argument("--canonical-targets", type=Path, required=True)
    annual_target_panel.add_argument("--transition-season", type=int, default=2024)
    annual_target_panel.add_argument("--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet")
    annual_target_panel.add_argument("--names", type=Path, default=PLAYER_NAMES)
    annual_target_panel.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    annual_target_panel.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    annual_target_panel.set_defaults(func=command_build_canonical_annual_target_panel)

    trajectories = subparsers.add_parser(
        "build-time-decayed-trajectories",
        help="Build a no-future-leakage time-decayed annual normal-RAPM baseline.",
    )
    trajectories.add_argument("--targets", type=Path, required=True)
    trajectories.add_argument("--names", type=Path, default=PLAYER_NAMES)
    trajectories.add_argument(
        "--candidate-decays", type=_float_list, default=(0.50, 0.65, 0.80, 0.90)
    )
    trajectories.add_argument(
        "--candidate-exposure-powers", type=_float_list, default=(0.0, 0.5, 1.0)
    )
    trajectories.add_argument(
        "--selection-origins", type=_season_list, default=(2018, 2019, 2020, 2021)
    )
    trajectories.add_argument(
        "--diagnostic-origins", type=_season_list, default=(2022, 2023)
    )
    trajectories.add_argument("--minimum-side-possessions", type=float, default=1000.0)
    trajectories.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    trajectories.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    trajectories.set_defaults(func=command_build_time_decayed_trajectory)

    annual_variance = subparsers.add_parser(
        "build-annual-observation-variance",
        help="Build annual normal-RAPM CR0 observation-variance diagnostics.",
    )
    annual_variance.add_argument("--targets", type=Path, required=True)
    annual_variance.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    annual_variance.add_argument("--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet")
    annual_variance.add_argument("--segments", type=Path, default=SILVER_ROOT / "possession_lineup_segments.parquet")
    annual_variance.add_argument("--transition-season", type=int, default=2024)
    annual_variance.add_argument("--seasons", type=_season_list)
    annual_variance.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    annual_variance.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    annual_variance.set_defaults(func=command_build_annual_observation_variance)

    state_space = subparsers.add_parser(
        "build-state-space-trajectory",
        help="Select a causal annual state-space challenger against frozen time decay.",
    )
    state_space.add_argument("--targets", type=Path, required=True)
    state_space.add_argument("--observation-variance", type=Path, required=True)
    state_space.add_argument("--names", type=Path, required=True)
    state_space.add_argument("--time-decay-trajectories", type=Path, required=True)
    state_space.add_argument("--candidate-phis", type=_float_list, default=(0.50, 0.65, 0.80, 0.90))
    state_space.add_argument("--candidate-process-sds", type=_float_list, default=(0.25, 0.50, 1.00, 2.00))
    state_space.add_argument("--selection-origins", type=_season_list, default=(2018, 2019, 2020, 2021))
    state_space.add_argument("--diagnostic-origins", type=_season_list, default=(2022, 2023))
    state_space.add_argument("--minimum-side-possessions", type=float, default=1000.0)
    state_space.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    state_space.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    state_space.set_defaults(func=command_build_state_space_trajectory)

    aging_projection = subparsers.add_parser(
        "build-aging-projection",
        help="Select walk-forward aging adjustments and project players and teams.",
    )
    aging_projection.add_argument("--trajectories", type=Path, required=True)
    aging_projection.add_argument("--targets", type=Path, required=True)
    aging_projection.add_argument(
        "--player-sheets-dir", type=Path, default=LEGACY_PLAYER_SHEETS
    )
    aging_projection.add_argument(
        "--selection-origins", type=_season_list, default=(2018, 2019, 2020, 2021)
    )
    aging_projection.add_argument(
        "--diagnostic-origins", type=_season_list, default=(2022, 2023)
    )
    aging_projection.add_argument("--projection-origin", type=int, default=2026)
    aging_projection.add_argument("--minimum-side-possessions", type=float, default=1000.0)
    aging_projection.add_argument("--alpha", type=float, default=25.0)
    aging_projection.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    aging_projection.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    aging_projection.set_defaults(func=command_build_aging_projection)

    expected_possession = subparsers.add_parser(
        "build-expected-possession-points",
        help="Cross-fit player-neutral expected points from possession-start context.",
    )
    expected_possession.add_argument(
        "--context", type=Path, default=SILVER_ROOT / "possession_start_context.parquet"
    )
    expected_possession.add_argument("--test-seasons", type=_season_list, default=(2024, 2025))
    expected_possession.add_argument("--alpha", type=float, default=0.01)
    expected_possession.add_argument("--max-iter", type=int, default=300)
    expected_possession.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    expected_possession.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    expected_possession.set_defaults(func=command_build_expected_possession_points)

    annual_spm = subparsers.add_parser(
        "fit-single-season-spm",
        help="Fit season-held-out and final single-season statistical plus-minus.",
    )
    annual_spm.add_argument("--features", type=Path, required=True)
    annual_spm.add_argument("--targets", type=Path, required=True)
    annual_spm.add_argument("--reference-run", type=Path, required=True)
    annual_spm.add_argument("--names", type=Path, default=PLAYER_NAMES)
    annual_spm.add_argument(
        "--external-raw-root", type=Path, default=BRONZE_ROOT / "external_impact"
    )
    annual_spm.add_argument(
        "--output-seasons", type=_season_list, default=tuple(range(2017, 2025))
    )
    annual_spm.add_argument("--minimum-possessions", type=float, default=1000.0)
    annual_spm.add_argument(
        "--additional-offense-features", type=_text_list, default=()
    )
    annual_spm.add_argument(
        "--additional-defense-features", type=_text_list, default=()
    )
    annual_spm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    annual_spm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    annual_spm.set_defaults(func=command_fit_single_season_spm)

    forward_annual_spm = subparsers.add_parser(
        "build-forward-annual-spm-priors",
        help="Fit each annual SPM only on earlier seasons for next-season tests.",
    )
    forward_annual_spm.add_argument("--features", type=Path, required=True)
    forward_annual_spm.add_argument("--targets", type=Path, required=True)
    forward_annual_spm.add_argument("--reference-run", type=Path, required=True)
    forward_annual_spm.add_argument("--contract", type=Path, required=True)
    forward_annual_spm.add_argument(
        "--output-seasons", type=_season_list, default=tuple(range(2017, 2024))
    )
    forward_annual_spm.add_argument("--minimum-training-seasons", type=int, default=3)
    forward_annual_spm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    forward_annual_spm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    forward_annual_spm.set_defaults(func=command_build_forward_annual_spm_priors)

    oof_annual_spm = subparsers.add_parser(
        "build-oof-annual-spm-priors",
        help="Convert leave-one-season-out SPM predictions into annual RAPM centers.",
    )
    oof_annual_spm.add_argument("--spm-run", type=Path, required=True)
    oof_annual_spm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    oof_annual_spm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    oof_annual_spm.set_defaults(func=command_build_oof_annual_spm_priors)

    annual_aio = subparsers.add_parser(
        "build-annual-aio-ratings",
        help="Build decomposed annual ratings from full-SPM-center normal RAPM.",
    )
    annual_aio.add_argument("--priors", type=Path, required=True)
    annual_aio.add_argument("--seasons", type=_season_list, default=tuple(range(2017, 2025)))
    annual_aio.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    annual_aio.add_argument("--names", type=Path, default=PLAYER_NAMES)
    annual_aio.add_argument("--lambda-off", type=float, default=3000.0)
    annual_aio.add_argument("--lambda-def", type=float, default=3000.0)
    annual_aio.add_argument("--lambda-home", type=float, default=300.0)
    annual_aio.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    annual_aio.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    annual_aio.set_defaults(func=command_build_annual_aio_ratings)

    rolling_peaks = subparsers.add_parser(
        "build-rolling-rapm-peaks",
        help="Build independent three-year and five-year normal-RAPM peaks.",
    )
    rolling_peaks.add_argument("--contract", type=Path, required=True)
    rolling_peaks.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    rolling_peaks.add_argument("--names", type=Path, default=PLAYER_NAMES)
    rolling_peaks.add_argument(
        "--player-sheets-dir", type=Path, default=LEGACY_PLAYER_SHEETS
    )
    rolling_peaks.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    rolling_peaks.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    rolling_peaks.set_defaults(func=command_build_rolling_rapm_peaks)

    peak_uncertainty = subparsers.add_parser(
        "quantify-rolling-peak-uncertainty",
        help="Refit and reselect rolling RAPM peaks inside each whole-game bootstrap draw.",
    )
    peak_uncertainty.add_argument("--contract", type=Path, required=True)
    peak_uncertainty.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    peak_uncertainty.add_argument("--names", type=Path, default=PLAYER_NAMES)
    peak_uncertainty.add_argument("--player-sheets-dir", type=Path, default=LEGACY_PLAYER_SHEETS)
    peak_uncertainty.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    peak_uncertainty.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    peak_uncertainty.add_argument("--draws", type=int, default=1000)
    peak_uncertainty.add_argument("--seed", type=int, default=20260812)
    peak_uncertainty.set_defaults(func=command_quantify_rolling_peak_uncertainty)

    current_spm = subparsers.add_parser(
        "confirm-current-spm",
        help="Score the frozen annual SPM once on a new current-data season.",
    )
    current_spm.add_argument("--features", type=Path, required=True)
    current_spm.add_argument("--frozen-spm-run", type=Path, required=True)
    current_spm.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    current_spm.add_argument(
        "--segments", type=Path, default=SILVER_ROOT / "possession_lineup_segments.parquet"
    )
    current_spm.add_argument("--names", type=Path, default=PLAYER_NAMES)
    current_spm.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    current_spm.add_argument("--season", type=int, default=2025)
    current_spm.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    current_spm.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    current_spm.set_defaults(func=command_confirm_current_spm)

    current_spm_diagnostics = subparsers.add_parser(
        "diagnose-current-spm",
        help="Diagnose a frozen current-season SPM miss without model tuning.",
    )
    current_spm_diagnostics.add_argument("--confirmation-run", type=Path, required=True)
    current_spm_diagnostics.add_argument("--current-features", type=Path, required=True)
    current_spm_diagnostics.add_argument("--reference-features", type=Path, required=True)
    current_spm_diagnostics.add_argument("--frozen-spm-run", type=Path, required=True)
    current_spm_diagnostics.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    current_spm_diagnostics.add_argument(
        "--segments", type=Path, default=SILVER_ROOT / "possession_lineup_segments.parquet"
    )
    current_spm_diagnostics.add_argument("--season", type=int, default=2025)
    current_spm_diagnostics.add_argument("--comparison-season", type=int, default=2024)
    current_spm_diagnostics.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    current_spm_diagnostics.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    current_spm_diagnostics.set_defaults(func=command_diagnose_current_spm)

    defense_ridge_nested = subparsers.add_parser(
        "run-annual-defense-ridge-nested",
        help="Select annual defensive ridge regularization on past seasons only.",
    )
    defense_ridge_nested.add_argument("--features", type=Path, required=True)
    defense_ridge_nested.add_argument("--targets", type=Path, required=True)
    defense_ridge_nested.add_argument("--frozen-spm-run", type=Path, required=True)
    defense_ridge_nested.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "annual_defense_ridge_nested_v1.json",
    )
    defense_ridge_nested.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    defense_ridge_nested.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    defense_ridge_nested.set_defaults(func=command_run_annual_defense_ridge_nested)

    defense_features_nested = subparsers.add_parser(
        "run-annual-defense-features-nested",
        help="Select predeclared annual defense feature blocks on past seasons only.",
    )
    defense_features_nested.add_argument("--features", type=Path, required=True)
    defense_features_nested.add_argument("--targets", type=Path, required=True)
    defense_features_nested.add_argument("--frozen-spm-run", type=Path, required=True)
    defense_features_nested.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "annual_defense_features_nested_v1.json",
    )
    defense_features_nested.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    defense_features_nested.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    defense_features_nested.set_defaults(func=command_run_annual_defense_features_nested)

    ratings_api = subparsers.add_parser(
        "serve-ratings",
        help="Serve pinned annual, rolling, peak, and player rating data as JSON.",
    )
    ratings_api.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "api" / "ratings_v1.json",
    )
    ratings_api.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_ROOT / "models",
        help="Directory containing the annual_aio_ratings and rolling_rapm_peaks families.",
    )
    ratings_api.add_argument("--host", default="127.0.0.1")
    ratings_api.add_argument("--port", type=int, default=8765)
    ratings_api.set_defaults(func=command_serve_ratings)

    web_snapshot = subparsers.add_parser(
        "build-web-snapshot",
        help="Export derived rating data for the static web client.",
    )
    web_snapshot.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "api" / "ratings_v2.json",
    )
    web_snapshot.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT / "models"
    )
    web_snapshot.add_argument(
        "--aging-curve",
        type=Path,
        default=PROJECT_ROOT / "rapm" / "outputs" / "aging" / "aging_curve_delta.csv",
    )
    web_snapshot.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "web" / "public" / "data",
    )
    web_snapshot.add_argument(
        "--spm-run",
        type=Path,
        default=ARTIFACT_ROOT
        / "models"
        / "single_season_spm"
        / "single_season_spm_v1_18496a1348",
    )
    web_snapshot.add_argument(
        "--player-sheets-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "playersheets" / "year_totals",
    )
    web_snapshot.add_argument(
        "--features",
        type=Path,
        default=ARTIFACT_ROOT
        / "features"
        / "statistical_impact"
        / "statistical_features_v2_907e4ee4b9"
        / "features.parquet",
    )
    web_snapshot.add_argument(
        "--walk-forward-run",
        type=Path,
        default=ARTIFACT_ROOT
        / "models"
        / "annual_spm_priors"
        / "annual_spm_priors_v1_1107680642",
    )
    web_snapshot.add_argument(
        "--walk-backward-run",
        type=Path,
        default=ARTIFACT_ROOT
        / "models"
        / "aging_balanced_validation"
        / "aging_balanced_validation_v1_ec5122d5a3",
    )
    web_snapshot.add_argument(
        "--aging-projection-run",
        type=Path,
        default=ARTIFACT_ROOT
        / "models"
        / "aging_projection"
        / "aging_projection_v1_6a288b493e",
    )
    web_snapshot.add_argument(
        "--win-probability-run",
        type=Path,
        default=ARTIFACT_ROOT
        / "models"
        / "win_probability_lineup"
        / "wp_pregame_ablation_v3_cdbcea84ee",
    )
    web_snapshot.add_argument("--shards", type=int, default=128)
    web_snapshot.set_defaults(func=command_build_web_snapshot)

    compare = subparsers.add_parser(
        "compare-rapm",
        help="Compare fixed RAPM penalties on one chronological diagnostic fold.",
    )
    compare.add_argument("--seasons", type=_season_list, required=True)
    compare.add_argument("--cache-dir", type=Path, default=LEGACY_POSSESSION_CACHE)
    compare.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    compare.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    compare.add_argument("--snapshot-id")
    compare.add_argument(
        "--lambda-pairs",
        type=_lambda_pairs,
        default=_lambda_pairs(
            "500:500,1000:1000,2000:2000,3000:3000,6000:6000,2000:4500"
        ),
    )
    compare.add_argument("--lambda-home", type=float, default=300.0)
    compare.add_argument("--game-types", default="regular")
    compare.add_argument("--no-home", action="store_true")
    compare.set_defaults(func=command_compare_rapm)

    walk = subparsers.add_parser(
        "walk-forward-rapm",
        help="Compare RAPM candidates across chronological outer seasons.",
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
        "fit-win-probability",
        help="Fit the chronological state-only win-probability baseline.",
    )
    win_probability.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    win_probability.add_argument(
        "--train-seasons", type=_text_list, default=("2024-25",)
    )
    win_probability.add_argument(
        "--test-seasons", type=_text_list, default=("2025-26",)
    )
    win_probability.add_argument("--interval-seconds", type=int, default=30)
    win_probability.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    win_probability.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    win_probability.add_argument("--snapshot-id")
    win_probability.set_defaults(func=command_fit_win_probability)

    wp_compare = subparsers.add_parser(
        "compare-win-probability",
        help="Compare state-only WP with time-safe pregame Elo.",
    )
    wp_compare.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    wp_compare.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_compare.add_argument("--train-seasons", type=_text_list, default=("2024-25",))
    wp_compare.add_argument("--test-seasons", type=_text_list, default=("2025-26",))
    wp_compare.add_argument("--interval-seconds", type=int, default=30)
    wp_compare.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_compare.add_argument("--seed", type=int, default=7)
    wp_compare.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_compare.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_compare.add_argument("--snapshot-id")
    wp_compare.set_defaults(func=command_compare_win_probability)

    espn_ingest = subparsers.add_parser(
        "ingest-espn-win-probability",
        help="Cache ESPN play-level win probabilities for canonical games.",
    )
    espn_ingest.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    espn_ingest.add_argument("--seasons", type=_text_list, default=("2025-26",))
    espn_ingest.add_argument(
        "--raw-root", type=Path, default=BRONZE_ROOT / "espn_win_probability"
    )
    espn_ingest.add_argument(
        "--output",
        type=Path,
        default=SILVER_ROOT / "espn_win_probability_index.parquet",
    )
    espn_ingest.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    espn_ingest.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    espn_ingest.add_argument("--max-workers", type=int, default=4)
    espn_ingest.set_defaults(func=command_ingest_espn_win_probability)

    wp_benchmark = subparsers.add_parser(
        "benchmark-win-probability",
        help="Compare the local WP model with ESPN on identical matched play states.",
    )
    wp_benchmark.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    wp_benchmark.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_benchmark.add_argument(
        "--espn-index",
        type=Path,
        default=SILVER_ROOT / "espn_win_probability_index.parquet",
    )
    wp_benchmark.add_argument("--model-run", type=Path, required=True)
    wp_benchmark.add_argument("--clock-tolerance-seconds", type=float, default=1.0)
    wp_benchmark.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_benchmark.add_argument("--seed", type=int, default=7)
    wp_benchmark.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_benchmark.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_benchmark.set_defaults(func=command_benchmark_win_probability)

    wp_lineup = subparsers.add_parser(
        "compare-wp-lineup-strength",
        help="Compare Elo WP with leakage-safe prior-season starter RAPM strength.",
    )
    wp_lineup.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    wp_lineup.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_lineup.add_argument(
        "--player-games", type=Path, default=SILVER_ROOT / "player_games.parquet"
    )
    wp_lineup.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    wp_lineup.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    wp_lineup.add_argument("--legacy-cache", type=Path, default=LEGACY_POSSESSION_CACHE)
    wp_lineup.add_argument(
        "--espn-index",
        type=Path,
        default=SILVER_ROOT / "espn_win_probability_index.parquet",
    )
    wp_lineup.add_argument("--train-season", default="2024-25")
    wp_lineup.add_argument("--test-season", default="2025-26")
    wp_lineup.add_argument("--skip-espn", action="store_true")
    wp_lineup.add_argument("--interval-seconds", type=int, default=30)
    wp_lineup.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_lineup.add_argument("--seed", type=int, default=7)
    wp_lineup.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_lineup.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_lineup.set_defaults(func=command_compare_wp_lineup_strength)

    wp_possession = subparsers.add_parser(
        "compare-wp-possession",
        help="Compare pregame-context WP with leakage-safe possession-start control.",
    )
    wp_possession.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    wp_possession.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_possession.add_argument("--train-season", default="2024-25")
    wp_possession.add_argument("--test-season", default="2025-26")
    wp_possession.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_possession.add_argument("--seed", type=int, default=7)
    wp_possession.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_possession.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_possession.set_defaults(func=command_compare_wp_possession)

    wp_stage1 = subparsers.add_parser(
        "compare-wp-stage1",
        help="Compare frozen logistic, spline GAM, and histogram GBM on two outer folds.",
    )
    wp_stage1.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    wp_stage1.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_stage1.add_argument("--interval-seconds", type=int, default=30)
    wp_stage1.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_stage1.add_argument("--seed", type=int, default=7)
    wp_stage1.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_stage1.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_stage1.set_defaults(func=command_compare_wp_stage1)

    wp_mlp = subparsers.add_parser(
        "compare-wp-mlp",
        help="Compare a fixed five-seed 64x64 feed-forward MLP with logistic.",
    )
    wp_mlp.add_argument(
        "--event-states", type=Path, default=SILVER_ROOT / "event_states.parquet"
    )
    wp_mlp.add_argument(
        "--game-dim", type=Path, default=SILVER_ROOT / "game_dim.parquet"
    )
    wp_mlp.add_argument("--interval-seconds", type=int, default=30)
    wp_mlp.add_argument("--bootstrap-repetitions", type=int, default=5000)
    wp_mlp.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    wp_mlp.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    wp_mlp.set_defaults(func=command_compare_wp_mlp)

    rapm_lineups = subparsers.add_parser(
        "compare-rapm-lineups",
        help="Compare start, terminal, and fractional lineup assignment on identical possessions.",
    )
    rapm_lineups.add_argument(
        "--possessions", type=Path, default=SILVER_ROOT / "possessions.parquet"
    )
    rapm_lineups.add_argument(
        "--segments",
        type=Path,
        default=SILVER_ROOT / "possession_lineup_segments.parquet",
    )
    rapm_lineups.add_argument("--bootstrap-repetitions", type=int, default=5000)
    rapm_lineups.add_argument("--seed", type=int, default=7)
    rapm_lineups.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    rapm_lineups.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    rapm_lineups.set_defaults(func=command_compare_rapm_lineups)

    inp = subparsers.add_parser(
        "benchmark-inpredictable",
        help="Compare the local neutral WP surface with Inpredictable's public calculator.",
    )
    inp.add_argument("--model-run", type=Path, required=True)
    inp.add_argument("--max-workers", type=int, default=4)
    inp.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    inp.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    inp.set_defaults(func=command_benchmark_inpredictable)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
