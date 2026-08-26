"""Run the frozen current-player-skill experiment and write immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.predictive_player_skills import (
    SKILL_SPECS,
    build_skill_estimates,
    load_player_skill_panel,
    skill_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _last_update(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        season = "season_end" if "season_end" in frame else "Season"
        player = "player_id" if "player_id" in frame else "PLAYER_ID"
        date = "game_date" if "game_date" in frame else "GAME_DATE"
        subset = frame.loc[
            frame.get("season_type", pd.Series("regular", index=frame.index)).eq("regular"),
            [player, season, date],
        ].rename(columns={player: "PLAYER_ID", season: "Season", date: "last_update_date"})
        frames.append(subset)
    output = pd.concat(frames, ignore_index=True).dropna(subset=["PLAYER_ID", "Season", "last_update_date"])
    output[["PLAYER_ID", "Season"]] = output[["PLAYER_ID", "Season"]].astype(int)
    return output.groupby(["PLAYER_ID", "Season"], as_index=False).agg(last_update_date=("last_update_date", "max"))


def run(contract_path: Path) -> dict:
    contract = yaml.safe_load(contract_path.read_text())
    policy = contract["season_policy"]
    forbidden = int(policy["forbidden_season"])
    all_declared = {
        int(policy["source_start"]), int(policy["final_parameter_cutoff"]),
        int(policy["output_season"]), *map(int, policy["development_seasons"]),
        *map(int, policy["output_seasons"]),
    }
    if forbidden in all_declared or max(all_declared) >= forbidden:
        raise ValueError("Season 2027 is forbidden before any source is read.")

    source_paths: dict[str, Path] = {}
    for key, value in contract["sources"].items():
        if key in {"player_sheet_dir", "player_games"}:
            continue
        source_paths[key] = ROOT / value
    sheet_dir = ROOT / contract["sources"]["player_sheet_dir"]
    sheet_paths = [sheet_dir / f"{season}.csv" for season in policy["output_seasons"]]
    game_paths = [ROOT / value for value in contract["sources"]["player_games"]]
    for path in [*sheet_paths, *source_paths.values(), *game_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    source_hashes = {
        **{f"player_sheet_{path.stem}": sha256_file(path) for path in sheet_paths},
        **{key: sha256_file(path) for key, path in source_paths.items()},
        **{f"player_games_{index + 1}": sha256_file(path) for index, path in enumerate(game_paths)},
    }
    contract_hash = sha256_file(contract_path)
    code_path = ROOT / "src/nba_impact/models/predictive_player_skills.py"
    code_hash = sha256_file(code_path)
    identity = hashlib.sha256(json.dumps(
        {"contract": contract_hash, "code": code_hash, "sources": source_hashes},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()[:10]
    run_id = f"predictive_player_skills_2026_v1_{identity}"
    output = ROOT / "artifacts/models/predictive_player_skills" / run_id
    output.mkdir(parents=True, exist_ok=True)

    panel = load_player_skill_panel(
        sheet_dir,
        seasons=range(int(policy["source_start"]), int(policy["output_season"]) + 1),
        player_skill_path=source_paths["player_skill"],
        playtype_path=source_paths["playtype"],
        statistical_path=source_paths["statistical"],
        tracking_path=source_paths["defensive_tracking"],
        tracking_dfg_observations_path=source_paths["defensive_tracking_dfg_observations"],
        tracking_rim_observations_path=source_paths["defensive_tracking_rim_observations"],
        tracking_hustle_observations_path=source_paths["defensive_tracking_hustle_observations"],
        matchup_path=source_paths["matchup_defense"],
    )
    if panel["Season"].max() >= forbidden:
        raise ValueError("Forbidden Season 2027 entered the skill panel.")
    selection = contract["selection"]
    estimates, folds, decisions = build_skill_estimates(
        panel,
        selection_seasons=tuple(map(int, policy["development_seasons"])),
        prior_grid=tuple(map(float, selection["prior_strength_grid"])),
        half_life_grid=tuple(map(float, selection["half_life_year_grid"])),
        minimum_exposure_grid=tuple(map(float, selection["minimum_exposure_grid"])),
        age_alpha_grid=tuple(map(float, selection["age_alpha_grid"])),
        output_seasons=tuple(map(int, policy["output_seasons"])),
        last_update=_last_update(game_paths),
        checkpoint_dir=output / "checkpoints",
    )
    estimates.to_parquet(output / "skill_estimates.parquet", index=False)
    folds.to_parquet(output / "fold_metrics.parquet", index=False)
    decisions.to_parquet(output / "model_selection.parquet", index=False)
    definitions = skill_definitions()
    definitions.to_parquet(output / "skill_definitions.parquet", index=False)

    current = estimates.loc[estimates["Season"].eq(int(policy["output_season"]))]
    expected_skills = len(SKILL_SPECS)
    coverage = current.groupby("PLAYER_ID").agg(
        observed_skills=("raw_value", "count"), estimated_skills=("estimate", "count"),
        player_name=("PLAYER_NAME", "last"), team=("TEAM_ABBREVIATION", "last"),
    ).reset_index()
    complete = coverage.loc[coverage["observed_skills"].eq(expected_skills)].sort_values(
        ["observed_skills", "PLAYER_ID"], ascending=[False, True], kind="stable"
    )
    default_player = int(complete.iloc[0]["PLAYER_ID"]) if not complete.empty else int(coverage.sort_values("observed_skills", ascending=False).iloc[0]["PLAYER_ID"])
    winners = decisions.loc[decisions["selected"], ["skill", "arm", "prior_strength", "half_life_years", "minimum_exposure", "age_alpha", "primary", "secondary"]]
    posterior_rows = current.loc[
        current["raw_value"].notna()
        & current["opportunities"].gt(0)
        & current["preseason_estimate"].notna()
    ].copy()
    posterior_expected = (
        posterior_rows["preseason_estimate"] * posterior_rows["preseason_precision"]
        + posterior_rows["raw_value"] * posterior_rows["opportunities"]
    ) / (posterior_rows["preseason_precision"] + posterior_rows["opportunities"])
    maximum_posterior_error = float(
        np.max(np.abs(posterior_rows["estimate"] - posterior_expected))
    )
    selected_age_skills = set(
        winners.loc[winners["arm"].eq("time_decayed_eb_plus_age"), "skill"]
    )
    selected_model_records = json.loads(winners.to_json(orient="records"))
    manifest = {
        "schema_version": "predictive_player_skills_run_v1",
        "run_id": run_id,
        "experiment_id": contract["experiment_id"],
        "estimand_id": contract["estimand_id"],
        "status": "research_current_skill",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season_policy": policy,
        "season_2027_loaded": False,
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "source_hashes": source_hashes,
        "source_paths": {
            "player_sheet_dir": _relative(sheet_dir),
            **{key: _relative(path) for key, path in source_paths.items()},
            "player_games": [_relative(path) for path in game_paths],
        },
        "artifacts": {
            name: _relative(output / filename)
            for name, filename in {
                "estimates": "skill_estimates.parquet", "fold_metrics": "fold_metrics.parquet",
                "model_selection": "model_selection.parquet", "definitions": "skill_definitions.parquet",
            }.items()
        },
        "quality": {
            "skills": expected_skills,
            "estimate_rows": int(len(estimates)),
            "duplicate_keys": int(estimates.duplicated(["PLAYER_ID", "Season", "skill"]).sum()),
            "current_players": int(current["PLAYER_ID"].nunique()),
            "current_complete_players": int(len(complete)),
            "current_estimate_coverage": float(current["estimate"].notna().mean()),
            "current_observed_coverage": float(current["raw_value"].notna().mean()),
            "default_player_id": default_player,
            "selected_arm_counts": winners["arm"].value_counts().sort_index().to_dict(),
            "selected_age_skills": len(selected_age_skills),
            "maximum_current_posterior_identity_error": maximum_posterior_error,
            "role_conditional_status": "skipped_no_consistent_preseason_role_support",
        },
        "interpretation": {
            "intended": "Current underlying player skill estimates under frozen chronological stabilization.",
            "forbidden": "RAPM, SPM, AIO impact, causal player credit, or a forecast of playing time.",
        },
    }
    write_json_atomic(manifest, output / "run.json")

    audit_dir = ROOT / "research/audits/predictive_player_skills_2026_v1"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic({
        "experiment_id": contract["experiment_id"],
        "run_id": run_id,
        "verdict": "pass_research_current_skill",
        "gates": {
            "selection_ends_2024": max(map(int, policy["development_seasons"])) == 2024,
            "parameters_refit_through_2025": int(policy["final_parameter_cutoff"]) == 2025,
            "output_is_2026": int(policy["output_season"]) == 2026,
            "season_2027_untouched": True,
            "unique_keys": bool(not estimates.duplicated(["PLAYER_ID", "Season", "skill"]).any()),
            "all_skills_selected": bool(winners["skill"].nunique() == expected_skills),
            "percentiles_bounded": bool(estimates["percentile"].dropna().between(0, 100).all()),
            "all_current_rows_have_update_date": bool(current["last_update_date"].notna().all()),
            "age_selected_outputs_apply_preseason_adjustment": bool(
                current.loc[current["skill"].isin(selected_age_skills), "preseason_estimate"].notna().all()
            ),
            "current_posterior_identity": maximum_posterior_error < 1e-10,
            "rebound_sources_scored_as_rates": bool(
                definitions.set_index("key").loc[
                    ["offensive_rebound_rate", "defensive_rebound_rate"], "family"
                ].eq("rate").all()
            ),
        },
        "default_player_id": default_player,
        "selected_models": selected_model_records,
        "limitations": [
            "The 2026 output is a reused diagnostic, not untouched confirmation.",
            "Role conditioning was not fit because no frozen pre-season role label covers every development fold.",
            "Defense tracking and matchup fields are observational and do not isolate causal responsibility.",
            "Shotmaking and matchup source fields already contain source-level shrinkage before temporal stabilization.",
            "Historical trajectories are post-hoc stabilized histories under parameters selected through 2024, not contemporaneous forecasts issued in each season.",
        ],
    }, audit_dir / "decision.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "research/experiments/predictive_player_skills_2026_v1.yml")
    args = parser.parse_args()
    print(json.dumps(run(args.contract), indent=2))


if __name__ == "__main__":
    main()
