"""Frozen, past-only PULSE comparison with common final player support."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.canonical_pulse import fit_pulse_season, game_metrics, predict_next_season_games
from nba_impact.models.pulse_validation import load_pulse_validation
from nba_impact.models.stint_rapm import build_stint_design, load_canonical_stints
from research.run_external_all_in_one_benchmark_v2 import component_frame, name_dimension, named_frame
from research.rapm_lab.run_wp_chronology_release import official_scores, paired_rmse

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "research/audits/benchmark_coverage_20260903/benchmark_plan.json"
PULSE = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a"
STINTS = ROOT / "data/lake/silver/canonical_lineup_stints"
SOURCES = ROOT / "research/rapm_lab/data/external/benchmark_20260903"
PIPM = ROOT / "research/rapm_lab/outputs/pipm_reconstruction/pipm_reconstruction_v1_e0625de5fe"
RAPTOR = ROOT / "research/rapm_lab/outputs/raptor_reconstruction/raptor_reconstruction_v1_938d1becf9"
OUTPUT = ROOT / "research/rapm_lab/outputs/pulse_external_common"


def external_panels() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    downloads = Path.home() / "Downloads"
    paths = [downloads / "EPM_All_Seasons.csv", downloads / (
        "lebron-data-2026-2025-2024-2023-2022-2021-2020-2019-2018-2017-2016-2015-2014-2013-2012-2011-2010.csv"
    ), downloads / "DARKO - Daily Adjusted and Regressed Kalman Optimized projections - Full DPM History.csv",
        downloads / "MAMBAVALUES.xlsx - Sheet1.csv", SOURCES / "annual_sources.parquet"]
    epm, lebron, darko, mamba = [pd.read_csv(path, low_memory=False) for path in paths[:4]]
    names = name_dimension(epm, lebron)
    panels, coverage = [], []
    for label, frame, id_col, season, off, defense in (
        ("EPM", epm, "EPM_player_id", "EPM_season", "EPM_off", "EPM_def"),
        ("LEBRON", lebron, "nba_id", "Season", "O-LEBRON", "D-LEBRON"),
        ("DARKO DPM", darko, "nba_id", "season", "o_dpm", "d_dpm"),
        ("CourtSignal PIPM reconstruction", pd.read_parquet(PIPM / "reconstructions.parquet"),
         "PLAYER_ID", "season", "pipm_offense", "pipm_defense"),
        ("CourtSignal RAPTOR reconstruction", pd.read_parquet(RAPTOR / "reconstructions.parquet"),
         "PLAYER_ID", "season", "raptor_offense", "raptor_defense"),
    ):
        if frame.duplicated([id_col, season]).any():
            raise ValueError(f"Ambiguous {label} source keys.")
        panels.append(component_frame(frame, candidate=label, id_column=id_col,
            season_column=season, offense_column=off, defense_column=defense))
    annual = pd.read_parquet(paths[-1])
    for label, frame, name, year, off, defense in (
        ("xRAPM", annual.loc[annual.source.eq("xrapm")], "player_name_xrapm", "season", "xrapm_offense", "xrapm_defense"),
        ("BPM 2.0", annual.loc[annual.source.eq("bpm")], "player_name_bpm", "season", "bpm_offense", "bpm_defense"),
        ("MAMBA", mamba.loc[mamba.Season.le(2024)], "Player", "Season", "Offense", "Defense"),
    ):
        table, matches = named_frame(frame, names, candidate=label, name_column=name,
            season_column=year, offense_column=off, defense_column=defense)
        panels.append(table)
        coverage.append(matches)
    return pd.concat(panels), pd.concat(coverage), paths + [
        run / name for run in (PIPM, RAPTOR) for name in ("run.json", "reconstructions.parquet")]


def common_coefficients(ratings, source, common, home):
    """Center on source-year matched exposure, then leave every unmatched coefficient zero."""
    keep = np.isin(source.players, list(common))
    selected = ratings.set_index("PLAYER_ID").reindex(source.players[keep])
    beta = np.zeros(source.X.shape[1])
    for offset, side, weights, sign in (
        (0, "offense", source.off_possessions, 1),
        (len(source.players), "defense", source.def_possessions, -1),
    ):
        values = selected[side].to_numpy(float)
        values = sign * (values - np.average(values, weights=weights[keep])) / 100
        beta[offset + np.flatnonzero(keep)] = values
    beta[-1] = home
    return beta


def score_scope(tables, names, source, target, home, intercept, scores, season, scope):
    finite_ids = [set(table.loc[np.isfinite(table[["offense", "defense"]]).all(axis=1), "PLAYER_ID"])
                  for table in (tables[name] for name in names)]
    positive_exposure = source.players[np.minimum(source.off_possessions, source.def_possessions) > 0]
    common = set.intersection(set(positive_exposure), *finite_ids)
    if not common:
        raise ValueError("No common finite player support.")
    keep = np.isin(target.players, list(common))
    exposure = (target.off_possessions[keep].sum() + target.def_possessions[keep].sum()) / (
        target.off_possessions.sum() + target.def_possessions.sum())
    frames, folds, coefficients = [], [], []
    for name in names:
        beta = common_coefficients(tables[name], source, common, home)
        predictions = predict_next_season_games(source, target, beta, intercept)
        predictions = predictions.drop(columns="actual_margin").merge(
            scores.loc[scores.season.eq(season + 1), ["gameid", "actual_margin"]].rename(columns={"gameid": "game_id"}),
            on="game_id", how="left", validate="one_to_one")
        if not np.isfinite(predictions[["actual_margin", "predicted_margin"]]).all().all():
            raise ValueError("Missing official game scores or nonfinite predictions.")
        identifiers = {"candidate": name, "rating_season": season, "outcome_season": season + 1, "scope": scope}
        frames.append(predictions.assign(**identifiers))
        folds.append({**identifiers, "matched_players": len(common), "matched_exposure": float(exposure), **game_metrics(predictions)})
        coefficients.append(pd.DataFrame({"PLAYER_ID": source.players, "offense_coefficient": beta[:len(source.players)],
            "defense_coefficient": beta[len(source.players):-1], "included": np.isin(source.players, list(common))}).assign(**identifiers))
    return pd.concat(frames), folds, pd.concat(coefficients)


def main():
    plan = json.loads(PLAN.read_text())
    saved_games, _ = load_pulse_validation(PULSE)
    pulse_manifest = json.loads((PULSE / "run.json").read_text())
    priors = pd.read_parquet(PULSE / "validation_priors.parquet")
    external, identity_coverage, input_paths = external_panels()
    scores = official_scores(2016, 2026)
    games, folds, coefficients, replay = [], [], [], []
    names = plan["main_models"] + [plan["internal_control"]]
    for season in range(2015, 2026):
        ratings, _, source, fits = fit_pulse_season(season, priors, STINTS, pulse_manifest["config"])
        target = build_stint_design(load_canonical_stints(STINTS, (season + 1,)))
        original = predict_next_season_games(source, target, *fits["pulse"])
        saved = saved_games.loc[saved_games.candidate.eq("pulse") & saved_games.rating_season.eq(season)]
        compared = original.merge(saved, on="game_id", suffixes=("_new", "_saved"), validate="one_to_one")
        delta = float(abs(compared.predicted_margin_new - compared.predicted_margin_saved).max())
        if len(compared) != len(saved) or len(compared) != len(original) or delta > 1e-8:
            raise ValueError("Frozen PULSE replay differs from saved chronological predictions.")
        replay.append({"rating_season": season, "maximum_prediction_difference": delta})
        tables = {name: group for name, group in external.loc[external.rating_season.eq(season)].groupby("candidate")}
        for label, prefix in (("PULSE", "pulse"), ("Normal RAPM", "rapm")):
            tables[label] = ratings[["PLAYER_ID", f"{prefix}_offense", f"{prefix}_defense"]].rename(
                columns={f"{prefix}_offense": "offense", f"{prefix}_defense": "defense"})
        for scope, candidates in [("main", names)] + ([("with_mamba", names + ["MAMBA"])] if season <= 2024 else []):
            scored, metrics, betas = score_scope(tables, candidates, source, target, fits["rapm"][0][-1], fits["rapm"][1], scores, season, scope)
            games.append(scored)
            folds.extend(metrics)
            coefficients.append(betas)
        print(f"common benchmark {season}->{season + 1}; PULSE replay max delta={delta:.2g}", flush=True)
    games, folds = pd.concat(games, ignore_index=True), pd.DataFrame(folds)
    summary = folds.groupby(["scope", "candidate"], as_index=False).agg(
        folds=("outcome_season", "nunique"), games=("games", "sum"), mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"), mean_calibration_slope=("calibration_slope", "mean"),
        min_matched_exposure=("matched_exposure", "min"), max_matched_exposure=("matched_exposure", "max"))
    summary["rmse"] = np.sqrt(summary.mse)
    intervals = []
    for scope, frame in games.groupby("scope"):
        candidates = ["PULSE"] + sorted(set(frame.candidate) - {"PULSE"})
        values, draws = paired_rmse(frame, candidates, plan)
        for i, name in enumerate(candidates[1:], start=1):
            intervals.append({"scope": scope, "other": name, "pulse_minus_other_rmse": float(values[0] - values[i]),
                "lower_95": float(np.quantile(draws[:, 0] - draws[:, i], .025)),
                "upper_95": float(np.quantile(draws[:, 0] - draws[:, i], .975))})
    save_run(plan, input_paths, games, folds, summary, intervals, coefficients, replay, identity_coverage)


def save_run(plan, input_paths, games, folds, summary, intervals, coefficients, replay, identity_coverage):
    paths = input_paths + [PLAN, Path(__file__), SOURCES / "manifest.json",
        PULSE / "validation_games.parquet", PULSE / "validation_priors.parquet", PULSE / "run.json"]
    paths += [ROOT / path for path in ("research/run_external_all_in_one_benchmark_v2.py",
        "research/rapm_lab/run_wp_chronology_release.py")]
    paths += [ROOT / "src/nba_impact/models" / name for name in ["canonical_pulse.py", "stint_rapm.py", "pulse_validation.py", "external_impact_benchmark.py"]]
    for year in range(2015, 2027):
        paths.append(STINTS / f"season={year}/regular.parquet")
        if year >= 2016:
            paths.append(ROOT / f"data/lake/bronze/official_game_scores/project_season={year}/regular.parquet")
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else f"provided/{path.name}": sha256_file(path) for path in paths}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"pulse_external_common_v1_{identity}"
    if output.exists():
        raise FileExistsError(f"Preserve immutable run: {output}")
    output.mkdir(parents=True)
    tables = {"game_predictions": games, "fold_metrics": folds, "summary": summary,
        "paired_intervals": pd.DataFrame(intervals), "scored_coefficients": pd.concat(coefficients),
        "identity_coverage": identity_coverage}
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    files = {path.name: sha256_file(path) for path in output.iterdir()}
    write_json_atomic({"run_id": output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "reused_evidence_common_support_diagnostic", "plan": plan, "input_hashes": hashes,
        "file_hashes": files, "pulse_replay": replay,
        "limitations": ["Observed next-season lineups; not a preseason forecast.",
            "Historical outcomes influenced model development; not untouched confirmation.",
            "External historical source files do not certify point-in-time training provenance.",
            "Reconstructions are CourtSignal implementations, not source-author ratings.",
            "RAPTOR uses its saved pooled 2014-2022 box mapping and 2014-2018 on/off mapping; early ratings are not past-only fits.",
            "PIPM uses its saved fixed-coefficient reconstruction. Neither reconstruction was refitted for this benchmark."]}, output / "run.json")
    print(summary.to_string(index=False))
    print(output, flush=True)


if __name__ == "__main__":
    main()
