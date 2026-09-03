"""Export checked summary data only. Never export player coefficients or game rows."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.pulse_validation import load_pulse_validation

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "pulse_external_common_v1_c500545ce4"
RUN = ROOT / "research/rapm_lab/outputs/pulse_external_common" / RUN_ID
RICH_RUN_ID = "target_window_spm_aio_v1_8e028133cb"
RICH_RUN = ROOT / "artifacts/research/target_window_spm_aio" / RICH_RUN_ID


def verify_benchmark_inputs(manifest, run=RUN):
    for filename, expected in manifest["file_hashes"].items():
        if sha256_file(run / filename) != expected:
            raise ValueError(f"Changed benchmark artifact: {filename}")
    for filename, expected in manifest["input_hashes"].items():
        path = (
            Path.home() / "Downloads" / filename.removeprefix("provided/")
            if filename.startswith("provided/")
            else ROOT / filename
        )
        if sha256_file(path) != expected:
            raise ValueError(f"Changed benchmark input: {filename}")


def validate_pulse_replay(manifest):
    _, pulse_folds = load_pulse_validation(
        ROOT / "artifacts/models/pulse" / manifest["plan"]["pulse_source_run"]
    )
    if not pulse_folds.training_end.lt(pulse_folds.rating_season).all():
        raise ValueError("PULSE training cutoff failed.")
    replay = manifest["pulse_replay"]
    complete = {row["rating_season"] for row in replay} == set(range(2015, 2026))
    exact = all(row["maximum_prediction_difference"] <= 1e-8 for row in replay)
    if not complete or not exact:
        raise ValueError("PULSE replay failed.")


def build_panel(scope, end, manifest, games, summary, coefficients):
    part = games.loc[games.scope.eq(scope)]
    extra = ["MAMBA"] if scope == "with_mamba" else []
    expected = set(manifest["plan"]["main_models"] + ["Normal RAPM"] + extra)
    if set(part.candidate) != expected or set(part.outcome_season) != set(range(2016, end + 1)):
        raise ValueError("Incomplete benchmark candidates or years.")
    if not part.outcome_season.eq(part.rating_season + 1).all():
        raise ValueError("Benchmark must score the next season.")
    actual = part.pivot(
        index=["outcome_season", "game_id"], columns="candidate", values="actual_margin"
    )
    if actual.isna().any().any() or not actual.eq(actual.PULSE, axis=0).all().all():
        raise ValueError("Benchmark games differ between models.")
    betas = coefficients.loc[coefficients.scope.eq(scope)]
    mask = betas.pivot(index=["rating_season", "PLAYER_ID"], columns="candidate", values="included")
    if mask.isna().any().any() or not mask.eq(mask.PULSE, axis=0).all().all():
        raise ValueError("Benchmark player support differs between models.")
    if betas.loc[~betas.included, ["offense_coefficient", "defense_coefficient"]].ne(0).any().any():
        raise ValueError("Excluded players have nonzero ratings.")
    errors = part.assign(error=(part.predicted_margin - part.actual_margin) ** 2)
    measured = errors.groupby(["candidate", "outcome_season"]).error.mean().groupby("candidate").mean().pow(.5)
    rows = summary.loc[summary.scope.eq(scope)].set_index("candidate")
    summaries_match = np.allclose(measured, rows.loc[measured.index, "rmse"], atol=1e-10)
    if not np.isfinite(part[["predicted_margin", "actual_margin"]]).all().all() or not summaries_match:
        raise ValueError("Benchmark summary disagrees with its predictions.")
    records = [
        {
            "candidate": "RAPM" if name == "Normal RAPM" else name,
            "folds": int(row.folds),
            "aggregate_rmse": float(row.rmse),
            "mean_correlation": float(row.mean_correlation),
            "mean_calibration_slope": float(row.mean_calibration_slope),
        }
        for name, row in rows.sort_values("rmse").iterrows()
    ]
    return {
        "scope": scope,
        "outcome_start": 2016,
        "outcome_end": end,
        "games": len(actual),
        "rows": records,
        "matched_exposure_min": float(rows.min_matched_exposure.min()),
        "matched_exposure_max": float(rows.max_matched_exposure.max()),
    }


def source_rating_years():
    return [
        {"model": label, "start": start, "end": end}
        for label, start, end in (
            ("EPM", 2002, 2026),
            ("LEBRON", 2010, 2026),
            ("DARKO DPM", 1997, 2026),
            ("xRAPM", 1997, 2026),
            ("BPM 2.0", 2014, 2026),
            ("CourtSignal PIPM reconstruction", 1997, 2026),
            ("CourtSignal RAPTOR reconstruction", 2014, 2026),
            ("MAMBA", 2015, 2024),
        )
    ]


def build_rich_prior_test(run=RICH_RUN):
    manifest = json.loads((run / "run.json").read_text())
    if manifest["run_id"] != RICH_RUN_ID or manifest["status"] != "reused_diagnostic_complete":
        raise ValueError("Unexpected Rich SPM prior run.")
    for filename in ("fold_metrics.parquet", "game_predictions.parquet", "paired_key_comparisons.parquet"):
        if sha256_file(run / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Changed Rich SPM artifact: {filename}")
    for source in ("runner", "contract"):
        entry = manifest["sources"][source]
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Changed Rich SPM source: {source}")
    contract = manifest["contract"]
    if contract["spm"]["training_rule"] != "expanding_history_ending_before_rating_season":
        raise ValueError("Rich SPM training cutoff failed.")

    candidates = {
        "box15_9y_normal_aio": "PULSE (Box15 prior)",
        "rich_spm_9y_normal_aio": "Rich SPM prior",
    }
    games = pd.read_parquet(run / "game_predictions.parquet")
    games = games.loc[
        games.candidate.isin(candidates) & games.test_season.between(2016, 2026)
    ].copy()
    if not games.test_season.eq(games.rating_season + 1).all():
        raise ValueError("Rich SPM comparison must score the next season.")
    actual = games.pivot(index=["test_season", "game_id"], columns="candidate", values="actual_margin")
    if actual.isna().any().any() or not actual.eq(actual.iloc[:, 0], axis=0).all().all():
        raise ValueError("Rich SPM comparison games differ.")

    folds = pd.read_parquet(run / "fold_metrics.parquet")
    folds = folds.loc[
        folds.candidate.isin(candidates) & folds.test_season.between(2016, 2026)
    ]
    rows = []
    for candidate, label in candidates.items():
        part = folds.loc[folds.candidate.eq(candidate)]
        if set(part.test_season) != set(range(2016, 2027)):
            raise ValueError(f"Incomplete Rich SPM comparison: {candidate}")
        rows.append({
            "candidate": label,
            "folds": int(part.test_season.nunique()),
            "aggregate_rmse": float(np.sqrt(part.mse.mean())),
            "mean_correlation": float(part.correlation.mean()),
            "mean_calibration_slope": float(part.calibration_slope.mean()),
        })
    paired = pd.read_parquet(run / "paired_key_comparisons.parquet")
    paired = paired.loc[
        paired.candidate_a.eq("box15_9y_normal_aio")
        & paired.candidate_b.eq("rich_spm_9y_normal_aio")
    ].squeeze()
    if paired.empty or int(paired.folds) != 11:
        raise ValueError("Missing Rich SPM paired comparison.")
    return {
        "run_id": RICH_RUN_ID,
        "outcome_start": 2016,
        "outcome_end": 2026,
        "games": len(actual),
        "rows": rows,
        "pulse_minus_rich_mse": float(paired.a_minus_b_mse),
        "lower_95": float(paired.lower_95),
        "upper_95": float(paired.upper_95),
    }


def build_payload(run=RUN):
    manifest = json.loads((run / "run.json").read_text())
    if manifest["run_id"] != RUN_ID:
        raise ValueError("Unexpected external benchmark run.")
    verify_benchmark_inputs(manifest, run)
    validate_pulse_replay(manifest)
    games = pd.read_parquet(run / "game_predictions.parquet")
    summary = pd.read_parquet(run / "summary.parquet")
    coefficients = pd.read_parquet(run / "scored_coefficients.parquet")
    intervals = pd.read_parquet(run / "paired_intervals.parquet")
    panels = [
        build_panel(scope, end, manifest, games, summary, coefficients)
        for scope, end in (("main", 2026), ("with_mamba", 2025))
    ]
    return {"run_id": RUN_ID, "panels": panels, "rich_prior_test": build_rich_prior_test(),
        "paired_intervals": intervals.to_dict("records"),
        "limitations": manifest["limitations"], "source_rating_years": source_rating_years()}


if __name__ == "__main__":
    payload = build_payload()
    output = ROOT / "web/public/data/external-benchmark.json"
    write_json_atomic(payload, output)
    print(output)
