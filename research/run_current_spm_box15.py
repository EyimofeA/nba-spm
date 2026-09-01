"""Fit the first dated Box15 current-SPM foundation."""

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
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit
from nba_impact.models.predictive_spm import aggregate_dated_box15_features


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "current_spm_box15_v1"
ANNUAL_BOX = ROOT / (
    "artifacts/research/historical_box15_extension/"
    "historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet"
)
TARGET_RUN = ROOT / (
    "artifacts/research/target_window_spm_aio/"
    "target_window_spm_aio_v1_be05a18f9b"
)


def _contract(path: Path) -> dict:
    payload = json.loads(json.dumps(yaml.safe_load(path.read_text()), default=str))
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_foundation",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    if payload["target"] != {"horizon_seasons": 9, "variant": "normal"}:
        raise ValueError("The foundation must use nine-season normal RAPM targets.")
    if payload["model"]["box_half_life_days"] != [365, 730]:
        raise ValueError("The first Box15 current-SPM search must remain 365/730 days.")
    return payload


def _cutoffs(season: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{season - 1}-11-01", f"{season}-04-01", freq="W-MON")


def _training_panel(annual: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    selected = targets.loc[
        targets["horizon"].eq(9) & targets["target_variant"].eq("normal")
    ].copy()
    panel = annual.merge(
        selected,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def fit_current_ratings(
    history: pd.DataFrame,
    panel: pd.DataFrame,
    contract: dict,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in contract["rating_seasons"]:
        train = panel.loc[panel["Window_End"].lt(int(season))].copy()
        if train.empty:
            raise ValueError(f"Current-SPM season {season} has no earlier target rows.")
        models = {
            side: _fit(
                train,
                BOX_PIPM_STYLE_FEATURES,
                f"target_{side}",
                float(contract["model"][f"{side}_alpha"]),
            )
            for side in ("offense", "defense")
        }
        for half_life in contract["model"]["box_half_life_days"]:
            for cutoff in _cutoffs(int(season)):
                available = history.loc[history["game_date"].lt(cutoff)]
                if available.empty:
                    continue
                features = aggregate_dated_box15_features(
                    available,
                    cutoff_date=cutoff,
                    half_life_days=float(half_life),
                    rate_prior_possessions=float(
                        contract["model"]["rate_prior_possessions"]
                    ),
                )
                valid = features[list(BOX_PIPM_STYLE_FEATURES)].notna().all(axis=1)
                features = features.loc[valid].copy()
                if features.empty:
                    continue
                output = features[
                    ["PLAYER_ID", "OffPoss", "DefPoss", "cutoff_date", "half_life_days"]
                ].copy()
                output["rating_season"] = int(season)
                output["training_target_end"] = int(train["Window_End"].max())
                for side, model in models.items():
                    output[side] = model.predict(features.loc[:, BOX_PIPM_STYLE_FEATURES])
                output["net"] = output["offense"] + output["defense"]
                rows.append(output)
    ratings = pd.concat(rows, ignore_index=True)
    if ratings.duplicated(
        ["rating_season", "cutoff_date", "half_life_days", "PLAYER_ID"]
    ).any():
        raise AssertionError("Current-SPM rating keys must be unique.")
    if not (ratings["training_target_end"] < ratings["rating_season"]).all():
        raise AssertionError("Current-SPM target chronology is invalid.")
    if not np.allclose(ratings["offense"] + ratings["defense"], ratings["net"]):
        raise AssertionError("Current-SPM offense plus defense must equal net.")
    return ratings.sort_values(
        ["rating_season", "cutoff_date", "half_life_days", "net"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/current_spm_box15_v1.yml",
    )
    parser.add_argument(
        "--dated-run",
        type=Path,
        default=ROOT
        / "artifacts/research/dated_box15_history/dated_box15_history_v1_65f2fa929c",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/research/current_spm_box15",
    )
    args = parser.parse_args()

    contract = _contract(args.contract)
    dated_manifest = json.loads((args.dated_run / "run.json").read_text())
    target_manifest = json.loads((TARGET_RUN / "run.json").read_text())
    if dated_manifest["run_id"] != contract["input_run"]:
        raise ValueError("Dated Box15 input run does not match the contract.")
    if target_manifest["run_id"] != contract["target_run"]:
        raise ValueError("Target run does not match the contract.")

    history = pd.read_parquet(args.dated_run / "player_games.parquet")
    annual = pd.read_parquet(ANNUAL_BOX)
    targets = pd.read_parquet(TARGET_RUN / "targets.parquet")
    ratings = fit_current_ratings(history, _training_panel(annual, targets), contract)
    hashes = {
        "contract": sha256_file(args.contract),
        "runner": sha256_file(Path(__file__)),
        "dated_aggregator": sha256_file(
            ROOT / "src/nba_impact/models/predictive_spm.py"
        ),
        "dated_history": sha256_file(args.dated_run / "player_games.parquet"),
        "annual_box": sha256_file(ANNUAL_BOX),
        "targets": sha256_file(TARGET_RUN / "targets.parquet"),
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = args.output_root / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_foundation_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "source_hashes": hashes,
        "quality": {
            "rows": int(len(ratings)),
            "players": int(ratings["PLAYER_ID"].nunique()),
            "cutoffs": int(ratings["cutoff_date"].nunique()),
            "first_cutoff": str(ratings["cutoff_date"].min().date()),
            "last_cutoff": str(ratings["cutoff_date"].max().date()),
            "component_identity_max_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
        },
        "files": {"ratings.parquet": sha256_file(output / "ratings.parquet")},
        "forbidden_interpretation": (
            "These are unvalidated dated statistical ratings, not a current-AIO winner."
        ),
    }
    write_json_atomic(manifest, output / "run.json")
    print(output)


if __name__ == "__main__":
    main()
