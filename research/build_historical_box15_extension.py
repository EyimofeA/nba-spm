#!/usr/bin/env python3
"""Build 1997-2026 Box15 inputs and canonical five-year RAPM targets."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.full_spm_features import build_rolling_five_year_features
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.rapm import RapmConfig, load_legacy_possessions
from nba_impact.models.rolling_rapm_peaks import fit_rolling_rapm_window


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "historical_box15_extension_v1"
CONTRACT = ROOT / "research/experiments/historical_box15_extension_v1.yml"
CACHE = ROOT / "rapm/data/possession_cache"
HISTORICAL_SHEETS = (
    ROOT / "data/lake/bronze/historical_player_sheets/year_totals"
)
CURRENT_FEATURES = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_8be676bd0f/annual_features.parquet"
)
CURRENT_ROLLING = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/rolling_ratings.parquet"
)
OUTPUT_ROOT = ROOT / "artifacts/research/historical_box15_extension"
COUNT_COLUMNS = tuple(feature.removesuffix("_p100") for feature in BOX_PIPM_STYLE_FEATURES)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment ID changed.")
    if contract["scope"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    if tuple(contract["box15"]["features"]) != BOX_PIPM_STYLE_FEATURES:
        raise ValueError("The historical feature list differs from Box15.")
    return contract


def _annual_exposure(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    away = frame[[f"a{index}" for index in range(1, 6)]].to_numpy(dtype=np.int64)
    home = frame[[f"h{index}" for index in range(1, 6)]].to_numpy(dtype=np.int64)
    home_offense = frame["home_poss"].to_numpy(dtype=bool)
    offense = np.where(home_offense[:, None], home, away)
    defense = np.where(home_offense[:, None], away, home)
    off = pd.Series(offense.ravel()).value_counts(sort=False)
    deff = pd.Series(defense.ravel()).value_counts(sort=False)
    players = sorted(set(off.index) | set(deff.index))
    return pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Season": int(season),
            "Poss_Off": [int(off.get(player, 0)) for player in players],
            "Poss_Def": [int(deff.get(player, 0)) for player in players],
        }
    )


def _historical_features(season: int, exposure: pd.DataFrame) -> pd.DataFrame:
    path = HISTORICAL_SHEETS / f"{season}.csv"
    source = pd.read_csv(path, low_memory=False)
    required = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "OffPoss",
        "DefPoss",
        "POSS",
        "FGA",
        "FGM",
        *COUNT_COLUMNS,
    }
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"Historical player sheet {season} lacks {missing}.")
    source["FG2A"] = source["FG2A"].fillna(source["FGA"] - source["FG3A"])
    source["FG2M"] = source["FG2M"].fillna(source["FGM"] - source["FG3M"])
    denominator = pd.to_numeric(source["OffPoss"], errors="coerce")
    denominator_source = np.where(denominator.gt(0), "OffPoss", "POSS")
    denominator = denominator.where(
        denominator.gt(0), pd.to_numeric(source["POSS"], errors="coerce")
    )
    defense_denominator = pd.to_numeric(source["DefPoss"], errors="coerce")
    defense_denominator = defense_denominator.where(
        defense_denominator.gt(0), pd.to_numeric(source["POSS"], errors="coerce")
    )
    rows = source[["PLAYER_ID", "PLAYER_NAME"]].copy()
    rows["Window_End"] = int(season)
    rows["feature_possessions"] = denominator
    rows["feature_def_possessions"] = defense_denominator
    rows["feature_possession_source"] = denominator_source
    for count, feature in zip(COUNT_COLUMNS, BOX_PIPM_STYLE_FEATURES, strict=True):
        rows[feature] = 100.0 * pd.to_numeric(source[count], errors="coerce") / denominator
    rows = rows.merge(
        exposure.drop(columns="Season"), on="PLAYER_ID", how="left", validate="one_to_one"
    )
    rows[["Poss_Off", "Poss_Def"]] = rows[["Poss_Off", "Poss_Def"]].fillna(0)
    rows = rows.loc[rows["feature_possessions"].gt(0)].copy()
    return rows


def _target_frame(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.rename(
        columns={
            "offense": "target_offense",
            "defense": "target_defense",
            "net": "target_net",
            "window_end": "Window_End",
        }
    )[
        [
            "PLAYER_ID",
            "Window_End",
            "target_offense",
            "target_defense",
            "target_net",
            "Poss_Off",
            "Poss_Def",
        ]
    ]


def main() -> None:
    contract = _load_contract()
    historical_features: list[pd.DataFrame] = []
    annual_ratings: list[pd.DataFrame] = []
    rolling_ratings: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    frames: deque[pd.DataFrame] = deque(maxlen=5)

    for season in range(1997, 2019):
        frame = load_legacy_possessions(CACHE, (season,), game_types=("regular",))
        exposure = _annual_exposure(frame, season)
        if season <= 2013:
            historical_features.append(_historical_features(season, exposure))

        annual_config = RapmConfig(
            seasons=(season,),
            lambda_off=3000,
            lambda_def=3000,
            lambda_home=300,
            game_types=("regular",),
            data_scope="historical_box15_annual_zero_prior",
        )
        annual, annual_quality = fit_rolling_rapm_window(
            frame,
            annual_config,
            window_start=season,
            window_end=season,
            minimum_possessions_per_window_season=0,
        )
        annual["Season"] = season
        annual_ratings.append(annual)
        quality_rows.append({"horizon": "1y", **annual_quality})

        frames.append(frame)
        if len(frames) == 5:
            start = season - 4
            window = pd.concat(frames, ignore_index=True)
            rolling_config = RapmConfig(
                seasons=tuple(range(start, season + 1)),
                lambda_off=3000,
                lambda_def=3000,
                lambda_home=300,
                game_types=("regular",),
                data_scope="historical_box15_rolling_5y_zero_prior",
            )
            rolling, rolling_quality = fit_rolling_rapm_window(
                window,
                rolling_config,
                window_start=start,
                window_end=season,
                minimum_possessions_per_window_season=0,
            )
            rolling_ratings.append(rolling)
            quality_rows.append({"horizon": "5y", **rolling_quality})
        print(f"{season}: loaded and fitted", flush=True)

    historical_annual = pd.concat(annual_ratings, ignore_index=True)
    historical_rolling = pd.concat(rolling_ratings, ignore_index=True)
    historical_annual_features = pd.concat(historical_features, ignore_index=True)
    current_annual = pd.read_parquet(CURRENT_FEATURES)
    current_annual = current_annual[
        ["PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *BOX_PIPM_STYLE_FEATURES]
    ].copy()
    annual_features = pd.concat(
        [
            historical_annual_features[
                [
                    "PLAYER_ID",
                    "Window_End",
                    "feature_possessions",
                    "feature_def_possessions",
                    *BOX_PIPM_STYLE_FEATURES,
                ]
            ].rename(
                columns={
                    "feature_possessions": "OffPoss",
                    "feature_def_possessions": "DefPoss",
                }
            ),
            current_annual,
        ],
        ignore_index=True,
    ).sort_values(["Window_End", "PLAYER_ID"], kind="stable")
    if annual_features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Historical annual features contain duplicate keys.")
    observed_seasons = sorted(annual_features["Window_End"].astype(int).unique())
    if observed_seasons != list(range(1997, 2027)):
        raise ValueError("Annual Box15 features do not cover 1997-2026.")

    selected = {
        "offense": BOX_PIPM_STYLE_FEATURES,
        "defense": BOX_PIPM_STYLE_FEATURES,
    }
    five_year_features = build_rolling_five_year_features(
        annual_features,
        None,
        selected,
        window_ends=tuple(range(2001, 2027)),
    )

    current_rolling = pd.read_parquet(CURRENT_ROLLING)
    rebuilt_2018 = _target_frame(
        historical_rolling.loc[historical_rolling["window_end"].eq(2018)]
    )
    pinned_2018 = _target_frame(
        current_rolling.loc[current_rolling["window_end"].eq(2018)]
    )
    overlap = rebuilt_2018.merge(
        pinned_2018,
        on=["PLAYER_ID", "Window_End"],
        suffixes=("_rebuilt", "_pinned"),
        validate="one_to_one",
    )
    overlap_rows = []
    for component in ("offense", "defense", "net"):
        left = overlap[f"target_{component}_rebuilt"]
        right = overlap[f"target_{component}_pinned"]
        overlap_rows.append(
            {
                "window_end": 2018,
                "component": component,
                "matched_players": len(overlap),
                "maximum_absolute_error": float((left - right).abs().max()),
                "correlation": float(pd.concat([left, right], axis=1).corr().iloc[0, 1]),
            }
        )
    overlap_validation = pd.DataFrame(overlap_rows)
    if overlap_validation["maximum_absolute_error"].max() > 1e-8:
        raise ValueError("Rebuilt 2018 rolling RAPM differs from the pinned target.")

    historical_targets = _target_frame(
        historical_rolling.loc[historical_rolling["window_end"].le(2017)]
    )
    current_targets = _target_frame(current_rolling)
    five_year_targets = pd.concat(
        [historical_targets, current_targets], ignore_index=True
    ).sort_values(["Window_End", "PLAYER_ID"], kind="stable")
    if sorted(five_year_targets["Window_End"].unique()) != list(range(2001, 2027)):
        raise ValueError("Five-year targets do not cover 2001-2026.")
    identity_error = (
        five_year_targets["target_offense"]
        + five_year_targets["target_defense"]
        - five_year_targets["target_net"]
    ).abs().max()
    if identity_error > 1e-8:
        raise ValueError("Historical target component identity failed.")

    source_paths = {
        "contract": CONTRACT,
        "current_features": CURRENT_FEATURES,
        "current_rolling_targets": CURRENT_ROLLING,
        "builder": Path(__file__),
        **{
            f"possession_cache_{season}": CACHE / f"matchups_{season}.parquet"
            for season in range(1997, 2019)
        },
        **{
            f"historical_player_sheet_{season}": HISTORICAL_SHEETS / f"{season}.csv"
            for season in range(1997, 2014)
        },
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "historical_annual_rapm.parquet": historical_annual,
        "historical_rolling_5y_rapm.parquet": historical_rolling,
        "annual_box15_features.parquet": annual_features,
        "five_year_box15_features.parquet": five_year_features,
        "five_year_targets.parquet": five_year_targets,
        "source_coverage.parquet": pd.concat(
            [
                historical_annual_features[
                    ["PLAYER_ID", "Window_End", "feature_possessions", "feature_possession_source"]
                ],
                pd.DataFrame(
                    {
                        "PLAYER_ID": current_annual["PLAYER_ID"],
                        "Window_End": current_annual["Window_End"],
                        "feature_possessions": current_annual["OffPoss"],
                        "feature_possession_source": "OffPoss",
                    }
                ),
            ],
            ignore_index=True,
        ),
        "target_overlap_validation.parquet": overlap_validation,
        "fit_quality.parquet": pd.DataFrame(quality_rows),
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "validated_research_input",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "quality": {
            "annual_feature_seasons": [1997, 2026],
            "complete_five_year_window_ends": [2001, 2026],
            "historical_rate_fallback_seasons": [1997, 2000],
            "maximum_2018_target_overlap_error": float(
                overlap_validation["maximum_absolute_error"].max()
            ),
            "maximum_component_identity_error": float(identity_error),
            "season_2027_loaded": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashes[name],
            }
            for name, path in source_paths.items()
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(run["quality"], indent=2))


if __name__ == "__main__":
    main()
