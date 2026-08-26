"""Incremental teammate-context test for the frozen five-year SPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.factor_reconstruction import weighted_metrics
from nba_impact.models.factor_target_spm import (
    CONTEXT_SOURCES,
    _fit_predict,
    build_related_feature_panel,
)


EXPERIMENT_ID = "five_year_spm_teammate_context_v1"
SIDES = ("offense", "defense")
CONTEXT_BANKS = {
    "offense": (
        "teammate_spacing",
        "teammate_creation",
        "teammate_rim_pressure",
        "teammate_turnover_to_load",
        "teammate_offensive_load",
        "teammate_oreb",
    ),
    "defense": (
        "teammate_dreb",
        "teammate_dreb_contests",
        "teammate_event_stops",
        "teammate_deflections",
        "teammate_rim_points_saved",
        "teammate_contested_shots",
    ),
}


def pool_five_year_context(
    annual: pd.DataFrame,
    *,
    window_ends: tuple[int, ...],
) -> pd.DataFrame:
    """Pool annual teammate context over each player's identical five-year window."""
    outputs = []
    for end in window_ends:
        window = annual.loc[annual["Season"].between(end - 4, end)].copy()
        player_ids = window["PLAYER_ID"].drop_duplicates().astype(int)
        pooled = pd.DataFrame({"PLAYER_ID": player_ids, "Window_End": end})
        for side, fields in CONTEXT_BANKS.items():
            weight_column = "OffPoss" if side == "offense" else "DefPoss"
            weights = pd.to_numeric(window[weight_column], errors="coerce").clip(lower=0)
            for field in fields:
                values = pd.to_numeric(window[field], errors="coerce")
                valid = values.notna() & weights.gt(0)
                numerator = (values.where(valid, 0.0) * weights.where(valid, 0.0)).groupby(
                    window["PLAYER_ID"]
                ).sum()
                denominator = weights.where(valid, 0.0).groupby(window["PLAYER_ID"]).sum()
                pooled[field] = pooled["PLAYER_ID"].map(
                    (numerator / denominator.replace(0.0, np.nan)).to_dict()
                )
        outputs.append(pooled)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year teammate context has duplicate keys.")
    return result


def _score(
    frame: pd.DataFrame,
    *,
    candidate: str,
    season: int,
    stage: str,
    target_prefix: str = "target",
    prediction_prefix: str,
) -> list[dict]:
    rows = []
    for side in (*SIDES, "net"):
        rows.append(
            {
                "stage": stage,
                "season": season,
                "candidate": candidate,
                "side": side,
                "players": int(len(frame)),
                **weighted_metrics(
                    frame[f"{target_prefix}_{side}"].to_numpy(dtype=float),
                    frame[f"{prediction_prefix}_{side}"].to_numpy(dtype=float),
                    frame["sample_weight"].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def run_five_year_spm_context(
    *,
    player_sheet_dir: str | Path,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    baseline_predictions_path: str | Path,
    five_year_targets_path: str | Path,
    annual_targets_path: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    contract = json.loads(Path(contract_path).read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected five-year SPM context experiment ID.")
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Five-year SPM context test requires a frozen contract.")
    if contract["untouched_confirmation_season"] in contract["rating_seasons"]:
        raise ValueError("Season 2027 must remain untouched.")

    source_paths = {
        "playtype": Path(playtype_path),
        "dfg": Path(dfg_path),
        "rim_dfg": Path(rim_dfg_path),
        "hustle": Path(hustle_path),
        "baseline_predictions": Path(baseline_predictions_path),
        "five_year_targets": Path(five_year_targets_path),
        "annual_targets": Path(annual_targets_path),
        "contract": Path(contract_path),
        "source_code": Path(__file__),
        "annual_context_builder": Path(build_related_feature_panel.__code__.co_filename),
    }
    for season in range(2014, 2027):
        source_paths[f"player_sheet_{season}"] = Path(player_sheet_dir) / f"{season}.parquet"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}

    annual, annual_quality = build_related_feature_panel(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
        seasons=tuple(range(2014, 2027)),
    )
    rating_seasons = tuple(int(value) for value in contract["rating_seasons"])
    context = pool_five_year_context(annual, window_ends=rating_seasons)
    baseline = pd.read_parquet(baseline_predictions_path)
    baseline = baseline.loc[baseline["target_kind"].eq("five_year")].copy()
    baseline = baseline.rename(
        columns={
            "prior_offense_per_100": "baseline_offense",
            "prior_defense_per_100": "baseline_defense",
            "prior_net_per_100": "baseline_net",
        }
    )
    targets = pd.read_parquet(five_year_targets_path)
    panel = baseline.merge(
        targets[
            [
                "PLAYER_ID",
                "Window_End",
                "target_offense",
                "target_defense",
                "target_net",
                "Poss_Off",
                "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).merge(
        context,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1)
    )
    for side in SIDES:
        panel[f"residual_{side}"] = (
            panel[f"target_{side}"] - panel[f"baseline_{side}"]
        )

    selection = int(contract["selection_rating_season"])
    alphas = tuple(float(value) for value in contract["ridge_alphas"])
    selected = {}
    selection_rows = []
    for side in SIDES:
        train = panel.loc[panel["Window_End"].lt(selection)]
        score = panel.loc[panel["Window_End"].eq(selection)]
        candidates = []
        for alpha in alphas:
            correction, _ = _fit_predict(
                train,
                score,
                features=CONTEXT_BANKS[side],
                target=f"residual_{side}",
                weight="sample_weight",
                alpha=alpha,
            )
            adjusted = score[f"baseline_{side}"].to_numpy(dtype=float) + correction
            metric = weighted_metrics(
                score[f"target_{side}"].to_numpy(dtype=float),
                adjusted,
                score["sample_weight"].to_numpy(dtype=float),
            )
            row = {
                "stage": "selection",
                "season": selection,
                "candidate": "five_year_spm_context",
                "side": side,
                "alpha": alpha,
                "players": int(len(score)),
                **metric,
            }
            candidates.append(row)
            selection_rows.append(row)
        winner = min(candidates, key=lambda row: (row["weighted_rmse"], row["alpha"]))
        selected[side] = float(winner["alpha"])

    prediction_rows = []
    metric_rows = []
    for season in tuple(value for value in rating_seasons if value >= selection):
        train = panel.loc[panel["Window_End"].lt(season)]
        score = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3:
            raise ValueError(f"Rating season {season} has fewer than three training ends.")
        for side in SIDES:
            correction, _ = _fit_predict(
                train,
                score,
                features=CONTEXT_BANKS[side],
                target=f"residual_{side}",
                weight="sample_weight",
                alpha=selected[side],
            )
            score[f"context_correction_{side}"] = correction
            score[f"adjusted_{side}"] = score[f"baseline_{side}"] + correction
        score["adjusted_net"] = score["adjusted_offense"] + score["adjusted_defense"]
        score["context_correction_net"] = (
            score["context_correction_offense"] + score["context_correction_defense"]
        )
        stage = "selection" if season == selection else "diagnostic"
        metric_rows.extend(
            _score(
                score,
                candidate="five_year_spm",
                season=season,
                stage=stage,
                prediction_prefix="baseline",
            )
        )
        metric_rows.extend(
            _score(
                score,
                candidate="five_year_spm_context",
                season=season,
                stage=stage,
                prediction_prefix="adjusted",
            )
        )
        prediction_rows.append(score)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    same_window_metrics = pd.DataFrame(metric_rows)

    annual_targets = pd.read_parquet(annual_targets_path).rename(
        columns={"Season": "test_season"}
    )
    required_annual = {
        "PLAYER_ID",
        "test_season",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    }
    if missing := sorted(required_annual - set(annual_targets.columns)):
        raise ValueError(f"Annual targets are missing {missing}.")
    if annual_targets.duplicated(["PLAYER_ID", "test_season"]).any():
        raise ValueError("Annual target keys are not unique.")
    next_rows = []
    next_predictions = []
    for rating_season in tuple(int(value) for value in contract["next_season_rating_seasons"]):
        rated = predictions.loc[predictions["Window_End"].eq(rating_season)].copy()
        target = annual_targets.loc[
            annual_targets["test_season"].eq(rating_season + 1)
        ].copy()
        joined = rated.merge(
            target[
                [
                    "PLAYER_ID",
                    "target_offense",
                    "target_defense",
                    "target_net",
                    "Poss_Off",
                    "Poss_Def",
                ]
            ],
            on="PLAYER_ID",
            how="inner",
            suffixes=("", "_next"),
            validate="one_to_one",
        )
        joined["sample_weight"] = np.sqrt(
            joined[["Poss_Off_next", "Poss_Def_next"]].min(axis=1).clip(lower=1)
        )
        joined["rating_season"] = rating_season
        joined["test_season"] = rating_season + 1
        next_rows.extend(
            _score(
                joined,
                candidate="five_year_spm",
                season=rating_season + 1,
                stage="next_season_diagnostic",
                prediction_prefix="baseline",
            )
        )
        next_rows.extend(
            _score(
                joined,
                candidate="five_year_spm_context",
                season=rating_season + 1,
                stage="next_season_diagnostic",
                prediction_prefix="adjusted",
            )
        )
        next_predictions.append(joined)
    next_season_metrics = pd.DataFrame(next_rows)
    next_season_predictions = pd.concat(next_predictions, ignore_index=True)

    config = {
        "contract": contract,
        "context_banks": {side: list(values) for side, values in CONTEXT_BANKS.items()},
        "selected_alphas": selected,
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = (
        Path(artifact_root)
        / "research"
        / "five_year_spm_context"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(selection_rows).to_parquet(output / "alpha_selection.parquet", index=False)
    same_window_metrics.to_parquet(output / "same_window_metrics.parquet", index=False)
    next_season_metrics.to_parquet(output / "next_season_metrics.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    next_season_predictions.to_parquet(output / "next_season_predictions.parquet", index=False)

    same_net = same_window_metrics.loc[same_window_metrics["side"].eq("net")]
    next_net = next_season_metrics.loc[next_season_metrics["side"].eq("net")]
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "annual_feature_quality": annual_quality,
            "panel_rows": int(len(panel)),
            "context_key_coverage": float(
                panel[list(CONTEXT_SOURCES)].notna().all(axis=1).mean()
            ),
            "rows_by_rating_season": {
                str(key): int(value)
                for key, value in panel["Window_End"].value_counts().sort_index().items()
            },
            "season_2027_loaded": False,
            "net_identity_max_error": float(
                (
                    predictions["adjusted_offense"]
                    + predictions["adjusted_defense"]
                    - predictions["adjusted_net"]
                ).abs().max()
            ),
        },
        "same_window_net_metrics": same_net.to_dict("records"),
        "next_season_net_metrics": next_net.to_dict("records"),
        "decision_rule": (
            "Retain only as a research candidate if the correction improves both later "
            "same-window folds and does not worsen either next-season fold."
        ),
        "caveats": [
            "The context model is a chronological ridge correction to the exact frozen five-year SPM, not a full joint refit.",
            "Season 2025 is an inspected failure and Season 2026 is heavily reused; neither is untouched confirmation.",
            "Same-team context may capture team and scheme strength. It is not player skill or a causal teammate effect.",
            "Annual TEAM_ID makes context approximate for traded players.",
            "DFG and rim-DFG observations end in 2025, so the 2026 defense context is source-constrained.",
        ],
        "paths": {
            "alpha_selection": "alpha_selection.parquet",
            "same_window_metrics": "same_window_metrics.parquet",
            "next_season_metrics": "next_season_metrics.parquet",
            "predictions": "predictions.parquet",
            "next_season_predictions": "next_season_predictions.parquet",
        },
        "forbidden_interpretation": (
            "Production model, causal teammate value, independent confirmation, or complete defense ceiling."
        ),
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest
