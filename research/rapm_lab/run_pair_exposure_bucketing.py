"""Test low-exposure pair shrinkage before any higher-order unit expansion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics, offense_defense_lineups
from nba_impact.models.rapm import build_design, load_current_player_names, load_unified_terminal_possessions
from nba_impact.models.standalone_unit_rapm import fit_unit_rapm, predict_unit_rapm, unit_slot_coverage
from research.rapm_lab.run_standalone_unit_rapm import _fit_reference, _paired_game_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/pair_exposure_bucketing_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/pair_exposure_bucketing"


def _score(frame, prediction):
    return game_margin_metrics(frame.reset_index(drop=True), prediction)


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(contract["seasons"])
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    offense, defense = offense_defense_lineups(frame)
    stages = {}
    fits = {}
    for stage_name in ("selection", "diagnostic"):
        stage = contract[stage_name]
        train = np.isin(design.seasons, stage["train"])
        test = design.seasons == stage["test"]
        _, reference_metrics, reference_games = _fit_reference(
            frame,
            design,
            train_seasons=tuple(stage["train"]),
            test_season=stage["test"],
        )
        candidates = []
        if stage_name == "selection":
            specs = [
                (strategy, float(penalty))
                for strategy in contract["strategies"]
                for penalty in contract["unit_penalties"]
            ]
        else:
            winner = stages["selection"]["winner"]
            specs = [(winner["strategy"], winner["unit_penalty"])]
        for strategy, penalty in specs:
            print(f"{stage_name} pair {strategy} lambda={penalty:g}", flush=True)
            fit = fit_unit_rapm(
                offense[train],
                defense[train],
                design.home_offense[train],
                design.y[train],
                order=2,
                unit_penalty=penalty,
                minimum_exposure=contract["minimum_exposure"],
                home_penalty=contract["home_penalty"],
                penalty_strategy=strategy,
                maximum_iterations=500,
            )
            prediction = predict_unit_rapm(
                fit,
                offense[test],
                defense[test],
                design.home_offense[test],
            )
            metrics, games = _score(frame.loc[test], prediction)
            candidates.append(
                {
                    "strategy": strategy,
                    "unit_penalty": penalty,
                    "units": int(len(fit.combinations)),
                    "test_slot_coverage": unit_slot_coverage(fit, offense[test], defense[test]),
                    "rmse_delta_vs_one_player": metrics["margin_rmse"] - reference_metrics["margin_rmse"],
                    **metrics,
                }
            )
            fits[(stage_name, strategy, penalty)] = (fit, games)
        table = pd.DataFrame(candidates).sort_values(["margin_rmse", "unit_penalty", "strategy"], kind="stable")
        winner = table.iloc[0].to_dict()
        stages[stage_name] = {
            "reference": reference_metrics,
            "candidates": table.to_dict("records"),
            "winner": winner,
        }
        if stage_name == "diagnostic":
            selected_fit, selected_games = fits[(stage_name, winner["strategy"], winner["unit_penalty"])]
            stages[stage_name]["paired_bootstrap_vs_one_player"] = _paired_game_bootstrap(reference_games, selected_games)

    selected = stages["diagnostic"]["winner"]
    fit, _ = fits[("diagnostic", selected["strategy"], selected["unit_penalty"])]
    names = load_current_player_names(
        ROOT / "rapm/data/all_names.csv",
        ROOT / "data/lake/silver/player_games.parquet",
    ).set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()
    n = len(fit.combinations)
    ratings = pd.DataFrame(
        {
            "players": ["|".join(str(int(value)) for value in row) for row in fit.combinations],
            "pair": [" + ".join(names.get(int(value), str(int(value))) for value in row) for row in fit.combinations],
            "offense_per_100": 100.0 * fit.coefficients[:n],
            "defense_per_100": -100.0 * fit.coefficients[n : 2 * n],
        }
    )
    ratings["net_per_100"] = ratings["offense_per_100"] + ratings["defense_per_100"]
    identity = hashlib.sha256(json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"pair_exposure_bucketing_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output / "pair_ratings.parquet", index=False)
    passes = (
        selected["rmse_delta_vs_one_player"] < 0
        and selected["strategy"] != "hard_floor"
        and stages["diagnostic"]["paired_bootstrap_vs_one_player"]["rmse_delta_95_high"] < 0
    )
    manifest = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": contract,
        "selection": stages["selection"],
        "diagnostic": stages["diagnostic"],
        "passes_higher_order_gate": passes,
        "decision": "expand_to_higher_orders" if passes else "stop_after_pair",
        "artifact": "pair_ratings.parquet",
        "forbidden_interpretation": "Causal two-player chemistry or individual player impact.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
