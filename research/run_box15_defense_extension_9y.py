#!/usr/bin/env python3
"""Test small defensive residual additions to the nine-year Box15 prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic

try:
    import run_target_window_spm_aio as target
except ModuleNotFoundError:
    from research import run_target_window_spm_aio as target


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/experiments/box15_defense_extension_9y_v1.yml"
SOURCE_RUN = ROOT / (
    "artifacts/research/target_window_spm_aio/"
    "target_window_spm_aio_v1_8e028133cb"
)
RICH_ANNUAL = target.RICH_ANNUAL
MECHANISM_ANNUAL = target.MECHANISM_ANNUAL


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_contract(path: Path = CONTRACT) -> dict:
    contract = yaml.safe_load(path.read_text())
    if contract.get("experiment_id") != path.stem:
        raise ValueError("Experiment ID must match the contract filename.")
    rating = tuple(map(int, contract["rating_seasons"]))
    tests = tuple(map(int, contract["test_seasons"]))
    if tests != tuple(season + 1 for season in rating):
        raise ValueError("Every rating season must predict the next season.")
    if contract["source_model"]["candidate"] != "box15_9y_normal":
        raise ValueError("The Box15 control changed.")
    return contract


def _feature_panel(contract: dict) -> pd.DataFrame:
    rich = pd.read_parquet(RICH_ANNUAL)
    mechanism = pd.read_parquet(MECHANISM_ANNUAL)
    fields = tuple(
        dict.fromkeys(
            feature
            for features in contract["candidates"].values()
            for feature in features
        )
    )
    extra = tuple(field for field in fields if field not in rich.columns)
    panel = rich.merge(
        mechanism[["PLAYER_ID", "Window_End", *extra]],
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    if missing := sorted(set(fields) - set(panel.columns)):
        raise ValueError(f"Feature panel misses {missing}.")
    if panel[list(fields)].isna().any().any():
        raise ValueError("Selected defensive inputs contain missing values.")
    return panel[["PLAYER_ID", "Window_End", *fields]]


def _learning_panel(contract: dict) -> pd.DataFrame:
    source = str(contract["source_model"]["candidate"])
    priors = pd.read_parquet(SOURCE_RUN / "priors.parquet")
    priors = priors.loc[priors["candidate"].eq(source)].drop(columns="candidate")
    targets = pd.read_parquet(SOURCE_RUN / "targets.parquet")
    targets = targets.loc[
        targets["horizon"].eq(int(contract["source_model"]["target_horizon"]))
        & targets["target_variant"].eq(contract["source_model"]["target_variant"])
    ]
    panel = targets.merge(
        priors,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).merge(
        _feature_panel(contract),
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    panel["residual_target"] = (
        panel["target_defense"] - panel["prior_defense_per_100"]
    )
    return panel


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )


def _select_alpha(
    train: pd.DataFrame, features: tuple[str, ...], alphas: tuple[float, ...]
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        season_mse = []
        for held_season in sorted(train["Window_End"].unique()):
            inner = train.loc[train["Window_End"].ne(held_season)]
            held = train.loc[train["Window_End"].eq(held_season)]
            if inner.empty or held.empty:
                continue
            model = _pipeline(alpha)
            model.fit(
                inner.loc[:, features],
                inner["residual_target"],
                ridge__sample_weight=inner["sample_weight"],
            )
            error = held["residual_target"].to_numpy(dtype=float) - model.predict(
                held.loc[:, features]
            )
            season_mse.append(
                float(np.average(error**2, weights=held["sample_weight"]))
            )
        rows.append(
            {
                "alpha": alpha,
                "training_seasons": len(season_mse),
                "mean_weighted_mse": float(np.mean(season_mse)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["mean_weighted_mse", "alpha"], kind="stable"
    )
    return float(scores.iloc[0]["alpha"]), scores


def _fit_priors(
    panel: pd.DataFrame, contract: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    alphas = tuple(map(float, contract["residual_model"]["alpha_grid"]))
    scale = float(contract["residual_model"]["residual_scale"])
    for season in map(int, contract["rating_seasons"]):
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 2 or test.empty:
            continue
        for candidate, values in contract["candidates"].items():
            features = tuple(values)
            prior = test[
                [
                    "PLAYER_ID",
                    "Window_End",
                    "prior_offense_per_100",
                    "prior_defense_per_100",
                ]
            ].copy()
            selected_alpha = np.nan
            if features:
                selected_alpha, scores = _select_alpha(train, features, alphas)
                model = _pipeline(selected_alpha)
                model.fit(
                    train.loc[:, features],
                    train["residual_target"],
                    ridge__sample_weight=train["sample_weight"],
                )
                prior["prior_defense_per_100"] += scale * model.predict(
                    test.loc[:, features]
                )
                for record in scores.to_dict("records"):
                    selections.append(
                        {
                            "rating_season": season,
                            "candidate": candidate,
                            "selected": record["alpha"] == selected_alpha,
                            "feature_count": len(features),
                            **record,
                        }
                    )
            prior["prior_net_per_100"] = (
                prior["prior_offense_per_100"]
                + prior["prior_defense_per_100"]
            )
            prior["candidate"] = candidate
            rows.append(prior)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(selections)


def _paired_bootstrap(
    games: pd.DataFrame, control: str, *, draws: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    control_games = games.loc[games["candidate"].eq(control)]
    for candidate in sorted(games["candidate"].unique()):
        if candidate in {control, "zero_prior_rapm"} or not candidate.endswith("_aio"):
            continue
        challenger = games.loc[games["candidate"].eq(candidate)]
        seasons = sorted(set(control_games["test_season"]) & set(challenger["test_season"]))
        differences = []
        wins = 0
        for season in seasons:
            left = control_games.loc[control_games["test_season"].eq(season)].set_index("game_id").sort_index()
            right = challenger.loc[challenger["test_season"].eq(season)].set_index("game_id").sort_index()
            if not left.index.equals(right.index):
                raise ValueError(f"{control} and {candidate} score different games.")
            delta = right["squared_error"].to_numpy() - left["squared_error"].to_numpy()
            differences.append(delta)
            wins += float(delta.mean()) < 0
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = np.mean(
                [values[rng.integers(0, len(values), len(values))].mean() for values in differences]
            )
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "candidate": candidate,
                "reference": control,
                "folds": len(seasons),
                "candidate_minus_reference_mse": float(np.mean([value.mean() for value in differences])),
                "lower_95": float(low),
                "upper_95": float(high),
                "probability_candidate_better": float(np.mean(samples < 0)),
                "candidate_fold_wins": int(wins),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_contract(contract_path)
    experiment_id = str(contract["experiment_id"])
    output_root = ROOT / "artifacts/research" / experiment_id.removesuffix("_v1")
    sources = {
        "contract": contract_path,
        "runner": Path(__file__),
        "source_manifest": SOURCE_RUN / "run.json",
        "source_targets": SOURCE_RUN / "targets.parquet",
        "source_priors": SOURCE_RUN / "priors.parquet",
        "rich_features": RICH_ANNUAL,
        "mechanism_features": MECHANISM_ANNUAL,
    }
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = output_root / f"{experiment_id}_{identity}"
    if (output / "run.json").exists():
        print(output)
        return

    panel = _learning_panel(contract)
    priors, selections = _fit_priors(panel, contract)
    games, ratings, coverage = target._evaluate_priors(priors, contract)
    folds, summary = target._game_metrics(games)
    control = f"{contract['source_model']['candidate']}_aio"
    paired = _paired_bootstrap(
        games,
        control,
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )
    control_rmse = float(
        summary.loc[summary["candidate"].eq(control), "equal_season_rmse"].iloc[0]
    )
    decisions = paired.merge(
        summary[["candidate", "equal_season_rmse", "mean_correlation"]],
        on="candidate",
        validate="one_to_one",
    )
    decisions["rmse_gain"] = control_rmse - decisions["equal_season_rmse"]
    decisions["passes_exploratory_gate"] = (
        decisions["rmse_gain"].ge(float(contract["evaluation"]["practical_rmse_gate"]))
        & decisions["upper_95"].lt(0)
    )

    outputs = {
        "priors.parquet": priors,
        "residual_alpha_selection.parquet": selections,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "paired_bootstrap.parquet": paired,
        "decisions.parquet": decisions,
        "prior_coverage.parquet": coverage,
    }
    for name, frame in outputs.items():
        _atomic_parquet(frame, output / name)
    run = {
        "run_id": output.name,
        "experiment_id": experiment_id,
        "status": f"{contract['status']}_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashes[name],
            }
            for name, path in sources.items()
        },
        "quality": {
            "learning_rows": len(panel),
            "prior_rows": len(priors),
            "game_prediction_rows": len(games),
            "rating_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "identical_games_within_fold": True,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "rows": len(frame),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.loc[summary["candidate"].str.endswith("_aio")].to_string(index=False))
    print("\nDefense extensions versus Box15")
    print(decisions.sort_values("equal_season_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
