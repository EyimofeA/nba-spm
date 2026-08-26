"""Run the fixed Phase B SPM defensive-source comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.defensive_tracking_features import DEFENSIVE_TRACKING_FEATURES
from nba_impact.data.manifest import sha256_file
from nba_impact.data.matchup_defense_features import MATCHUP_DEFENSE_FEATURES
from nba_impact.models.predictive_spm import _predictive_metrics
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model
from run_spm_defense_source_audit import (
    ARTIFACT_PATHS,
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT,
    OFFICIAL_DFG,
    OFFICIAL_RIM,
    PUBLIC_DFG,
    PUBLIC_RIM,
    _load_manifest,
    _source_family,
    _tracking_observation_keys,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "artifacts/models/canonical_annual_target_panel/canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
PHASE_A_DECISION = DEFAULT_OUTPUT / "decision.json"
OUTPUT = DEFAULT_OUTPUT / "phase_b"


def _defense_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _tracking_matrix(
    artifact: Path,
    selected: list[str],
    observed: dict[str, set[tuple[int, int]]],
) -> pd.DataFrame:
    columns = [feature for feature in selected if feature in DEFENSIVE_TRACKING_FEATURES]
    frame = pd.read_parquet(
        artifact / "features.parquet", columns=["PLAYER_ID", "Season", *columns]
    )
    keys = list(zip(frame["PLAYER_ID"].astype(int), frame["Season"].astype(int), strict=True))
    for feature in columns:
        family = _source_family(feature)
        is_observed = np.fromiter(
            (key in observed[family] for key in keys), dtype=bool, count=len(keys)
        )
        frame.loc[~is_observed, feature] = np.nan
        frame[f"observed__{family}"] = is_observed
    return frame


def _source_matrix(
    *,
    version: str,
    selected: list[str],
    public_base: pd.DataFrame,
) -> pd.DataFrame:
    if version == "control":
        tracking_path = ARTIFACT_PATHS["public_tracking"]
        matchup_path = ARTIFACT_PATHS["public_matchup"]
        tracking_manifest = _load_manifest(tracking_path)
        observed = _tracking_observation_keys(
            tracking_manifest, dfg_path=PUBLIC_DFG, rim_path=PUBLIC_RIM
        )
    elif version == "challenger":
        tracking_path = ARTIFACT_PATHS["latest_tracking"]
        matchup_path = ARTIFACT_PATHS["latest_matchup"]
        tracking_manifest = _load_manifest(tracking_path)
        observed = _tracking_observation_keys(
            tracking_manifest, dfg_path=OFFICIAL_DFG, rim_path=OFFICIAL_RIM
        )
    else:
        raise ValueError(f"Unknown source version {version}.")

    tracking = _tracking_matrix(tracking_path, selected, observed)
    matchup_columns = [feature for feature in selected if feature in MATCHUP_DEFENSE_FEATURES]
    matchup = pd.read_parquet(
        matchup_path / "features.parquet",
        columns=["PLAYER_ID", "Season", *matchup_columns],
    ).rename(columns={"Season": "Window_End"})
    matchup["observed__matchup_defense"] = True
    tracking = tracking.rename(columns={"Season": "Window_End"})

    official = set(DEFENSIVE_TRACKING_FEATURES) | set(MATCHUP_DEFENSE_FEATURES)
    nonofficial = [feature for feature in selected if feature not in official]
    matrix = public_base[["PLAYER_ID", "Window_End", *nonofficial]].merge(
        tracking,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    ).merge(
        matchup,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    for family in ("official_dfg", "official_rim_dfg", "hustle", "matchup_defense"):
        column = f"observed__{family}"
        if column not in matrix:
            matrix[column] = False
        matrix[column] = matrix[column].fillna(False).astype(bool)
    missing = sorted(set(selected) - set(matrix))
    if missing:
        raise ValueError(f"{version} matrix lacks selected fields {missing}.")
    return matrix


def _bootstrap_difference(
    predictions: pd.DataFrame,
    *,
    draws: int = 5_000,
    seed: int = 20260826,
) -> pd.DataFrame:
    players = np.sort(predictions["PLAYER_ID"].unique())
    player_index = {int(player): index for index, player in enumerate(players)}
    row_player = predictions["PLAYER_ID"].map(player_index).to_numpy(dtype=int)
    seasons = sorted(predictions["test_season"].unique())
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        counts = rng.multinomial(len(players), np.full(len(players), 1.0 / len(players)))
        multiplicity = counts[row_player]
        differences = []
        for season in seasons:
            mask = predictions["test_season"].eq(season).to_numpy() & (multiplicity > 0)
            weight = predictions.loc[mask, "sample_weight"].to_numpy() * multiplicity[mask]
            actual = predictions.loc[mask, "target_defense"].to_numpy()
            control = predictions.loc[mask, "control_defense"].to_numpy()
            challenger = predictions.loc[mask, "challenger_defense"].to_numpy()
            control_rmse = np.sqrt(np.average((actual - control) ** 2, weights=weight))
            challenger_rmse = np.sqrt(
                np.average((actual - challenger) ** 2, weights=weight)
            )
            differences.append(challenger_rmse - control_rmse)
        rows.append(
            {"draw": draw, "challenger_minus_control_rmse": float(np.mean(differences))}
        )
    return pd.DataFrame(rows)


def _slice_rmse(frame: pd.DataFrame, prediction: str, mask: pd.Series) -> float:
    rows = frame.loc[mask]
    return float(
        np.sqrt(
            np.average(
                (rows["target_defense"] - rows[prediction]) ** 2,
                weights=rows["sample_weight"],
            )
        )
    )


def run_phase_b() -> dict:
    contract = yaml.safe_load(DEFAULT_CONTRACT.read_text())
    phase_a = json.loads(PHASE_A_DECISION.read_text())
    if not phase_a.get("phase_a_passed"):
        raise ValueError("Phase A did not permit a model fit.")
    if int(2027) in set(contract["development_seasons"]):
        raise ValueError("Season 2027 must remain untouched.")

    selected = _selected_single_season_features(ARTIFACT_PATHS["public_spm_run"])
    defense_features = list(selected["defense"])
    offense_features = list(selected["offense"])
    base = pd.read_parquet(
        ARTIFACT_PATHS["public_features"] / "features.parquet",
        columns=["PLAYER_ID", "Window_End", *sorted(set(defense_features + offense_features))],
    )
    control = _source_matrix(
        version="control", selected=defense_features, public_base=base
    )
    challenger = _source_matrix(
        version="challenger", selected=defense_features, public_base=base
    )
    targets = pd.read_parquet(TARGETS).rename(columns={"Season": "Window_End"})
    target_columns = [
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    ]
    base_targets = base.merge(
        targets[target_columns],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    base_targets["sample_weight"] = np.sqrt(
        np.minimum(base_targets["Poss_Off"], base_targets["Poss_Def"]).clip(lower=1)
    )
    offense_only = [feature for feature in offense_features if feature not in defense_features]
    control = base_targets[
        ["PLAYER_ID", "Window_End", *target_columns[2:], "sample_weight", *offense_only]
    ].merge(
        control,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    challenger = base_targets[["PLAYER_ID", "Window_End"]].merge(
        challenger,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    if control[["PLAYER_ID", "Window_End"]].to_records(index=False).tolist() != challenger[
        ["PLAYER_ID", "Window_End"]
    ].to_records(index=False).tolist():
        raise ValueError("Control and challenger row keys differ.")

    alpha = float(contract["phase_b"]["ridge_alpha"])
    test_seasons = tuple(int(value) for value in contract["phase_b"]["test_seasons"])
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for season in test_seasons:
        train_mask = control["Window_End"].lt(season)
        test_mask = control["Window_End"].eq(season)
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Empty chronological partition for {season}.")
        train_control, test_control = control.loc[train_mask], control.loc[test_mask]
        train_challenger, test_challenger = challenger.loc[train_mask], challenger.loc[test_mask]

        control_model = _defense_model(alpha)
        control_model.fit(
            train_control[defense_features],
            train_control["target_defense"],
            model__sample_weight=train_control["sample_weight"],
        )
        challenger_model = _defense_model(alpha)
        challenger_model.fit(
            train_challenger[defense_features],
            train_control["target_defense"],
            model__sample_weight=train_control["sample_weight"],
        )
        offense_model = _fit_model(
            _frozen_model("offense"),
            train_control,
            tuple(offense_features),
            "target_offense",
        )
        fold = test_control[
            [
                "PLAYER_ID",
                "Window_End",
                "target_offense",
                "target_defense",
                "target_net",
                "Poss_Off",
                "Poss_Def",
                "sample_weight",
            ]
        ].copy()
        fold["test_season"] = season
        fold["control_defense"] = control_model.predict(test_control[defense_features])
        fold["challenger_defense"] = challenger_model.predict(
            test_challenger[defense_features]
        )
        fold["frozen_offense"] = offense_model.predict(test_control[offense_features])
        fold["control_net"] = fold["frozen_offense"] + fold["control_defense"]
        fold["challenger_net"] = fold["frozen_offense"] + fold["challenger_defense"]
        fold["repaired_source_observed"] = test_challenger[
            [
                "observed__official_dfg",
                "observed__official_rim_dfg",
                "observed__hustle",
                "observed__matchup_defense",
            ]
        ].all(axis=1).to_numpy()
        predictions.append(fold)

        for arm in ("control", "challenger"):
            for component in ("defense", "net"):
                actual = fold[f"target_{component}"].to_numpy()
                predicted = fold[f"{arm}_{component}"].to_numpy()
                metric_rows.append(
                    {
                        "test_season": season,
                        "arm": arm,
                        "component": component,
                        "train_start": int(train_control["Window_End"].min()),
                        "train_end": int(train_control["Window_End"].max()),
                        "rows": len(fold),
                        **_predictive_metrics(
                            actual, predicted, fold["sample_weight"].to_numpy()
                        ),
                    }
                )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bootstrap = _bootstrap_difference(prediction_frame)
    defense = metrics.loc[metrics["component"].eq("defense")]
    control_rows = defense.loc[defense["arm"].eq("control")].set_index("test_season")
    challenger_rows = defense.loc[defense["arm"].eq("challenger")].set_index("test_season")
    rmse_difference = challenger_rows["weighted_rmse"] - control_rows["weighted_rmse"]
    mean_control = float(control_rows["weighted_rmse"].mean())
    mean_challenger = float(challenger_rows["weighted_rmse"].mean())
    interval = bootstrap["challenger_minus_control_rmse"].quantile([0.025, 0.975])
    probability_better = float(
        bootstrap["challenger_minus_control_rmse"].lt(0).mean()
    )
    correlation_change = float(
        (
            challenger_rows["weighted_correlation"]
            - control_rows["weighted_correlation"]
        ).mean()
    )
    calibration_worsening = float(
        (
            (challenger_rows["calibration_slope"] - 1).abs()
            - (control_rows["calibration_slope"] - 1).abs()
        ).mean()
    )
    exposure = np.minimum(prediction_frame["Poss_Off"], prediction_frame["Poss_Def"])
    exposure_changes = {}
    for label, mask in {
        "low": exposure.lt(1000),
        "high": exposure.ge(1000),
    }.items():
        exposure_changes[label] = _slice_rmse(
            prediction_frame, "challenger_defense", mask
        ) - _slice_rmse(prediction_frame, "control_defense", mask)

    gates = {
        "phase_a_passed": True,
        "mean_rmse_improvement_at_least_0_010": mean_control - mean_challenger >= 0.010,
        "wins_four_of_five_folds": int(rmse_difference.lt(0).sum()) >= 4,
        "bootstrap_probability_at_least_0_95": probability_better >= 0.95,
        "bootstrap_upper_bound_below_zero": float(interval.loc[0.975]) < 0,
        "weighted_correlation_noninferiority": correlation_change >= -0.010,
        "calibration_noninferiority": calibration_worsening <= 0.05,
        "low_exposure_safety": exposure_changes["low"] <= 0.020,
        "high_exposure_safety": exposure_changes["high"] <= 0.020,
        "season_2027_untouched": True,
    }
    promoted = all(gates.values())
    decision = {
        "experiment_id": contract["experiment_id"],
        "phase": "B_fixed_source_comparison",
        "status": "promote_to_downstream_aio" if promoted else "null_stop_before_aio",
        "promoted": promoted,
        "gates": gates,
        "mean_weighted_defense_rmse": {
            "control": mean_control,
            "challenger": mean_challenger,
            "challenger_minus_control": mean_challenger - mean_control,
        },
        "fold_wins": int(rmse_difference.lt(0).sum()),
        "weighted_correlation_change": correlation_change,
        "calibration_distance_worsening": calibration_worsening,
        "exposure_rmse_changes": exposure_changes,
        "bootstrap": {
            "draws": len(bootstrap),
            "probability_challenger_better": probability_better,
            "interval_95": [float(interval.loc[0.025]), float(interval.loc[0.975])],
        },
        "inputs": {
            "contract_sha256": sha256_file(DEFAULT_CONTRACT),
            "phase_a_decision_sha256": sha256_file(PHASE_A_DECISION),
            "targets_sha256": sha256_file(TARGETS),
            "public_features_sha256": sha256_file(
                ARTIFACT_PATHS["public_features"] / "features.parquet"
            ),
            "latest_tracking_sha256": sha256_file(
                ARTIFACT_PATHS["latest_tracking"] / "features.parquet"
            ),
            "latest_matchup_sha256": sha256_file(
                ARTIFACT_PATHS["latest_matchup"] / "features.parquet"
            ),
            "source_code_sha256": sha256_file(Path(__file__)),
        },
        "season_2027_loaded": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(OUTPUT / "outer_predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT / "paired_player_bootstrap.csv", index=False)
    (OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    report = [
        "# SPM defense source-transition Phase B",
        "",
        f"Decision: `{decision['status']}`.",
        "",
        f"Equal-season weighted defense RMSE changed from {mean_control:.4f} to {mean_challenger:.4f}.",
        f"The challenger won {decision['fold_wins']} of {len(test_seasons)} chronological folds.",
        f"Paired player-bootstrap probability of improvement: {probability_better:.3f}.",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{value}`" for name, value in gates.items()],
        "",
        "Season 2025 and 2026 were not used. Season 2027 was not loaded.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    return decision


if __name__ == "__main__":
    print(json.dumps(run_phase_b(), indent=2))
