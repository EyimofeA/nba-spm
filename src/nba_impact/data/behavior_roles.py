"""Behavior-only annual player roles with chronological stability diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROLE_INPUT_FEATURES = (
    "FG2A_p100", "FG3A_p100", "FTA_p100", "OREB_p100",
    "drives_p100", "touches_p100", "front_court_touches_p100",
    "paint_touches_p100", "post_touches_p100", "elbow_touches_p100",
    "time_of_possession_p100", "passes_made_p100", "potential_assists_p100",
    "avg_seconds_per_touch", "avg_dribbles_per_touch",
    "self_created_point_share", "assisted_three_share", "pull_up_attempt_share",
    "drive_pass_rate", "creation_load_p100", "interior_role_load",
    "at_rim_frequency", "short_mid_frequency", "long_mid_frequency",
    "corner3_frequency", "arc3_frequency", "rebound_contests_p100",
    "dreb_chances_p100", "all_shot_zero_dribble_share",
    "all_shot_three_plus_dribble_share", "jump_shot_zero_dribble_share",
    "jump_shot_three_plus_dribble_share",
)
ROLE_AXIS_FEATURES = tuple(f"role_axis_{index}" for index in range(1, 7))
ROLE_AFFINITY_FEATURES = tuple(f"role_affinity_{index}" for index in range(7))
ROLE_MODEL_FEATURES = (*ROLE_AXIS_FEATURES, *ROLE_AFFINITY_FEATURES)
FORBIDDEN_ROLE_TOKENS = (
    "age", "height", "position", "minutes", "games", "onoff", "plus_minus",
    "rapm", "spm", "accuracy", "fg_pct", "points_above", "pts_p100",
)


@dataclass(frozen=True)
class BehaviorRoleConfig:
    development_seasons: tuple[int, ...] = (2014, 2015, 2016, 2017, 2018)
    n_clusters: int = 8
    n_axes: int = 6
    minimum_descriptor_coverage: float = 0.80
    seed: int = 20260817
    stability_seeds: tuple[int, ...] = (11, 29, 47, 71, 101)
    minimum_seed_ari: float = 0.90
    minimum_adjacent_same_role: float = 0.50
    minimum_adjacent_axis_cosine: float = 0.75
    minimum_cluster_share: float = 0.02
    maximum_cluster_share: float = 0.30


def _season_robust_scale(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for season, rows in frame.groupby("Season").groups.items():
        subset = frame.loc[rows, ROLE_INPUT_FEATURES]
        median = subset.median()
        filled = subset.fillna(median).fillna(0.0)
        iqr = filled.quantile(0.75) - filled.quantile(0.25)
        fallback = filled.std(ddof=0).replace(0.0, np.nan)
        scale = iqr.where(iqr.gt(1e-9), fallback).fillna(1.0)
        output.loc[rows, ROLE_INPUT_FEATURES] = (
            (filled - median.fillna(0.0)) / scale
        ).clip(-5.0, 5.0).to_numpy()
    return output


def _deterministic_pca(matrix: np.ndarray, n_axes: int) -> PCA:
    pca = PCA(n_components=n_axes, svd_solver="full")
    pca.fit(matrix)
    for row in range(len(pca.components_)):
        pivot = int(np.argmax(np.abs(pca.components_[row])))
        if pca.components_[row, pivot] < 0:
            pca.components_[row] *= -1.0
    return pca


def _ordered_clusters(model: KMeans) -> tuple[dict[int, int], np.ndarray]:
    order = sorted(
        range(len(model.cluster_centers_)),
        key=lambda row: tuple(np.round(model.cluster_centers_[row], 12)),
    )
    mapping = {old: new for new, old in enumerate(order)}
    return mapping, model.cluster_centers_[order]


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1), denominator,
        out=np.zeros(len(left), dtype=float), where=denominator > 0,
    )


def compute_behavior_roles(
    annual_features: pd.DataFrame,
    role_context: pd.DataFrame,
    *,
    config: BehaviorRoleConfig = BehaviorRoleConfig(),
) -> dict[str, object]:
    """Fit roles on development seasons and apply the fixed map to later years."""
    if "Season" not in annual_features and "Window_End" in annual_features:
        annual_features = annual_features.rename(columns={"Window_End": "Season"})
    keys = ["PLAYER_ID", "Season"]
    if annual_features.duplicated(keys).any() or role_context.duplicated(keys).any():
        raise ValueError("Behavior-role inputs must have unique player-season keys.")
    if forbidden := sorted(
        feature for feature in ROLE_INPUT_FEATURES
        if any(token in feature.lower() for token in FORBIDDEN_ROLE_TOKENS)
    ):
        raise ValueError(f"Behavior-role contract contains forbidden inputs {forbidden}.")
    needed_annual = set(ROLE_INPUT_FEATURES) - set(role_context.columns)
    if missing := sorted(needed_annual - set(annual_features.columns)):
        raise ValueError(f"Annual feature table is missing role inputs {missing}.")
    context_columns = [
        feature for feature in ROLE_INPUT_FEATURES if feature in role_context.columns
    ]
    frame = annual_features[keys + sorted(needed_annual)].merge(
        role_context[keys + context_columns], on=keys, how="left", validate="one_to_one"
    )
    for feature in ROLE_INPUT_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame["role_descriptor_coverage"] = frame[list(ROLE_INPUT_FEATURES)].notna().mean(axis=1)
    frame["role_eligible"] = frame["role_descriptor_coverage"].ge(
        config.minimum_descriptor_coverage
    )
    eligible = frame.loc[frame["role_eligible"]].copy()
    eligible = _season_robust_scale(eligible)
    development = eligible.loc[eligible["Season"].isin(config.development_seasons)]
    if development["Season"].nunique() != len(config.development_seasons):
        raise ValueError("One or more role development seasons have no eligible rows.")
    if len(development) < config.n_clusters * 20:
        raise ValueError("Role development sample is too small for the requested clusters.")

    development_matrix = development.loc[:, ROLE_INPUT_FEATURES].to_numpy(dtype=float)
    full_matrix = eligible.loc[:, ROLE_INPUT_FEATURES].to_numpy(dtype=float)
    pca = _deterministic_pca(development_matrix, config.n_axes)
    development_axes = pca.transform(development_matrix)
    full_axes = pca.transform(full_matrix)
    kmeans = KMeans(
        n_clusters=config.n_clusters, n_init=50, random_state=config.seed,
        algorithm="lloyd",
    ).fit(development_axes)
    mapping, ordered_centers = _ordered_clusters(kmeans)
    raw_labels = kmeans.predict(full_axes)
    labels = np.array([mapping[int(label)] for label in raw_labels], dtype=int)
    distances = np.linalg.norm(
        full_axes[:, None, :] - ordered_centers[None, :, :], axis=2
    )
    development_mask = eligible["Season"].isin(config.development_seasons).to_numpy()
    temperature = float(np.median(np.min(distances[development_mask], axis=1) ** 2))
    temperature = max(temperature, 1e-6)
    logits = -(distances**2) / (2.0 * temperature)
    logits -= logits.max(axis=1, keepdims=True)
    affinities = np.exp(logits)
    affinities /= affinities.sum(axis=1, keepdims=True)

    assignments = eligible[[*keys, "role_descriptor_coverage"]].copy()
    assignments["role_cluster"] = [f"role_{label}" for label in labels]
    assignments["role_distance"] = distances[np.arange(len(labels)), labels]
    assignments["role_confidence"] = affinities.max(axis=1)
    for index, feature in enumerate(ROLE_AXIS_FEATURES):
        assignments[feature] = full_axes[:, index]
    for index in range(config.n_clusters):
        assignments[f"role_affinity_{index}"] = affinities[:, index]

    out_of_sample = ~assignments["Season"].isin(config.development_seasons)
    reference_labels = labels[out_of_sample.to_numpy()]
    seed_rows = []
    for seed in config.stability_seeds:
        alternate = KMeans(
            n_clusters=config.n_clusters, n_init=10, random_state=seed,
            algorithm="lloyd",
        ).fit(development_axes)
        alternate_labels = alternate.predict(full_axes[out_of_sample.to_numpy()])
        seed_rows.append(
            {"seed": seed, "adjusted_rand": adjusted_rand_score(reference_labels, alternate_labels)}
        )
    seed_stability = pd.DataFrame(seed_rows)

    prior = assignments.copy()
    prior["Season"] += 1
    prior = prior.rename(
        columns={
            "role_cluster": "previous_role_cluster",
            **{feature: f"previous_{feature}" for feature in ROLE_AXIS_FEATURES},
        }
    )
    adjacent = assignments.merge(
        prior[[*keys, "previous_role_cluster", *(f"previous_{f}" for f in ROLE_AXIS_FEATURES)]],
        on=keys, how="inner", validate="one_to_one",
    )
    adjacent["same_role"] = adjacent["role_cluster"].eq(adjacent["previous_role_cluster"])
    adjacent["axis_cosine"] = _cosine(
        adjacent[list(ROLE_AXIS_FEATURES)].to_numpy(dtype=float),
        adjacent[[f"previous_{f}" for f in ROLE_AXIS_FEATURES]].to_numpy(dtype=float),
    )
    adjacent_evaluation = adjacent.loc[
        ~adjacent["Season"].isin(config.development_seasons)
    ].copy()
    adjacent_summary = adjacent_evaluation.groupby("Season", as_index=False).agg(
        matched_players=("PLAYER_ID", "size"),
        same_role_rate=("same_role", "mean"),
        median_axis_cosine=("axis_cosine", "median"),
    )

    cluster_share = (
        assignments.loc[out_of_sample, "role_cluster"].value_counts(normalize=True)
        .reindex([f"role_{index}" for index in range(config.n_clusters)], fill_value=0.0)
        .rename_axis("role_cluster").reset_index(name="share")
    )
    median_seed_ari = float(seed_stability["adjusted_rand"].median())
    adjacent_same = float(adjacent_evaluation["same_role"].mean())
    adjacent_cosine = float(adjacent_evaluation["axis_cosine"].median())
    min_share = float(cluster_share["share"].min())
    max_share = float(cluster_share["share"].max())
    coverage = float(frame["role_eligible"].mean())
    gates = {
        "coverage": coverage >= 0.90,
        "seed_stability": median_seed_ari >= config.minimum_seed_ari,
        "adjacent_same_role": adjacent_same >= config.minimum_adjacent_same_role,
        "adjacent_axis_cosine": adjacent_cosine >= config.minimum_adjacent_axis_cosine,
        "minimum_cluster_share": min_share >= config.minimum_cluster_share,
        "maximum_cluster_share": max_share <= config.maximum_cluster_share,
    }

    loadings = pd.DataFrame(
        pca.components_, index=ROLE_AXIS_FEATURES, columns=ROLE_INPUT_FEATURES
    ).reset_index(names="role_axis")
    centroid_profiles = pd.DataFrame(
        pca.inverse_transform(ordered_centers), columns=ROLE_INPUT_FEATURES
    )
    centroid_profiles.insert(0, "role_cluster", [f"role_{i}" for i in range(config.n_clusters)])
    return {
        "assignments": assignments,
        "seed_stability": seed_stability,
        "adjacent_stability": adjacent_summary,
        "cluster_share": cluster_share,
        "axis_loadings": loadings,
        "centroid_profiles": centroid_profiles,
        "metrics": {
            "rows": int(len(frame)), "eligible_rows": int(len(assignments)),
            "coverage": coverage, "development_rows": int(len(development)),
            "out_of_sample_rows": int(out_of_sample.sum()),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "median_seed_adjusted_rand": median_seed_ari,
            "adjacent_pairs": int(len(adjacent_evaluation)),
            "adjacent_same_role_rate": adjacent_same,
            "median_adjacent_axis_cosine": adjacent_cosine,
            "minimum_out_of_sample_cluster_share": min_share,
            "maximum_out_of_sample_cluster_share": max_share,
            "gates": gates, "passed": bool(all(gates.values())),
        },
    }


def build_behavior_roles(
    annual_features_path: str | Path,
    role_context_path: str | Path,
    *,
    artifact_root: str | Path,
    config: BehaviorRoleConfig = BehaviorRoleConfig(),
) -> dict:
    result = compute_behavior_roles(
        pd.read_parquet(annual_features_path), pd.read_parquet(role_context_path),
        config=config,
    )
    config_payload = {
        **config.__dict__,
        "development_seasons": list(config.development_seasons),
        "stability_seeds": list(config.stability_seeds),
        "role_input_features": list(ROLE_INPUT_FEATURES),
        "model_candidate_features": list(ROLE_MODEL_FEATURES),
        "source_hashes": {
            "annual_features": sha256_file(annual_features_path),
            "role_context": sha256_file(role_context_path),
            "source_code": sha256_file(Path(__file__)),
        },
    }
    identity = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"behavior_roles_v1_{identity}"
    output = Path(artifact_root) / "features" / "behavior_roles" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name in (
        "assignments", "seed_stability", "adjacent_stability", "cluster_share",
        "axis_loadings", "centroid_profiles",
    ):
        path = output / f"{name}.parquet"
        result[name].to_parquet(path, index=False)
        paths[f"{name}_path"] = str(path.resolve())
    passed = result["metrics"]["passed"]
    run = {
        "run_id": run_id, "dataset": "annual_behavior_roles_v1",
        "status": "validated_research_input" if passed else "research_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_payload, "quality": result["metrics"],
        "model_candidate_features": list(ROLE_MODEL_FEATURES) if passed else [],
        **paths, "artifact_path": str(output.resolve()),
        "forbidden_interpretation": (
            "Clusters describe observed offensive and rebounding behavior. They are not "
            "positions, player value, causal roles, or evidence that a player would retain "
            "the same impact in a counterfactual role."
        ),
        "caveats": [
            "The fixed role map is trained only on 2014-18 and applied unchanged through 2024.",
            "All inputs are season-relative behavior measures; age, size, listed position, minutes, games, efficiency, on/off, and RAPM are excluded.",
            "The pinned Gabriel role-context source does not declare a license, so derived roles remain research-only.",
            "Hard clusters are descriptive labels. Continuous axes and drop-one affinities are the only eligible AIO candidates.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
