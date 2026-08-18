"""Separate offense and defense deployment roles.

The role maps describe player usage. They do not estimate player value. The
offense map uses offensive actions and playtype shares. The defense map uses
defensive assignment and activity volume, including the offensive-role mix of
the players guarded.
"""

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
from sklearn.metrics import adjusted_rand_score, silhouette_score

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.matchup_defense_features import _read_archive


OFFENSE_BASE_INPUTS = (
    "FG2A_p100", "FG3A_p100", "FTA_p100", "OREB_p100", "drives_p100",
    "touches_p100", "front_court_touches_p100", "paint_touches_p100",
    "post_touches_p100", "elbow_touches_p100", "time_of_possession_p100",
    "passes_made_p100", "potential_assists_p100", "avg_seconds_per_touch",
    "avg_dribbles_per_touch", "self_created_point_share",
    "assisted_three_share", "pull_up_attempt_share", "drive_pass_rate",
    "creation_load_p100", "interior_role_load", "at_rim_frequency",
    "short_mid_frequency", "long_mid_frequency", "corner3_frequency",
    "arc3_frequency",
)
OFFENSE_DRIBBLE_INPUTS = (
    "all_shot_zero_dribble_share", "all_shot_three_plus_dribble_share",
    "jump_shot_zero_dribble_share", "jump_shot_three_plus_dribble_share",
)
PLAYTYPE_NAMES = (
    "cut", "hand_off", "iso", "misc", "off_screen", "post", "pr_ball",
    "pr_roll", "putback", "spot", "transition",
)
OFFENSE_PLAYTYPE_INPUTS = tuple(f"playtype_share_{name}" for name in PLAYTYPE_NAMES)
OFFENSE_ROLE_INPUTS = (
    *OFFENSE_BASE_INPUTS, *OFFENSE_DRIBBLE_INPUTS, *OFFENSE_PLAYTYPE_INPUTS,
)

DEFENSE_ANNUAL_INPUTS = ("rebound_contest_share", "dreb_contested_share")
DEFENSE_TRACKING_ROLE_INPUTS = (
    "dfg_attempts_p100", "rim_dfga_p100", "contested_2pt_p100",
    "contested_3pt_p100",
)
DEFENSE_MATCHUP_ROLE_INPUTS = (
    "def_matchup_switches_p100", "def_matchup_fga_p100",
    "def_matchup_3pa_share", "def_matchup_potential_assists_p100",
    "def_matchup_help_fga_p100",
)


@dataclass(frozen=True)
class RoleMapConfig:
    prefix: str
    development_seasons: tuple[int, ...]
    n_axes: int
    candidate_clusters: tuple[int, ...]
    minimum_descriptor_coverage: float = 0.80
    seed: int = 20260817
    stability_seeds: tuple[int, ...] = (11, 29, 47, 71, 101)
    minimum_seed_ari: float = 0.90
    minimum_cluster_share: float = 0.015
    maximum_cluster_share: float = 0.36


OFFENSE_CONFIG = RoleMapConfig(
    prefix="off_role", development_seasons=(2014, 2015, 2016, 2017, 2018),
    n_axes=8, candidate_clusters=(6, 8, 10, 12),
)
DEFENSE_CONFIG = RoleMapConfig(
    prefix="def_role", development_seasons=(2018, 2019, 2020, 2021),
    n_axes=6, candidate_clusters=(5, 6, 7, 8, 9),
)


def _season_scale(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    for _, rows in frame.groupby("Season").groups.items():
        values = frame.loc[rows, features]
        median = values.median()
        filled = values.fillna(median).fillna(0.0)
        iqr = filled.quantile(0.75) - filled.quantile(0.25)
        fallback = filled.std(ddof=0).replace(0.0, np.nan)
        scale = iqr.where(iqr.gt(1e-9), fallback).fillna(1.0)
        output.loc[rows, features] = (
            (filled - median.fillna(0.0)) / scale
        ).clip(-5.0, 5.0).to_numpy()
    return output


def _fit_pca(matrix: np.ndarray, n_axes: int) -> PCA:
    pca = PCA(n_components=n_axes, svd_solver="full").fit(matrix)
    for row in range(len(pca.components_)):
        pivot = int(np.argmax(np.abs(pca.components_[row])))
        if pca.components_[row, pivot] < 0:
            pca.components_[row] *= -1.0
    return pca


def _ordered_centers(model: KMeans) -> tuple[dict[int, int], np.ndarray]:
    order = sorted(
        range(len(model.cluster_centers_)),
        key=lambda row: tuple(np.round(model.cluster_centers_[row], 12)),
    )
    return {old: new for new, old in enumerate(order)}, model.cluster_centers_[order]


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1), denominator,
        out=np.zeros(len(left), dtype=float), where=denominator > 0,
    )


def _playtype_shares(path: str | Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).rename(columns={"year": "Season"})
    required = {"PLAYER_ID", "Season", "playtype", "Poss"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Playtype role source is missing {missing}.")
    for column in ("PLAYER_ID", "Season", "Poss"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    aliases = {"tran": "transition", "oreb": "putback", "bh": "pr_ball"}
    frame["playtype"] = (
        frame["playtype"].astype(str).str.strip().str.lower().replace(aliases)
    )
    frame = frame.loc[frame["Season"].isin(seasons)].dropna(
        subset=["PLAYER_ID", "Season", "Poss"]
    )
    frame = frame.loc[frame["playtype"].isin(PLAYTYPE_NAMES)]
    pivot = frame.pivot_table(
        index=["PLAYER_ID", "Season"], columns="playtype", values="Poss",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=PLAYTYPE_NAMES, fill_value=0.0)
    totals = pivot.sum(axis=1)
    pivot = pivot.div(totals.where(totals.gt(0)), axis=0).fillna(0.0)
    pivot.columns = OFFENSE_PLAYTYPE_INPUTS
    output = pivot.reset_index()
    output["PLAYER_ID"] = output["PLAYER_ID"].astype(int)
    output["Season"] = output["Season"].astype(int)
    return output


def _fit_role_map(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    config: RoleMapConfig,
) -> dict[str, object]:
    keys = ["PLAYER_ID", "Season"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{config.prefix} input contains duplicate player-seasons.")
    work = frame[keys + list(features)].copy()
    for feature in features:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
    coverage_name = f"{config.prefix}_descriptor_coverage"
    work[coverage_name] = work[list(features)].notna().mean(axis=1)
    work = work.loc[
        work[coverage_name].ge(config.minimum_descriptor_coverage)
    ].copy()
    scaled = _season_scale(work, features)
    development = scaled.loc[
        scaled["Season"].isin(config.development_seasons)
    ].copy()
    if development["Season"].nunique() != len(config.development_seasons):
        raise ValueError(f"{config.prefix} development seasons are incomplete.")
    matrix = development.loc[:, features].to_numpy(dtype=float)
    full_matrix = scaled.loc[:, features].to_numpy(dtype=float)
    pca = _fit_pca(matrix, config.n_axes)
    development_axes = pca.transform(matrix)
    full_axes = pca.transform(full_matrix)
    out_of_sample = ~scaled["Season"].isin(config.development_seasons).to_numpy()

    candidates = []
    candidate_models: dict[int, KMeans] = {}
    for count in config.candidate_clusters:
        model = KMeans(
            n_clusters=count, n_init=50, random_state=config.seed,
            algorithm="lloyd",
        ).fit(development_axes)
        candidate_models[count] = model
        reference = model.predict(full_axes[out_of_sample])
        seed_ari = []
        for seed in config.stability_seeds:
            alternate = KMeans(
                n_clusters=count, n_init=10, random_state=seed,
                algorithm="lloyd",
            ).fit(development_axes)
            seed_ari.append(
                adjusted_rand_score(reference, alternate.predict(full_axes[out_of_sample]))
            )
        shares = pd.Series(reference).value_counts(normalize=True)
        candidates.append(
            {
                "clusters": count,
                "silhouette": float(silhouette_score(development_axes, model.labels_)),
                "median_seed_adjusted_rand": float(np.median(seed_ari)),
                "minimum_out_of_sample_share": float(shares.min()),
                "maximum_out_of_sample_share": float(shares.max()),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame.loc[
        candidate_frame["median_seed_adjusted_rand"].ge(config.minimum_seed_ari)
        & candidate_frame["minimum_out_of_sample_share"].ge(config.minimum_cluster_share)
        & candidate_frame["maximum_out_of_sample_share"].le(config.maximum_cluster_share)
    ]
    if eligible.empty:
        raise ValueError(f"No stable {config.prefix} cluster count passed the frozen gates.")
    selected_count = int(
        eligible.sort_values(
            ["silhouette", "clusters"], ascending=[False, True], kind="stable"
        ).iloc[0]["clusters"]
    )
    model = candidate_models[selected_count]
    mapping, centers = _ordered_centers(model)
    raw_labels = model.predict(full_axes)
    labels = np.array([mapping[int(label)] for label in raw_labels], dtype=int)
    distances = np.linalg.norm(full_axes[:, None, :] - centers[None, :, :], axis=2)
    development_mask = scaled["Season"].isin(config.development_seasons).to_numpy()
    temperature = max(
        float(np.median(np.min(distances[development_mask], axis=1) ** 2)), 1e-6
    )
    logits = -(distances**2) / (2.0 * temperature)
    logits -= logits.max(axis=1, keepdims=True)
    affinities = np.exp(logits)
    affinities /= affinities.sum(axis=1, keepdims=True)

    assignments = scaled[keys + [coverage_name]].copy()
    assignments[f"{config.prefix}_cluster"] = [
        f"{config.prefix}_{label}" for label in labels
    ]
    assignments[f"{config.prefix}_distance"] = distances[
        np.arange(len(labels)), labels
    ]
    assignments[f"{config.prefix}_confidence"] = affinities.max(axis=1)
    axis_features = tuple(
        f"{config.prefix}_axis_{index}" for index in range(1, config.n_axes + 1)
    )
    affinity_features = tuple(
        f"{config.prefix}_affinity_{index}" for index in range(selected_count)
    )
    for index, feature in enumerate(axis_features):
        assignments[feature] = full_axes[:, index]
    for index, feature in enumerate(affinity_features):
        assignments[feature] = affinities[:, index]

    previous = assignments.copy()
    previous["Season"] += 1
    previous = previous.rename(
        columns={
            f"{config.prefix}_cluster": "previous_cluster",
            **{feature: f"previous_{feature}" for feature in axis_features},
        }
    )
    adjacent = assignments.merge(
        previous[[*keys, "previous_cluster", *(f"previous_{x}" for x in axis_features)]],
        on=keys, how="inner", validate="one_to_one",
    )
    adjacent["same_role"] = adjacent[f"{config.prefix}_cluster"].eq(
        adjacent["previous_cluster"]
    )
    adjacent["axis_cosine"] = _cosine(
        adjacent[list(axis_features)].to_numpy(dtype=float),
        adjacent[[f"previous_{x}" for x in axis_features]].to_numpy(dtype=float),
    )
    adjacent = adjacent.loc[
        ~adjacent["Season"].isin(config.development_seasons)
    ].copy()
    selected = candidate_frame.loc[
        candidate_frame["clusters"].eq(selected_count)
    ].iloc[0]
    metrics = {
        "rows": int(len(frame)),
        "eligible_rows": int(len(assignments)),
        "coverage": float(len(assignments) / len(frame)),
        "development_rows": int(len(development)),
        "out_of_sample_rows": int(out_of_sample.sum()),
        "input_features": len(features),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selected_clusters": selected_count,
        "silhouette": float(selected["silhouette"]),
        "median_seed_adjusted_rand": float(selected["median_seed_adjusted_rand"]),
        "adjacent_pairs": int(len(adjacent)),
        "adjacent_same_role_rate": float(adjacent["same_role"].mean()),
        "median_adjacent_axis_cosine": float(adjacent["axis_cosine"].median()),
        "minimum_out_of_sample_cluster_share": float(
            selected["minimum_out_of_sample_share"]
        ),
        "maximum_out_of_sample_cluster_share": float(
            selected["maximum_out_of_sample_share"]
        ),
    }
    return {
        "assignments": assignments,
        "cluster_candidates": candidate_frame,
        "axis_loadings": pd.DataFrame(
            pca.components_, index=axis_features, columns=features
        ).reset_index(names="role_axis"),
        "centroid_profiles": pd.DataFrame(
            pca.inverse_transform(centers), columns=features
        ).assign(**{f"{config.prefix}_cluster": [
            f"{config.prefix}_{index}" for index in range(selected_count)
        ]}),
        "adjacent_stability": adjacent,
        "metrics": metrics,
        "model_features": (*axis_features, *affinity_features[:-1]),
    }


def _defensive_matchup_context(
    archive_root: str | Path,
    offense_assignments: pd.DataFrame,
    seasons: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, float]]:
    def load_source(season: int) -> tuple[pd.DataFrame, Path]:
        """Load one canonical regular-season matchup source for a project season.

        Older seasons retain the pinned archive source.  The current NBA Stats
        parquet exports have the same per-matchup fields, so 2025--26 can use
        them without changing the role definition or silently dropping the
        opponent-role-assignment block.
        """
        root = Path(archive_root)
        candidates = (
            root / "shufinskiy_nba_data" / "revision=e829d46" / "matchups" / f"season={season}",
            root / "nba_data_archive" / "matchups" / f"season={season}" / "regular.parquet",
            root / "official_nba_stats_v3" / "matchups" / f"season={season}" / "regular.parquet",
            root / f"season={season}",
        )
        for candidate in candidates:
            if candidate.is_dir():
                archives = sorted(candidate.glob("*.tar.xz"))
                if len(archives) == 1:
                    frame, _ = _read_archive(archives[0])
                    return frame, archives[0]
            elif candidate.exists():
                return pd.read_parquet(candidate), candidate
        raise ValueError(f"No canonical matchup source found for season {season}.")

    affinity_columns = tuple(
        column for column in offense_assignments
        if column.startswith("off_role_affinity_")
    )
    outputs = []
    hashes = {}
    role_coverage = {}
    for season in seasons:
        frame, source_path = load_source(season)
        columns = [
            "person_id", "matchups_person_id", "partial_possessions", "switches_on",
            "matchup_field_goals_attempted", "matchup_three_pointers_attempted",
            "matchup_potential_assists", "matchup_blocks", "help_blocks",
            "help_field_goals_attempted", "help_field_goals_made",
        ]
        work = frame[columns].copy()
        for column in columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(
            subset=["person_id", "matchups_person_id", "partial_possessions"]
        )
        work = work.loc[work["partial_possessions"].gt(0)].copy()
        season_roles = offense_assignments.loc[
            offense_assignments["Season"].eq(season), ["PLAYER_ID", *affinity_columns]
        ]
        work = work.merge(
            season_roles, left_on="person_id", right_on="PLAYER_ID", how="left",
            validate="many_to_one",
        )
        role_covered = work[affinity_columns[0]].notna()
        means = season_roles[list(affinity_columns)].mean()
        for column in affinity_columns:
            work[column] = work[column].fillna(means[column]) * work["partial_possessions"]
        aggregations = {
            "matchup_possessions": ("partial_possessions", "sum"),
            "switches": ("switches_on", "sum"),
            "fga": ("matchup_field_goals_attempted", "sum"),
            "three_pa": ("matchup_three_pointers_attempted", "sum"),
            "potential_assists": ("matchup_potential_assists", "sum"),
            "blocks": ("matchup_blocks", "sum"),
            "help_blocks": ("help_blocks", "sum"),
            "help_fga": ("help_field_goals_attempted", "sum"),
            "help_fgm": ("help_field_goals_made", "sum"),
            **{column: (column, "sum") for column in affinity_columns},
        }
        defender = work.groupby("matchups_person_id", as_index=False).agg(**aggregations)
        defender = defender.rename(columns={"matchups_person_id": "PLAYER_ID"})
        possessions = defender["matchup_possessions"]
        defender["Season"] = season
        rate_sources = {
            "switches": "def_matchup_switches_p100",
            "fga": "def_matchup_fga_p100",
            "potential_assists": "def_matchup_potential_assists_p100",
            "blocks": "def_matchup_blocks_p100",
            "help_blocks": "def_matchup_help_blocks_p100",
            "help_fga": "def_matchup_help_fga_p100",
        }
        for source, destination in rate_sources.items():
            defender[destination] = 100.0 * defender[source] / possessions
        defender["def_matchup_3pa_share"] = (
            defender["three_pa"] / defender["fga"].where(defender["fga"].gt(0))
        ).fillna(0.0)
        attempts = defender["help_fga"]
        league_help_pct = (
            float(defender["help_fgm"].sum() / attempts.sum())
            if attempts.sum() > 0 else 0.0
        )
        defender["def_matchup_help_fg_pct_allowed_eb"] = (
            defender["help_fgm"] + 100.0 * league_help_pct
        ) / (attempts + 100.0)
        for index, column in enumerate(affinity_columns):
            defender[f"def_opponent_off_role_affinity_{index}"] = (
                defender[column] / possessions
            )
        keep = [
            "PLAYER_ID", "Season", "matchup_possessions",
            *DEFENSE_MATCHUP_ROLE_INPUTS,
            "def_matchup_blocks_p100", "def_matchup_help_blocks_p100",
            "def_matchup_help_fg_pct_allowed_eb",
            *(f"def_opponent_off_role_affinity_{i}" for i in range(len(affinity_columns))),
        ]
        outputs.append(defender[keep])
        hashes[str(source_path.resolve())] = sha256_file(source_path)
        role_coverage[str(season)] = float(
            work.loc[role_covered, "partial_possessions"].sum()
            / work["partial_possessions"].sum()
        )
    return pd.concat(outputs, ignore_index=True), hashes, role_coverage


def build_side_roles(
    annual_features_path: str | Path,
    dribble_context_path: str | Path,
    playtype_source_path: str | Path,
    defensive_tracking_path: str | Path,
    matchup_archive_root: str | Path,
    *,
    artifact_root: str | Path,
    offense_seasons: tuple[int, ...] = tuple(range(2014, 2025)),
    defense_seasons: tuple[int, ...] = tuple(range(2018, 2025)),
) -> dict:
    """Build chronologically frozen, side-specific role maps."""
    annual = pd.read_parquet(annual_features_path).rename(columns={"Window_End": "Season"})
    dribble = pd.read_parquet(dribble_context_path)
    playtypes = _playtype_shares(playtype_source_path, offense_seasons)
    offense = annual.loc[
        annual["Season"].isin(offense_seasons), ["PLAYER_ID", "Season", *OFFENSE_BASE_INPUTS]
    ].merge(
        dribble[["PLAYER_ID", "Season", *OFFENSE_DRIBBLE_INPUTS]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    ).merge(
        playtypes, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    )
    offense_result = _fit_role_map(offense, OFFENSE_ROLE_INPUTS, OFFENSE_CONFIG)

    matchup_context, matchup_hashes, matchup_role_coverage = _defensive_matchup_context(
        matchup_archive_root, offense_result["assignments"], defense_seasons
    )
    tracking = pd.read_parquet(defensive_tracking_path)
    opponent_inputs = tuple(
        column for column in matchup_context
        if column.startswith("def_opponent_off_role_affinity_")
    )
    defense_inputs = (
        *DEFENSE_ANNUAL_INPUTS, *DEFENSE_TRACKING_ROLE_INPUTS,
        *DEFENSE_MATCHUP_ROLE_INPUTS, *opponent_inputs,
    )
    defense = annual.loc[
        annual["Season"].isin(defense_seasons),
        ["PLAYER_ID", "Season", *DEFENSE_ANNUAL_INPUTS],
    ].merge(
        tracking[["PLAYER_ID", "Season", *DEFENSE_TRACKING_ROLE_INPUTS]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    ).merge(
        matchup_context[["PLAYER_ID", "Season", *DEFENSE_MATCHUP_ROLE_INPUTS, *opponent_inputs]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    )
    defense_result = _fit_role_map(defense, defense_inputs, DEFENSE_CONFIG)

    config = {
        "offense_seasons": list(offense_seasons),
        "defense_seasons": list(defense_seasons),
        "offense_config": OFFENSE_CONFIG.__dict__,
        "defense_config": DEFENSE_CONFIG.__dict__,
        "offense_inputs": list(OFFENSE_ROLE_INPUTS),
        "defense_inputs": list(defense_inputs),
        "source_hashes": {
            "annual_features": sha256_file(annual_features_path),
            "dribble_context": sha256_file(dribble_context_path),
            "playtype": sha256_file(playtype_source_path),
            "defensive_tracking": sha256_file(defensive_tracking_path),
            "matchup_archives": matchup_hashes,
            "builder": sha256_file(Path(__file__)),
        },
        "matchup_offense_role_coverage_by_season": matchup_role_coverage,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"side_roles_v1_{identity}"
    output = Path(artifact_root) / "features" / "side_roles" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for side, result in (("offense", offense_result), ("defense", defense_result)):
        result["assignments"].to_parquet(output / f"{side}_assignments.parquet", index=False)
        result["cluster_candidates"].to_parquet(
            output / f"{side}_cluster_candidates.parquet", index=False
        )
        result["axis_loadings"].to_parquet(output / f"{side}_axis_loadings.parquet", index=False)
        result["centroid_profiles"].to_parquet(
            output / f"{side}_centroid_profiles.parquet", index=False
        )
        result["adjacent_stability"].to_parquet(
            output / f"{side}_adjacent_stability.parquet", index=False
        )
    matchup_context.to_parquet(output / "defensive_matchup_context.parquet", index=False)
    run = {
        "run_id": run_id,
        "dataset": "annual_side_specific_behavior_roles_v1",
        "status": "validated_research_input",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": {
            "offense": offense_result["metrics"],
            "defense": defense_result["metrics"],
        },
        "model_features": {
            "offense": list(offense_result["model_features"]),
            "defense": list(defense_result["model_features"]),
        },
        "paths": {
            "offense_assignments": str((output / "offense_assignments.parquet").resolve()),
            "defense_assignments": str((output / "defense_assignments.parquet").resolve()),
            "defensive_matchup_context": str((output / "defensive_matchup_context.parquet").resolve()),
        },
        "artifact_path": str(output.resolve()),
        "interpretation": (
            "Roles describe observed deployment. Cluster labels are anonymous and are not "
            "BBall Index grades, positions, player value, or role-fit counterfactuals."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run
