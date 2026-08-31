#!/usr/bin/env python3
"""Build scale and lineup-separation diagnostics for fixed-window RAPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csgraph, load_npz

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "rapm_publication_diagnostics_v1"
CONTRACT = ROOT / "research/experiments/rapm_publication_diagnostics_v1.yml"
ROLLING = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77"
INTERVALS = ROOT / "research/rapm_lab/outputs/production_5y_rapm_intervals/production_5y_rapm_intervals_v1_e86fb09750"
EXTERNAL = ROOT / "research/rapm_lab/outputs/external_reproduction_benchmark/external_reproduction_benchmark_v1_0a95702214"
ANNUAL_TARGETS = ROOT / "artifacts/models/canonical_annual_target_panel/canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
FIVE_YEAR_TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"


def _max_partner(block, exposure: np.ndarray, players: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = block.tocoo()
    keep = matrix.row != matrix.col
    row = matrix.row[keep]
    col = matrix.col[keep]
    shared = matrix.data[keep].astype(float)
    cosine = shared / np.sqrt(np.maximum(exposure[row] * exposure[col], 1.0))
    share = shared / np.maximum(np.minimum(exposure[row], exposure[col]), 1.0)
    order = np.lexsort((players[col], -cosine, row))
    sorted_row = row[order]
    first = np.r_[True, sorted_row[1:] != sorted_row[:-1]]
    chosen = order[first]
    summary = pd.DataFrame(
        {
            "PLAYER_ID": players[row[chosen]],
            "most_linked_teammate_id": players[col[chosen]],
            "maximum_teammate_cosine": cosine[chosen],
            "maximum_teammate_shared_exposure_fraction": share[chosen],
        }
    )
    pairs = pd.DataFrame(
        {
            "player_1": players[row],
            "player_2": players[col],
            "shared_possessions": shared,
            "teammate_cosine": cosine,
            "shared_exposure_fraction": share,
        }
    )
    pairs = pairs.loc[pairs["player_1"].lt(pairs["player_2"])]
    return summary, pairs


def _connectivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals = pd.read_parquet(INTERVALS / "ratings.parquet")
    ratings = pd.read_parquet(ROLLING / "rolling_ratings.parquet")
    names = ratings[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID")
    player_rows = []
    pair_rows = []
    graph_rows = []
    for window_end in range(2018, 2027):
        matrix_dir = ROLLING / "lambda_matrices" / f"5y_end_{window_end}"
        xtx = load_npz(matrix_dir / "train_xtx.npz").tocsr()
        players = np.load(matrix_dir / "player_ids.npy").astype(np.int64)
        off = np.load(matrix_dir / "train_off_possessions.npy").astype(float)
        deff = np.load(matrix_dir / "train_def_possessions.npy").astype(float)
        n = len(players)
        active = (off > 0) & (deff > 0)
        off_summary, off_pairs = _max_partner(xtx[:n, :n], off, players)
        def_summary, def_pairs = _max_partner(xtx[n : 2 * n, n : 2 * n], deff, players)
        off_summary = off_summary.rename(columns={column: f"offense_{column}" for column in off_summary if column != "PLAYER_ID"})
        def_summary = def_summary.rename(columns={column: f"defense_{column}" for column in def_summary if column != "PLAYER_ID"})
        adjacency = (xtx[:n, :n] + xtx[n : 2 * n, n : 2 * n]).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        active_adjacency = adjacency[active][:, active]
        component_count, active_labels = csgraph.connected_components(
            active_adjacency, directed=False
        )
        active_players = players[active]
        active_off_degree = np.diff(xtx[:n, :n].indptr)[active] - 1
        active_def_degree = np.diff(xtx[n : 2 * n, n : 2 * n].indptr)[active] - 1
        component_sizes = pd.Series(active_labels).value_counts()
        metadata = pd.DataFrame(
            {
                "PLAYER_ID": active_players,
                "window_start": window_end - 4,
                "window_end": window_end,
                "lineup_graph_component": active_labels,
                "lineup_graph_component_size": pd.Series(active_labels).map(component_sizes).to_numpy(),
                "offense_distinct_teammates": active_off_degree,
                "defense_distinct_teammates": active_def_degree,
            }
        ).merge(off_summary, on="PLAYER_ID", how="left", validate="one_to_one").merge(def_summary, on="PLAYER_ID", how="left", validate="one_to_one")
        uncertainty = intervals.loc[intervals["window_end"].eq(window_end)]
        metadata = metadata.merge(
            uncertainty[["PLAYER_ID", "Poss_Off", "Poss_Def", "offense_se", "defense_se", "net_se"]],
            on="PLAYER_ID",
            how="left",
            validate="one_to_one",
        )
        metadata["minimum_side_possessions"] = np.minimum(metadata["Poss_Off"], metadata["Poss_Def"])
        metadata["maximum_teammate_cosine"] = metadata[["offense_maximum_teammate_cosine", "defense_maximum_teammate_cosine"]].max(axis=1)
        metadata["lineup_separation_score"] = 1.0 - metadata["maximum_teammate_cosine"]
        metadata["separation_status"] = np.select(
            [metadata["minimum_side_possessions"].lt(1000), metadata["maximum_teammate_cosine"].ge(0.80)],
            ["low_exposure", "high_teammate_collinearity"],
            default="standard",
        )
        player_rows.append(metadata)
        for side, pairs in (("offense", off_pairs), ("defense", def_pairs)):
            pairs["side"] = side
            pairs["window_end"] = window_end
            pair_rows.append(pairs.nlargest(100, "teammate_cosine"))
        graph_rows.append(
            {
                "window_end": window_end,
                "players": int(active.sum()),
                "evaluation_only_players_excluded": int((~active).sum()),
                "connected_components": int(component_count),
                "largest_component_players": int(component_sizes.max()),
                "isolated_players": int((component_sizes == 1).sum()),
            }
        )
    players = pd.concat(player_rows, ignore_index=True).merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    pairs = pd.concat(pair_rows, ignore_index=True)
    return players, pairs, pd.DataFrame(graph_rows)


def _weighted_metrics(frame: pd.DataFrame, scope: str, component: str, exposure_group: str) -> dict:
    reference = frame["reference"].to_numpy(dtype=float)
    ours = frame["courtsignal"].to_numpy(dtype=float)
    weight = np.sqrt(
        np.clip(frame["minimum_side_possessions"].to_numpy(dtype=float), 1, None)
    )
    mean_x = float(np.average(reference, weights=weight))
    mean_y = float(np.average(ours, weights=weight))
    weighted_slope = float(np.sum(weight * (reference - mean_x) * (ours - mean_y)) / np.sum(weight * (reference - mean_x) ** 2))
    return {
        "scope": scope,
        "component": component,
        "exposure_group": exposure_group,
        "rows": len(frame),
        "pearson": float(np.corrcoef(reference, ours)[0, 1]),
        "spearman": float(pd.Series(reference).corr(pd.Series(ours), method="spearman")),
        "unweighted_slope": float(np.polyfit(reference, ours, 1)[0]),
        "exposure_weighted_slope": weighted_slope,
        "court_to_reference_sd_ratio": float(np.std(ours, ddof=1) / np.std(reference, ddof=1)),
        "court_mean": float(np.mean(ours)),
        "reference_mean": float(np.mean(reference)),
    }


def _scale_audit() -> pd.DataFrame:
    matched = pd.read_parquet(EXTERNAL / "matched_rows.parquet")
    annual = matched.loc[
        matched["source"].eq("Ryan Davis annual RAPM")
        & matched["comparison"].eq("RAPM")
        & matched["scope"].astype(str).str.fullmatch(r"\d{4}")
    ].copy()
    annual["Season"] = annual["scope"].astype(int)
    exposure = pd.read_parquet(ANNUAL_TARGETS)[["PLAYER_ID", "Season", "Poss_Off", "Poss_Def"]]
    annual = annual.merge(exposure, on=["PLAYER_ID", "Season"], how="inner", validate="many_to_one")
    annual["minimum_side_possessions"] = np.minimum(annual["Poss_Off"], annual["Poss_Def"])
    five = matched.loc[
        matched["source"].eq("Ryan Davis multi-year RAPM")
        & matched["comparison"].eq("5-year RAPM")
        & matched["scope"].astype(str).str.fullmatch(r"\d{4}-\d{4}")
    ].copy()
    five["Window_End"] = five["scope"].str[-4:].astype(int)
    five_exposure = pd.read_parquet(FIVE_YEAR_TARGETS)[["PLAYER_ID", "Window_End", "Poss_Off", "Poss_Def"]]
    five = five.merge(five_exposure, on=["PLAYER_ID", "Window_End"], how="inner", validate="many_to_one")
    five["minimum_side_possessions"] = np.minimum(five["Poss_Off"], five["Poss_Def"])
    rows = []
    for label, frame in (("annual_2014_2023", annual), ("five_year_exact_windows", five)):
        for component, group in frame.groupby("component"):
            rows.append(_weighted_metrics(group, label, component, "all"))
            ranked = group.copy()
            ranked["exposure_group"] = pd.qcut(ranked["minimum_side_possessions"], 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
            for exposure_group, subgroup in ranked.groupby("exposure_group", observed=True):
                rows.append(_weighted_metrics(subgroup, label, component, str(exposure_group)))
    for season, frame in annual.groupby("Season"):
        for component, group in frame.groupby("component"):
            rows.append(_weighted_metrics(group, f"annual_{season}", component, "all"))
    for scope, frame in five.groupby("scope"):
        for component, group in frame.groupby("component"):
            rows.append(_weighted_metrics(group, f"five_year_{scope}", component, "all"))
    return pd.DataFrame(rows)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["inputs"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    connectivity, pairs, graph = _connectivity()
    scale = _scale_audit()
    sources = {
        "contract": CONTRACT,
        "rolling_manifest": ROLLING / "run.json",
        "interval_manifest": INTERVALS / "run.json",
        "external_manifest": EXTERNAL / "run.json",
        "annual_targets": ANNUAL_TARGETS,
        "five_year_targets": FIVE_YEAR_TARGETS,
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/rapm_publication_diagnostics" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "player_connectivity.parquet": connectivity,
        "strongest_teammate_pairs.parquet": pairs,
        "graph_summary.parquet": graph,
        "external_scale_audit.parquet": scale,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    pooled = scale.loc[scale["exposure_group"].eq("all") & scale["scope"].isin(["annual_2014_2023", "five_year_exact_windows"])]
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_publication_diagnostics_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "player_window_rows": len(connectivity),
            "missing_names": int(connectivity["PLAYER_NAME"].isna().sum()),
            "maximum_graph_components": int(graph["connected_components"].max()),
            "season_2027_loaded": False,
        },
        "results": {
            "separation_status_counts": connectivity["separation_status"].value_counts().to_dict(),
            "pooled_external_scale": pooled.to_dict(orient="records"),
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(graph.to_string(index=False))
    print(connectivity["separation_status"].value_counts().to_string())
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
