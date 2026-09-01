#!/usr/bin/env python3
"""Benchmark internal and public all-in-one ratings on next-season games."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.external_impact_benchmark import normalize_player_name
from nba_impact.models.impact_validation_suite import paired_game_mse_intervals
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit as fit_box
from research.run_aio_prior_complementarity import (
    BOX_ANNUAL,
    CHECKPOINT_ROOT,
    TARGET_WINDOWS,
    _load_matrix,
    _standalone_prediction,
    _uniform_basis,
    canonical_frame_hash,
    coefficient_center,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "external_all_in_one_benchmark_v2"
INTERNAL_RUN = ROOT / "artifacts/research/aio_prior_complementarity/aio_prior_complementarity_v1_4d83e381af"
EXTERNAL_ANNUAL = ROOT / "artifacts/models/external_impact_benchmark/external_impact_benchmark_v1_bab43a4087/external_annual.parquet"
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
INTERNAL_CANDIDATES = {
    "current_control__box15": "Box15",
    "current_control__rich": "Rich elastic SPM",
    "target_excluded__def_residual_outcome": "Defense residual challenger",
}
INTERNAL_AIO_CANDIDATES = {
    "Box15": "Box15 + RAPM",
    "Box15 (2014+)": "Box15 (2014+) + RAPM",
    "Rich elastic SPM": "Rich elastic SPM + RAPM",
    "Defense residual challenger": "Defense residual + RAPM",
}
STRICT_CANDIDATES = (
    "Box15",
    "Box15 (2014+)",
    "Rich elastic SPM",
    "Defense residual challenger",
    "EPM",
    "LEBRON",
    "MAMBA",
    "DARKO DPM",
    "PIPM",
    "RAPTOR",
    "BPM 2.0",
    "xRAPM",
)

SPREADSHEET_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATIONSHIP_NAMESPACE = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_RELATIONSHIP_NAMESPACE = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one value-only XLSX sheet without adding an Excel dependency."""
    with ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(
            item
            for item in workbook.find(f"{SPREADSHEET_NAMESPACE}sheets")
            if item.attrib["name"] == sheet_name
        )
        relationship_id = sheet.attrib[f"{RELATIONSHIP_NAMESPACE}id"]
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            item.attrib["Target"]
            for item in relationships
            if item.attrib["Id"] == relationship_id
        )
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{SPREADSHEET_NAMESPACE}t"))
                for item in strings.findall(f"{SPREADSHEET_NAMESPACE}si")
            ]
        worksheet = ElementTree.fromstring(archive.read(target))

    rows = []
    for row in worksheet.iter(f"{SPREADSHEET_NAMESPACE}row"):
        values = {}
        for cell in row.findall(f"{SPREADSHEET_NAMESPACE}c"):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
            value_node = cell.find(f"{SPREADSHEET_NAMESPACE}v")
            value = None if value_node is None else value_node.text
            if value is not None and cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
            values[column] = value
        rows.append(values)
    if not rows:
        raise ValueError(f"{path.name}:{sheet_name} is empty.")
    header = rows[0]
    columns = tuple(header)
    return pd.DataFrame(
        [{header[column]: row.get(column) for column in columns} for row in rows[1:]]
    )


def fit_box15_2014_onward() -> pd.DataFrame:
    """Fit the frozen Box15 learner using no rating season before 2014."""
    features = pd.read_parquet(BOX_ANNUAL).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(TARGET_WINDOWS)
    targets = targets.loc[
        targets["horizon"].eq(9) & targets["target_variant"].eq("normal")
    ].rename(columns={"Window_End": "Season"})
    panel = features.merge(targets, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    rows = []
    for rating_season in range(2017, 2025):
        train = panel.loc[panel["Season"].between(2014, rating_season - 1)]
        test = panel.loc[panel["Season"].eq(rating_season)]
        if train.empty or test.empty:
            raise ValueError(f"Box15 2014+ fold {rating_season} has an empty partition.")
        output = test[["PLAYER_ID"]].copy()
        output["rating_season"] = rating_season
        output["candidate"] = "Box15 (2014+)"
        for side, alpha in (("offense", 300.0), ("defense", 1_000.0)):
            model = fit_box(train, BOX_PIPM_STYLE_FEATURES, f"target_{side}", alpha)
            output[side] = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
        output["net"] = output["offense"] + output["defense"]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def season_end(value: object) -> int:
    """Return the end-year convention used by the repository."""
    text = str(value).strip()
    if "-" not in text:
        return int(float(text))
    first, second = text.split("-", maxsplit=1)
    start = int(first)
    if len(second) == 2:
        century = start // 100 * 100
        end = century + int(second)
        if end < start:
            end += 100
        return end
    return int(second)


def component_frame(
    frame: pd.DataFrame,
    *,
    candidate: str,
    id_column: str,
    season_column: str,
    offense_column: str,
    defense_column: str,
) -> pd.DataFrame:
    output = frame[[id_column, season_column, offense_column, defense_column]].copy()
    output.columns = ["PLAYER_ID", "rating_season", "offense", "defense"]
    output["PLAYER_ID"] = pd.to_numeric(output["PLAYER_ID"], errors="coerce")
    output["rating_season"] = output["rating_season"].map(season_end)
    output["offense"] = pd.to_numeric(output["offense"], errors="coerce")
    output["defense"] = pd.to_numeric(output["defense"], errors="coerce")
    output = output.dropna().astype({"PLAYER_ID": int, "rating_season": int})
    output["candidate"] = candidate
    output["net"] = output["offense"] + output["defense"]
    return output.drop_duplicates(["candidate", "rating_season", "PLAYER_ID"], keep="last")


def name_dimension(epm: pd.DataFrame, lebron: pd.DataFrame) -> pd.DataFrame:
    epm_names = epm[["EPM_player_id", "EPM_player_name", "EPM_season"]].rename(
        columns={"EPM_player_id": "PLAYER_ID", "EPM_player_name": "player_name", "EPM_season": "rating_season"}
    )
    lebron_names = lebron[["nba_id", "Player", "Season"]].rename(
        columns={"nba_id": "PLAYER_ID", "Player": "player_name", "Season": "rating_season"}
    )
    names = pd.concat([epm_names, lebron_names], ignore_index=True)
    names["PLAYER_ID"] = pd.to_numeric(names["PLAYER_ID"], errors="coerce")
    names["rating_season"] = names["rating_season"].map(season_end)
    names["normalized_name"] = names["player_name"].map(normalize_player_name)
    names = names.dropna(subset=["PLAYER_ID", "normalized_name"]).astype({"PLAYER_ID": int})
    conflicts = names.groupby(["rating_season", "normalized_name"])["PLAYER_ID"].nunique()
    conflicts = set(conflicts.loc[conflicts.gt(1)].index)
    keep = [
        (season, name) not in conflicts
        for season, name in names[["rating_season", "normalized_name"]].itertuples(index=False, name=None)
    ]
    return names.loc[keep, ["rating_season", "normalized_name", "PLAYER_ID"]].drop_duplicates()


def named_frame(
    frame: pd.DataFrame,
    names: pd.DataFrame,
    *,
    candidate: str,
    name_column: str,
    season_column: str,
    offense_column: str,
    defense_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = frame[[name_column, season_column, offense_column, defense_column]].copy()
    source["rating_season"] = source[season_column].map(season_end)
    source["normalized_name"] = source[name_column].map(normalize_player_name)
    source["offense"] = pd.to_numeric(source[offense_column], errors="coerce")
    source["defense"] = pd.to_numeric(source[defense_column], errors="coerce")
    source = source.dropna(subset=["normalized_name", "offense", "defense"])
    source = source.drop_duplicates(["rating_season", "normalized_name"], keep="last")
    matched = source.merge(names, on=["rating_season", "normalized_name"], how="left", validate="one_to_one")
    coverage = matched.groupby("rating_season", as_index=False).agg(
        source_players=("normalized_name", "size"),
        matched_players=("PLAYER_ID", lambda x: int(x.notna().sum())),
    )
    coverage["candidate"] = candidate
    coverage["identity_match_rate"] = coverage["matched_players"] / coverage["source_players"]
    output = matched.dropna(subset=["PLAYER_ID"]).astype({"PLAYER_ID": int})
    output["candidate"] = candidate
    output["net"] = output["offense"] + output["defense"]
    return output[["PLAYER_ID", "rating_season", "candidate", "offense", "defense", "net"]], coverage


def load_panels(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    sources = {
        "internal_priors": INTERNAL_RUN / "priors.parquet",
        "internal_training_scope": INTERNAL_RUN / "base_prior_selections.parquet",
        "box15_features": BOX_ANNUAL,
        "nine_year_targets": TARGET_WINDOWS,
        "darko_history": args.darko_history,
        "epm": args.epm,
        "lebron": args.lebron,
        "mamba": args.mamba,
        "pipm": args.pipm,
        "raptor": args.raptor,
        "bpm_xrapm": EXTERNAL_ANNUAL,
    }
    epm = pd.read_csv(args.epm, low_memory=False)
    lebron = pd.read_csv(args.lebron, low_memory=False)
    names = name_dimension(epm, lebron)
    panels = [
        component_frame(
            epm,
            candidate="EPM",
            id_column="EPM_player_id",
            season_column="EPM_season",
            offense_column="EPM_off",
            defense_column="EPM_def",
        ),
        component_frame(
            lebron,
            candidate="LEBRON",
            id_column="nba_id",
            season_column="Season",
            offense_column="O-LEBRON",
            defense_column="D-LEBRON",
        ),
    ]
    darko_history = read_xlsx_sheet(args.darko_history, "Full DPM History")
    panels.append(
        component_frame(
            darko_history,
            candidate="DARKO DPM",
            id_column="nba_id",
            season_column="season",
            offense_column="o_dpm",
            defense_column="d_dpm",
        )
    )
    coverage_rows = []

    mamba, coverage = named_frame(
        pd.read_csv(args.mamba), names,
        candidate="MAMBA", name_column="Player", season_column="Season",
        offense_column="Offense", defense_column="Defense",
    )
    panels.append(mamba)
    coverage_rows.append(coverage)

    pipm, coverage = named_frame(
        pd.read_csv(args.pipm, low_memory=False), names,
        candidate="PIPM", name_column="Player", season_column="Season",
        offense_column="O-PIPM", defense_column="D-PIPM",
    )
    panels.append(pipm)
    coverage_rows.append(coverage)

    raptor, coverage = named_frame(
        pd.read_csv(args.raptor, low_memory=False), names,
        candidate="RAPTOR", name_column="player_name", season_column="season",
        offense_column="raptor_offense", defense_column="raptor_defense",
    )
    panels.append(raptor)
    coverage_rows.append(coverage)

    external = pd.read_parquet(EXTERNAL_ANNUAL)
    for candidate, name_column, offense_column, defense_column in (
        ("BPM 2.0", "player_name_bpm", "bpm_offense", "bpm_defense"),
        ("xRAPM", "player_name_xrapm", "xrapm_offense", "xrapm_defense"),
    ):
        subset = external.dropna(subset=[name_column, offense_column, defense_column])
        panel, coverage = named_frame(
            subset, names,
            candidate=candidate, name_column=name_column, season_column="season",
            offense_column=offense_column, defense_column=defense_column,
        )
        panels.append(panel)
        coverage_rows.append(coverage)

    internal = pd.read_parquet(INTERNAL_RUN / "priors.parquet")
    internal["candidate_full"] = internal["design"] + "__" + internal["candidate"]
    internal = internal.loc[internal["candidate_full"].isin(INTERNAL_CANDIDATES)].copy()
    internal["candidate"] = internal["candidate_full"].map(INTERNAL_CANDIDATES)
    internal = internal.rename(
        columns={"prior_offense": "offense", "prior_defense": "defense", "prior_net": "net"}
    )
    panels.append(internal[["PLAYER_ID", "rating_season", "candidate", "offense", "defense", "net"]])
    panels.append(fit_box15_2014_onward())

    if args.darko:
        sources["darko_2019_snapshot"] = args.darko
        darko = pd.read_csv(args.darko).rename(
            columns={"nba_id": "PLAYER_ID", "o_dpm": "offense", "d_dpm": "defense"}
        )
        darko["rating_season"] = 2018
        darko["candidate"] = "DARKO preseason"
        darko["net"] = darko["offense"] + darko["defense"]
        panels.append(darko[["PLAYER_ID", "rating_season", "candidate", "offense", "defense", "net"]])

    ratings = pd.concat(panels, ignore_index=True)
    ratings = ratings.loc[ratings["rating_season"].between(2017, 2024)].copy()
    if ratings.duplicated(["candidate", "rating_season", "PLAYER_ID"]).any():
        duplicates = ratings.loc[
            ratings.duplicated(["candidate", "rating_season", "PLAYER_ID"], keep=False),
            ["candidate", "rating_season", "PLAYER_ID"],
        ]
        raise ValueError(f"Duplicate normalized rating keys:\n{duplicates.head()}")
    identity_error = (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
    if identity_error > 1e-8:
        raise ValueError(f"Metric side identity failed: {identity_error}")
    identity_coverage = pd.concat(coverage_rows, ignore_index=True)
    identity_coverage = identity_coverage.loc[
        identity_coverage["rating_season"].between(2017, 2024)
    ].copy()
    return ratings, identity_coverage, sources


def exposure_coverage(matrix, player_ids: set[int]) -> float:
    n = len(matrix.players)
    mask = np.isin(matrix.players, list(player_ids))
    columns = np.concatenate([mask, mask, [False]])
    numerator = float(np.abs(matrix.game_design[:, columns]).sum())
    denominator = float(np.abs(matrix.game_design[:, : 2 * n]).sum())
    return numerator / denominator if denominator else math.nan


def score_scope(
    ratings: pd.DataFrame,
    *,
    scope: str,
    seasons: tuple[int, ...],
    candidates: tuple[str, ...] | None = None,
    strict_intersection: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_rows = []
    coverage_rows = []
    factor_cache = {}
    for season in seasons:
        matrix = _load_matrix(season)
        season_ratings = ratings.loc[ratings["rating_season"].eq(season)]
        available = set(season_ratings["candidate"])
        selected = tuple(candidates or sorted(available))
        selected = tuple(candidate for candidate in selected if candidate in available)
        if candidates and len(selected) != len(candidates):
            continue
        common_players: set[int] | None = None
        if strict_intersection:
            common_players = set.intersection(
                *(
                    set(season_ratings.loc[season_ratings["candidate"].eq(candidate), "PLAYER_ID"])
                    for candidate in selected
                )
            )
        for candidate in selected:
            prior = season_ratings.loc[season_ratings["candidate"].eq(candidate)].copy()
            if common_players is not None:
                prior = prior.loc[prior["PLAYER_ID"].isin(common_players)]
            center, _ = coefficient_center(
                prior.rename(columns={"offense": "prior_offense", "defense": "prior_defense"}),
                matrix,
            )
            predicted = _standalone_prediction(matrix, center)
            predictions = [(candidate, predicted)]
            if candidate in INTERNAL_AIO_CANDIDATES:
                base, offense_update, defense_update = _uniform_basis(
                    matrix,
                    center,
                    3_000.0,
                    4_500.0,
                    300.0,
                    factor_cache,
                )
                predictions.append(
                    (
                        INTERNAL_AIO_CANDIDATES[candidate],
                        base + offense_update + defense_update,
                    )
                )
            for output_candidate, output_prediction in predictions:
                frame = pd.DataFrame(
                    {
                        "scope": scope,
                        "candidate": output_candidate,
                        "rating_season": season,
                        "outcome_season": season + 1,
                        "game_id": matrix.game_ids,
                        "actual_margin": matrix.actual_margin,
                        "predicted_margin": output_prediction,
                    }
                )
                game_rows.append(frame)
            player_ids = set(prior["PLAYER_ID"])
            for output_candidate, _ in predictions:
                coverage_rows.append(
                    {
                        "scope": scope,
                        "candidate": output_candidate,
                        "rating_season": season,
                        "outcome_season": season + 1,
                        "rated_players": len(player_ids),
                        "lineup_slot_coverage": exposure_coverage(matrix, player_ids),
                    }
                )
    return pd.concat(game_rows, ignore_index=True), pd.DataFrame(coverage_rows)


def fold_metrics(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, candidate, season), frame in games.groupby(
        ["scope", "candidate", "outcome_season"], sort=True
    ):
        actual = frame["actual_margin"].to_numpy(dtype=float)
        predicted = frame["predicted_margin"].to_numpy(dtype=float)
        variance = float(np.var(predicted))
        rows.append(
            {
                "scope": scope,
                "candidate": candidate,
                "outcome_season": int(season),
                "games": len(frame),
                "mse": float(np.mean((actual - predicted) ** 2)),
                "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                "calibration_slope": float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance else math.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, candidate), frame in folds.groupby(["scope", "candidate"], sort=True):
        mse = float(frame["mse"].mean())
        rows.append(
            {
                "scope": scope,
                "candidate": candidate,
                "folds": len(frame),
                "mean_mse": mse,
                "aggregate_rmse": math.sqrt(mse),
                "mean_correlation": float(frame["correlation"].mean()),
                "mean_calibration_slope": float(frame["calibration_slope"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "mean_mse", "candidate"], kind="stable")


def correlation_table(ratings: pd.DataFrame, seasons: tuple[int, ...], candidates: tuple[str, ...]) -> pd.DataFrame:
    selected = ratings.loc[
        ratings["rating_season"].isin(seasons) & ratings["candidate"].isin(candidates),
        ["rating_season", "PLAYER_ID", "candidate", "net"],
    ]
    wide = selected.pivot_table(index=["rating_season", "PLAYER_ID"], columns="candidate", values="net")
    wide = wide.dropna(subset=list(candidates))
    rows = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            rows.append(
                {
                    "left_candidate": left,
                    "right_candidate": right,
                    "matched_player_seasons": len(wide),
                    "pearson": float(wide[left].corr(wide[right])),
                    "spearman": float(wide[left].corr(wide[right], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epm", type=Path, default=DEFAULT_DOWNLOADS / "EPM_All_Seasons.csv")
    parser.add_argument(
        "--lebron", type=Path,
        default=DEFAULT_DOWNLOADS / "lebron-data-2026-2025-2024-2023-2022-2021-2020-2019-2018-2017-2016-2015-2014-2013-2012-2011-2010.csv",
    )
    parser.add_argument("--mamba", type=Path, default=DEFAULT_DOWNLOADS / "MAMBAVALUES.xlsx - Sheet1.csv")
    parser.add_argument("--pipm", type=Path, default=DEFAULT_DOWNLOADS / "PIPM Player Finder through 2021 - Database.csv")
    parser.add_argument("--raptor", type=Path, default=DEFAULT_DOWNLOADS / "Data/modern_RAPTOR_by_player.csv")
    parser.add_argument("--darko", type=Path)
    parser.add_argument(
        "--darko-history",
        type=Path,
        default=DEFAULT_DOWNLOADS / "Data/DARKO - Daily Adjusted and Regressed Kalman Optimized projections.xlsx",
    )
    args = parser.parse_args()

    ratings, identity_coverage, sources = load_panels(args)
    broad_ratings = ratings.loc[ratings["candidate"].ne("DARKO preseason")]
    broad_games, broad_coverage = score_scope(
        broad_ratings, scope="all_available_2017_2024", seasons=tuple(range(2017, 2025))
    )
    strict_games, strict_coverage = score_scope(
        ratings,
        scope="strict_common_2017_2020",
        seasons=(2017, 2018, 2019, 2020),
        candidates=STRICT_CANDIDATES,
        strict_intersection=True,
    )
    holdout_candidates = tuple(
        candidate
        for candidate in (
            "Box15", "Box15 (2014+)", "Rich elastic SPM", "Defense residual challenger",
            "EPM", "LEBRON", "MAMBA", "DARKO DPM", "BPM 2.0", "xRAPM",
        )
        if candidate in set(ratings.loc[ratings["rating_season"].eq(2024), "candidate"])
    )
    holdout_games, holdout_coverage = score_scope(
        ratings,
        scope="trained_through_2023_holdout",
        seasons=(2024,),
        candidates=holdout_candidates,
        strict_intersection=True,
    )
    frames = [broad_games, strict_games, holdout_games]
    coverages = [broad_coverage, strict_coverage, holdout_coverage]
    if "DARKO preseason" in set(ratings["candidate"]):
        darko_candidates = STRICT_CANDIDATES + ("DARKO preseason",)
        darko_games, darko_coverage = score_scope(
            ratings,
            scope="darko_2019_snapshot",
            seasons=(2018,),
            candidates=darko_candidates,
            strict_intersection=True,
        )
        frames.append(darko_games)
        coverages.append(darko_coverage)
    games = pd.concat(frames, ignore_index=True)
    coverage = pd.concat(coverages, ignore_index=True)
    folds = fold_metrics(games)
    aggregate = aggregate_metrics(folds)

    interval_rows = []
    for scope in ("strict_common_2017_2020", "trained_through_2023_holdout", "darko_2019_snapshot"):
        frame = games.loc[games["scope"].eq(scope)].rename(columns={"outcome_season": "season"})
        if frame.empty:
            continue
        result = paired_game_mse_intervals(frame, draws=5_000, seed=20260901)
        result["scope"] = scope
        interval_rows.append(result)
    intervals = pd.concat(interval_rows, ignore_index=True)
    correlations = correlation_table(ratings, (2017, 2018, 2019, 2020), STRICT_CANDIDATES)
    training_scope = pd.read_parquet(sources["internal_training_scope"])
    training_scope = training_scope.loc[
        training_scope["design"].eq("current_control")
        & training_scope["candidate"].isin(("box15", "rich"))
        & training_scope["rating_season"].between(2017, 2024),
        [
            "design", "candidate", "rating_season", "input_season", "side",
            "training_start", "training_end", "training_seasons", "selected_features",
        ],
    ].copy()
    restricted_box_scope = pd.DataFrame(
        [
            {
                "design": "current_control",
                "candidate": "box15_2014_onward",
                "rating_season": season,
                "input_season": season,
                "side": side,
                "training_start": 2014,
                "training_end": season - 1,
                "training_seasons": season - 2014,
                "selected_features": len(BOX_PIPM_STYLE_FEATURES),
            }
            for season in range(2017, 2025)
            for side in ("offense", "defense")
        ]
    )
    training_scope = pd.concat([training_scope, restricted_box_scope], ignore_index=True)

    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    runner_sha256 = sha256_file(Path(__file__))
    identity_payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_hashes": source_hashes,
        "runner_sha256": runner_sha256,
        "matrix_root": str(CHECKPOINT_ROOT.relative_to(ROOT)),
        "strict_candidates": STRICT_CANDIDATES,
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/external_all_in_one_benchmark" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    tables = {
        "ratings.parquet": ratings,
        "identity_coverage.parquet": identity_coverage,
        "game_predictions.parquet": games,
        "lineup_coverage.parquet": coverage,
        "fold_metrics.parquet": folds,
        "aggregate_metrics.parquet": aggregate,
        "paired_intervals.parquet": intervals,
        "metric_correlations.parquet": correlations,
        "internal_training_scope.parquet": training_scope,
    }
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_external_benchmark_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "all_available_rating_seasons": list(range(2017, 2025)),
            "strict_common_rating_seasons": [2017, 2018, 2019, 2020],
            "trained_through_2023_rating_season": 2024,
            "darko_snapshot": "day one of 2018-19; different information timing",
        },
        "method": {
            "score": "equal-season mean next-season whole-game margin MSE",
            "lineup_weights": "observed next-season lineups; evaluation device, not deployable forecast",
            "missing_players": "zero relative impact after exposure centering",
            "strict_scope": "same player intersection and same games for every candidate",
            "bootstrap": "5000 paired whole-game draws within season",
            "internal_aio_update": "one-season terminal-lineup RAPM with 3000 offense, 4500 defense, and 300 home penalties",
            "external_update": "external metrics remain standalone to avoid double-counting embedded on-off evidence",
        },
        "sources": {
            name: {"file": path.name, "sha256": source_hashes[name]}
            for name, path in sources.items()
        },
        "runner_sha256": runner_sha256,
        "quality": {
            "games": int(games[["scope", "outcome_season", "game_id"]].drop_duplicates().shape[0]),
            "rating_rows": len(ratings),
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
            "row_set_sha256": canonical_frame_hash(games, ["scope", "candidate", "outcome_season", "game_id"]),
        },
        "files": {},
        "forbidden_interpretation": "Different coverage periods are not a head-to-head ranking. DARKO has different information timing. All outcomes are reused evidence.",
    }
    for name, table in tables.items():
        table.to_parquet(output / name, index=False)
        run["files"][name] = {
            "rows": len(table),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
