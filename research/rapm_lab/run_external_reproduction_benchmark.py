"""Reproduce public plus-minus variants and audit CourtSignal RAPM agreement.

The output separates exact-key, exact-window checks from deliberately weaker
cross-estimand comparisons.  External ratings are comparators, never labels or
ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "research/rapm_lab/data/external"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/external_reproduction_benchmark"
HORIZON_RATINGS = (
    ROOT
    / "research/rapm_lab/outputs/rapm_target_horizon_bakeoff"
    / "rapm_target_horizon_bakeoff_v1_7c70e278cb/ratings.parquet"
)
WP_RATINGS = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_wp_rapm"
    / "rolling_5y_wp_rapm_v1_39800d31b3/ratings.parquet"
)
LEGACY_28Y = next((ROOT / "rapm/outputs/rapm_results").glob("Core_Rapm_full_1997*24.csv"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def season_end(value: object) -> int:
    parts = str(value).split("-")
    end = parts[-1]
    if len(end) != 2:
        return int(end)
    start = int(parts[0])
    resolved = (start // 100) * 100 + int(end)
    return resolved + 100 if resolved < start else resolved


def ryan_window_bounds(value: object) -> tuple[int, int]:
    """Map Ryan Davis labels such as 2018-23 to season-end years 2019-23."""
    source_start = int(str(value).split("-")[0])
    return source_start + 1, season_end(value)


def finite_number(value: str) -> float:
    return np.nan if value in {"null", "undefined", "NaN"} else float(value)


def parse_xrapm_3y(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    pattern = re.compile(
        r'player_pages/(\d+)\.html">([^<]+)</td>\s*'
        r'<td[^>]*>([^<]*)</td>\s*'
        r'<td[^>]*>([-+]?\d+(?:\.\d+)?)\s*\([^)]*\)</td>\s*'
        r'<td[^>]*>([-+]?\d+(?:\.\d+)?)\s*\([^)]*\)</td>\s*'
        r'<td[^>]*[^>]*>([-+]?\d+(?:\.\d+)?)\s*\([^)]*\)</td>'
    )
    rows = []
    for player_id, name, team, offense, points_allowed, total in pattern.findall(text):
        rows.append(
            {
                "PLAYER_ID": int(player_id),
                "player_name": name.strip(),
                "team": team.strip(),
                "reference_offense": float(offense),
                "reference_defense": -float(points_allowed),
                "reference_net": float(total),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["PLAYER_ID"].duplicated().any():
        raise ValueError("xRAPM 3-year table did not parse to unique NBA player IDs")
    return frame


def parse_darko_wowy(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    pattern = re.compile(
        r'\{nba_id:(\d+),season:(\d+),.*?player_name:"([^"]+)".*?'
        r'wowy_rapm:([^,}]+),wowy_orapm:([^,}]+),wowy_drapm:([^,}]+),'
        r'exposure:([^,}]+),season_possessions:([^,}]+),minutes:([^,}]+),bpm:([^,}]+),'
    )
    rows = []
    for values in pattern.findall(text):
        player_id, season, name, net, offense, defense, exposure, possessions, minutes, bpm = values
        rows.append(
            {
                "PLAYER_ID": int(player_id),
                "season": int(season),
                "player_name": name,
                "reference_offense": finite_number(offense),
                "reference_defense": finite_number(defense),
                "reference_net": finite_number(net),
                "exposure": finite_number(exposure),
                "season_possessions": finite_number(possessions),
                "minutes": finite_number(minutes),
                "bpm": finite_number(bpm),
            }
        )
    frame = pd.DataFrame(rows)
    expected_season = int(re.search(r"(\d{4})", path.stem).group(1))
    if frame.empty or set(frame["season"]) != {expected_season}:
        raise ValueError(f"DARKO WOWY season {expected_season} did not parse")
    if frame[["PLAYER_ID", "season"]].duplicated().any():
        raise ValueError("DARKO WOWY contains duplicate player-season keys")
    return frame


def _clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["reference", "courtsignal"]
    )


def comparison_metrics(frame: pd.DataFrame) -> dict:
    clean = _clean_pairs(frame)
    if len(clean) < 3:
        return {
            "n": int(len(clean)),
            "pearson": np.nan,
            "spearman": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }
    reference = clean["reference"].to_numpy(float)
    courtsignal = clean["courtsignal"].to_numpy(float)
    slope, intercept = np.polyfit(reference, courtsignal, 1)
    error = courtsignal - reference
    return {
        "n": int(len(clean)),
        "pearson": float(np.corrcoef(reference, courtsignal)[0, 1]),
        "spearman": float(clean[["reference", "courtsignal"]].corr(method="spearman").iloc[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def append_comparison(
    metric_rows: list[dict],
    matched_frames: list[pd.DataFrame],
    *,
    source: str,
    comparison: str,
    scope: str,
    component: str,
    frame: pd.DataFrame,
    method_status: str,
    note: str,
) -> None:
    clean = _clean_pairs(frame.copy())
    metrics = comparison_metrics(clean)
    metric_rows.append(
        {
            "source": source,
            "comparison": comparison,
            "scope": scope,
            "component": component,
            "method_status": method_status,
            "note": note,
            **metrics,
        }
    )
    if not clean.empty:
        keep = [
            column
            for column in ("PLAYER_ID", "player_name", "season", "window_start", "window_end")
            if column in clean
        ]
        output = clean[keep + ["reference", "courtsignal"]].copy()
        if "season" in output:
            output["season"] = output["season"].astype(str)
        output.insert(0, "component", component)
        output.insert(0, "scope", scope)
        output.insert(0, "comparison", comparison)
        output.insert(0, "source", source)
        matched_frames.append(output)


def compare_components(
    metric_rows: list[dict],
    matched_frames: list[pd.DataFrame],
    merged: pd.DataFrame,
    *,
    source: str,
    comparison: str,
    scope: str,
    method_status: str,
    note: str,
    components: tuple[str, ...] = ("offense", "defense", "net"),
) -> None:
    for component in components:
        frame = merged.copy()
        frame["reference"] = frame[f"reference_{component}"]
        frame["courtsignal"] = frame[f"courtsignal_{component}"]
        append_comparison(
            metric_rows,
            matched_frames,
            source=source,
            comparison=comparison,
            scope=scope,
            component=component,
            frame=frame,
            method_status=method_status,
            note=note,
        )


def unique_name_join(reference: pd.DataFrame, courtsignal: pd.DataFrame) -> pd.DataFrame:
    left = reference.copy()
    right = courtsignal.copy()
    left["name_key"] = left["player_name"].map(normalize_name)
    right["name_key"] = right["player_name"].map(normalize_name)
    left = left.loc[~left["name_key"].duplicated(keep=False)]
    right = right.loc[~right["name_key"].duplicated(keep=False)]
    return left.merge(right.drop(columns="player_name"), on="name_key", how="inner", validate="one_to_one")


def reproduce_aupm(frame: pd.DataFrame) -> tuple[pd.Series, float]:
    columns = ["OnOffRtg", "NET_RATING", "BLK_per100_def", "DREB_per100_def", "SumAbove"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    reproduced = (
        -6.357797067568494
        + 0.058647 * values["OnOffRtg"]
        + 0.282983 * values["NET_RATING"]
        - 0.143842 * values["BLK_per100_def"]
        + 0.122480 * values["DREB_per100_def"]
        + 0.007007 * values["SumAbove"]
    )
    difference = (reproduced - pd.to_numeric(frame["AuPM"], errors="coerce")).abs()
    return reproduced, float(difference.max())


def minutes_weighted_player_seasons(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    valid = frame.dropna(subset=["PLAYER_ID", "Season", value]).copy()
    valid["PLAYER_ID"] = valid["PLAYER_ID"].astype(int)
    valid["weight"] = pd.to_numeric(valid["MIN"], errors="coerce").fillna(0.0).clip(lower=0.0)
    valid["weighted"] = valid[value] * valid["weight"]
    grouped = valid.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
        weighted=("weighted", "sum"),
        weight=("weight", "sum"),
        unweighted=(value, "mean"),
        player_name=("Player", "first"),
    )
    grouped["reference_net"] = np.where(
        grouped["weight"].gt(0), grouped["weighted"] / grouped["weight"], grouped["unweighted"]
    )
    return grouped.rename(columns={"Season": "season"})


def build_game_level_pm() -> tuple[pd.DataFrame, dict]:
    games = pd.read_parquet(ROOT / "data/lake/silver/game_dim.parquet")
    games = games.loc[
        games["season_type"].eq("regular") & games["season_end"].isin([2024, 2025, 2026])
    ].copy()
    possessions = pd.read_parquet(
        ROOT / "data/lake/silver/possessions.parquet", columns=["game_id", "season_type"]
    )
    possessions = possessions.loc[possessions["season_type"].eq("regular")]
    counts = possessions.groupby("game_id").size().rename("total_possessions")
    games = games.merge(counts, on="game_id", how="inner", validate="one_to_one")
    games["team_possessions"] = games["total_possessions"] / 2.0
    games = games.loc[games["team_possessions"].gt(0)].reset_index(drop=True)
    games["target"] = 100.0 * games["home_margin"] / games["team_possessions"]

    player_games = pd.read_parquet(
        ROOT / "data/lake/silver/player_games.parquet",
        columns=["game_id", "player_id", "player_name", "team_side", "played", "minutes_seconds"],
    )
    player_games = player_games.loc[
        player_games["game_id"].isin(games["game_id"])
        & player_games["played"]
        & player_games["player_id"].notna()
        & player_games["minutes_seconds"].fillna(0).gt(0)
    ].copy()
    player_games["player_id"] = player_games["player_id"].astype(int)
    players = np.sort(player_games["player_id"].unique())
    player_index = {player: index for index, player in enumerate(players)}
    game_index = dict(zip(games["game_id"], range(len(games))))
    max_period = games.set_index("game_id")["max_period"].clip(lower=4)
    game_seconds = 48 * 60 + (max_period - 4) * 5 * 60
    player_games["share"] = player_games["minutes_seconds"] / player_games["game_id"].map(game_seconds)
    rows = player_games["game_id"].map(game_index).to_numpy(int)
    columns = player_games["player_id"].map(player_index).to_numpy(int)
    signs = np.where(player_games["team_side"].eq("home"), 1.0, -1.0)
    values = player_games["share"].to_numpy(float) * signs
    player_design = sparse.csr_matrix((values, (rows, columns)), shape=(len(games), len(players)))
    design = player_design
    target = games["target"].to_numpy(float)

    def evaluate(train_seasons: list[int], test_season: int, alpha: float) -> dict:
        train = games["season_end"].isin(train_seasons).to_numpy()
        test = games["season_end"].eq(test_season).to_numpy()
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        model.fit(design[train], target[train])
        prediction = model.predict(design[test])
        error = prediction - target[test]
        return {
            "alpha": float(alpha),
            "train_seasons": train_seasons,
            "test_season": test_season,
            "games": int(test.sum()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "correlation": float(np.corrcoef(prediction, target[test])[0, 1]),
        }

    selection = [evaluate([2024], 2025, alpha) for alpha in (1, 3, 10, 30, 100, 300, 1000)]
    selected = min(selection, key=lambda row: (row["rmse"], row["alpha"]))["alpha"]
    diagnostic = evaluate([2024, 2025], 2026, selected)
    model = Ridge(alpha=selected, fit_intercept=True, solver="lsqr")
    model.fit(design, target)
    names = (
        player_games.sort_values("game_id")
        .drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
    )
    ratings = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "player_name": [names.get(player, str(player)) for player in players],
            "gpm_net": model.coef_,
        }
    )
    exposure = player_games.groupby("player_id")["minutes_seconds"].sum() / 60.0
    ratings["minutes"] = ratings["PLAYER_ID"].map(exposure).fillna(0.0)
    manifest = {
        "estimand": "game-level home margin per 100 estimated from signed player minute shares",
        "selected_alpha": selected,
        "selection": selection,
        "diagnostic": diagnostic,
        "games": int(len(games)),
        "players": int(len(players)),
        "home_effect": float(model.intercept_),
        "home_term_parameterization": "unpenalized intercept because every target is home margin",
    }
    return ratings, manifest


def run() -> dict:
    started = time.perf_counter()
    horizons = pd.read_parquet(HORIZON_RATINGS).rename(
        columns={
            "offense": "courtsignal_offense",
            "defense": "courtsignal_defense",
            "net": "courtsignal_net",
        }
    )
    metrics: list[dict] = []
    matched: list[pd.DataFrame] = []
    quality: list[dict] = []

    def source_record(name: str, path: Path, *, grain: str, status: str, note: str, rows: int) -> None:
        quality.append(
            {
                "source": name,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "rows": int(rows),
                "grain": grain,
                "status": status,
                "note": note,
            }
        )

    # Ryan Davis annual RAPM and luck-adjusted RAPM: exact ID-season joins.
    annual_path = EXTERNAL / "user_downloads/ryan_davis_annual_rapm.csv"
    annual = pd.read_csv(annual_path)
    annual["season"] = annual["season"].map(season_end)
    annual = annual.rename(columns={"playerId": "PLAYER_ID", "playerName": "player_name"})
    internal_annual = horizons.loc[horizons["horizon"].eq("1y")].assign(
        season=lambda frame: frame["window_end"].astype(int)
    )
    source_record(
        "Ryan Davis annual RAPM",
        annual_path,
        grain="NBA player ID x season",
        status="exact_key_scope",
        note="Annual regular-season RAPM; 2014-23 overlaps CourtSignal.",
        rows=len(annual),
    )
    for variant, prefix in (("RAPM", "RAPM"), ("Luck-adjusted RAPM", "LA_RAPM")):
        reference = annual.assign(
            reference_offense=annual[f"{prefix}__Off"],
            reference_defense=annual[f"{prefix}__Def"],
            reference_net=annual[prefix],
        )
        merged = reference.merge(
            internal_annual,
            on=["PLAYER_ID", "season"],
            how="inner",
            suffixes=("", "_internal"),
            validate="one_to_one",
        )
        for season in sorted(merged["season"].unique()):
            compare_components(
                metrics,
                matched,
                merged.loc[merged["season"].eq(season)],
                source="Ryan Davis annual RAPM",
                comparison=variant,
                scope=str(season),
                method_status=(
                    "exact_key_scope" if variant == "RAPM" else "different_estimand"
                ),
                note=(
                    "Same NBA ID and season; implementation and source possession rules may differ."
                    if variant == "RAPM"
                    else "Same NBA ID and season, but luck-adjusted RAPM is compared with normal RAPM."
                ),
            )
        compare_components(
            metrics,
            matched,
            merged,
            source="Ryan Davis annual RAPM",
            comparison=variant,
            scope="2014-2023 pooled",
            method_status=(
                "exact_key_scope" if variant == "RAPM" else "different_estimand"
            ),
            note=(
                "Pooled player-seasons; season-level rows are also reported."
                if variant == "RAPM"
                else "Exact player-season keys, different luck-adjusted versus normal estimands."
            ),
        )

    # Ryan Davis labels are start-year to end-year, so 2018-23 is five NBA
    # seasons ending 2019 through 2023. The file contains three- and five-year
    # panels interleaved.
    multi_path = EXTERNAL / "user_downloads/ryan_davis_multi_rapm.csv"
    multi = pd.read_csv(multi_path).rename(columns={"playerId": "PLAYER_ID", "playerName": "player_name"})
    bounds = multi["season"].map(ryan_window_bounds)
    multi["window_start"] = bounds.str[0]
    multi["window_end"] = bounds.str[1]
    multi["horizon_years"] = multi["window_end"] - multi["window_start"] + 1
    multi = multi.assign(
        reference_offense=lambda frame: frame["RAPM__Off"],
        reference_defense=lambda frame: frame["RAPM__Def"],
        reference_net=lambda frame: frame["RAPM"],
    )
    source_record(
        "Ryan Davis multi-year RAPM",
        multi_path,
        grain="NBA player ID x rolling window",
        status="exact_key_scope",
        note="Source start-year labels are mapped to exact three- and five-season end-year windows.",
        rows=len(multi),
    )
    for horizon in (3, 5):
        reference = multi.loc[multi["horizon_years"].eq(horizon)]
        internal = horizons.loc[horizons["horizon"].eq(f"{horizon}y")]
        merged = reference.merge(
            internal,
            on=["PLAYER_ID", "window_start", "window_end"],
            how="inner",
            suffixes=("", "_internal"),
            validate="one_to_one",
        )
        for end in sorted(merged["window_end"].unique()):
            subset = merged.loc[merged["window_end"].eq(end)]
            start = int(subset["window_start"].iloc[0])
            compare_components(
                metrics,
                matched,
                subset,
                source="Ryan Davis multi-year RAPM",
                comparison=f"{horizon}-year RAPM",
                scope=f"{start}-{end}",
                method_status="exact_key_scope",
                note=f"Same NBA ID and exact {horizon}-season window; source label is one year earlier at the start.",
            )
        compare_components(
            metrics,
            matched,
            merged,
            source="Ryan Davis multi-year RAPM",
            comparison=f"{horizon}-year RAPM",
            scope=f"exact {horizon}-year windows pooled",
            method_status="exact_key_scope",
            note=f"Pooled exact {horizon}-season player-window rows.",
        )

    # Current xRAPM page: same 2024-26 span, but explicitly unequal season weights.
    xrapm_path = EXTERNAL / "xrapm/RAPM_3y.html"
    xrapm = parse_xrapm_3y(xrapm_path)
    internal_three = horizons.loc[
        horizons["horizon"].eq("3y")
        & horizons["window_start"].eq(2024)
        & horizons["window_end"].eq(2026)
    ]
    merged = xrapm.merge(internal_three, on="PLAYER_ID", how="inner", validate="one_to_one")
    compare_components(
        metrics,
        matched,
        merged,
        source="xRAPM",
        comparison="Current three-year RAPM",
        scope="2024-2026",
        method_status="same_window_weight_mismatch",
        note="Same player IDs and seasons; xRAPM downweights 2024 and 2025 while CourtSignal uses equal possession weight.",
    )
    source_record(
        "xRAPM",
        xrapm_path,
        grain="NBA player ID x current three-year table",
        status="same_window_weight_mismatch",
        note="2024-26 seasons, unequal xRAPM season weights.",
        rows=len(xrapm),
    )

    # DARKO WOWY: official season averages, intentionally a different estimand.
    darko_paths = sorted((EXTERNAL / "darko_wowy").glob("season_*.html"))
    darko = pd.concat([parse_darko_wowy(path) for path in darko_paths], ignore_index=True)
    merged = darko.merge(
        internal_annual,
        on=["PLAYER_ID", "season"],
        how="inner",
        suffixes=("", "_internal"),
        validate="one_to_one",
    )
    for season in sorted(merged["season"].unique()):
        compare_components(
            metrics,
            matched,
            merged.loc[merged["season"].eq(season)],
            source="DARKO WOWY",
            comparison="WOWY RAPM season average",
            scope=str(season),
            method_status="different_estimand",
            note="DARKO daily synthetic game-level RAPM average versus retrospective possession RAPM.",
        )
    compare_components(
        metrics,
        matched,
        merged,
        source="DARKO WOWY",
        comparison="WOWY RAPM season average",
        scope="2017-2026 pooled",
        method_status="different_estimand",
        note="Agreement check only; model timing, priors, and estimand differ.",
    )
    for path in darko_paths:
        source_record(
            "DARKO WOWY",
            path,
            grain="NBA player ID x season",
            status="different_estimand",
            note="Official published season-average WOWY RAPM page.",
            rows=len(darko.loc[darko["season"].eq(int(re.search(r"\d{4}", path.stem).group()))]),
        )

    # RAPTOR's on/off component: exact season and exact normalized names, but a
    # different model and exposure treatment from CourtSignal RAPM.
    raptor_path = ROOT / "data/raw/site_Data/full_raptor.csv"
    raptor = pd.read_csv(raptor_path).rename(columns={"player_name": "player_name"})
    raptor = raptor.assign(
        reference_offense=raptor["raptor_onoff_offense"],
        reference_defense=raptor["raptor_onoff_defense"],
        reference_net=raptor["raptor_onoff_total"],
    )
    raptor["name_key"] = raptor["player_name"].map(normalize_name)
    internal_names = internal_annual.assign(name_key=internal_annual["PLAYER_NAME"].map(normalize_name))
    valid_left = ~raptor[["name_key", "season"]].duplicated(keep=False)
    valid_right = ~internal_names[["name_key", "season"]].duplicated(keep=False)
    merged = raptor.loc[valid_left].merge(
        internal_names.loc[valid_right],
        on=["name_key", "season"],
        how="inner",
        suffixes=("", "_internal"),
        validate="one_to_one",
    )
    for season in sorted(merged["season"].unique()):
        compare_components(
            metrics,
            matched,
            merged.loc[merged["season"].eq(season)],
            source="FiveThirtyEight RAPTOR",
            comparison="RAPTOR on/off",
            scope=str(season),
            method_status="different_estimand",
            note="RAPTOR on/off component versus CourtSignal RAPM; exact normalized name and season only.",
        )
    compare_components(
        metrics,
        matched,
        merged,
        source="FiveThirtyEight RAPTOR",
        comparison="RAPTOR on/off",
        scope="2014-2022 pooled",
        method_status="different_estimand",
        note="No fuzzy matching; ambiguous normalized names are excluded.",
    )
    qualified_raptor = merged.loc[pd.to_numeric(merged["mp"], errors="coerce").ge(1000)]
    compare_components(
        metrics,
        matched,
        qualified_raptor,
        source="FiveThirtyEight RAPTOR",
        comparison="RAPTOR on/off, 1000+ minutes",
        scope="2014-2022 pooled",
        method_status="different_estimand",
        note="Exposure-qualified RAPTOR on/off component; still a different estimand from CourtSignal RAPM.",
    )
    source_record(
        "FiveThirtyEight RAPTOR",
        raptor_path,
        grain="BBRef player ID x season",
        status="different_estimand",
        note="Official RAPTOR data file; on/off fields only.",
        rows=len(raptor),
    )

    # Local historical AuPM reproduction.
    aupm_path = ROOT / "data/processed/merged_per100_with_rTS_AuPM.csv"
    aupm = pd.read_csv(
        aupm_path,
        usecols=[
            "PLAYER_ID",
            "Player",
            "Season",
            "MIN",
            "AuPM",
            "OnOffRtg",
            "NET_RATING",
            "BLK_per100_def",
            "DREB_per100_def",
            "SumAbove",
        ],
    )
    aupm["AuPM_reproduced"], aupm_error = reproduce_aupm(aupm)
    aupm_player_season = minutes_weighted_player_seasons(aupm, "AuPM_reproduced")
    merged = aupm_player_season.merge(
        internal_annual,
        on=["PLAYER_ID", "season"],
        how="inner",
        validate="one_to_one",
    )
    for season in sorted(merged["season"].unique()):
        frame = merged.loc[merged["season"].eq(season)].assign(
            reference=lambda value: value["reference_net"],
            courtsignal=lambda value: value["courtsignal_net"],
        )
        append_comparison(
            metrics,
            matched,
            source="Local legacy AuPM",
            comparison="Reproduced AuPM",
            scope=str(season),
            component="net",
            frame=frame,
            method_status="different_estimand",
            note="Minutes-weighted team rows; historical local formula, not canonical Ben Taylor AuPM.",
        )
    append_comparison(
        metrics,
        matched,
        source="Local legacy AuPM",
        comparison="Reproduced AuPM",
        scope="2014-2024 pooled",
        component="net",
        frame=merged.assign(reference=merged["reference_net"], courtsignal=merged["courtsignal_net"]),
        method_status="different_estimand",
        note="Formula reproduction is exact to stored rounding; agreement with RAPM is descriptive.",
    )
    source_record(
        "Local legacy AuPM",
        aupm_path,
        grain="NBA player-team-season",
        status="reproduced_legacy_formula",
        note=f"Maximum stored-versus-recomputed AuPM error {aupm_error:.3g}.",
        rows=len(aupm),
    )

    # PBPStats 2024 on-court net: not an on-minus-off or RAPM estimate.
    pbp_path = ROOT / "data/raw/site_Data/wowy/player_large.csv"
    pbp = pd.read_csv(
        pbp_path,
        usecols=["EntityId", "Name", "PlusMinus", "OffPoss", "DefPoss"],
    ).assign(
        PLAYER_ID=lambda frame: frame["EntityId"].astype(int),
        player_name=lambda frame: frame["Name"],
        reference_net=lambda frame: 100.0
        * frame["PlusMinus"]
        / ((frame["OffPoss"] + frame["DefPoss"]) / 2.0),
    )
    internal_2024 = internal_annual.loc[internal_annual["season"].eq(2024)]
    merged = pbp.merge(internal_2024, on="PLAYER_ID", how="inner", validate="one_to_one")
    append_comparison(
        metrics,
        matched,
        source="PBPStats local export",
        comparison="Raw on-court plus-minus per 100",
        scope="2024",
        component="net",
        frame=merged.assign(reference=merged["reference_net"], courtsignal=merged["courtsignal_net"]),
        method_status="different_estimand",
        note="Raw on-court scoring margin per average team possession, not on-minus-off and not lineup adjusted.",
    )
    source_record(
        "PBPStats local export",
        pbp_path,
        grain="NBA player x 2024 regular season",
        status="different_estimand",
        note="Local get-totals export; source manifest is incomplete.",
        rows=len(pbp),
    )

    # Reproduce a transparent game-level PM-style ridge on 2024-26.
    gpm_ratings, gpm_manifest = build_game_level_pm()
    internal_2024_2026 = horizons.loc[
        horizons["horizon"].eq("3y")
        & horizons["window_start"].eq(2024)
        & horizons["window_end"].eq(2026)
    ]
    merged = gpm_ratings.merge(internal_2024_2026, on="PLAYER_ID", how="inner", validate="one_to_one")
    append_comparison(
        metrics,
        matched,
        source="CourtSignal reproduction",
        comparison="GPM-style game-level ridge",
        scope="2024-2026",
        component="net",
        frame=merged.assign(reference=merged["gpm_net"], courtsignal=merged["courtsignal_net"]),
        method_status="different_estimand",
        note="Final game margins and signed player minute shares versus possession-lineup RAPM.",
    )
    quality.append(
        {
            "source": "CourtSignal reproduction",
            "path": "data/lake/silver/player_games.parquet + game_dim.parquet + possessions.parquet",
            "sha256": "multiple_files_recorded_in_run_sources",
            "rows": gpm_manifest["games"],
            "grain": "game x signed player minute share",
            "status": "reproduced",
            "note": "GPM-style ridge; not a claim of exact Thinking Basketball WOWYR reproduction.",
        }
    )

    # User-downloaded long-span files: exact normalized names, with explicit mismatch labels.
    legacy = pd.read_csv(LEGACY_28Y).rename(
        columns={"Name": "player_name", "Off": "courtsignal_offense", "RAPM": "courtsignal_net"}
    )
    legacy["courtsignal_defense"] = -legacy["Def"]
    long_specs = [
        (
            "Downloaded 28-year RAPM",
            EXTERNAL / "user_downloads/rapm_1997_2024_28y.csv",
            {"Player": "player_name", "Offense": "reference_offense", "Defense": "raw_defense", "Total": "reference_net"},
            "1997-2024",
            legacy,
            "Exact scope, name-only join, and different legacy engine; useful reproduction check, not current-model validation.",
        ),
        (
            "Downloaded same-age coach-adjusted RAPM",
            EXTERNAL / "user_downloads/rapm_1997_2024_same_age_coach.csv",
            {"Player": "player_name", "Off ": "reference_offense", "Def": "raw_defense", "Tot": "reference_net"},
            "1997-2024 versus CourtSignal age-27 1997-2026",
            pd.read_parquet(
                ROOT / "research/rapm_lab/outputs/age_adjusted_rapm/age_adjusted_full_1997_2026_v1_1765feaffc/ratings.parquet"
            ).rename(
                columns={
                    "player_name": "player_name",
                    "age27_offense": "courtsignal_offense",
                    "age27_defense": "courtsignal_defense",
                    "age27_net": "courtsignal_net",
                }
            ),
            "Scope and coach-control mismatch; name-only reference comparison.",
        ),
    ]
    for source, path, columns, scope, ours, note in long_specs:
        reference = pd.read_csv(path).rename(columns=columns)
        reference["reference_defense"] = -reference["raw_defense"]
        joined = unique_name_join(reference, ours)
        compare_components(
            metrics,
            matched,
            joined,
            source=source,
            comparison="Long-span downloaded reference",
            scope=scope,
            method_status="invalid_direct",
            note=note,
        )
        source_record(
            source,
            path,
            grain="player name x full span",
            status="invalid_direct",
            note=note,
            rows=len(reference),
        )

    weighted_path = EXTERNAL / "user_downloads/rapm_2022_2024_weighted.csv"
    weighted = pd.read_csv(weighted_path).rename(
        columns={"Player": "player_name", "Offense": "reference_offense", "Defense": "raw_defense", "Total": "reference_net"}
    )
    weighted["reference_defense"] = -weighted["raw_defense"]
    ours_three = horizons.loc[
        horizons["horizon"].eq("3y")
        & horizons["window_start"].eq(2022)
        & horizons["window_end"].eq(2024)
    ].rename(columns={"PLAYER_NAME": "player_name"})
    joined = unique_name_join(weighted, ours_three)
    compare_components(
        metrics,
        matched,
        joined,
        source="Downloaded weighted RAPM",
        comparison="Three-year weighted RAPM",
        scope="2022-2024",
        method_status="same_window_weight_mismatch",
        note="Reference weights are 0.6/0.8/1.0 and age-adjusted; CourtSignal uses equal possession weight and no age adjustment.",
    )
    source_record(
        "Downloaded weighted RAPM",
        weighted_path,
        grain="player name x 2022-24 window",
        status="same_window_weight_mismatch",
        note="Name-only join; explicit season-weight and age mismatch.",
        rows=len(weighted),
    )

    wp_path = EXTERNAL / "user_downloads/wp_rapm_2018.csv"
    wp_reference = pd.read_csv(wp_path).rename(
        columns={
            "Player": "player_name",
            "Effect on Win Probability, per 100 possessions": "reference_net",
        }
    )
    wp_internal = pd.read_parquet(WP_RATINGS)
    wp_internal = wp_internal.loc[
        wp_internal["window_start"].eq(2014) & wp_internal["window_end"].eq(2018)
    ].rename(columns={"net_wp_percentage_points_per_100": "courtsignal_net"})
    joined = unique_name_join(wp_reference, wp_internal)
    append_comparison(
        metrics,
        matched,
        source="Downloaded WP RAPM",
        comparison="WP impact",
        scope="2018 reference versus 2014-2018 CourtSignal",
        component="net",
        frame=joined.assign(reference=joined["reference_net"], courtsignal=joined["courtsignal_net"]),
        method_status="invalid_direct",
        note="One-season reference versus five-year CourtSignal window; scale and WP surface may also differ.",
    )
    source_record(
        "Downloaded WP RAPM",
        wp_path,
        grain="player name x 2018",
        status="invalid_direct",
        note="No exact one-season CourtSignal WP-RAPM artifact for 2018.",
        rows=len(wp_reference),
    )

    unresolved_path = EXTERNAL / "user_downloads/rapm_2017_2025.csv"
    unresolved = pd.read_csv(unresolved_path)
    source_record(
        "Downloaded 2017-2025 RAPM",
        unresolved_path,
        grain="player name x nine-season span",
        status="not_compared_no_exact_artifact",
        note="Total-only name-keyed reference; no exact 2017-25 CourtSignal fit was present. Do not compare it with 2014-26.",
        rows=len(unresolved),
    )

    metrics_frame = pd.DataFrame(metrics).sort_values(
        ["method_status", "source", "comparison", "scope", "component"], kind="stable"
    )
    matched_frame = pd.concat(matched, ignore_index=True, sort=False)
    quality_frame = pd.DataFrame(quality)
    sources = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            HORIZON_RATINGS,
            WP_RATINGS,
            LEGACY_28Y,
            ROOT / "data/raw/site_Data/full_raptor.csv",
            ROOT / "data/processed/merged_per100_with_rTS_AuPM.csv",
            ROOT / "data/raw/site_Data/wowy/player_large.csv",
            ROOT / "data/lake/silver/player_games.parquet",
            ROOT / "data/lake/silver/game_dim.parquet",
            ROOT / "data/lake/silver/possessions.parquet",
        ]
        + sorted((EXTERNAL / "user_downloads").glob("*.csv"))
        + [xrapm_path]
        + darko_paths
    }
    identity = hashlib.sha256(
        json.dumps(
            {"runner": sha256_file(Path(__file__)), "sources": sources}, sort_keys=True
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"external_reproduction_benchmark_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_parquet(output / "comparison_metrics.parquet", index=False)
    matched_frame.to_parquet(output / "matched_rows.parquet", index=False)
    quality_frame.to_parquet(output / "source_quality.parquet", index=False)
    gpm_ratings.to_parquet(output / "gpm_ratings.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "research_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "sources": sources,
        "quality": {
            "comparison_rows": int(len(metrics_frame)),
            "matched_rating_rows": int(len(matched_frame)),
            "source_records": int(len(quality_frame)),
            "aupm_maximum_reproduction_error": aupm_error,
        },
        "game_level_pm": gpm_manifest,
        "paths": {
            "comparison_metrics": "comparison_metrics.parquet",
            "matched_rows": "matched_rows.parquet",
            "source_quality": "source_quality.parquet",
            "gpm_ratings": "gpm_ratings.parquet",
        },
        "forbidden_interpretation": (
            "Correlation with an external rating is not predictive validation or proof of correctness. "
            "Rows marked different_estimand, weight mismatch, or invalid_direct cannot establish model parity."
        ),
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
