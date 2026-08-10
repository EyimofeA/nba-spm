"""External BPM and xRAPM benchmark for three-season statistical impact windows."""

from __future__ import annotations

import math
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic


XRAPM_URL = "https://xrapm.com/table_pages/xRAPM_{season}.html"
BPM_URL = "https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
USER_AGENT = "nba-impact-lab/0.1 research benchmark"


class _TableParser(HTMLParser):
    def __init__(self, table_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id = table_id
        self.in_table = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == self.table_id:
            self.in_table = True
        elif self.in_table and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join("".join(self.cell_parts).split()))
            self.in_cell = False
        elif self.in_table and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = []
        elif self.in_table and tag == "table":
            if self.row:
                self.rows.append(self.row)
            self.row = []
            self.in_table = False


def normalize_player_name(value: object) -> str:
    text = str(value).translate(
        str.maketrans({"ı": "i", "Ł": "L", "ł": "l", "Đ": "D", "đ": "d"})
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.replace("’", "'").replace(".", "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return " ".join(tokens)


def _parse_numeric(value: object) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else float("nan")


def _table_rows(html: str, table_id: str) -> list[list[str]]:
    parser = _TableParser(table_id)
    parser.feed(html)
    if not parser.rows:
        raise ValueError(f"HTML does not contain table #{table_id}.")
    return parser.rows


def parse_xrapm_html(html: str, season: int) -> pd.DataFrame:
    rows = _table_rows(html, "sortableTable")
    header = next(row for row in rows if row[:4] == ["Player", "Offense", "Defense(*)", "Total"])
    records = []
    for row in rows[rows.index(header) + 1 :]:
        if len(row) < 4 or row[0] == "Player":
            continue
        offense, defense_allowed, total = map(_parse_numeric, row[1:4])
        if not np.isfinite([offense, defense_allowed, total]).all():
            continue
        records.append(
            {
                "season": season,
                "player_name_xrapm": row[0],
                "normalized_name": normalize_player_name(row[0]),
                "xrapm_offense": offense,
                "xrapm_defense": -defense_allowed,
                "xrapm_net": total,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty or frame["normalized_name"].duplicated().any():
        raise ValueError(f"xRAPM {season} has no rows or duplicate normalized names.")
    identity_error = (frame["xrapm_offense"] + frame["xrapm_defense"] - frame["xrapm_net"]).abs()
    if identity_error.max() > 0.11:
        raise ValueError(f"xRAPM {season} offense-defense identity exceeds rounding tolerance.")
    return frame


def parse_bpm_html(html: str, season: int) -> pd.DataFrame:
    rows = _table_rows(html, "advanced")
    header = next(row for row in rows if "Player" in row and "BPM" in row and "MP" in row)
    indices = {
        name: header.index(name)
        for name in ("Player", "Team", "MP", "OBPM", "DBPM", "BPM")
    }
    records = []
    for row in rows[rows.index(header) + 1 :]:
        if len(row) <= max(indices.values()) or row[indices["Player"]] == "Player":
            continue
        values = {
            name: _parse_numeric(row[index])
            for name, index in indices.items()
            if name not in {"Player", "Team"}
        }
        if not np.isfinite(list(values.values())).all() or values["MP"] <= 0:
            continue
        name = row[indices["Player"]].rstrip("*").strip()
        records.append(
            {
                "season": season,
                "player_name_bpm": name,
                "normalized_name": normalize_player_name(name),
                "team": row[indices["Team"]],
                "minutes": values["MP"],
                "bpm_offense": values["OBPM"],
                "bpm_defense": values["DBPM"],
                "bpm_net": values["BPM"],
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"BPM {season} has no parsed rows.")
    duplicate_names = frame.loc[
        frame["normalized_name"].duplicated(keep=False), "normalized_name"
    ].unique()
    if len(duplicate_names):
        keep = ~frame["normalized_name"].isin(duplicate_names) | frame["team"].str.fullmatch(
            r"\d+TM"
        )
        frame = frame.loc[keep].copy()
    if frame["normalized_name"].duplicated().any():
        raise ValueError(f"BPM {season} cannot resolve duplicate normalized player names.")
    identity_error = (frame["bpm_offense"] + frame["bpm_defense"] - frame["bpm_net"]).abs()
    if identity_error.max() > 0.11:
        raise ValueError(f"BPM {season} offense-defense identity exceeds rounding tolerance.")
    return frame


def _download_html(url: str, destination: Path, *, attempts: int = 5) -> dict:
    if destination.exists() and destination.stat().st_size > 1_000:
        return {"download_status": "cached"}
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 60),
            )
            response.raise_for_status()
            if len(response.content) < 1_000:
                raise ValueError("HTML response is unexpectedly small.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.write_bytes(response.content)
            os.replace(partial, destination)
            return {"download_status": "downloaded"}
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}: {error}") from error


def acquire_external_impact_pages(
    raw_root: str | Path,
    *,
    seasons: tuple[int, ...] = tuple(range(2017, 2025)),
) -> dict:
    root = Path(raw_root)
    records = []
    for season in seasons:
        for source, template, table_id in (
            ("xrapm", XRAPM_URL, "sortableTable"),
            ("basketball_reference_bpm", BPM_URL, "advanced"),
        ):
            url = template.format(season=season)
            destination = root / source / f"season={season}" / "page.html"
            status = _download_html(url, destination)
            html = destination.read_text(encoding="utf-8")
            _table_rows(html, table_id)
            records.append(
                {
                    "source": source,
                    "season": season,
                    "url": url,
                    "path": str(destination.resolve()),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "retrieved_or_verified_at": datetime.now(timezone.utc).isoformat(),
                    **status,
                }
            )
            time.sleep(1.0)
    manifest = {
        "dataset": "external_impact_benchmarks",
        "grain": "source season HTML page",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(seasons),
        "files": records,
    }
    write_json_atomic(manifest, root / "manifest.json")
    return manifest


def _weighted_average(group: pd.DataFrame, column: str) -> float:
    valid = group[column].notna() & group["minutes"].gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(group.loc[valid, column], weights=group.loc[valid, "minutes"]))


def _correlation_rows(frame: pd.DataFrame, scope: str, window_end: int | None) -> list[dict]:
    rows = []
    for external in ("bpm", "xrapm"):
        for component in ("offense", "defense", "net"):
            left = f"prior_{component}_per_100"
            right = f"{external}_{component}"
            valid = frame[[left, right]].dropna()
            if len(valid) < 3:
                continue
            rows.append(
                {
                    "scope": scope,
                    "window_end": window_end,
                    "external_metric": external,
                    "component": component,
                    "matched_player_windows": len(valid),
                    "pearson": float(valid[left].corr(valid[right], method="pearson")),
                    "spearman": float(valid[left].corr(valid[right], method="spearman")),
                }
            )
    return rows


def build_external_impact_benchmark(
    priors_path: str | Path,
    features_path: str | Path,
    names_path: str | Path,
    raw_root: str | Path,
    *,
    artifact_root: str | Path,
    window_ends: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024),
    minimum_window_possessions_per_side: float = 3_000.0,
) -> dict:
    raw = Path(raw_root)
    annual_frames = []
    source_quality = []
    for season in range(min(window_ends) - 2, max(window_ends) + 1):
        x_path = raw / "xrapm" / f"season={season}" / "page.html"
        b_path = raw / "basketball_reference_bpm" / f"season={season}" / "page.html"
        xrapm = parse_xrapm_html(x_path.read_text(encoding="utf-8"), season)
        bpm = parse_bpm_html(b_path.read_text(encoding="utf-8"), season)
        annual = bpm.merge(
            xrapm,
            on=["season", "normalized_name"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        source_quality.append(
            {
                "season": season,
                "bpm_rows": len(bpm),
                "xrapm_rows": len(xrapm),
                "matched_rows": int(annual["_merge"].eq("both").sum()),
                "bpm_match_rate": float(annual["_merge"].eq("both").sum() / len(bpm)),
                "xrapm_match_rate": float(annual["_merge"].eq("both").sum() / len(xrapm)),
            }
        )
        annual_frames.append(annual.drop(columns="_merge"))
    annual = pd.concat(annual_frames, ignore_index=True)

    aggregate_rows = []
    for window_end in window_ends:
        window = annual.loc[annual["season"].between(window_end - 2, window_end)]
        for name, group in window.groupby("normalized_name", sort=False):
            row = {
                "Window_End": window_end,
                "normalized_name": name,
                "external_seasons": int(group["season"].nunique()),
                "external_minutes": float(group["minutes"].sum()),
            }
            for metric in ("bpm", "xrapm"):
                for component in ("offense", "defense", "net"):
                    row[f"{metric}_{component}"] = _weighted_average(
                        group, f"{metric}_{component}"
                    )
            aggregate_rows.append(row)
    aggregates = pd.DataFrame(aggregate_rows)

    priors = pd.read_parquet(priors_path)
    features = pd.read_parquet(features_path)[
        ["PLAYER_ID", "Window_End", "OffPoss", "DefPoss"]
    ]
    names = pd.read_csv(names_path)[["PLAYER_ID", "PLAYER_NAME"]]
    names["normalized_name"] = names["PLAYER_NAME"].map(normalize_player_name)
    ambiguous_names = set(
        names.loc[names["normalized_name"].duplicated(keep=False), "normalized_name"]
    )
    names.loc[names["normalized_name"].isin(ambiguous_names), "normalized_name"] = pd.NA
    matched = (
        priors.loc[priors["Window_End"].isin(window_ends)]
        .merge(features, on=["PLAYER_ID", "Window_End"], validate="one_to_one")
        .merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
        .merge(
            aggregates,
            on=["Window_End", "normalized_name"],
            how="left",
            validate="many_to_one",
        )
    )
    matched["high_exposure"] = matched["OffPoss"].ge(
        minimum_window_possessions_per_side
    ) & matched["DefPoss"].ge(minimum_window_possessions_per_side)

    metric_rows = _correlation_rows(matched, "all_matched", None)
    metric_rows += _correlation_rows(
        matched.loc[matched["high_exposure"]], "high_exposure", None
    )
    for window_end in window_ends:
        metric_rows += _correlation_rows(
            matched.loc[matched["Window_End"].eq(window_end)],
            "window",
            window_end,
        )
    metrics = pd.DataFrame(metric_rows)
    quality = pd.DataFrame(source_quality)
    coverage = (
        matched.groupby("Window_End", as_index=False)
        .agg(
            spm_rows=("PLAYER_ID", "size"),
            bpm_matched=("bpm_net", lambda values: int(values.notna().sum())),
            xrapm_matched=("xrapm_net", lambda values: int(values.notna().sum())),
            high_exposure_rows=("high_exposure", "sum"),
        )
    )
    coverage["bpm_match_rate"] = coverage["bpm_matched"] / coverage["spm_rows"]
    coverage["xrapm_match_rate"] = coverage["xrapm_matched"] / coverage["spm_rows"]

    run_id = f"external_impact_benchmark_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "external_impact_benchmark" / run_id
    output.mkdir(parents=True, exist_ok=False)
    matched.to_parquet(output / "matched_player_windows.parquet", index=False)
    annual.to_parquet(output / "external_annual.parquet", index=False)
    metrics.to_parquet(output / "correlations.parquet", index=False)
    coverage.to_parquet(output / "coverage.parquet", index=False)
    quality.to_parquet(output / "source_quality.parquet", index=False)
    source_files = sorted(raw.glob("*/*/page.html"))
    run = {
        "run_id": run_id,
        "model_family": "external_impact_benchmark",
        "estimand": "association_between_three_season_spm_and_external_metrics",
        "status": "descriptive_benchmark_not_ground_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "window_ends": list(window_ends),
            "aggregation": "NBA-minutes-weighted annual values over T-2 through T",
            "minimum_window_possessions_per_side": minimum_window_possessions_per_side,
            "xrapm_defense_sign": "converted from lower-is-better to positive-good",
            "sources": {"xrapm": XRAPM_URL, "bpm": BPM_URL},
            "source_hashes": {str(path.resolve()): sha256_file(path) for path in source_files},
            "priors_path": str(Path(priors_path).resolve()),
            "features_path": str(Path(features_path).resolve()),
        },
        "quality": {
            "matched_player_windows": int(len(matched)),
            "duplicate_matched_keys": int(matched.duplicated(["PLAYER_ID", "Window_End"]).sum()),
            "minimum_bpm_source_match_rate": float(quality["bpm_match_rate"].min()),
            "minimum_xrapm_source_match_rate": float(quality["xrapm_match_rate"].min()),
            "minimum_spm_to_bpm_match_rate": float(coverage["bpm_match_rate"].min()),
            "minimum_spm_to_xrapm_match_rate": float(coverage["xrapm_match_rate"].min()),
            "ambiguous_name_count": len(ambiguous_names),
        },
        "metrics": metrics.to_dict(orient="records"),
        "caveats": [
            "External metrics are comparators, not ground truth.",
            "SPM is a three-season retrodiction; annual BPM and xRAPM are "
            "minutes-weighted to the same window.",
            "xRAPM is itself a prior-informed RAPM and is therefore not "
            "independent of box and lineup information.",
            "Name-only external joins exclude ambiguous normalized names.",
        ],
        "artifact_path": str(output.resolve()),
    }
    if not math.isfinite(run["quality"]["minimum_spm_to_xrapm_match_rate"]):
        raise ValueError("External benchmark coverage is not finite.")
    write_json_atomic(run, output / "run.json")
    return run
