#!/usr/bin/env python3
"""Parse a public PIPM reconstruction and compare it with CourtSignal models.

The source is The Basketball Database, not an original Jacob Goldstein release.
Its 2023-24 page contains zero-valued PIPM fields, so scored comparisons stop at
2022-23.  Exact NBA player IDs are recovered from each player link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pipm_reference_comparison_v1"
SOURCE_ROOT = "https://www.thebasketballdatabase.com"
SEASON_LABELS = tuple(f"{start}-{str(start + 1)[-2:]}" for start in range(2014, 2024))


class _PipmTableParser(HTMLParser):
    """Read table cells without accidentally concatenating hidden rank spans."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, str | None]]] = []
        self._row: list[dict[str, str | None]] | None = None
        self._cell: dict[str, str | None] | None = None
        self._in_body = False

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = {"text": "", "data_order": attributes.get("data-order"), "href": None}
        elif tag == "a" and self._cell is not None:
            self._cell["href"] = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"] = str(self._cell["text"] or "") + data

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self._row is not None:
            self._cell["text"] = " ".join(str(self._cell["text"] or "").split())
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag == "tbody":
            self._in_body = False


def parse_pipm_html(path: Path, season_label: str) -> pd.DataFrame:
    parser = _PipmTableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for cells in parser.rows:
        if len(cells) < 8:
            continue
        match = re.search(r"(\d+)RegularSeasonBoxScore", str(cells[0]["href"] or ""))
        if not match:
            continue
        values = [float(str(cells[index]["data_order"])) for index in range(2, 8)]
        rows.append(
            {
                "PLAYER_ID": int(match.group(1)),
                "PLAYER_NAME": cells[0]["text"],
                "TEAM_ABBREVIATION": cells[1]["text"],
                "minutes": values[0],
                "off_possessions": values[1],
                "pipm_offense": values[2],
                "def_possessions": values[3],
                "pipm_defense": values[4],
                "pipm_net": values[5],
                "season_label": season_label,
                "rating_season": int(season_label[:4]) + 1,
                "source_url": f"{SOURCE_ROOT}/{season_label}RegularSeasonPlayerPIPM.html",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No PIPM rows parsed from {path}")
    # The source emits one row per team stint for traded players.  Build the
    # season value by possession-weighting each side, then restore net=off+def.
    if frame.duplicated(["PLAYER_ID", "rating_season"]).any():
        combined = []
        for (_, _), group in frame.groupby(["PLAYER_ID", "rating_season"], sort=False):
            off_possessions = float(group["off_possessions"].sum())
            def_possessions = float(group["def_possessions"].sum())
            offense = float(
                np.average(group["pipm_offense"], weights=group["off_possessions"])
                if off_possessions > 0
                else group["pipm_offense"].mean()
            )
            defense = float(
                np.average(group["pipm_defense"], weights=group["def_possessions"])
                if def_possessions > 0
                else group["pipm_defense"].mean()
            )
            combined.append(
                {
                    "PLAYER_ID": int(group["PLAYER_ID"].iloc[0]),
                    "PLAYER_NAME": group["PLAYER_NAME"].iloc[0],
                    "TEAM_ABBREVIATION": "/".join(sorted(group["TEAM_ABBREVIATION"].unique())),
                    "minutes": float(group["minutes"].sum()),
                    "off_possessions": off_possessions,
                    "pipm_offense": offense,
                    "def_possessions": def_possessions,
                    "pipm_defense": defense,
                    "pipm_net": offense + defense,
                    "season_label": season_label,
                    "rating_season": int(group["rating_season"].iloc[0]),
                    "source_url": group["source_url"].iloc[0],
                }
            )
        frame = pd.DataFrame(combined)
    if frame.duplicated(["PLAYER_ID", "rating_season"]).any():
        raise ValueError(f"Duplicate player-season keys remain in {path}")
    identity_error = (frame["pipm_offense"] + frame["pipm_defense"] - frame["pipm_net"]).abs()
    if identity_error.max() > 0.011:
        raise ValueError(f"PIPM component identity failed in {path}: {identity_error.max()}")
    return frame


def _component_metrics(matched: pd.DataFrame, model: str, season: int | str) -> list[dict]:
    output = []
    for component in ("offense", "defense", "net"):
        x = matched[f"{component}_{model}"].to_numpy(dtype=float)
        y = matched[f"pipm_{component}"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(y, x, 1)
        output.append(
            {
                "model": model,
                "rating_season": str(season),
                "component": component,
                "players": len(matched),
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
                "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                "mean_difference": float(np.mean(x - y)),
                "model_on_pipm_slope": float(slope),
                "model_on_pipm_intercept": float(intercept),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument(
        "--bakeoff-root",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/aio_prior_bakeoff"
            / "aio_prior_bakeoff_v1_0a3591a402"
        ),
    )
    parser.add_argument("--min-possessions", type=float, default=1000.0)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    source_paths = {label: args.html_root / f"{label}.html" for label in SEASON_LABELS}
    reference = pd.concat(
        [parse_pipm_html(path, label) for label, path in source_paths.items()],
        ignore_index=True,
    )
    reference["min_possessions"] = np.minimum(
        reference["off_possessions"], reference["def_possessions"]
    )
    season_quality = (
        reference.groupby("rating_season", as_index=False)
        .agg(
            rows=("PLAYER_ID", "size"),
            eligible_players=("min_possessions", lambda values: int((values >= args.min_possessions).sum())),
            pipm_net_sd=("pipm_net", lambda values: float(np.std(values))),
            nonzero_pipm_rows=("pipm_net", lambda values: int((values != 0).sum())),
        )
    )

    priors = pd.read_parquet(args.bakeoff_root / "priors.parquet")
    priors = priors.loc[priors["candidate"].eq("box_pipm_style_prior")].rename(
        columns={
            "prior_offense_per_100": "offense_box_prior",
            "prior_defense_per_100": "defense_box_prior",
            "prior_net_per_100": "net_box_prior",
            "Window_End": "rating_season",
        }
    )
    posterior = pd.read_parquet(args.bakeoff_root / "posterior_ratings.parquet")
    posterior = posterior.loc[posterior["candidate"].eq("box_pipm_style_prior")].rename(
        columns={
            "offense": "offense_box_aio",
            "defense": "defense_box_aio",
            "net": "net_box_aio",
            "Poss_Off": "aio_off_possessions",
            "Poss_Def": "aio_def_possessions",
        }
    )
    matched = reference.merge(
        priors[
            ["PLAYER_ID", "rating_season", "offense_box_prior", "defense_box_prior", "net_box_prior"]
        ],
        on=["PLAYER_ID", "rating_season"],
        how="inner",
        validate="one_to_one",
    ).merge(
        posterior[
            ["PLAYER_ID", "rating_season", "offense_box_aio", "defense_box_aio", "net_box_aio", "aio_off_possessions", "aio_def_possessions"]
        ],
        on=["PLAYER_ID", "rating_season"],
        how="inner",
        validate="one_to_one",
    )
    matched["aio_min_possessions"] = np.minimum(
        matched["aio_off_possessions"], matched["aio_def_possessions"]
    )
    matched = matched.loc[
        matched["min_possessions"].ge(args.min_possessions)
        & matched["aio_min_possessions"].ge(args.min_possessions)
        & matched["rating_season"].between(2021, 2023)
    ].copy()
    if matched.empty:
        raise ValueError("No eligible PIPM/model rows matched.")

    comparisons = []
    for season, frame in matched.groupby("rating_season"):
        for model in ("box_prior", "box_aio"):
            comparisons.extend(_component_metrics(frame, model, int(season)))
    for model in ("box_prior", "box_aio"):
        comparisons.extend(_component_metrics(matched, model, "pooled_2021_2023"))
    comparison_frame = pd.DataFrame(comparisons)

    config = {
        "source": "The Basketball Database PIPM reconstruction",
        "source_urls": [f"{SOURCE_ROOT}/{label}RegularSeasonPlayerPIPM.html" for label in SEASON_LABELS],
        "source_hashes": {label: sha256_file(path) for label, path in source_paths.items()},
        "bakeoff_run": args.bakeoff_root.name,
        "bakeoff_hashes": {
            "priors": sha256_file(args.bakeoff_root / "priors.parquet"),
            "posterior_ratings": sha256_file(args.bakeoff_root / "posterior_ratings.parquet"),
        },
        "min_possessions_each_source": args.min_possessions,
        "scored_seasons": [2021, 2022, 2023],
        "runner_hash": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.artifact_root / "research/pipm_reference_comparison" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    reference.to_parquet(output / "reference.parquet", index=False)
    matched.to_parquet(output / "matched_players.parquet", index=False)
    comparison_frame.to_parquet(output / "comparisons.parquet", index=False)
    season_quality.to_parquet(output / "season_quality.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "reference_rows": len(reference),
            "matched_rows": len(matched),
            "matched_seasons": sorted(matched["rating_season"].unique().tolist()),
            "duplicate_reference_keys": int(reference.duplicated(["PLAYER_ID", "rating_season"]).sum()),
            "season_2024_nonzero_rows": int(
                season_quality.loc[season_quality["rating_season"].eq(2024), "nonzero_pipm_rows"].iloc[0]
            ),
        },
        "paths": {
            "reference": "reference.parquet",
            "matched_players": "matched_players.parquet",
            "comparisons": "comparisons.parquet",
            "season_quality": "season_quality.parquet",
        },
        "caveats": [
            "This is a third-party published-formula reconstruction, not verified original Goldstein PIPM data.",
            "The source's 2023-24 page has zero PIPM values and is retained for QA but excluded from scored comparisons.",
            "The source contains unstable extreme values for tiny samples; both source and AIO possession filters are applied.",
            "Correlation measures agreement, not predictive superiority or model validity.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(season_quality.to_string(index=False))
    print(comparison_frame.to_string(index=False))
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
