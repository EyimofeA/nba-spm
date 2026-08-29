#!/usr/bin/env python3
"""Export a compact, named historical RAPM/SPM/AIO bundle for the site."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "historical_impact_web_bundle_v1"
RATINGS_RUN = (
    ROOT
    / "artifacts/research/historical_box15_ratings"
    / "historical_box15_ratings_v1_d65c267829"
)
HISTORICAL_SHEETS = ROOT / "data/lake/bronze/historical_player_sheets/year_totals"
CURRENT_SHEETS = (
    ROOT
    / "data/lake/bronze/gabriel_player_sheets"
    / "revision=54b57cf/year_totals"
)
OUTPUT_ROOT = ROOT / "artifacts/releases/historical_impact_web_bundle"
HISTORICAL_NAME_ALIASES = {
    (471, 1997): "Lionel Simmons",
    (775, 1997): "Melvin Booker",
}


def _names() -> pd.DataFrame:
    frames = []
    for season in range(1997, 2027):
        if season <= 2013:
            source = pd.read_csv(
                HISTORICAL_SHEETS / f"{season}.csv",
                usecols=["PLAYER_ID", "PLAYER_NAME"],
            )
        else:
            source = pd.read_parquet(
                CURRENT_SHEETS / f"{season}.parquet",
                columns=["PLAYER_ID", "PLAYER_NAME"],
            )
        source = source.drop_duplicates("PLAYER_ID")
        source["Season"] = season
        frames.append(source)
    names = pd.concat(frames, ignore_index=True)
    names = pd.concat(
        [
            names,
            pd.DataFrame(
                [
                    {"PLAYER_ID": player_id, "Season": season, "PLAYER_NAME": name}
                    for (player_id, season), name in HISTORICAL_NAME_ALIASES.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    if names.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Player-season names are not unique.")
    return names


def main() -> None:
    rapm = pd.read_parquet(RATINGS_RUN / "rapm_ratings.parquet")
    spm = pd.read_parquet(RATINGS_RUN / "spm_priors.parquet").rename(
        columns={
            "Window_End": "Season",
            "prior_offense_per_100": "spm_offense",
            "prior_defense_per_100": "spm_defense",
            "prior_net_per_100": "spm_net",
            "information_status": "spm_information_status",
        }
    )
    aio = pd.read_parquet(RATINGS_RUN / "aio_ratings.parquet").rename(
        columns={"information_status": "aio_information_status"}
    )
    names = _names()
    panel = rapm.merge(
        spm[
            [
                "PLAYER_ID",
                "Season",
                "spm_offense",
                "spm_defense",
                "spm_net",
                "spm_information_status",
            ]
        ],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).merge(
        aio[
            [
                "PLAYER_ID",
                "Season",
                "aio_offense",
                "aio_defense",
                "aio_net",
                "aio_information_status",
            ]
        ],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).merge(
        names,
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    )
    if panel["PLAYER_NAME"].isna().any():
        missing = panel.loc[panel["PLAYER_NAME"].isna(), ["PLAYER_ID", "Season"]]
        raise ValueError(f"Historical bundle lacks names: {missing.head().to_dict('records')}")
    panel = panel.sort_values(["Season", "aio_net", "rapm_net"], ascending=[True, False, False])

    source_paths = {
        "ratings_run": RATINGS_RUN / "run.json",
        "rapm": RATINGS_RUN / "rapm_ratings.parquet",
        "spm": RATINGS_RUN / "spm_priors.parquet",
        "aio": RATINGS_RUN / "aio_ratings.parquet",
        "exporter": Path(__file__),
        **{
            f"player_names_{season}": (
                HISTORICAL_SHEETS / f"{season}.csv"
                if season <= 2013
                else CURRENT_SHEETS / f"{season}.parquet"
            )
            for season in range(1997, 2027)
        },
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    seasons_root = output / "seasons"
    seasons_root.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "historical_impact_panel.parquet", index=False)

    files = {}
    for season, frame in panel.groupby("Season", sort=True):
        records = []
        for row in frame.itertuples(index=False):
            record = {
                "id": int(row.PLAYER_ID),
                "name": row.PLAYER_NAME,
                "season": int(row.Season),
                "possessions": {
                    "offense": int(row.Poss_Off),
                    "defense": int(row.Poss_Def),
                },
                "rapm": {
                    "offense": float(row.rapm_offense),
                    "defense": float(row.rapm_defense),
                    "net": float(row.rapm_net),
                },
            }
            if pd.notna(row.spm_net):
                record["spm"] = {
                    "offense": float(row.spm_offense),
                    "defense": float(row.spm_defense),
                    "net": float(row.spm_net),
                    "status": row.spm_information_status,
                }
            if pd.notna(row.aio_net):
                record["aio"] = {
                    "offense": float(row.aio_offense),
                    "defense": float(row.aio_defense),
                    "net": float(row.aio_net),
                    "status": row.aio_information_status,
                }
            records.append(record)
        path = seasons_root / f"{int(season)}.json"
        path.write_text(json.dumps(records, separators=(",", ":")))
        files[str(path.relative_to(output))] = {
            "sha256": sha256_file(path),
            "rows": len(records),
            "bytes": path.stat().st_size,
        }

    panel_path = output / "historical_impact_panel.parquet"
    files[panel_path.name] = {
        "sha256": sha256_file(panel_path),
        "rows": len(panel),
        "bytes": panel_path.stat().st_size,
    }
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "local_site_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "rapm": [1997, 2026],
            "spm": [2001, 2026],
            "aio": [2001, 2026],
        },
        "quality": {
            "rows": len(panel),
            "player_name_coverage": float(panel["PLAYER_NAME"].notna().mean()),
            "duplicate_player_seasons": int(panel.duplicated(["PLAYER_ID", "Season"]).sum()),
            "season_2027_loaded": False,
        },
        "manual_name_aliases": [
            {"player_id": player_id, "season": season, "player_name": name}
            for (player_id, season), name in HISTORICAL_NAME_ALIASES.items()
        ],
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashes[name],
            }
            for name, path in source_paths.items()
        },
        "files": files,
        "publication_boundary": (
            "This bundle is ready for local UI integration. It is not copied "
            "into web/public or deployed until the site snapshot contract and "
            "copy are updated."
        ),
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(run["quality"], indent=2))


if __name__ == "__main__":
    main()
