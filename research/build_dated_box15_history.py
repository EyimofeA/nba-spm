"""Build the dated Box15 player-game ledger used by current-SPM research."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.blocked_game_box_features import (
    BOX_COUNTS,
    _game_id,
    build_player_game_box15_ledger,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "dated_box15_history_v1"


def _contract(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_data_build",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    seasons = tuple(int(value) for value in payload["seasons"])
    if seasons != (2021, 2022, 2023, 2024, 2025, 2026):
        raise ValueError("The first dated Box15 build must remain 2021 through 2026.")
    return json.loads(json.dumps(payload, default=str))


def _game_dates(possessions: pd.DataFrame) -> pd.DataFrame:
    required = {"gameid", "date"}
    if missing := sorted(required - set(possessions.columns)):
        raise ValueError(f"Possessions are missing game-date fields {missing}.")
    games = possessions.loc[:, ["gameid", "date"]].copy()
    games["game_id"] = games.pop("gameid").map(_game_id)
    raw_date = games.pop("date")
    numeric = pd.to_numeric(raw_date, errors="coerce")
    encoded = numeric.notna() & numeric.between(19_000_000, 21_000_000)
    parsed = pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns]")
    parsed.loc[encoded] = pd.to_datetime(
        numeric.loc[encoded].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="raise",
    )
    parsed.loc[~encoded] = pd.to_datetime(raw_date.loc[~encoded], errors="coerce")
    games["game_date"] = parsed.dt.normalize()
    if games.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("Each game must have exactly one date.")
    return games.drop_duplicates("game_id").sort_values("game_id").reset_index(drop=True)


def build_history(contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    source = contract["sources"]
    canonical_dates = pd.read_parquet(
        ROOT / source["canonical_player_games"],
        columns=["game_id", "season_end", "season_type", "game_date"],
    )
    canonical_dates = canonical_dates.loc[
        canonical_dates["season_type"].eq("regular")
        & canonical_dates["season_end"].isin(contract["seasons"]),
        ["game_id", "game_date"],
    ].drop_duplicates()
    canonical_dates["game_id"] = canonical_dates["game_id"].map(_game_id)
    canonical_dates["fallback_game_date"] = pd.to_datetime(
        canonical_dates.pop("game_date"), errors="raise"
    ).dt.normalize()
    if canonical_dates.duplicated("game_id").any():
        raise ValueError("Canonical player games contain conflicting game dates.")
    frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    hashes: dict[str, str] = {}
    for season in contract["seasons"]:
        possession_path = ROOT / source["possession_template"].format(season=season)
        possessions = pd.read_parquet(possession_path)
        ledger, quality = build_player_game_box15_ledger(
            possessions,
            espn_player_box_path=ROOT / source["espn_player_box"],
            nba_player_game_box_path=ROOT / source["nba_player_game_box"],
            gabriel_pbp_root=ROOT / source["gabriel_pbp_root"],
            season=int(season),
        )
        dates = _game_dates(possessions).merge(
            canonical_dates, on="game_id", how="left", validate="one_to_one"
        )
        dates["game_date"] = dates["game_date"].fillna(dates["fallback_game_date"])
        dates = dates.drop(columns="fallback_game_date")
        ledger = ledger.merge(dates, on="game_id", how="left", validate="many_to_one")
        if ledger["game_date"].isna().any():
            raise ValueError(f"Season {season} contains a player-game without a date.")
        ledger["season"] = int(season)
        frames.append(ledger)
        quality_rows.append({"season": int(season), **quality})
        hashes[f"possessions_{season}"] = sha256_file(possession_path)

    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["game_date", "game_id", "PLAYER_ID"], kind="stable")
    if history.duplicated(["season", "game_id", "PLAYER_ID"]).any():
        raise AssertionError("Dated Box15 player-game keys must be unique.")
    if history[list(BOX_COUNTS)].lt(0).any().any():
        raise AssertionError("Dated Box15 counts cannot be negative.")
    if not history.groupby("season").apply(
        lambda rows: rows["OffPoss"].sum() == rows["DefPoss"].sum(),
        include_groups=False,
    ).all():
        raise AssertionError("Offensive and defensive lineup slots must balance.")
    quality_frame = pd.DataFrame(quality_rows).sort_values("season")
    minimum = float(contract["acceptance"]["minimum_player_game_source_join_coverage"])
    if quality_frame["player_game_source_join_coverage"].lt(minimum).any():
        raise ValueError("Player-game source coverage fell below the contract minimum.")
    for field in ("espn_player_box", "nba_player_game_box"):
        hashes[field] = sha256_file(ROOT / source[field])
    hashes["canonical_player_games"] = sha256_file(
        ROOT / source["canonical_player_games"]
    )
    return history.reset_index(drop=True), quality_frame.reset_index(drop=True), hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/dated_box15_history_v1.yml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/research/dated_box15_history",
    )
    args = parser.parse_args()

    contract = _contract(args.contract)
    history, quality, source_hashes = build_history(contract)
    source_hashes["contract"] = sha256_file(args.contract)
    source_hashes["runner"] = sha256_file(Path(__file__))
    identity = hashlib.sha256(
        json.dumps(
            {"contract": contract, "source_hashes": source_hashes},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{identity}"
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    history.to_parquet(output / "player_games.parquet", index=False)
    quality.to_parquet(output / "source_quality.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_data_ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "source_hashes": source_hashes,
        "quality": {
            "seasons": [int(value) for value in contract["seasons"]],
            "games": int(quality["games"].sum()),
            "player_game_rows": int(len(history)),
            "minimum_join_coverage": float(
                quality["player_game_source_join_coverage"].min()
            ),
            "nba_fallback_games": int(quality["nba_fallback_games"].sum()),
        },
        "files": {
            "player_games.parquet": sha256_file(output / "player_games.parquet"),
            "source_quality.parquet": sha256_file(output / "source_quality.parquet"),
        },
        "forbidden_interpretation": "This artifact contains dated inputs, not player ratings.",
    }
    write_json_atomic(manifest, output / "run.json")
    print(output)


if __name__ == "__main__":
    main()
