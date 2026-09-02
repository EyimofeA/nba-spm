"""Fit a methodology-aligned RAPTOR box reconstruction and publish 2014-2026.

FiveThirtyEight published the RAPTOR inputs and final box/on-off blend but not
the fitted box coefficients. This runner learns the missing box mapping from
the official 2014-2022 table, applies it to the same annual feature contract
through 2026, and combines it with CourtSignal's on/off reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players
from sklearn.linear_model import RidgeCV


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "artifacts/research/full_feature_2014_2026/panels/full_spm_features_2014_2026_v1_60323ba959/annual_features.parquet"
SOURCE = ROOT / "research/rapm_lab/data/external/fivethirtyeight_raptor/modern_RAPTOR_by_player.csv"
ONOFF_ROOT = ROOT / "research/rapm_lab/outputs/raptor_onoff_proxy"
SHEETS = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/raptor_reconstruction"
ALPHAS = np.array([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])


def fold_name(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in text if not unicodedata.combining(c)).lower())


def latest_onoff() -> tuple[dict, pd.DataFrame]:
    runs = sorted(ONOFF_ROOT.glob("raptor_onoff_proxy_v1_*/run.json"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("No RAPTOR on/off reconstruction was available")
    path = runs[-1]
    return json.loads(path.read_text()), pd.read_parquet(path.parent / "matches.parquet")


def prepare_features() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(FEATURES).rename(columns={"Window_End": "season"})
    names = {int(row["id"]): row["full_name"] for row in nba_players.get_players()}
    panel["player"] = panel["PLAYER_ID"].map(names)
    panel["name_key"] = panel["player"].map(fold_name)
    teams = []
    for season in sorted(panel["season"].unique()):
        sheet = pd.read_parquet(SHEETS / f"{int(season)}.parquet", columns=["PLAYER_ID", "TEAM_ABBREVIATION"])
        sheet = sheet.drop_duplicates("PLAYER_ID", keep="last")
        sheet["season"] = int(season)
        teams.append(sheet)
    panel = panel.merge(pd.concat(teams, ignore_index=True), on=["PLAYER_ID", "season"], how="left")
    excluded = {"PLAYER_ID", "season", "player", "name_key", "TEAM_ABBREVIATION"}
    feature_columns = [column for column in panel.columns if column not in excluded and pd.api.types.is_numeric_dtype(panel[column])]
    return panel, feature_columns


def main() -> None:
    panel, feature_columns = prepare_features()
    source = pd.read_csv(SOURCE)
    source["name_key"] = source["player_name"].map(fold_name)
    match_panel = (
        panel.loc[panel["player"].notna()]
        .sort_values("OffPoss", ascending=False)
        .drop_duplicates(["season", "name_key"], keep="first")
    )
    training = match_panel.merge(source, on=["season", "name_key"], how="inner", validate="one_to_one")

    train_x = training[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = train_x.median().fillna(0)
    train_x = train_x.fillna(medians)
    means = train_x.mean()
    scales = train_x.std().replace(0, 1).fillna(1)
    train_z = (train_x - means) / scales
    all_z = (panel[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians) - means) / scales
    weights = np.sqrt(training[["OffPoss", "DefPoss"]].min(axis=1).clip(lower=1))

    selected_alphas = {}
    for side in ("offense", "defense"):
        target = training[f"raptor_box_{side}"]
        complete = target.notna()
        model = RidgeCV(alphas=ALPHAS).fit(
            train_z.loc[complete], target.loc[complete], sample_weight=weights.loc[complete]
        )
        panel[f"box_{side}"] = model.predict(all_z)
        selected_alphas[side] = float(model.alpha_)
    panel["box_net"] = panel["box_offense"] + panel["box_defense"]

    onoff_run, onoff = latest_onoff()
    onoff = onoff.rename(columns={
        "proxy_offense": "onoff_offense",
        "proxy_defense": "onoff_defense",
        "proxy_net": "onoff_net",
    })
    panel = panel.merge(
        onoff[["PLAYER_ID", "season", "onoff_offense", "onoff_defense", "onoff_net"]],
        on=["PLAYER_ID", "season"], how="left", validate="one_to_one",
    )
    for side in ("offense", "defense"):
        panel[f"raptor_{side}"] = 0.85 * panel[f"box_{side}"] + 0.21 * panel[f"onoff_{side}"]
    panel["raptor_net"] = panel["raptor_offense"] + panel["raptor_defense"]

    matched = (
        panel.loc[panel["player"].notna()]
        .sort_values("OffPoss", ascending=False)
        .drop_duplicates(["season", "name_key"], keep="first")
        .merge(source, on=["season", "name_key"], how="inner", suffixes=("", "_source"), validate="one_to_one")
    )
    metrics = []
    for family, source_family in (("box", "raptor_box"), ("onoff", "raptor_onoff"), ("raptor", "raptor")):
        for side, source_side in (("offense", "offense"), ("defense", "defense"), ("net", "total")):
            x = matched[f"{family}_{side}"]
            source_key = f"{source_family}_{source_side}"
            if source_key in panel.columns:
                source_key = f"{source_key}_source"
            y = matched[source_key]
            complete = pd.DataFrame({"x": x, "y": y}).dropna()
            metrics.append({
                "family": family,
                "component": side,
                "matched_rows": int(len(complete)),
                "pearson": float(complete["x"].corr(complete["y"])),
                "r_squared": float(1 - ((complete["x"] - complete["y"]) ** 2).sum() / ((complete["y"] - complete["y"].mean()) ** 2).sum()),
                "rmse": float(np.sqrt(np.mean((complete["x"] - complete["y"]) ** 2))),
            })
    metrics_frame = pd.DataFrame(metrics)

    config = {"alphas": ALPHAS.tolist(), "selected_alphas": selected_alphas, "blend": {"box": 0.85, "onoff": 0.21}, "feature_count": len(feature_columns)}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"raptor_reconstruction_v1_{digest}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)
    keep = [
        "PLAYER_ID", "player", "TEAM_ABBREVIATION", "season", "OffPoss", "DefPoss",
        "box_offense", "box_defense", "box_net", "onoff_offense", "onoff_defense", "onoff_net",
        "raptor_offense", "raptor_defense", "raptor_net",
    ]
    panel[keep].to_parquet(output / "reconstructions.parquet", index=False)
    matched.to_parquet(output / "source_matches.parquet", index=False)
    metrics_frame.to_parquet(output / "agreement.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "methodology_aligned_reconstruction",
        "seasons": sorted(panel["season"].astype(int).unique().tolist()),
        "config": config,
        "onoff_run_id": onoff_run["run_id"],
        "metrics": metrics_frame.to_dict("records"),
        "limitations": [
            "FiveThirtyEight did not publish the fitted RAPTOR box coefficients.",
            "The box mapping is fitted to the official 2014-2022 source table.",
            "The final 0.85 box plus 0.21 on/off weights are the published rounded weights.",
        ],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(metrics_frame.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
