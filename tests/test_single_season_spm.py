from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_impact.models.single_season_spm import fit_single_season_spm


def _external_pages(players: list[str], season: int) -> tuple[str, str]:
    bpm_rows = "".join(
        f"<tr><td>{name}</td><td>AAA</td><td>{1000 + 10 * index}</td>"
        f"<td>{index - 1:.1f}</td><td>{1 - index / 2:.1f}</td>"
        f"<td>{index / 2:.1f}</td></tr>"
        for index, name in enumerate(players)
    )
    xrapm_rows = "".join(
        f"<td>{name}</td><td>{index - 1:.1f} (50)</td>"
        f"<td>{index / 2 - 1:.1f} (50)</td>"
        f"<td>{index / 2:.1f} (50)</td></tr>"
        for index, name in enumerate(players)
    )
    bpm = (
        '<table id="advanced"><tr><th>Player</th><th>Team</th><th>MP</th>'
        "<th>OBPM</th><th>DBPM</th><th>BPM</th></tr>"
        f"{bpm_rows}</table>"
    )
    xrapm = (
        '<table id="sortableTable"><tr><th>Player</th><th>Offense</th>'
        f"<th>Defense(*)</th><th>Total</th></tr>{xrapm_rows}</table>"
    )
    return bpm, xrapm


def test_single_season_spm_builds_oof_and_final_outputs(tmp_path: Path) -> None:
    players = ["Player One", "Player Two", "Player Three", "Player Four"]
    features = []
    targets = []
    for season in (2017, 2018, 2019, 2020):
        for player_id, name in enumerate(players, start=1):
            signal = player_id - 2.5 + 0.1 * (season - 2017)
            features.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": season,
                    "OffPoss": 2000.0,
                    "DefPoss": 2000.0,
                    "f1": signal,
                    "f2": signal**2,
                }
            )
            targets.append(
                {
                    "PLAYER_ID": player_id,
                    "Season": season,
                    "target_offense": signal,
                    "target_defense": 0.5 * signal,
                    "target_net": 1.5 * signal,
                    "Poss_Off": 2000.0,
                    "Poss_Def": 2000.0,
                }
            )
        bpm, xrapm = _external_pages(players, season)
        for source, html in (
            ("basketball_reference_bpm", bpm),
            ("xrapm", xrapm),
        ):
            page = tmp_path / "raw" / source / f"season={season}" / "page.html"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(html)

    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.parquet"
    names_path = tmp_path / "names.csv"
    pd.DataFrame(features).to_parquet(features_path, index=False)
    pd.DataFrame(targets).to_parquet(targets_path, index=False)
    pd.DataFrame(
        {"PLAYER_ID": range(1, 5), "PLAYER_NAME": players}
    ).to_csv(names_path, index=False)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "run.json").write_text(
        json.dumps(
            {
                "selected_features": {
                    "offense": ["f1", "f2", "f1_trend"],
                    "defense": ["f1"],
                }
            }
        )
    )

    run = fit_single_season_spm(
        features_path,
        targets_path,
        reference,
        names_path,
        tmp_path / "raw",
        artifact_root=tmp_path,
        output_seasons=(2017, 2018, 2019, 2020),
    )
    output = Path(run["artifact_path"])
    oof = pd.read_parquet(output / "oof_predictions.parquet")
    leaderboard = pd.read_parquet(output / "leaderboard.parquet")
    disagreement = pd.read_parquet(output / "defensive_disagreements.parquet")
    assert len(oof) == 16
    assert len(leaderboard) == 16
    assert len(disagreement) == 16
    assert not oof.duplicated(["PLAYER_ID", "Season"]).any()
    assert run["models"]["offense"]["features"] == ["f1", "f2"]
    assert run["quality"]["xrapm_matched_rows"] == 16
    assert run["quality"]["high_exposure_xrapm_matched_rows"] == 16
    assert run["quality"]["nonfinite_prediction_values"] == 0
    assert "season_2017_high_exposure" in {
        row["scope"] for row in run["metrics"]["external"]
    }
    assert disagreement["high_exposure"].all()
