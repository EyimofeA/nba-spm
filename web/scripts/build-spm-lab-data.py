"""Build the localhost-only SPM feature-research payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "web/local-data/spm-lab.json"


def latest_complete_run() -> Path:
    runs = sorted(
        (ROOT / "artifacts/models/five_year_spm_feature_research").glob(
            "five_year_spm_feature_research_v1_*/run.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    complete = [path.parent for path in runs if (path.parent / "aio_metrics.parquet").exists()]
    if not complete:
        raise FileNotFoundError("No complete five-year SPM feature-research run exists.")
    return complete[-1]


def clean(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def rating_rows(run_path: Path) -> list[dict]:
    spm = pd.read_parquet(run_path / "spm_predictions.parquet")
    aio = pd.read_parquet(run_path / "aio_ratings.parquet")
    active = aio.loc[
        aio["variant"].eq("selected_combined")
        & aio["Poss_Off"].gt(0)
        & aio["Poss_Def"].gt(0),
        ["PLAYER_ID", "rating_season", "PLAYER_NAME"],
    ].rename(columns={"rating_season": "Season"})
    active = active.dropna(subset=["PLAYER_NAME"]).drop_duplicates(
        ["PLAYER_ID", "Season"], keep="last"
    )
    outputs = []
    for metric, frame, columns in (
        (
            "spm",
            spm,
            {
                "prior_offense_per_100": "offense",
                "prior_defense_per_100": "defense",
                "prior_net_per_100": "net",
            },
        ),
        ("aio", aio, {"offense": "offense", "defense": "defense", "net": "net"}),
    ):
        scope = frame.loc[frame["variant"].isin(("baseline", "selected_combined"))].copy()
        scope = scope.rename(columns={"Window_End": "Season", "rating_season": "Season"})
        if "PLAYER_NAME" in scope:
            scope = scope.drop(columns="PLAYER_NAME")
        scope = scope.merge(
            active, on=["PLAYER_ID", "Season"], how="inner", validate="many_to_one"
        )
        selected = scope.loc[scope["variant"].eq("selected_combined"), [
            "PLAYER_ID", "Season", "PLAYER_NAME", *columns
        ]].rename(columns={source: f"selected_{target}" for source, target in columns.items()})
        baseline = scope.loc[scope["variant"].eq("baseline"), [
            "PLAYER_ID", "Season", *columns
        ]].rename(columns={source: f"baseline_{target}" for source, target in columns.items()})
        merged = selected.merge(baseline, on=["PLAYER_ID", "Season"], validate="one_to_one")
        merged["metric"] = metric
        for side in ("offense", "defense", "net"):
            merged[f"delta_{side}"] = merged[f"selected_{side}"] - merged[f"baseline_{side}"]
        outputs.append(merged)
    return clean(pd.concat(outputs, ignore_index=True))


def build(run_path: Path) -> dict:
    manifest = json.loads((run_path / "run.json").read_text())
    decisions = pd.read_parquet(run_path / "feature_group_decisions.parquet")
    aio = pd.read_parquet(run_path / "aio_metrics.parquet")
    wide = aio.pivot(index="test_season", columns="variant", values=["margin_rmse", "margin_correlation"])
    validation = pd.DataFrame(
        {
            "test_season": wide.index.astype(int),
            "baseline_rmse": wide[("margin_rmse", "baseline")].to_numpy(),
            "selected_rmse": wide[("margin_rmse", "selected_combined")].to_numpy(),
            "rmse_delta": (
                wide[("margin_rmse", "selected_combined")]
                - wide[("margin_rmse", "baseline")]
            ).to_numpy(),
            "baseline_correlation": wide[("margin_correlation", "baseline")].to_numpy(),
            "selected_correlation": wide[("margin_correlation", "selected_combined")].to_numpy(),
        }
    )
    payload = {
        "run_id": manifest["run_id"],
        "scope": "localhost_only",
        "seasons": [2021, 2022, 2023, 2024, 2025, 2026],
        "stabilization": manifest["stabilization_contract"],
        "selection_gate": manifest["selection_gate"],
        "decisions": clean(decisions),
        "validation": clean(validation),
        "ratings": rating_rows(run_path),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    payload = build(args.run or latest_complete_run())
    print(json.dumps({"run_id": payload["run_id"], "rows": len(payload["ratings"])}, indent=2))


if __name__ == "__main__":
    main()
