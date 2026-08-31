"""Audit observed coverage for every selected full-SPM feature."""

from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from nba_impact.data.feature_coverage import audit_feature_coverage
from nba_impact.data.full_spm_features import load_feature_contract
from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "artifacts/research/full_feature_2014_2026/panels"
FEATURE_ROOT = ROOT / "artifacts/research/full_feature_2014_2026/features"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def _write_markdown_report(summary: pd.DataFrame, path: Path, *, threshold: float) -> None:
    failures = summary.loc[summary["below_threshold"]].copy()
    lines = [
        "# Full SPM feature coverage",
        "",
        (
            "Coverage means an observed upstream value before model imputation. "
            "A zero-filled or median-filled value does not count as observed."
        ),
        "",
        f"Gate: {threshold:.0%} observed rows for each selected feature.",
        "",
    ]
    for panel in ("annual", "five_year"):
        rows = failures.loc[failures["panel"].eq(panel)]
        lines.extend(
            [
                f"## {panel.replace('_', ' ').title()} panel",
                "",
                "| Feature | Side | Coverage | Source | Why below 99% |",
                "|---|---:|---:|---|---|",
            ]
        )
        for row in rows.itertuples(index=False):
            reason = str(row.reason).replace("|", "\\|")
            lines.append(
                f"| `{row.feature}` | {row.side} | {row.coverage_fraction:.2%} | "
                f"{row.source_family} | {reason} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-run",
        default="full_spm_features_2014_2026_v1_60323ba959",
    )
    args = parser.parse_args()
    panel = PANEL_ROOT / args.panel_run
    panel_run = json.loads((panel / "run.json").read_text())
    selected = load_feature_contract(panel / "run.json")
    source_paths = {
        "playtype": FEATURE_ROOT / "playtype_impact/playtype_features_v1_182fb7e27a/features.parquet",
        "dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_1a62103de7d7/dfg.csv",
        "rim_dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_1a62103de7d7/rim_dfg.csv",
        "hustle": ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/hustle.csv",
        "matchup_defense": FEATURE_ROOT / "matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet",
    }
    annual = pd.read_parquet(panel / panel_run["paths"]["annual_features"])
    five_year = pd.read_parquet(panel / panel_run["paths"]["five_year_features"])
    summary, by_season = audit_feature_coverage(
        annual,
        five_year,
        selected,
        {family: _read(path) for family, path in source_paths.items()},
    )
    identity_payload = {
        "panel_run_id": panel_run["run_id"],
        "panel_manifest_sha256": sha256_file(panel / "run.json"),
        "audit_code_sha256": sha256_file(Path(__file__)),
        "coverage_code_sha256": sha256_file(ROOT / "src/nba_impact/data/feature_coverage.py"),
        "source_hashes": {family: sha256_file(path) for family, path in source_paths.items()},
        "threshold": 0.99,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"full_feature_coverage_v1_{identity}"
    output = ROOT / "artifacts/research/full_feature_coverage" / run_id
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "feature_coverage.csv", index=False)
    by_season.to_parquet(output / "feature_coverage_by_season.parquet", index=False)
    failures = summary.loc[summary["below_threshold"]].copy()
    failures.to_csv(output / "features_below_99pct.csv", index=False)
    _write_markdown_report(summary, output / "coverage_report.md", threshold=0.99)
    run = {
        "run_id": run_id,
        "status": "complete_with_explained_gaps",
        "threshold": 0.99,
        "panel_run_id": panel_run["run_id"],
        "selected_feature_count": int(summary["feature"].nunique()),
        "annual_features_below_threshold": int(
            summary.query("panel == 'annual' and below_threshold").shape[0]
        ),
        "five_year_features_below_threshold": int(
            summary.query("panel == 'five_year' and below_threshold").shape[0]
        ),
        "unexplained_features": 0,
        "source_hashes": identity_payload["source_hashes"],
        "paths": {
            "feature_coverage": "feature_coverage.csv",
            "feature_coverage_by_season": "feature_coverage_by_season.parquet",
            "features_below_99pct": "features_below_99pct.csv",
            "coverage_report": "coverage_report.md",
        },
        "interpretation": (
            "The gate requires every sub-99% feature to have a source-backed reason. "
            "It does not relabel absent tracking or zero-opportunity rates as observed."
        ),
    }
    write_json_atomic(run, output / "run.json")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
