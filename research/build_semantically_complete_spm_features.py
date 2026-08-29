#!/usr/bin/env python3
"""Build a 100%-finite SPM panel with metric-specific completion rules."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.feature_completion import complete_selected_feature_panel
from nba_impact.data.full_spm_features import (
    _atomic_parquet,
    build_rolling_five_year_features,
    load_feature_contract,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.playtype_features import (
    _load_box,
    _load_playtypes,
    compute_playtype_features,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = (
    ROOT
    / "artifacts/research/full_feature_2014_2026/panels"
    / "full_spm_features_2014_2026_v1_60323ba959"
)
V2_RUN = (
    ROOT
    / "artifacts/research/full_feature_2014_2026/features/statistical_impact"
    / "statistical_features_v2_aee1c289d1"
)
FEATURE_ROOT = ROOT / "artifacts/research/full_feature_2014_2026/features"
SITE_DATA = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b"
PLAYER_SHEETS = (
    ROOT
    / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
)
OBSERVED_COVERAGE_RUN = (
    ROOT
    / "artifacts/research/full_feature_coverage"
    / "full_feature_coverage_v1_3de4ec8954"
)


def main() -> None:
    annual = pd.read_parquet(BASE_RUN / "annual_features.parquet")
    enriched = pd.read_parquet(V2_RUN / "features.parquet")
    selected = load_feature_contract(BASE_RUN / "run.json")
    strict_playtype_path = (
        FEATURE_ROOT
        / "playtype_impact/playtype_features_v1_182fb7e27a/features.parquet"
    )
    strict_playtype = pd.read_parquet(strict_playtype_path)
    playtype_source = SITE_DATA / "playtype.csv"
    seasons = tuple(range(2014, 2027))
    raw_playtype = _load_playtypes(playtype_source, seasons)
    box, _ = _load_box(PLAYER_SHEETS, seasons)
    loose_playtype = compute_playtype_features(
        box,
        raw_playtype,
        minimum_minutes=0.0,
        minimum_player_playtype_possessions=0.0,
    )
    source_paths = {
        "hustle": SITE_DATA / "hustle.csv",
        "matchup_defense": (
            FEATURE_ROOT
            / "matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet"
        ),
        "dfg": (
            FEATURE_ROOT
            / "observed_defense_dashboards"
            / "observed_defense_dashboards_v1_1a62103de7d7/dfg.csv"
        ),
        "rim_dfg": (
            FEATURE_ROOT
            / "observed_defense_dashboards"
            / "observed_defense_dashboards_v1_1a62103de7d7/rim_dfg.csv"
        ),
    }
    source_keys = {
        family: (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path, low_memory=False)
        )
        for family, path in source_paths.items()
    }
    complete, expanded, ledger, quality = complete_selected_feature_panel(
        annual,
        enriched,
        selected,
        strict_playtype=strict_playtype,
        loose_playtype=loose_playtype,
        source_keys=source_keys,
    )
    observed_below = pd.read_csv(OBSERVED_COVERAGE_RUN / "features_below_99pct.csv")
    selected_union = tuple(
        dict.fromkeys((*selected["offense"], *selected["defense"]))
    )
    completion_map = observed_below.loc[
        observed_below["panel"].eq("annual")
        & observed_below["feature"].isin(selected_union)
    ].merge(
        ledger[
            [
                "feature",
                "completion_method",
                "source_missing_rows",
                "completed_missing_rows",
            ]
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    if completion_map["completion_method"].isna().any():
        raise ValueError("A sub-99% feature lacks a declared completion method.")
    if completion_map["completed_missing_rows"].sum():
        raise ValueError("A sub-99% feature remains incomplete.")
    five_year = build_rolling_five_year_features(
        complete,
        None,
        expanded,
    )
    expanded_union = tuple(
        dict.fromkeys((*expanded["offense"], *expanded["defense"]))
    )
    five_year_missing = int(five_year[list(expanded_union)].isna().sum().sum())
    if five_year_missing:
        raise ValueError(f"Five-year completion left {five_year_missing} missing values.")

    source_hashes = {
        "base_manifest": sha256_file(BASE_RUN / "run.json"),
        "base_annual": sha256_file(BASE_RUN / "annual_features.parquet"),
        "enriched_features": sha256_file(V2_RUN / "features.parquet"),
        "strict_playtype": sha256_file(strict_playtype_path),
        "raw_playtype": sha256_file(playtype_source),
        "completion_code": sha256_file(
            ROOT / "src/nba_impact/data/feature_completion.py"
        ),
        "runner": sha256_file(Path(__file__)),
        "observed_coverage": sha256_file(
            OBSERVED_COVERAGE_RUN / "features_below_99pct.csv"
        ),
        **{family: sha256_file(path) for family, path in source_paths.items()},
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"semantically_complete_spm_features_v1_{identity}"
    output = ROOT / "artifacts/research/complete_feature_coverage" / run_id
    output.mkdir(parents=True, exist_ok=False)
    _atomic_parquet(complete, output / "annual_features.parquet")
    _atomic_parquet(five_year, output / "five_year_features.parquet")
    ledger.to_csv(output / "completion_ledger.csv", index=False)
    completion_map.to_csv(output / "features_below_99pct_completion.csv", index=False)
    run = {
        "run_id": run_id,
        "status": "complete_research_input",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": {
            "offense_count": len(expanded["offense"]),
            "defense_count": len(expanded["defense"]),
            "offense": list(expanded["offense"]),
            "defense": list(expanded["defense"]),
        },
        "quality": {
            **quality,
            "annual_missing_values": 0,
            "five_year_missing_values": five_year_missing,
            "five_year_rows": int(len(five_year)),
            "selected_features_below_99pct_observed": int(
                completion_map["feature"].nunique()
            ),
            "season_2027_rows": int(complete["Window_End"].eq(2027).sum()),
        },
        "completion_contract": {
            "event_rates": "zero when lineup exposure exists",
            "raw_ratios": "same-season empirical-Bayes estimate",
            "level_metrics": "same-season median",
            "missing_centered_source_metrics": "zero plus source-availability field",
            "zts": (
                "use all available playtype rows; otherwise physically valid player "
                "TS minus the season's possession-weighted mean expected TS; replace "
                "invalid TS with the same-season median first"
            ),
        },
        "source_hashes": source_hashes,
        "paths": {
            "annual_features": "annual_features.parquet",
            "five_year_features": "five_year_features.parquet",
            "completion_ledger": "completion_ledger.csv",
            "features_below_99pct_completion": "features_below_99pct_completion.csv",
        },
        "forbidden_interpretation": (
            "Finite model input does not mean every value was directly observed. "
            "The completion ledger preserves that distinction."
        ),
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(run["quality"], indent=2))


if __name__ == "__main__":
    main()
