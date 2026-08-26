"""Package the 2026 retrospective SPM and predictive AIO research rows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/experiments/courtsignal_2026_research_bundle_v1.yml"
SPM_RUN = ROOT / "artifacts/models/single_season_spm/single_season_spm_v1_47b3bd9b17"
AIO_RUN = (
    ROOT
    / "artifacts/models/predictive_current_aio"
    / "predictive_current_aio_2026_v1_c18e2472ec"
)
OUTPUT_ROOT = ROOT / "artifacts/models/courtsignal_2026_research_bundle"


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("bundle_id") != "courtsignal_2026_research_bundle_v1":
        raise ValueError("Unexpected 2026 research bundle contract.")
    if contract.get("season_2027_policy") != "forbidden_and_not_loaded":
        raise ValueError("The bundle must forbid Season 2027.")
    spm_source = SPM_RUN / "oof_predictions.parquet"
    aio_source = AIO_RUN / "selected_2026_ratings.parquet"
    prior_source = (
        ROOT
        / "artifacts/models/predictive_spm_trajectory_ablation"
        / "predictive_spm_trajectory_ablation_v1_8d310a2ad6"
        / "selected_predictions.parquet"
    )
    hashes = {
        "contract": sha256_file(CONTRACT),
        "spm": sha256_file(spm_source),
        "aio": sha256_file(aio_source),
        "predictive_spm": sha256_file(prior_source),
        "source_code": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    run_id = f"courtsignal_2026_research_bundle_v1_{identity}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)

    spm = pd.read_parquet(spm_source)
    spm = spm.loc[spm["Season"].eq(2026)].copy()
    if spm.empty or spm["PLAYER_ID"].duplicated().any():
        raise ValueError("The 2026 OOF SPM rows are empty or duplicated.")
    spm = spm.rename(
        columns={
            "spm_offense": "retrospective_spm_offense",
            "spm_defense": "retrospective_spm_defense",
            "spm_net": "retrospective_spm_net",
            "Poss_Off": "season_possessions_offense",
            "Poss_Def": "season_possessions_defense",
        }
    )
    aio = pd.read_parquet(aio_source).rename(
        columns={
            "offense": "predictive_aio_offense",
            "defense": "predictive_aio_defense",
            "net": "predictive_aio_net",
            "Poss_Off": "effective_history_possessions_offense",
            "Poss_Def": "effective_history_possessions_defense",
        }
    )
    prior = pd.read_parquet(prior_source)
    prior_ids = set(prior.loc[prior["Target_Season"].eq(2026), "PLAYER_ID"].astype(int))
    columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "season_possessions_offense",
        "season_possessions_defense",
        "retrospective_spm_offense",
        "retrospective_spm_defense",
        "retrospective_spm_net",
    ]
    table = spm[columns].merge(
        aio[
            [
                "PLAYER_ID",
                "predictive_aio_offense",
                "predictive_aio_defense",
                "predictive_aio_net",
                "effective_history_possessions_offense",
                "effective_history_possessions_defense",
            ]
        ],
        on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )
    if (
        table[
            ["predictive_aio_offense", "predictive_aio_defense", "predictive_aio_net"]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Every active 2026 SPM player must resolve to the AIO design.")
    table["predictive_spm_prior_available"] = table["PLAYER_ID"].isin(prior_ids)
    table["season"] = 2026
    table["retrospective_spm_status"] = "research_defense_weak"
    table["predictive_aio_status"] = "research_champion_reused_diagnostic"
    table = table.sort_values("predictive_aio_net", ascending=False).reset_index(
        drop=True
    )
    table.to_parquet(output / "ratings_2026.parquet", index=False)
    run = {
        "run_id": run_id,
        "bundle_id": contract["bundle_id"],
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season": 2026,
        "rows": int(len(table)),
        "predictive_spm_prior_coverage": float(
            table["predictive_spm_prior_available"].mean()
        ),
        "source_hashes": hashes,
        "paths": {"ratings": "ratings_2026.parquet"},
        "component_identity": {
            "retrospective_spm_max_error": float(
                (
                    table["retrospective_spm_net"]
                    - table["retrospective_spm_offense"]
                    - table["retrospective_spm_defense"]
                )
                .abs()
                .max()
            ),
            "predictive_aio_max_error": float(
                (
                    table["predictive_aio_net"]
                    - table["predictive_aio_offense"]
                    - table["predictive_aio_defense"]
                )
                .abs()
                .max()
            ),
        },
        "forbidden_interpretation": (
            "The SPM is retrospective. The AIO is a lineup-conditioned 2026 forecast "
            "with reused diagnostics. Neither has untouched 2027 confirmation."
        ),
    }
    write_json_atomic(run, output / "run.json")
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
