from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research/audits/predictive_player_skills_2026_v1/decision.json"


def _load_run() -> tuple[dict, Path]:
    decision = json.loads(AUDIT.read_text())
    run_dir = ROOT / "artifacts/models/predictive_player_skills" / decision["run_id"]
    return json.loads((run_dir / "run.json").read_text()), run_dir


def test_frozen_skill_audit_passes_without_future_season() -> None:
    decision = json.loads(AUDIT.read_text())
    run, _ = _load_run()
    assert decision["verdict"] == "pass_research_current_skill"
    assert all(decision["gates"].values())
    assert run["season_policy"]["development_seasons"] == list(range(2019, 2025))
    assert run["season_policy"]["final_parameter_cutoff"] == 2025
    assert run["season_policy"]["output_season"] == 2026
    assert run["season_policy"]["forbidden_season"] == 2027
    assert run["season_2027_loaded"] is False
    assert run["quality"]["skills"] == 34
    assert run["quality"]["current_players"] == 558
    assert run["quality"]["current_complete_players"] == 303
    assert run["quality"]["role_conditional_status"].startswith("skipped_")


def test_frozen_skill_artifacts_are_complete_and_portable() -> None:
    run, run_dir = _load_run()
    serialized = json.dumps(run, allow_nan=False)
    assert str(ROOT) not in serialized
    assert all(not Path(path).is_absolute() for path in run["artifacts"].values())
    assert all(not Path(path).is_absolute() for path in run["source_paths"].values() if isinstance(path, str))

    definitions = pd.read_parquet(run_dir / "skill_definitions.parquet")
    estimates = pd.read_parquet(run_dir / "skill_estimates.parquet")
    decisions = pd.read_parquet(run_dir / "model_selection.parquet")
    assert definitions["key"].is_unique
    assert len(definitions) == 34
    assert not estimates.duplicated(["PLAYER_ID", "Season", "skill"]).any()
    assert estimates["Season"].max() == 2026
    assert 2027 not in set(estimates["Season"])
    current = estimates.loc[estimates["Season"].eq(2026)]
    has_evidence = (
        current["raw_value"].notna() | current["preseason_estimate"].notna()
    )
    assert current.loc[has_evidence, "estimate"].notna().all()
    assert current.loc[~has_evidence, "estimate"].isna().all()
    assert current["percentile"].dropna().between(0, 100).all()
    assert current["last_update_date"].notna().all()
    assert decisions.loc[decisions["selected"]].groupby("skill").size().eq(1).all()
    skipped_roles = decisions.loc[decisions["arm"].eq("role_conditional")]
    assert len(skipped_roles) == 34
    assert skipped_roles["status"].eq("skipped").all()
