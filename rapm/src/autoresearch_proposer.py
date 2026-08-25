#!/usr/bin/env python3
"""Autoresearch proposer — creates next gen_NNN/build.py from templates + mutations."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from paths import FEATURES_DIR, ensure_dirs

ensure_dirs()
CANDIDATES = FEATURES_DIR / "candidates"
STATE = FEATURES_DIR / "autoresearch_state.json"

# Ordered queue of built-in proposals (each must have build.py after materialize).
PROPOSAL_TEMPLATES: list[dict] = [
    {
        "code": "tracking_interactions_v1",
        "description": "NEW: tier2 tracking ratios (drives/touches/passing interactions)",
        "type": "build",
        "use_columns": "new_only",
        "alpha": 1000,
        "c_grid": [2.0],
        "search_folds_only": True,
        "build_template": "tracking_interactions",
    },
    {
        "code": "derived_box_v2_residual",
        "description": "NEW: gen006 derived features + minutes residual SPM",
        "type": "build",
        "use_columns": "new_only",
        "alpha": 1000,
        "c_grid": [2.0],
        "search_folds_only": True,
        "residual": True,
        "build_template": "derived_box_v1",
    },
    {
        "code": "playtype_ppp_v1",
        "description": "NEW: playtype PPP from staging",
        "type": "build",
        "use_columns": "new_only",
        "staging_run": "curator_20260703_1459",
        "alpha": 1000,
        "c_grid": [2.0],
        "search_folds_only": True,
        "build_template": "playtype_ppp",
    },
    {
        "code": "box_tracking_blend_v1",
        "description": "NEW: blend derived box + tracking interaction features",
        "type": "build",
        "use_columns": "new_only",
        "alpha": 1000,
        "c_grid": [2.0, 4.0],
        "search_folds_only": True,
        "build_template": "box_tracking_blend",
    },
]

BUILD_TEMPLATES: dict[str, str] = {
    "derived_box_v1": "gen_006/build.py",
    "playtype_ppp": "gen_007/build.py",
    "tracking_interactions": "gen_008/build.py",
    "box_tracking_blend": "gen_009/build.py",
}


def list_gen_ids() -> list[int]:
    out = []
    for p in CANDIDATES.glob("gen_*"):
        m = re.match(r"gen_(\d+)$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def next_gen_id() -> int:
    ids = list_gen_ids()
    return (max(ids) if ids else 0) + 1


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"materialized": [], "completed": []}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def materialize_gen(gen_id: int, proposal: dict) -> Path:
    """Write manifest + copy/synthesize build.py for a proposal."""
    gen_dir = CANDIDATES / f"gen_{gen_id:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in proposal.items() if k != "build_template"}
    (gen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tpl = proposal.get("build_template")
    if tpl and tpl in BUILD_TEMPLATES:
        src = CANDIDATES / BUILD_TEMPLATES[tpl]
        if src.exists():
            shutil.copy2(src, gen_dir / "build.py")
    elif not (gen_dir / "build.py").exists():
        raise FileNotFoundError(f"No build.py for gen_{gen_id:03d} template={tpl}")
    return gen_dir


def code_exists(code: str) -> bool:
    for p in CANDIDATES.glob("gen_*/manifest.json"):
        try:
            if json.loads(p.read_text()).get("code") == code:
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def propose_next(materialize: bool = True) -> int | None:
    """Return next gen_id to run; optionally create candidate folder."""
    state = load_state()
    for prop in PROPOSAL_TEMPLATES:
        code = prop["code"]
        if code in state["materialized"] or code_exists(code):
            if code not in state["materialized"]:
                state["materialized"].append(code)
                save_state(state)
            continue
        gen_id = next_gen_id()
        if materialize:
            materialize_gen(gen_id, prop)
            state["materialized"].append(code)
            state.setdefault("gen_by_code", {})[code] = gen_id
            save_state(state)
            print(f"PROPOSED gen_{gen_id:03d} code={code}", flush=True)
        return gen_id

    # Template queue empty — try LLM proposer
    try:
        from llm_proposer import propose_llm

        if materialize:
            gen_id = propose_llm()
            if gen_id is not None:
                print(f"PROPOSED_LLM gen_{gen_id:03d}", flush=True)
            return gen_id
        return next_gen_id()
    except Exception as e:
        print(f"LLM_PROPOSE_SKIP {e}", flush=True)
        return None


def pending_gens(outputs_done_dir: Path) -> list[int]:
    """Gens with build.py that lack foundry .done marker."""
    pending = []
    for gid in list_gen_ids():
        done = outputs_done_dir / f"foundry_g{gid}.done"
        manifest = CANDIDATES / f"gen_{gid:03d}" / "manifest.json"
        build = CANDIDATES / f"gen_{gid:03d}" / "build.py"
        if manifest.exists() and build.exists() and not done.exists():
            cfg = json.loads(manifest.read_text())
            if not cfg.get("deprecated"):
                pending.append(gid)
    return sorted(pending)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pending":
        from paths import OUTPUTS

        print(pending_gens(OUTPUTS))
    else:
        gid = propose_next()
        print(gid or "queue_empty")
