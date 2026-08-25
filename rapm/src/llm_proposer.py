#!/usr/bin/env python3
"""LLM proposer — writes candidates/gen_NNN/build.py for autoresearch."""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import textwrap
import urllib.request
from pathlib import Path

from autoresearch_proposer import CANDIDATES, code_exists, load_state, next_gen_id, save_state
from candidate_build import apply_candidate_build
from paths import DATA, FEATURES_DIR, PROJECT_ROOT, ensure_dirs

ensure_dirs()

PROPOSALS_LOG = FEATURES_DIR / "llm_proposals.jsonl"
PROMPT_PATH = FEATURES_DIR / "proposer_prompt.md"

SYSTEM_PROMPT = """You are a basketball analytics feature engineer for an NBA SPM prior foundry.

Write Python `build(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` that adds NEW columns
prefixed `new_`. Do NOT use leakage: no OnOffRtg, OnDefRtg, plus-minus, team ratings, or labels Off/Def/RAPM.

Rules:
- Input `feats` is player-window rows from spm_features_windows.parquet (box + tracking per-100 cols).
- Return (augmented DataFrame, list of new column names you created).
- Use only vectorized pandas/numpy. Handle missing columns with if col in df.columns checks.
- Be creative: ratios, interactions, window z-scores, role composites, age curves.
- 5–15 new features max. Simpler beats sprawling.

Output ONLY valid Python for build.py — no markdown fences, no explanation."""


def _gather_context() -> str:
    parts = []
    imp = FEATURES_DIR / "improvement_log.md"
    if imp.exists():
        parts.append("## improvement_log (recent)\n" + imp.read_text()[-4000:])
    tsv = FEATURES_DIR / "results.tsv"
    if tsv.exists():
        parts.append("## results.tsv (last rows)\n" + "\n".join(tsv.read_text().splitlines()[-15:]))
    ex = CANDIDATES / "gen_006" / "build.py"
    if ex.exists():
        parts.append("## example winning build.py (gen_006)\n" + ex.read_text())
    cat = FEATURES_DIR / "prepare.py"
    if cat.exists():
        parts.append("## prepare.py blocklist excerpt\n" + "\n".join(cat.read_text().splitlines()[:55]))
    try:
        import pandas as pd

        cols = pd.read_parquet(DATA / "spm_features_windows.parquet", columns=[]).columns.tolist()
        parts.append("## available parquet columns\n" + ", ".join(cols[:80]))
    except Exception:
        pass
    if PROMPT_PATH.exists():
        parts.append(PROMPT_PATH.read_text())
    return "\n\n".join(parts)


def _extract_python(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.S | re.I)
    if m:
        return m.group(1).strip()
    return raw


def _call_openai(user: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_anthropic(user: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def call_llm(user: str) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return _call_openai(user)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(user)
    raise RuntimeError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in Basketball/New SPM/.env")


def validate_build_py(source: str, gen_id: int) -> list[str]:
    ast.parse(source)
    gen_dir = CANDIDATES / f"gen_{gen_id:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / "build.py"
    header = '"""LLM-proposed features — autoresearch."""\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\n\n'
    if "def build(" not in source:
        raise ValueError("build() function missing")
    if not source.lstrip().startswith('"""') and not source.lstrip().startswith("from"):
        source = header + source
    path.write_text(source, encoding="utf-8")
    _, new_cols = apply_candidate_build(gen_id)
    if not new_cols:
        raise ValueError("build() returned no new columns")
    return new_cols


def propose_llm(code_hint: str | None = None, materialize: bool = True) -> int | None:
    """Generate next candidate via LLM. Returns gen_id or None on failure."""
    gen_id = next_gen_id()
    user = _gather_context()
    user += f"\n\nPropose gen_{gen_id:03d}. Code name slug: llm_{gen_id:03d}."
    if code_hint:
        user += f"\nUser direction: {code_hint}"
    user += "\nWrite build.py only."

    print(f"LLM_PROPOSE gen_{gen_id:03d} ...", flush=True)
    raw = call_llm(user)
    source = _extract_python(raw)
    new_cols = validate_build_py(source, gen_id)

    code = f"llm_{gen_id:03d}"
    if code_exists(code):
        code = f"llm_{gen_id:03d}_{len(new_cols)}feat"

    manifest = {
        "code": code,
        "description": f"LLM-proposed: {len(new_cols)} new features",
        "type": "build",
        "prior": "spmv2_subset",
        "use_columns": "new_only",
        "alpha": 1000,
        "c_grid": [2.0, 4.0],
        "search_folds_only": True,
        "llm": True,
        "feature_preview": new_cols[:15],
    }
    (CANDIDATES / f"gen_{gen_id:03d}" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    record = {"gen": gen_id, "code": code, "n_features": len(new_cols), "features": new_cols}
    with open(PROPOSALS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    state = load_state()
    state.setdefault("materialized", []).append(code)
    state.setdefault("llm_gens", []).append(gen_id)
    save_state(state)
    print(f"LLM_PROPOSED gen_{gen_id:03d} {len(new_cols)} features: {new_cols[:6]}...", flush=True)
    return gen_id


def main() -> None:
    hint = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    try:
        gid = propose_llm(code_hint=hint)
        print(gid or "fail")
    except Exception as e:
        print(f"LLM_PROPOSE_FAIL {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
