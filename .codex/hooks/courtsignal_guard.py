#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


COMPLETION = re.compile(r"\b(done|complete(?:d)?|finish(?:ed)?|implemented|fixed|ready|shipped)\b", re.I)
FAST_TESTS = (
    "uv",
    "run",
    "pytest",
    "-q",
    "tests/test_repository_boundaries.py",
    "tests/test_research_control_plane.py",
)


def _run(command: tuple[str, ...], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def _root(cwd: str) -> Path | None:
    result = _run(("git", "rev-parse", "--show-toplevel"), Path(cwd), timeout=5)
    if result.returncode:
        return None
    root = Path(result.stdout.strip())
    return root if (root / "src" / "nba_impact").is_dir() else None


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _session_start(root: Path) -> None:
    branch = _run(("git", "branch", "--show-current"), root, timeout=5).stdout.strip() or "detached"
    status = _run(("git", "status", "--porcelain=v1"), root, timeout=5).stdout.splitlines()
    untracked = sum(line.startswith("??") for line in status)
    tracked = len(status) - untracked
    context = (
        f"CourtSignal session: branch {branch}; {tracked} tracked and {untracked} untracked dirty paths. "
        "Read AGENTS.md and ROADMAP.md before edits. Preserve unrelated changes and stage explicit files only. "
        "Do not download, fit, deploy, or promote a research artifact unless the active user goal explicitly requires it."
    )
    _emit({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}})


def _stop(root: Path, event: dict[str, object]) -> None:
    if event.get("stop_hook_active") or not COMPLETION.search(str(event.get("last_assistant_message") or "")):
        _emit({"continue": True})
        return

    failures: list[str] = []
    for command in (("git", "diff", "--check"), FAST_TESTS):
        result = _run(command, root)
        if result.returncode:
            output = (result.stdout + result.stderr).strip()[-2000:]
            failures.append(f"{' '.join(command)}\n{output}")
    if failures:
        _emit({"decision": "block", "reason": "CourtSignal completion guard failed:\n\n" + "\n\n".join(failures)})
        return
    _emit({"continue": True})


def _self_test() -> None:
    assert COMPLETION.search("Implemented and ready.")
    assert not COMPLETION.search("I am still investigating.")
    assert FAST_TESTS[-1] == "tests/test_research_control_plane.py"


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        _self_test()
        return
    event = json.load(sys.stdin)
    root = _root(str(event.get("cwd") or "."))
    if root is None:
        _emit({"continue": True})
    elif event.get("hook_event_name") == "SessionStart":
        _session_start(root)
    elif event.get("hook_event_name") == "Stop":
        _stop(root, event)
    else:
        _emit({"continue": True})


if __name__ == "__main__":
    main()
