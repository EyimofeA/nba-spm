from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "nba_impact"
FORBIDDEN_ROOTS = {"paths", "rapm", "research", "zts"}


def test_production_package_does_not_import_legacy_projects() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            blocked = roots & FORBIDDEN_ROOTS
            if blocked:
                violations.append(f"{path.relative_to(PACKAGE)} imports {sorted(blocked)}")
    assert not violations, "\n".join(violations)
