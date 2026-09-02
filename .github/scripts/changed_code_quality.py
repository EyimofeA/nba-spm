#!/usr/bin/env python3
"""Reject new source bloat without failing on unchanged legacy debt."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

MAX_LINES = 500
MAX_COMPLEXITY = 22
BASELINE_PATH = Path(".github/quality-baseline.json")
SOURCE_ROOTS = (
    ".github/scripts/",
    "research/",
    "src/",
    "tests/",
    "web/app/",
    "web/scripts/",
    "web/tests/",
)
SOURCE_SUFFIXES = {".cjs", ".js", ".mjs", ".py", ".ts", ".tsx"}
EXPLICIT_ANY = re.compile(r"(?::|\bas\b|<)\s*any\b|\bArray\s*<\s*any\s*>")


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], check=check, capture_output=True, text=True
    )
    return result.stdout


def source_path(path: str) -> bool:
    return path.endswith(tuple(SOURCE_SUFFIXES)) and path.startswith(SOURCE_ROOTS)


def changed_paths(base: str) -> list[str]:
    paths: set[str] = set()
    commands = [
        ("diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"),
        ("diff", "--name-only", "--diff-filter=ACMR"),
        ("ls-files", "--others", "--exclude-standard"),
    ]
    for command in commands:
        paths.update(git(*command).splitlines())
    return sorted(path for path in paths if source_path(path) and Path(path).is_file())


def old_source(base: str, path: str) -> str:
    return git("show", f"{base}:{path}", check=False)


class Complexity(ast.NodeVisitor):
    def __init__(self) -> None:
        self.value = 1

    def visit_If(self, node: ast.If) -> None:
        self.value += 1
        self.generic_visit(node)

    visit_IfExp = visit_If
    visit_For = visit_If
    visit_AsyncFor = visit_If
    visit_While = visit_If
    visit_ExceptHandler = visit_If

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.value += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.value += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.value += max(0, len(node.cases) - 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


class Functions(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.values: dict[str, int] = {}

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = ".".join([*self.names, node.name])
        visitor = Complexity()
        for statement in node.body:
            visitor.visit(statement)
        self.values[name] = visitor.value
        self.names.append(node.name)
        self.generic_visit(node)
        self.names.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        self.generic_visit(node)
        self.names.pop()


def parse_python(source: str) -> tuple[dict[str, int], int]:
    if not source:
        return {}, 0
    tree = ast.parse(source)
    functions = Functions()
    functions.visit(tree)
    explicit_any = sum(
        isinstance(node, ast.Name) and node.id == "Any"
        or isinstance(node, ast.Attribute) and node.attr == "Any"
        or isinstance(node, ast.alias) and node.name == "Any"
        for node in ast.walk(tree)
    )
    return functions.values, explicit_any


def any_count(path: str, source: str) -> int:
    if path.endswith(".py"):
        return parse_python(source)[1]
    return len(EXPLICIT_ANY.findall(source))


def self_test() -> None:
    functions, explicit_any = parse_python(
        "from typing import Any\n"
        "def example(value: Any):\n"
        "    return value if value and value.ok else None\n"
    )
    assert functions == {"example": 3}
    assert explicit_any == 2
    assert any_count("example.ts", "const value: any = input as any") == 2


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Changed-code quality gate self-test passed.")
        return 0
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD^"
    baseline = json.loads(BASELINE_PATH.read_text())
    failures: list[str] = []
    paths = changed_paths(base)
    for path in paths:
        current = Path(path).read_text(encoding="utf-8")
        previous = old_source(base, path)
        current_lines = len(current.splitlines())
        previous_lines = max(
            len(previous.splitlines()), baseline["lines"].get(path, 0)
        )
        if current_lines > MAX_LINES and current_lines > previous_lines:
            failures.append(
                f"{path}: {current_lines} lines; limit is {MAX_LINES} and the old file had {previous_lines}"
            )
        previous_any = max(
            any_count(path, previous), baseline["explicit_any"].get(path, 0)
        )
        if any_count(path, current) > previous_any:
            failures.append(f"{path}: adds an explicit Any type")
        if path.endswith(".py"):
            current_functions, _ = parse_python(current)
            previous_functions, _ = parse_python(previous)
            for name, value in current_functions.items():
                old_value = max(
                    previous_functions.get(name, 0),
                    baseline["complexity"].get(f"{path}:{name}", 0),
                )
                if value > MAX_COMPLEXITY and value > old_value:
                    failures.append(
                        f"{path}:{name}: cyclomatic complexity {value}; limit is {MAX_COMPLEXITY}"
                    )
    if failures:
        print("Changed-code quality gate failed:", *failures, sep="\n- ")
        return 1
    print(f"Changed-code quality gate passed for {len(paths)} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
