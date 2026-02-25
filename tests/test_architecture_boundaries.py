"""Architecture dependency boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path


def _iter_strategy_python_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return sorted((root / "strategies").rglob("*.py"))


def test_strategies_package_does_not_import_research_backtest() -> None:
    violations: list[str] = []

    for path in _iter_strategy_python_files():
        rel_path = path.relative_to(path.parents[1])
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "research.backtest" or alias.name.startswith(
                        "research.backtest."
                    ):
                        violations.append(f"{rel_path}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "research.backtest" or module.startswith("research.backtest."):
                    violations.append(f"{rel_path}:{node.lineno} from {module} import ...")

    assert not violations, "strategies package must not depend on research.backtest:\n" + "\n".join(
        violations
    )
