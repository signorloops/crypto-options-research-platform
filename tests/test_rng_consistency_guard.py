"""Guardrails for RNG consistency in production code paths."""

from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = ("data", "research", "strategies")


def _iter_target_python_files():
    for dirname in TARGET_DIRS:
        root = REPO_ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            yield path


def _except_type_contains_exception(handler: ast.ExceptHandler) -> bool:
    """Return True when the except clause catches broad `Exception`."""
    node = handler.type
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id == "Exception"
    if isinstance(node, ast.Tuple):
        return any(isinstance(el, ast.Name) and el.id == "Exception" for el in node.elts)
    return False


def test_no_numpy_global_seed_calls_in_production_code():
    pattern = re.compile(r"\bnp\.random\.seed\(")
    offenders: list[str] = []

    for path in _iter_target_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert offenders == [], "Found forbidden np.random.seed usage:\n" + "\n".join(offenders)


def test_no_numpy_module_level_sampling_calls_in_production_code():
    pattern = re.compile(
        r"\bnp\.random\.(?!default_rng\b|Generator\b|SeedSequence\b|RandomState\b)[A-Za-z_]+\("
    )
    offenders: list[str] = []

    for path in _iter_target_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert offenders == [], "Found forbidden direct np.random sampling usage:\n" + "\n".join(offenders)


def test_no_hardcoded_default_rng_seed_in_core_execution_paths():
    pattern = re.compile(r"\bnp\.random\.default_rng\(\s*\d+\s*\)")
    targets = [
        "data/generators/synthetic.py",
        "research/backtest/engine.py",
        "research/execution/almgren_chriss.py",
        "research/risk/var.py",
        "strategies/market_making/ppo_agent.py",
    ]

    offenders: list[str] = []
    for rel in targets:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert offenders == [], "Found hardcoded default_rng integer seeds:\n" + "\n".join(offenders)


def test_no_except_exception_pass_in_production_code():
    offenders: list[str] = []

    for path in _iter_target_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if _except_type_contains_exception(handler) and any(
                    isinstance(stmt, ast.Pass) for stmt in handler.body
                ):
                    offenders.append(f"{rel}:{handler.lineno}")

    assert offenders == [], "Found forbidden 'except Exception: pass' usage:\n" + "\n".join(offenders)
