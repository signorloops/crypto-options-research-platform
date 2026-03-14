"""Regression guard for Python 3.9-safe arena imports."""

from importlib import import_module


def test_arena_module_imports_without_runtime_annotation_errors() -> None:
    module = import_module("research.backtest.arena")
    assert hasattr(module, "StrategyArena")
