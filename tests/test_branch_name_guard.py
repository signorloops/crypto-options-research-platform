"""Tests for branch name guard behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "branch_name_guard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("branch_name_guard_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load branch_name_guard module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_allows_standard_codex_branch_prefix(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_git_branches", lambda root: ["codex/review-findings"])
    monkeypatch.setattr(sys, "argv", ["branch_name_guard.py"])

    assert module.main() == 0


def test_main_still_blocks_forbidden_token_outside_allowed_prefix(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_git_branches", lambda root: ["feature/codex-review"])
    monkeypatch.setattr(sys, "argv", ["branch_name_guard.py"])

    assert module.main() == 2
