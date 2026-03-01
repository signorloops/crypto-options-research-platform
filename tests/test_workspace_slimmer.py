"""Tests for workspace slimming utility."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "workspace_slimmer.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("workspace_slimmer_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load workspace_slimmer module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_build_cleanup_plan_includes_safe_targets_and_excludes_venv_by_default(tmp_path):
    module = _load_module()
    _touch(tmp_path / ".pytest_cache" / "state")
    _touch(tmp_path / "artifacts" / "weekly.md")
    _touch(tmp_path / "pkg" / "__pycache__" / "mod.cpython-311.pyc")
    _touch(tmp_path / "venv" / "bin" / "python")

    plan = module._build_cleanup_plan(
        repo_root=tmp_path,
        include_results=False,
        include_venv=False,
        include_all_worktrees=False,
    )
    plan_paths = {item["path"] for item in plan}

    assert str((tmp_path / ".pytest_cache").resolve()) in plan_paths
    assert str((tmp_path / "artifacts").resolve()) in plan_paths
    assert str((tmp_path / "pkg" / "__pycache__").resolve()) in plan_paths
    assert str((tmp_path / "venv").resolve()) not in plan_paths


def test_build_cleanup_plan_can_include_venv(tmp_path):
    module = _load_module()
    _touch(tmp_path / "venv" / "bin" / "python")

    plan = module._build_cleanup_plan(
        repo_root=tmp_path,
        include_results=False,
        include_venv=True,
        include_all_worktrees=False,
    )
    plan_paths = {item["path"] for item in plan}
    assert str((tmp_path / "venv").resolve()) in plan_paths


def test_build_cleanup_plan_can_include_untracked_results(tmp_path, monkeypatch):
    module = _load_module()
    _touch(tmp_path / "results" / "sample_run.json")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="results/sample_run.json\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    plan = module._build_cleanup_plan(
        repo_root=tmp_path,
        include_results=True,
        include_venv=False,
        include_all_worktrees=False,
    )
    plan_paths = {item["path"] for item in plan}
    assert str((tmp_path / "results" / "sample_run.json").resolve()) in plan_paths


def test_list_worktree_roots_parses_porcelain_output(tmp_path, monkeypatch):
    module = _load_module()
    main_root = (tmp_path / "repo").resolve()
    secondary_root = (tmp_path / "repo-live-next").resolve()
    main_root.mkdir(parents=True)
    secondary_root.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                f"worktree {main_root}\n"
                "HEAD 1111111111111111111111111111111111111111\n"
                "branch refs/heads/master\n"
                f"worktree {secondary_root}\n"
                "HEAD 2222222222222222222222222222222222222222\n"
                "branch refs/heads/feature\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    roots = module._list_worktree_roots(main_root)

    assert roots == [main_root, secondary_root]


def test_build_cleanup_plan_can_scan_all_worktrees(tmp_path, monkeypatch):
    module = _load_module()
    main_root = (tmp_path / "repo").resolve()
    secondary_root = (tmp_path / "repo-live-next").resolve()
    main_root.mkdir(parents=True)
    secondary_root.mkdir(parents=True)
    _touch(secondary_root / ".pytest_cache" / "state")

    monkeypatch.setattr(module, "_list_worktree_roots", lambda _: [main_root, secondary_root])

    plan_single = module._build_cleanup_plan(
        repo_root=main_root,
        include_results=False,
        include_venv=False,
        include_all_worktrees=False,
    )
    plan_single_paths = {item["path"] for item in plan_single}
    assert str((secondary_root / ".pytest_cache").resolve()) not in plan_single_paths

    plan_all = module._build_cleanup_plan(
        repo_root=main_root,
        include_results=False,
        include_venv=False,
        include_all_worktrees=True,
    )
    plan_all_paths = {item["path"] for item in plan_all}
    assert str((secondary_root / ".pytest_cache").resolve()) in plan_all_paths
