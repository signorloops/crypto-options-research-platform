#!/usr/bin/env python3
"""Workspace slimming utility: report and optionally clean generated artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_DIR_TARGETS = [
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "artifacts",
    "logs",
    "htmlcov",
]
VENV_DIR_TARGETS = ["venv", ".venv", "env"]
SAFE_FILE_GLOB_TARGETS = [".coverage", ".coverage.*"]


@dataclass
class CleanupItem:
    path: str
    kind: str
    bytes: int


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for sub in path.rglob("*"):
        if sub.is_file():
            total += sub.stat().st_size
    return total


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    suffixes = ["KB", "MB", "GB", "TB"]
    size = float(value)
    for suffix in suffixes:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {suffix}"
    return f"{size:.2f} PB"


def _list_untracked_results(repo_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "results"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    paths: list[Path] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        candidate = (repo_root / text).resolve()
        if candidate.exists():
            paths.append(candidate)
    return paths


def _list_macos_metadata_files(repo_root: Path) -> list[Path]:
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(repo_root):
        # Never traverse into git internals.
        dirnames[:] = [name for name in dirnames if name != ".git"]
        if ".DS_Store" in filenames:
            matches.append((Path(current_root) / ".DS_Store").resolve())
    return matches


def _list_worktree_roots(repo_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return [repo_root.resolve()]

    roots: list[Path] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        raw_path = line.split(" ", 1)[1].strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).resolve()
        if candidate.exists():
            roots.append(candidate)

    if not roots:
        return [repo_root.resolve()]

    dedup: dict[str, Path] = {}
    for item in roots:
        dedup[str(item)] = item
    return list(dedup.values())


def _build_cleanup_plan(
    *,
    repo_root: Path,
    include_results: bool,
    include_venv: bool,
    include_all_worktrees: bool,
) -> list[dict[str, Any]]:
    plan: list[CleanupItem] = []
    roots = (
        _list_worktree_roots(repo_root)
        if include_all_worktrees
        else [repo_root.resolve()]
    )

    for root in roots:
        for name in SAFE_DIR_TARGETS:
            target = (root / name).resolve()
            if target.exists():
                plan.append(CleanupItem(path=str(target), kind="dir", bytes=_path_size(target)))

        for pattern in SAFE_FILE_GLOB_TARGETS:
            for target in root.glob(pattern):
                resolved = target.resolve()
                if resolved.is_file():
                    plan.append(
                        CleanupItem(path=str(resolved), kind="file", bytes=resolved.stat().st_size)
                    )

        for metadata_file in _list_macos_metadata_files(root):
            if metadata_file.is_file():
                plan.append(
                    CleanupItem(
                        path=str(metadata_file),
                        kind="file",
                        bytes=metadata_file.stat().st_size,
                    )
                )

        for pycache in root.rglob("__pycache__"):
            if pycache.is_dir():
                plan.append(
                    CleanupItem(path=str(pycache.resolve()), kind="dir", bytes=_path_size(pycache))
                )

        if include_venv:
            for name in VENV_DIR_TARGETS:
                target = (root / name).resolve()
                if target.exists():
                    plan.append(
                        CleanupItem(path=str(target), kind="dir", bytes=_path_size(target))
                    )

        if include_results:
            for file_path in _list_untracked_results(root):
                if file_path.is_file():
                    plan.append(
                        CleanupItem(path=str(file_path), kind="file", bytes=file_path.stat().st_size)
                    )

    dedup: dict[str, CleanupItem] = {}
    for item in plan:
        dedup[item.path] = item

    rows = sorted(dedup.values(), key=lambda item: (item.kind, item.path))
    return [{"path": row.path, "kind": row.kind, "bytes": row.bytes} for row in rows]


def _execute_cleanup(plan: list[dict[str, Any]]) -> int:
    removed = 0
    for item in plan:
        path = Path(item["path"]).resolve()
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Report or clean workspace bloat safely.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include untracked files under results/ in cleanup plan.",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Include local virtual environment directories (venv/.venv/env).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup plan. Default is dry-run report only.",
    )
    parser.add_argument(
        "--all-worktrees",
        action="store_true",
        help="Scan all worktrees in the same repository, not only --root.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    plan = _build_cleanup_plan(
        repo_root=root,
        include_results=args.include_results,
        include_venv=args.include_venv,
        include_all_worktrees=args.all_worktrees,
    )

    total_bytes = sum(int(item["bytes"]) for item in plan)
    print(f"Workspace slim plan items: {len(plan)}")
    print(f"Estimated reclaimable size: {_format_bytes(total_bytes)}")
    for item in plan:
        print(f"- {item['kind']}: {item['path']} ({_format_bytes(int(item['bytes']))})")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to execute cleanup.")
        return 0

    removed = _execute_cleanup(plan)
    print(f"Cleanup applied. Removed {removed} item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
