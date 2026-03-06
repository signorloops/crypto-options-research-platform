"""Shared git parsing helpers for weekly governance scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable


def parse_recent_change_entries(output: str) -> list[dict[str, str]]:
    """Parse tab-delimited git log rows into compact report entries."""
    entries: list[dict[str, str]] = []
    for raw in output.splitlines():
        parts = raw.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_hash, commit_date, subject = parts
        entries.append({"date": commit_date, "commit": commit_hash[:8], "subject": subject})
    return entries


def _is_shallow_repository(repo_root: Path, runner: Callable = subprocess.run) -> bool:
    completed = runner(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    return completed.stdout.strip().lower() == "true"


def collect_recent_changes(
    repo_root: Path,
    since_days: int,
    *,
    runner: Callable = subprocess.run,
) -> dict[str, Any]:
    shallow = _is_shallow_repository(repo_root, runner)
    completed = runner(
        [
            "git",
            "log",
            f"--since={since_days} days ago",
            "--pretty=format:%H%x09%ad%x09%s",
            "--date=short",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "executed": False,
            "since_days": since_days,
            "entries": [],
            "count": 0,
            "shallow": shallow,
            "error": completed.stderr.strip(),
        }
    entries = parse_recent_change_entries(completed.stdout)
    return {
        "executed": True,
        "since_days": since_days,
        "entries": entries,
        "count": len(entries),
        "shallow": shallow,
        "error": "",
    }


def detect_latest_tag(
    repo_root: Path,
    *,
    runner: Callable = subprocess.run,
) -> dict[str, Any]:
    completed = runner(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return {"executed": True, "tag": completed.stdout.strip(), "error": "", "source": "tag"}

    head_ref = runner(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head_ref.returncode != 0:
        return {"executed": False, "tag": "", "error": completed.stderr.strip(), "source": ""}
    return {
        "executed": True,
        "tag": f"HEAD-{head_ref.stdout.strip()}",
        "error": "",
        "source": "commit",
    }
