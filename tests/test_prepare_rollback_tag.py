"""Tests for rollback tag preparation automation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "prepare_rollback_tag.py"
)


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("demo\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "init")


def test_script_creates_annotated_backup_release_tag_and_json_report(tmp_path):
    _init_repo(tmp_path)
    output_json = tmp_path / "artifacts" / "rollback-tag.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["created"] is True
    assert report["reused_existing"] is False
    assert report["tag"].startswith("backup-release-")
    assert report["commit"]
    tag_details = _git(tmp_path, "for-each-ref", f"refs/tags/{report['tag']}", "--format=%(objecttype)")
    assert tag_details.stdout.strip() == "tag"


def test_script_reuses_existing_prefix_tag_on_same_commit(tmp_path):
    _init_repo(tmp_path)
    head_commit = _git(tmp_path, "rev-parse", "--short", "HEAD").stdout.strip()
    tag_name = f"backup-release-20260306-{head_commit}"
    _git(tmp_path, "tag", "-a", tag_name, "-m", "existing rollback baseline")
    output_json = tmp_path / "artifacts" / "rollback-tag.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["created"] is False
    assert report["reused_existing"] is True
    assert report["tag"] == tag_name


def test_script_fails_when_explicit_tag_points_to_other_commit(tmp_path):
    _init_repo(tmp_path)
    _git(tmp_path, "tag", "-a", "backup-release-fixed", "-m", "existing rollback baseline")
    (tmp_path / "next.txt").write_text("next\n", encoding="utf-8")
    _git(tmp_path, "add", "next.txt")
    _git(tmp_path, "commit", "-m", "next")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--tag",
            "backup-release-fixed",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "points to a different commit" in completed.stderr
