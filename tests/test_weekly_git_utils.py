"""Tests for weekly audit git-log parsing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.governance.weekly_git_utils import (
    collect_recent_changes,
    detect_latest_tag,
    parse_recent_change_entries,
)


def test_parse_recent_change_entries_skips_malformed_rows_and_truncates_hash():
    entries = parse_recent_change_entries(
        "abc123456789\t2026-03-05\tRefactor audit helpers\n"
        "malformed\n"
        "def987654321\t2026-03-06\tTighten arena tests\n"
    )

    assert entries == [
        {
            "date": "2026-03-05",
            "commit": "abc12345",
            "subject": "Refactor audit helpers",
        },
        {
            "date": "2026-03-06",
            "commit": "def98765",
            "subject": "Tighten arena tests",
        },
    ]


def test_collect_recent_changes_uses_runner_and_preserves_shallow_state(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "--is-shallow-repository"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="true\n", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="abc123456789\t2026-03-05\tRefactor audit helpers\n",
            stderr="",
        )

    report = collect_recent_changes(tmp_path, 7, runner=fake_run)

    assert report["executed"] is True
    assert report["shallow"] is True
    assert report["count"] == 1
    assert report["entries"][0]["commit"] == "abc12345"
    assert calls[1][:2] == ["git", "log"]


def test_detect_latest_tag_falls_back_to_head_hash_when_not_tagged(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "describe", "--tags"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no tag")
        return subprocess.CompletedProcess(cmd, 0, stdout="abc12345\n", stderr="")

    marker = detect_latest_tag(Path(tmp_path), runner=fake_run)

    assert marker == {
        "executed": True,
        "tag": "HEAD-abc12345",
        "error": "",
        "source": "commit",
    }
    assert calls[0][:3] == ["git", "describe", "--tags"]
