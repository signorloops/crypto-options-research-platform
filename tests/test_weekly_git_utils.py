"""Tests for weekly audit git-log parsing helpers."""

from __future__ import annotations

from scripts.governance.weekly_git_utils import parse_recent_change_entries


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
