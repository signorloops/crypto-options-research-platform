"""Shared git parsing helpers for weekly governance scripts."""

from __future__ import annotations


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
